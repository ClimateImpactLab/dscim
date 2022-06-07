from math import ceil
from pathlib import Path

import os, sys

USER = os.getenv("USER")
import dask
import dask.array as da
import dask.config
import numpy as np
import xarray as xr

from dask.distributed import Client, progress
from dscim.utils.functions import ce_func, mean_func

import yaml, time, argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--recipe",
    required=True,
    type=str,
    dest="recipe",
    metavar="<recipe>",
    help="Recipe",
)
parser.add_argument(
    "-c", "--ce", required=True, type=str, dest="ce", metavar="<ce>", help="ce"
)
parser.add_argument(
    "-e",
    "--eta",
    required=False,
    type=np.float64,
    dest="eta",
    metavar="<eta>",
    help="eta",
)
parser.add_argument(
    "-s",
    "--sector",
    required=False,
    type=str,
    dest="sector",
    metavar="<sector>",
    help="sector",
)
parser.add_argument(
    "-y",
    "--config",
    required=True,
    type=str,
    dest="config",
    metavar="<config>",
    help="config str",
)
parser.add_argument(
    "-o",
    "--outpath",
    required=True,
    type=str,
    dest="outpath",
    metavar="<outpath>",
    help="outpath str",
)
parser.add_argument("--zero", dest="zero", action="store_true")
parser.add_argument("--no-zero", dest="zero", action="store_false")

args = parser.parse_args()

bottom_coding_gdppc = 39.39265060424805

from dask.distributed import Client

if __name__ == "__ce_calculation__":

    client = Client(n_workers=35, memory_limit="9G", threads_per_worker=1)

with open(args.config, "r") as stream:
    loaded_config = yaml.safe_load(stream)
    params = loaded_config["sectors"][args.sector]

ZARR = Path(params["sector_path"])
histclim = params["histclim"]
delta = params["delta"]

ECON_ZARR = Path(
    "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr"
)

with xr.open_zarr(ZARR, chunks=None)[histclim] as ds:
    with xr.open_zarr(ECON_ZARR, chunks=None) as gdppc:

        ce_batch_dims = [i for i in gdppc.dims] + [
            i for i in ds.dims if i not in gdppc.dims and i != "batch"
        ]
        ce_batch_coords = {c: ds[c].values for c in ce_batch_dims}
        ce_batch_coords["region"] = [
            i for i in gdppc.region.values if i in ce_batch_coords["region"]
        ]
        ce_shapes = [len(ce_batch_coords[c]) for c in ce_batch_dims]
        ce_chunks = [xr.open_zarr(ZARR).chunks[c][0] for c in ce_batch_dims]
        print(ce_chunks)

template = xr.DataArray(
    da.empty(ce_shapes, chunks=ce_chunks),
    dims=ce_batch_dims,
    coords=ce_batch_coords,
)

# convert string eta to float
eta = float(args.eta)
print("Eta is: ", eta)
print(f"{args.config}")
print(f"{args.sector}")
print(args.outpath)


def ce_from_chunk(chunk, filepath, ce_type, bottom_code, histclim, delta, recipe, eta):

    year = chunk.year.values
    ssp = chunk.ssp.values
    model = chunk.model.values

    gdppc = (
        xr.open_zarr(ECON_ZARR, chunks=None)
        .sel(
            year=year, ssp=ssp, model=model, region=ce_batch_coords["region"], drop=True
        )
        .gdppc
    )

    if ce_type == "ce_no_cc":
        if args.zero == True:
            chunk[histclim] = xr.where(chunk[histclim] == 0, 0, 0)
        calculation = gdppc + chunk[histclim].mean("batch") - chunk[histclim]
    elif ce_type == "ce_cc":
        calculation = gdppc - chunk[delta]
    else:
        raise NotImplementedError("Pass 'ce_cc' or 'ce_no_cc' to ce_type.")

    if recipe == "adding_up":
        result = mean_func(
            np.maximum(
                calculation,
                bottom_code,
            ),
            "batch",
        )
    elif recipe == "risk_aversion":
        result = ce_func(
            np.maximum(
                calculation,
                bottom_code,
            ),
            "batch",
            eta=eta,
        )

    return result


other = xr.open_zarr(ZARR)

out = other.map_blocks(
    ce_from_chunk,
    kwargs=dict(
        filepath=ZARR,
        ce_type=args.ce,
        bottom_code=bottom_coding_gdppc,
        histclim=histclim,
        delta=delta,
        eta=eta,
        recipe=args.recipe,
    ),
    template=template,
)

out = out.astype(np.float32).rename(args.ce).to_dataset()

out.attrs["eta"] = args.eta
out.attrs["bottom code"] = bottom_coding_gdppc
out.attrs["histclim=0"] = args.zero
out.attrs["filepath"] = str(ZARR)

out.to_zarr(
    f"{args.outpath}/{args.recipe}_{args.ce}_eta{args.eta}.zarr",
    consolidated=True,
    mode="w",
)