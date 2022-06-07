import xarray as xr
import pandas as pd
from dscim.menu.simple_storage import EconVars
from dscim.utils.functions import gcms
import time, os, sys
import dask
import dask.array as da
from dask.distributed import Client
import yaml, time, argparse

parser = argparse.ArgumentParser()

# set up dask
dask.config.set(
    {
        "distributed.worker.memory.target": 0.7,
        "distributed.worker.memory.spill": 0.8,
        "distributed.worker.memory.pause": 0.9,
    }
)

client = Client(n_workers=40, memory_limit="9G", threads_per_worker=1)

ECON_ZARR = "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr"

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

# load config
with open(args.config, "r") as stream:
    loaded_config = yaml.safe_load(stream)
    params = loaded_config["sectors"][args.sector]

# get sector paths and variable names
path = Path(params["sector_path"])
histclim = params["histclim"]
delta = params["delta"]

with xr.open_zarr(path, chunks=None)[delta] as ds:
    with xr.open_zarr(ECON_ZARR, chunks=None) as gdppc:

        ce_batch_dims = [i for i in ds.dims]
        ce_batch_coords = {c: ds[c].values for c in ce_batch_dims}
        ce_batch_coords["region"] = [
            i for i in ds.region.values if i in gdppc.region.values
        ]
        ce_shapes = [len(ce_batch_coords[c]) for c in ce_batch_dims]
        ce_chunks = [xr.open_zarr(path).chunks[c][0] for c in ce_batch_dims]
        print(ce_chunks)

template = xr.DataArray(
    da.empty(ce_shapes, chunks=ce_chunks),
    dims=ce_batch_dims,
    coords=ce_batch_coords,
)


def chunk_func(
    damages,
):

    year = damages.year.values
    ssp = damages.ssp.values
    model = damages.model.values
    region = damages.region.values

    gdppc = (
        xr.open_zarr(ECON_ZARR, chunks=None)
        .sel(year=year, ssp=ssp, model=model, region=region, drop=True)
        .gdppc
    )

    # get damages as % of GDPpc
    shares = damages / gdppc

    # find the 1st/99th percentile of damage share
    # across batches and regions
    quantile = shares.quantile([0.01, 0.99], ["batch", "region"])

    # find the equivalent damages
    # if damage share is capped to 1st/99th percentile
    quantdams = quantile * gdppc

    # keep damages that are within cutoff,
    # otherwise replace with capped damages
    damages = xr.where(
        (shares <= quantile.sel(quantile=0.99, drop=True)),
        damages,
        quantdams.sel(quantile=0.99, drop=True),
    )

    damages = xr.where(
        (shares >= quantile.sel(quantile=0.01, drop=True)),
        damages,
        quantdams.sel(quantile=0.01, drop=True),
    )

    return damages


data = xr.open_zarr(path)

for var in [delta, histclim]:
    out = data[var].map_blocks(chunk_func, template=template).rename(var).to_dataset()
    outpath = path.replace(".zarr", "_clipped.zarr")
    out.to_zarr(outpath, mode="a", consolidated=True)
