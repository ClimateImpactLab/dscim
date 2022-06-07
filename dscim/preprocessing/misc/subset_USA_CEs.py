import xarray as xr
import pandas as pd
from p_tqdm import p_map
from itertools import product

eta_rhos = {
    2.0: 0.0,
    1.016010255: 9.149608e-05,
    1.244459066: 0.00197263997,
    1.421158116: 0.00461878399,
    1.567899395: 0.00770271076,
}

ces = ["ce_cc", "ce_no_cc"]
recipes = ["adding_up", "risk_aversion"]
CE_folder = ["/shares/gcp/integration/CE_library_epa_vsl_bc39"]
sectors = ["agriculture", "mortality", "energy", "labor"]
subset_zarr = False


def subset_USA_CE(args):

    ce, recipe, eta, CE_folder, sector = args

    ds = xr.open_zarr(f"{CE_folder}/{sector}/{recipe}_{ce}_eta{eta}.zarr")

    subset = ds.sel(region=[i for i in ds.region.values if "USA" in i])

    for var in subset.variables:
        subset[var].encoding.clear()

    subset.to_zarr(
        f"{CE_folder}/{sector}_USA/{recipe}_{ce}_eta{eta}.zarr",
        consolidated=True,
        mode="w",
    )


# run subsetting of CEs
p_map(
    subset_USA_CE,
    list(product(ces, recipes, eta_rhos.keys(), CE_folder, sectors)),
    num_cpus=20,
)

# subset the zarrs for USA
if subset_zarr == True:
    zarr = xr.open_zarr(
        "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr",
        consolidated=True,
        mode="w",
    )

    zarr = zarr.sel(region=[i for i in zarr.region.values if "USA" in i])

    for var in zarr.variables:
        zarr[var].encoding.clear()

    zarr.to_zarr(
        "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39_USA.zarr",
        consolidated=True,
    )
