import xarray as xr
import pandas as pd
import numpy as np
import os, sys
from numpy.testing import assert_allclose
from p_tqdm import p_map
from itertools import product

USER = os.getenv("USER")
eta_rhos = {
    "2.0": "0.0",
    "1.016010255": "9.149608e-05",
    "1.244459066": "0.00197263997",
    "1.421158116": "0.00461878399",
    "1.567899395": "0.00770271076",
}

USA = True

in_library = "/mnt/CIL_integration/damage_function_library/damage_function_library_ssp"
out_library = "/mnt/CIL_integration/damage_function_library/damage_function_library_rff"
sectors = [
    # "CAMEL_m5_c0.21.5",
    # "AMEL_m5",
    # "coastal_v0.21.5",
    "mortality_v4",
    "mortality_v5",
    # "energy",
    # "labor",
    # "agriculture",
]

if USA == True:
    sectors = [f"{i}_USA" for i in sectors]

if USA == False:

    # ssp GDP for fractionalizing damage functions
    ssp_gdp = (
        xr.open_zarr(
            "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr",
            consolidated=True,
        )
        .sum("region")
        .gdp
    )

    # get RFF data
    rff_gdp = (
        xr.open_dataset(
            f"/shares/gcp/integration/rff/socioeconomics/rff_global_socioeconomics.nc4"
        )
        .sel(region="world", drop=True)
        .gdp
    )
else:

    # ssp GDP for fractionalizing damage functions
    ssp_gdp = (
        xr.open_zarr(
            "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39_USA.zarr",
            consolidated=True,
        )
        .sum("region")
        .gdp
    )

    # get RFF data
    rff_gdp = (
        xr.open_dataset(
            f"/shares/gcp/integration/rff/socioeconomics/rff_USA_socioeconomics.nc4"
        )
        .sel(region="USA", drop=True)
        .gdp
    )

# get global consumption factors to extrapolate damage function
factors = rff_gdp.sel(year=slice(2100, 2300)) / rff_gdp.sel(year=2099)

# get RFF emulator weights
run_id = xr.open_dataset("/shares/gcp/integration/rff2/rffsp_fair_sequence.nc")
weights = (
    xr.open_dataset(
        f"/shares/gcp/integration/rff/damage_function_weights/damage_function_weights3.nc4",
    )
    .sel(rff_sp=run_id.rff_sp, drop=True)
    .value
)


def weight_df(args):

    sector, eta_rho, recipe, disc, file = args

    # get damage function as share of global GDP
    df = (
        xr.open_dataset(
            f"{in_library}/{sector}/{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_{file}.nc4"
        )
        / ssp_gdp
    )

    # pre-2100 weighted fractional damage functions
    rff = (df * weights).sum(["ssp", "model"])

    # save fractional damage function
    # rff.sel(year=slice(2020, 2099)).to_netcdf(
    #     f"{out_library}/{sector}/{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_fractional_{file}.nc4"
    # )

    # recover damage function as dollars instead of fraction
    rff = (rff * rff_gdp).sel(year=slice(2020, 2099))

    # post-2100 weighted damage functions
    post_2100 = rff.sel(year=2099) * factors

    dfs = xr.combine_by_coords([rff, post_2100])

    os.makedirs(f"{out_library}/{sector}", exist_ok=True)
    dfs.to_netcdf(
        f"{out_library}/{sector}/{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_{file}.nc4"
    )


p_map(
    weight_df,
    list(
        product(
            sectors,
            eta_rhos.items(),
            ["adding_up"],
            ["constant"],
            ["damage_function_coefficients"],
        )
    ),
    num_cpus=20,
)

p_map(
    weight_df,
    list(
        product(
            sectors,
            eta_rhos.items(),
            ["risk_aversion"],
            ["euler_ramsey"],
            ["damage_function_coefficients"],
        )
    ),
    num_cpus=20,
)
