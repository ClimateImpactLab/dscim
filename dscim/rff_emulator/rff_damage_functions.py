import xarray as xr
import pandas as pd
import numpy as np
import os, sys
from numpy.testing import assert_allclose
from p_tqdm import p_map
from itertools import product

# note from kit: in progress still!!

USER = os.getenv("USER")
eta_rhos = {
    "2.0": "0.0",
    "1.016010255": "9.149608e-05",
    "1.244459066": "0.00197263997",
    "1.421158116": "0.00461878399",
    "1.567899395": "0.00770271076",
}

sectors = [
    "CAMEL_m4_c0.21.4",
    "AMEL_m4",
    "coastal_v0.21.4",
    # "mortality_v4",
    # "energy",
    # "labor",
    # "agriculture",
]
USA=True

if USA == True:
    sectors = [f"{i}_USA" for i in sectors]
    ssp_gdp = "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39_USA.zarr"
    rff_gdp = f"/shares/gcp/integration/rff/socioeconomics/rff_USA_socioeconomics.nc4"
else:
    ssp_gdp = "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr"
    rff_gdp = "/shares/gcp/integration/rff/socioeconomics/rff_global_socioeconomics.nc4"
    
def weight_df(
    sector, 
    eta_rho, 
    recipe, 
    disc, 
    file,
    in_library,
    out_library,
    rff_gdp,
    ssp_gdp,
    weights,
):

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
    if fractional==True:
        rff.sel(year=slice(2020, 2099)).to_netcdf(
            f"{out_library}/{sector}/{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_fractional_{file}.nc4"
        )

    # recover damage function as dollars instead of fraction
    rff = (rff * rff_gdp).sel(year=slice(2020, 2099))

    # post-2100 weighted damage functions
    post_2100 = rff.sel(year=2099) * factors

    dfs = xr.combine_by_coords([rff, post_2100])

    os.makedirs(f"{out_library}/{sector}", exist_ok=True)
    dfs.to_netcdf(
        f"{out_library}/{sector}/{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_{file}.nc4"
    )

def rff_damage_functions(
    sectors,
    USA = False,
    ssp_gdp,
    rff_gdp,
    recipes=["adding_up", "risk_aversion"],
    discs=["constant", "euler_ramsey"],
    in_library = "/mnt/CIL_integration/damage_function_library/damage_function_library_ssp",
    out_library = "/mnt/CIL_integration/damage_function_library/damage_function_library_rff",
    runid_path = "/shares/gcp/integration/rff2/rffsp_fair_sequence.nc",
    weights_path = "/shares/gcp/integration/rff/damage_function_weights/damage_function_weights3.nc4",
    
    
):

# ssp GDP for fractionalizing damage functions
ssp_gdp = xr.open_zarr(ssp_gdp, consolidated=True).sum("region").gdp

# get RFF data
region = "USA" if USA==True else "world"
rff_gdp = xr.open_dataset(rff_gdp).sel(region=region, drop=True).gdp

# get global consumption factors to extrapolate damage function
factors = rff_gdp.sel(year=slice(2100, 2300)) / rff_gdp.sel(year=2099)

# get RFF emulator weights
run_id = xr.open_dataset(runid_path)
weights = xr.open_dataset(weights_path).sel(rff_sp=run_id.rff_sp, drop=True).value

p_map(
    weight_df,
    list(
        product(
            sectors,
            eta_rhos.items(),
            recipes,
            discs,
            ["damage_function_coefficients"],
        )
    ),
    num_cpus=20,
)

# make it work for the new style:
# weight_df(
#     sector, 
#     eta_rho, 
#     recipe, 
#     disc, 
#     file,
#     in_library,
#     out_library,
#     rff_gdp,
#     ssp_gdp,
#     weights,
# )