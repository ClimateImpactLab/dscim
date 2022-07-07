import xarray as xr
import pandas as pd
import numpy as np
from p_tqdm import p_map
from itertools import product
from functools import partial
import os, sys


def clean_simulation(draw):
    ds = pd.read_csv(
        f"/shares/gcp/social/rff/fivebeans3/emulate-fivebean-{draw}.csv",
        skiprows=9,
    )

    for i, name in enumerate(["model", "ssp"]):
        ds[name] = ds["name"].str.split("/", expand=True)[i]
    for i, name in enumerate(["year", "model"]):
        ds[name] = ds["model"].str.split(":", expand=True)[i]

    ds = ds.loc[ds.param != "error"]

    ds["model"] = ds.model.replace({"low": "IIASA GDP", "high": "OECD Env-Growth"})

    ds["rff_sp"] = draw

    ds["year"] = ds.year.astype(int)

    ds = ds.set_index(["model", "ssp", "rff_sp", "year"]).to_xarray()["value"]

    return ds


def clean_error(draw):

    ds = pd.read_csv(
        f"/shares/gcp/social/rff/fivebeans3/emulate-fivebean-{draw}.csv",
        skiprows=9,
    )

    for i, name in enumerate(["iso", "year"]):
        ds[name] = ds["name"].str.split(":", expand=True)[i]

    ds = ds.loc[ds.param == "error"]

    ds["rff_sp"] = draw

    ds["year"] = ds.year.astype(int)

    ds = ds.set_index(["iso", "year", "rff_sp"]).value.to_xarray()

    return ds


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
    factors,
    fractional=False,
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
    if fractional == True:
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
    eta_rhos,
    USA,
    ssp_gdp,
    rff_gdp,
    recipes_discs,
    in_library,
    out_library,
    runid_path,
    weights_path,
):

    # ssp GDP for fractionalizing damage functions
    ssp_gdp = xr.open_zarr(ssp_gdp, consolidated=True).sum("region").gdp

    # get RFF data
    region = "USA" if USA == True else "world"
    rff_gdp = xr.open_dataset(rff_gdp).sel(region=region, drop=True).gdp

    # get global consumption factors to extrapolate damage function
    factors = rff_gdp.sel(year=slice(2100, 2300)) / rff_gdp.sel(year=2099)

    # get RFF emulator weights
    run_id = xr.open_dataset(runid_path)
    weights = xr.open_dataset(weights_path).sel(rff_sp=run_id.rff_sp, drop=True).value

    for recipe, disc in recipes_discs:

        p_map(
            partial(
                weight_df,
                recipe=recipe,
                disc=disc,
                file="damage_function_coefficients",
                in_library=in_library,
                out_library=out_library,
                rff_gdp=rff_gdp,
                ssp_gdp=ssp_gdp,
                weights=weights,
                factors=factors,
            ),
            [i for i, j in product(sectors, eta_rhos.items())],
            [j for i, j in product(sectors, eta_rhos.items())],
            num_cpus=20,
        )
