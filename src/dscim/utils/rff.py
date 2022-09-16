import xarray as xr
import pandas as pd
import numpy as np
from p_tqdm import p_map
from itertools import product
from functools import partial
import os
import sys
from numpy.testing import assert_allclose
from datetime import datetime


def clean_simulation(
    draw,
    root,
):
    """
    Clean the weights csvs generated by the RFF emulator.
    This produces a file of the weights.

    Parameters
    ----------
    draw : int, weight draw
    root : str, root directory
    """

    ds = pd.read_csv(
        f"{root}/emulate-fivebean-{draw}.csv",
        skiprows=9,
    )

    # cleaning columns
    for i, name in enumerate(["model", "ssp"]):
        ds[name] = ds["name"].str.split("/", expand=True)[i]
    for i, name in enumerate(["year", "model"]):
        ds[name] = ds["model"].str.split(":", expand=True)[i]

    # dropping error rows and keeping only weights rows
    ds = ds.loc[ds.param != "error"]

    ds["model"] = ds.model.replace({"low": "IIASA GDP", "high": "OECD Env-Growth"})

    ds["rff_sp"] = draw

    ds["year"] = ds.year.astype(int)

    ds = ds.set_index(["model", "ssp", "rff_sp", "year"]).to_xarray()["value"]

    return ds


def clean_error(
    draw,
    root,
):
    """
    Clean the weights csvs generated by the RFF emulator.
    This produces a file of the errors (for diagnostics).

    Parameters
    ----------
    draw : int, weight draw
    root : str, root directory
    """

    ds = pd.read_csv(
        f"{root}/emulate-fivebean-{draw}.csv",
        skiprows=9,
    )

    # cleaning columns
    for i, name in enumerate(["iso", "year"]):
        ds[name] = ds["name"].str.split(":", expand=True)[i]

    # dropping weights rows and keeping only error rows
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
    pulse_year,
    fractional=False,
):
    """Weight, fractionalize, and combine SSP damage functions,
    then multiply by RFF GDP to return RFF damage functions.
    """

    # get damage function as share of global GDP
    df = (
        xr.open_dataset(
            f"{in_library}/{sector}/{pulse_year}/{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_{file}.nc4"
        )
        / ssp_gdp
    )

    # pre-2100 weighted fractional damage functions
    rff = (df * weights).sum(["ssp", "model"])

    # save fractional damage function
    if fractional:
        rff.sel(year=slice(2020, 2099)).to_netcdf(
            f"{out_library}/{sector}/{pulse_year}/{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_fractional_{file}.nc4"
        )

    # recover damage function as dollars instead of fraction
    rff = (rff * rff_gdp).sel(year=slice(2020, 2099))

    # post-2100 weighted damage functions
    post_2100 = rff.sel(year=2099) * factors

    dfs = xr.combine_by_coords([rff, post_2100])

    os.makedirs(f"{out_library}/{sector}/{pulse_year}/", exist_ok=True)
    dfs.to_netcdf(
        f"{out_library}/{sector}/{pulse_year}/{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_{file}.nc4"
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
    pulse_year,
):
    """Wrapper function for `weight_df()`."""

    # ssp GDP for fractionalizing damage functions
    ssp_gdp = xr.open_zarr(ssp_gdp, consolidated=True).sum("region").gdp

    # get RFF data
    region = "USA" if USA else "world"
    rff_gdp = xr.open_dataset(rff_gdp).sel(region=region, drop=True).gdp

    # get global consumption factors to extrapolate damage function
    factors = rff_gdp.sel(year=slice(2100, 2300)) / rff_gdp.sel(year=2099)

    # get RFF emulator weights
    run_id = xr.open_dataset(runid_path)
    weights = (
        xr.open_dataset(f"{weights_path}/damage_function_weights.nc4")
        .sel(rff_sp=run_id.rff_sp, drop=True)
        .value
    )

    for recipe_disc, sector, eta_rho in product(
        recipes_discs, sectors, eta_rhos.items()
    ):

        print(f"{datetime.now()} : {recipe_disc} {sector} {eta_rho}")

        weight_df(
            sector=sector,
            eta_rho=eta_rho,
            recipe=recipe_disc[0],
            disc=recipe_disc[1],
            file="damage_function_coefficients",
            in_library=in_library,
            out_library=out_library,
            rff_gdp=rff_gdp,
            ssp_gdp=ssp_gdp,
            weights=weights,
            factors=factors,
            pulse_year=pulse_year,
        )


def prep_rff_socioeconomics(
    inflation_path,
    rff_path,
    runid_path,
    out_path,
    USA,
):
    """Generate the global or domestic RFF socioeconomics file for use with the `dscim` MainRecipe."""

    # Load Fed GDP deflator
    fed_gdpdef = pd.read_csv(inflation_path).set_index("year")["gdpdef"].to_dict()

    # transform 2011 USD to 2019 USD
    inflation_adj = fed_gdpdef[2019] / fed_gdpdef[2011]

    # read in RFF data
    socioec = xr.open_dataset(rff_path)

    if not USA:
        print("Summing to globe.")
        socioec = socioec.sum("Country")
    else:
        print("USA output.")
        socioec = socioec.sel(Country="USA", drop=True)

    # interpolate with log -> linear interpolation -> exponentiate
    socioec = np.exp(
        np.log(socioec).interp({"Year": range(2020, 2301, 1)}, method="linear")
    ).rename({"runid": "rff_sp", "Year": "year", "Pop": "pop", "GDP": "gdp"})

    socioec["pop"] = socioec["pop"] * 1000
    socioec["gdp"] = socioec["gdp"] * 1e6 * inflation_adj

    # read in RFF runids and update coordinates with them
    run_id = xr.open_dataset(runid_path)
    socioec = socioec.sel(rff_sp=run_id.rff_sp, drop=True)

    if not USA:
        socioec.expand_dims({"region": ["world"]}).to_netcdf(
            f"{out_path}/rff_global_socioeconomics.nc4"
        )
    else:
        socioec.expand_dims({"region": ["USA"]}).to_netcdf(
            f"{out_path}/rff_USA_socioeconomics.nc4"
        )


def aggregate_rff_weights(
    root,
    output,
):
    """Wrapper function for `clean_simulation()` and `clean_error()`.
    Generates an aggregated file of RFF emulator weights.
    """

    # clean simulation files
    datasets = p_map(partial(clean_simulation, root=root), range(1, 10001, 1))

    # concatenate and interpolate
    concatenated = xr.concat(datasets, "rff_sp").interp(
        {"year": range(2010, 2101, 1)}, method="linear"
    )

    # reweight
    reweighted = concatenated / concatenated.sum(["model", "ssp"])

    # make sure weights sum to 1
    assert_allclose(reweighted.sum(["model", "ssp"]).values, 1)

    # describe and save file
    reweighted = reweighted.to_dataset()
    reweighted.attrs["version"] = 3
    reweighted.attrs[
        "description"
    ] = """
    This set of emulator weights is generated using this script:
    dscim/dscim/utils/rff.py -> aggregate_rff_weights
    It cleans and aggregates the emulator weights csvs, linearly interpolates them between 5 year intervals, reweights them to sum to 1, and converts to ncdf4 format.
    """
    os.makedirs(output, exist_ok=True)
    reweighted.to_netcdf(f"{output}/damage_function_weights.nc4")

    # save out error files

    error_datasets = p_map(partial(clean_error, root=root), range(1, 10001, 1))
    error_concatenated = xr.concat(error_datasets, "rff_sp")

    # describe and save file
    error_concatenated = error_concatenated.to_dataset()
    error_concatenated.attrs["version"] = 3
    error_concatenated.attrs[
        "description"
    ] = """
    This set of emulator weight errors is generated using this script:
    dscim/dscim/preprocessing/rff/aggregate_rff_weights.py
    It cleans and aggregates the emulator weights csvs for error rows only, and converts to ncdf4 format.
    """

    error_concatenated.to_netcdf(f"{output}/weights_errors.nc4")