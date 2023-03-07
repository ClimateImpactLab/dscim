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
import fsspec
import pyarrow
import gurobipy as gp
import time
from scipy.sparse import coo_matrix
from gurobipy import GRB


## Solve the optimization problem
def solve_optimization(ssp_df, rff_df):
    """Generate weights based on which to derive the weighted average of damage function coefficents
    across six SSP-growth models for a single RFF-SP

    This function applies an emulation scheme to calculate a set of weights, constrained to
    sum to unity, that, when used to take a weighted average of global GDP across SSP-growth models
    (3 SSPs X 2 IAMs), most closely recovers the global GDP in the RFF-SP simulation run that
    wish to emulate. The emulation scheme is estimated and applied separately for each 5-year period,
    of a single RFF-SP. Within each period, the scheme aims to interpolate between the SSP-growth models
    in order to match the country-level GDPs designated by the given RFF-SP. Empirically, it solves
    an optimization problem to minimize a weighted sum of country-level errors, taking country-level
    RFF-SP GDPs as weights

    Parameters
    ----------
    ssp_df : pd.DataFrame
        Dataset with country-level log per capita GDPs by SSP-growth models in 5-year increments, post-
        processed by the `process_ssp_sample` function
    rff_df : pd.DataFrame
        Dateset with country-level GDPs and log per capita GDPs for a single RFF-SP simulation run
    Returns
    ------
        Dataset with a set of SSP-growth model weights and country-level errors in 5-year increments
        for a single RFF-SP
    """

    ssp_df = ssp_df[(ssp_df.scenario != "SSP1") & (ssp_df.scenario != "SSP5")]

    output = []

    header = ["year", "param", "name", "value"]

    years = pd.unique(ssp_df.year)
    for year in years:
        sspidf = ssp_df[ssp_df.year == year]
        rffidf = rff_df[rff_df.year == year]

        if rffidf.shape[0] == 0:
            continue

        isoyears = pd.unique(sspidf.isoyear)

        # Create parameters list
        alphaparams = pd.unique(sspidf.yearscen)

        # Drop the first entry for each subgroup
        alphaparams_butfirst = np.delete(alphaparams, 0)
        params = np.concatenate((isoyears, alphaparams_butfirst))
        paramindex_alpha0 = len(isoyears)

        # Construct objective function
        if "weight" in rff_df.columns:
            weights = [
                rffidf.weight[(rffidf.isoyear == isoyear)].values[0]
                for isoyear in isoyears
            ]
        else:
            weights = np.ones(len(isoyears))
        objfunc = np.concatenate((weights, np.zeros(len(alphaparams_butfirst))))

        # Contruct constraints, all in the form of A x < b
        AA_rows = []
        AA_cols = []
        AA_data = []
        bb = []

        def add_AA_cell(row, col, value):
            AA_rows.append(row)
            AA_cols.append(col)
            AA_data.append(value)

        ## Rows defining absolute values

        # d_it > y_it - (sum_s>1 alpha_is y_sit + (1 - sum_s>1 alpha_is) y_1it)
        # -d_it - (sum_s>1 alpha_is y_sit - (sum_s>1 alpha_is) y_1it) < -y_it + y_1it

        # d_it > -(y_it - (sum_s>1 alpha_is y_sit + (1 - sum_s>1 alpha_is) y_1it))
        # -d_it + (sum_s>1 alpha_is y_sit - (sum_s>1 alpha_is) y_1it) < y_it - y_1it

        for ii, isoyear in enumerate(isoyears):
            subdf = sspidf[sspidf.isoyear == isoyear]
            if subdf.shape[0] == 0 or not np.any(rffidf.isoyear == isoyear):
                continue

            try:
                y1it = subdf.loginc[subdf.yearscen == alphaparams[0]].values[0]
            except Exception as ex:
                print(ii, isoyear, ex)
                print("Exception! Keep going..")  # KM added
                continue

            add_AA_cell(len(bb), ii, -1)
            add_AA_cell(len(bb) + 1, ii, -1)

            for jj, alphaparam in enumerate(alphaparams_butfirst):
                ysit = subdf.loginc[subdf.yearscen == alphaparam].values[0]

                add_AA_cell(len(bb), paramindex_alpha0 + jj, -ysit + y1it)
                add_AA_cell(len(bb) + 1, paramindex_alpha0 + jj, ysit - y1it)

            bb.append(-rffidf.loginc[(rffidf.isoyear == isoyear)].values[0] + y1it)
            bb.append(rffidf.loginc[(rffidf.isoyear == isoyear)].values[0] - y1it)

        constrindex_alphag10 = len(bb)

        # sum alpha_is < 1
        for jj in range(paramindex_alpha0, len(params)):
            add_AA_cell(constrindex_alphag10, jj, 1)
        bb.append(1)

        constrindex_end = constrindex_alphag10 + 1

        AA = coo_matrix(
            (AA_data, (AA_rows, AA_cols)), shape=(constrindex_end, len(params))
        )

        # Also constrained so that d_it > 0 and alpha_is > 0

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        mod = gp.Model("gourmet", env=env)
        xx = mod.addMVar(shape=len(objfunc), vtype=GRB.CONTINUOUS, name="xx")
        mod.setObjective(objfunc @ xx, GRB.MINIMIZE)
        bb = np.array(bb)
        mod.addConstr(AA @ xx <= bb)
        mod.optimize()
        errors = np.around(xx.X[:paramindex_alpha0], 6)
        alphas = np.around(xx.X[paramindex_alpha0:], 6)

        for ii, error in enumerate(errors):
            output.append([year, "error", isoyears[ii], error])

        output.append([year, "alpha", alphaparams[0], max(0, 1 - sum(alphas))])
        for ii, alpha in enumerate(alphas):
            output.append([year, "alpha", alphaparams_butfirst[ii], alpha])

    out_df = pd.DataFrame(output, columns=header)

    return out_df


# Process SSP sample
def process_ssp_sample(ssppath):
    """Clean SSP per capita GDP projections"""
    ssp_df = pd.read_csv(ssppath, skiprows=11)
    ssp_df = ssp_df[ssp_df.year >= 2010]
    ssp_df["loginc"] = np.log(ssp_df.value)
    ssp_df["isoyear"] = ssp_df.apply(lambda row: "%s:%d" % (row.iso, row.year), axis=1)
    ssp_df["yearscen"] = ssp_df.apply(
        lambda row: "%d:%s/%s" % (row.year, row.model, row.scenario), axis=1
    )

    return ssp_df


## Process RFF Sample
def process_rff_sample(i, rffpath, ssp_df, outdir, HEADER):
    """Clean raw socioeconomic projections from a single RFF-SP simulation run,
    pass the cleaned dataset to the `solve_optimization` function, and save outputs

    This produces a csv file of RFF emulator weights and country-level errors in 5-year
    increments for a single RFF-SP
    """

    read_feather = os.path.join(rffpath, "run_%d.feather" % i)
    rff_raw = pd.read_feather(read_feather)
    rff_raw.rename(columns={"Year": "year", "Country": "iso"}, inplace=True)

    # Fill missing data with mean across SSP scenarios of the same years
    rff_df = pd.DataFrame()
    for iso, group in rff_raw.groupby(["iso"]):
        minyear = min(group.year)
        before_all = ssp_df[(ssp_df.year < minyear) & (ssp_df.iso == iso)][
            ["iso", "year", "value"]
        ]
        before = before_all.groupby(["iso", "year"]).mean().reset_index()
        after = pd.DataFrame(
            dict(
                iso=iso,
                year=group.year,
                value=(88.58 / 98.71) * (group.GDP * 1e6) / (group.Pop * 1000),
            )
        )  # Get in per capita 2005 PPP-adjusted USD rff GDP
        all_year_df = pd.concat((before, after))
        rff_df = pd.concat((rff_df, all_year_df))

    rff_df["loginc"] = np.log(rff_df.value)
    rff_df["isoyear"] = rff_df.apply(lambda row: "%s:%d" % (row.iso, row.year), axis=1)

    rff_df = pd.merge(rff_df, rff_raw, on=["year", "iso"], how="left")

    rff_df["weight"] = (
        88.58 / 98.71
    ) * rff_df.GDP  # Adjust weight measurement from 2011 tp 2005 PPP USD

    # print(rff_df.iso[np.isnan(rff_df.weight)])
    rff_df.weight[np.isnan(rff_df.weight)] = np.exp(
        np.nanmean(np.log(rff_df.weight))
    )  # Fill missing value weights with sample mean

    out_df = solve_optimization(ssp_df, rff_df)

    write_file = os.path.join(outdir, "emulate-%d.csv") % i

    protocol = write_file.split("://")[0] if "://" in write_file else ""
    write_options = storage_options if protocol != "" else {}
    fs = fsspec.filesystem(protocol, **write_options)

    with fs.open(write_file, "w") as outf:
        outf.write(HEADER.strip() + "\n")
        out_df.to_csv(outf, index=False)
        # print("writing",write_file)


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

    for recipe_disc, sector, eta_rho in product(recipes_discs, sectors, eta_rhos):
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
