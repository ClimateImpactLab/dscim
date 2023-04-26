"""
Calculate damages from the projection system using VSL
"""

import os
import glob
import re
import logging
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from itertools import product
from functools import partial
from p_tqdm import p_map, p_umap
from dscim.menu.simple_storage import EconVars
from zarr.errors import GroupNotFoundError

logger = logging.getLogger(__name__)


def _parse_projection_filesys(input_path, query="exists==True"):
    """Retrieve projection system output structure to read files

    Parameters
    ---------

    input_path: str Directory containing all raw projection output
    quey: str String with a query object to filter the raw data paths pandas
    dataframe  (i.e "ssp=='SSP3' and exists==True")
    """

    # Projection elements
    rcp = ["rcp85", "rcp45"]
    gcm = [
        "ACCESS1-0",
        "CCSM4",
        "GFDL-CM3",
        "IPSL-CM5A-LR",
        "MIROC-ESM-CHEM",
        "bcc-csm1-1",
        "CESM1-BGC",
        "GFDL-ESM2G",
        "IPSL-CM5A-MR",
        "MPI-ESM-LR",
        "BNU-ESM",
        "CNRM-CM5",
        "GFDL-ESM2M",
        "MIROC5",
        "MPI-ESM-MR",
        "CanESM2",
        "CSIRO-Mk3-6-0",
        "inmcm4",
        "MIROC-ESM",
        "MRI-CGCM3",
        "NorESM1-M",
        "surrogate_GFDL-CM3_89",
        "surrogate_GFDL-ESM2G_11",
        "surrogate_CanESM2_99",
        "surrogate_GFDL-ESM2G_01",
        "surrogate_MRI-CGCM3_11",
        "surrogate_CanESM2_89",
        "surrogate_GFDL-CM3_94",
        "surrogate_MRI-CGCM3_01",
        "surrogate_CanESM2_94",
        "surrogate_GFDL-CM3_99",
        "surrogate_MRI-CGCM3_06",
        "surrogate_GFDL-ESM2G_06",
    ]
    model = ["high", "low"]
    ssp = [f"SSP{n}" for n in range(1, 6)]
    moddict = {"high": "OECD Env-Growth", "low": "IIASA GDP"}

    # Build file tree using available batches
    batch_folders = [
        re.search(r"batch\d+", dirname.name).group(0)
        for dirname in Path(input_path).iterdir()
        if re.search(r"batch*", dirname.name) is not None
    ]

    # DataFrame with cartesian product and test for existing paths
    df = pd.DataFrame(
        list(product(batch_folders, rcp, gcm, model, ssp)),
        columns=["batch", "rcp", "gcm", "model", "ssp"],
    )
    df["path"] = df.apply(
        lambda x: os.path.join(input_path, x.batch, x.rcp, x.gcm, x.model, x.ssp),
        axis=1,
    )
    df["exists"] = df.path.apply(lambda x: Path(x).exists())
    df["iam"] = df["model"].replace(moddict)

    return df.query(query)


def calculate_labor_impacts(input_path, file_prefix, variable, val_type):
    """Calculate impacts for labor results.

    Paramemters
    ----------
    input_path str
        Path to model/gcm/iam/rcp/ folder, usually from the
        `_parse_projection_filesys` function.
    file_prefix str
        Prefix of the MC output filenames
    variable str
        Variable to use within `xr.Dataset`
    val_type str
        Valuation type.

    Returns
    -------
        xr.Dataset object with per-capita monetary damages
    """

    damages_val = xr.open_dataset(f"{input_path}/{file_prefix}-{val_type}.nc4").sel(
        year=slice(2010, 2099)
    )
    damages_histclim = xr.open_dataset(
        f"{input_path}/{file_prefix}-histclim-{val_type}.nc4"
    ).sel(year=slice(2010, 2099))

    # labour needs indexing assigned
    damages_val_index = damages_val.assign_coords({"region": damages_val.regions})

    damages_histclim_index = damages_histclim.assign_coords(
        {"region": damages_histclim.regions}
    )

    # calculate the delta output
    impacts_delta = damages_val_index[variable] - damages_histclim_index[variable]

    # generate the histclim variable
    impacts = damages_histclim_index[variable].to_dataset(name=f"histclim_{variable}")
    # generate the delta variable
    impacts[f"delta_{variable}"] = impacts_delta

    return impacts


def concatenate_labor_damages(
    input_path,
    save_path,
    ec_cls,
    file_prefix="uninteracted_main_model",
    variable="rebased",
    val_type="wage-levels",
    format_file="netcdf",
    **kwargs,
):
    """Concatenate damages across batches.

    Parameters
    ----------
    input_path str
        Directory containing all raw projection output.
    ec_cls dscim.simple_storage.EconVars
        EconVars class with population and GDP data to rescale damages
    save_path str
        Path to save concatenated file in .zarr or .nc4 format
    file_prefix str
        Prefix of the MC output filenames
    variables list
        Variable names to extract from calculated damages
    format_file str
        Format to save file. Options are 'netcdf' or 'zarr'
    **kwargs
        Other options passed to the `_parse_projection_filesys`: query="rcp=='rcp45'"
    """

    # Load df with paths
    df = _parse_projection_filesys(
        input_path=input_path, query=kwargs.get("query", "exists==True")
    )

    # Process files by batches and save as .zarr files
    for i, g in df.groupby("batch"):
        logger.info(f"Processing damages in batch: {i}")
        list_damages_batch = []
        for idx, row in g.iterrows():
            try:
                ds = calculate_labor_impacts(
                    input_path=row.path,
                    file_prefix=file_prefix,
                    variable=variable,
                    val_type=val_type,
                )
                ds = ds.assign_coords(
                    {
                        "ssp": row.ssp,
                        "rcp": row.rcp,
                        "gcm": row.gcm,
                        "model": row.iam,
                        "batch": row.batch,
                    }
                )

                ds_exp = ds.expand_dims(["ssp", "rcp", "model", "gcm", "batch"])
                list_damages_batch.append(ds_exp)

            except Exception as e:
                logger.error(f"Error in batch{i}: {e}")
                pass

        # Concatenate file within batch
        conversion_value = 1.273526
        concat_ds = xr.combine_by_coords(list_damages_batch)
        for v in [f"histclim_{variable}", f"delta_{variable}"]:
            concat_ds[v] = (concat_ds[v] / ec_cls.econ_vars.pop) * -1 * conversion_value

        # Save file
        file_name = f"{variable}_{val_type}_{i}"
        path_to_file = os.path.join(save_path, file_name)

        # convert to float32
        concat_ds = concat_ds.astype(np.float32)
        logger.info(f"Concatenating and processing {i}")

        # save out
        if format_file == "zarr":
            to_store = concat_ds.copy()
            for var in to_store.variables:
                to_store[var].encoding.clear()

            to_store.to_zarr(f"{path_to_file}.zarr", mode="w", consolidated=True)
        elif format_file == "netcdf":
            concat_ds.to_netcdf(f"{path_to_file}.nc4")

    return concat_ds


def calculate_labor_batch_damages(batch, ec, input_path, save_path):
    print(f"Processing batch={batch} damages in {os.getpid()}")
    concatenate_labor_damages(
        input_path=input_path,
        save_path=save_path,
        ec_cls=ec,
        variable="rebased",
        val_type="wage-levels",
        format_file="zarr",
        query=f"exists==True&batch=='batch{batch}'",
    )
    print("Saved!")


def calculate_labor_damages(
    path_econ="/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr",
    input_path="/shares/gcp/outputs/labor/impacts-woodwork/mc_correct_rebasing_for_integration",
    save_path="/shares/gcp/integration/float32/input_data_histclim/labor_data/replication/",
):
    ec = EconVars(path_econ)
    # process in 3 rounds to limit memory usage
    for i in range(0, 3):
        partial_func = partial(
            calculate_labor_batch_damages,
            input_path=input_path,
            save_path=save_path,
            ec=ec,
        )
        print("Processing batches:")
        print(list(range(i * 5, i * 5 + 5)))
        p_umap(partial_func, list(range(i * 5, i * 5 + 5)))


def compute_ag_damages(
    input_path,
    save_path,
    pop,
    varname,
    query="exists==True",
    topcode=None,
    scalar=None,
    integration=False,
    batches=range(0, 15),
    num_cpus=15,
    file="/disaggregated_damages.nc4",
    vars=None,
    min_year=2010,
    max_year=2099,
):
    """
    Reshapes ag estimate runs for use in integration system,
    then converts to negative per capita damages.

    Parameters
    ----------
    input_path str or list
        Path to NetCDF4 files to be reshaped.
    file str
        Name of files to be globbed.
    integration bool
        If True, will format files to be integrated with other sectors.
    pop xr.DataArray
        Population data to convert ag damages into per capita terms.
    save_path str
        Path where files should be saved
    """
    if vars is None:
        vars = ["wc_no_reallocation"]

    if integration:
        assert (
            topcode is not None
        ), "Data is being processed for integration. Please pass a topcode."

    if type(input_path) == str:
        input_path = [input_path]

    paths = []
    for f in input_path:
        p = _parse_projection_filesys(input_path=f, query=query)
        paths.append(p)

    paths = pd.concat(paths)
    paths["path"] = paths.path + file

    def prep(ds):
        ds.coords["model"] = ds.model.str.replace("high", "OECD Env-Growth")
        ds.coords["model"] = ds.model.str.replace("low", "IIASA GDP")
        return ds

    def process_batch(g):
        i = g.batch.values[0]
        if i in [f"batch{j}" for j in batches]:

            print(f"Processing damages in {i}")
            ds = xr.open_mfdataset(g.path, preprocess=prep, parallel=True)[vars]
            attrs = ds.attrs
            ds = ds.sel({"year": slice(min_year, max_year)})
            # ag has missing 2099 damages so we fill these in with the 2098 damages
            ds = ds.reindex(year=range(min_year, max_year + 1), method="ffill")

            # squeeze ag-specific dimensions out so that
            # it can be stacked with other sectors
            if integration:
                ds = ds.sel(demand_topcode=topcode)
                print(f"Selecting topcode {topcode}.")
                for var in [
                    "continent",
                    "iam",
                    "Es_Ed",
                    "market_level",
                    "demand_topcode",
                ]:

                    if var in ds.coords:
                        attrs[var] = ds[var].values
                        ds = ds.drop(var)

            # get in per capita 2019 PPP-adjusted USD damages
            ds = (ds / pop) * -1 * 1.273526

            # replace infinite values with missings
            for var in ds.keys():
                ds[var] = xr.where(np.isinf(ds[var]), np.nan, ds[var])

            if scalar is not None:
                print("Scaling for reallocation.")
                ds["wc_reallocation"] = ds["wc_no_reallocation"] * scalar

            ds.attrs = attrs
            return ds
        else:
            print(f"Skipped {i}.")

    batches = p_map(
        process_batch, [g for i, g in paths.groupby("batch")], num_cpus=num_cpus
    )
    chunkies = {
        "rcp": 1,
        "region": 24378,
        "gcm": 1,
        "year": 10,
        "model": 1,
        "ssp": 1,
        "batch": 15,
    }
    batches = (
        xr.concat(batches, "batch", combine_attrs="override")
        .chunk(chunkies)
        .drop("variable")
        .squeeze()
    )
    batches = xr.where(np.isinf(batches), np.nan, batches)

    batches.rename({"wc_reallocation": varname})[varname].to_zarr(
        save_path, mode="a", consolidated=True
    )


def read_energy_files(df, seed="TINV_clim_price014_total_energy_fulladapt-histclim"):
    """Read energy CSV files and trasnform them to Xarray objects

    This function reads a dataframe with the filesystem metadata (from
    ``_parse_projection_filesys``) to read all CSV files in it with the desired
    ``seed`` and transform to xarray object adding the directory metadata as
    new dimensions, this will be helpful for data concatenation.

    This function is parallelized by ``read_energy_files_parallel``

    Parameters
    ---------
    df : pd.DataFrame
        DataFrame with projection system metadata by batch/RCP/IAM/GCM

    Returns
    -------
    None
        Saved data array with expanded damages from original CSV
    """

    for idx, row in df.iterrows():
        path = os.path.join(row.path, f"{seed}.csv")
        try:
            damages = pd.read_csv(path)
            damages = damages[damages.year >= 2010]
            damages_arr = damages.set_index(["region", "year"]).to_xarray()

            # Add dims to array
            damages_arr = damages_arr.expand_dims(
                {
                    "batch": [row.batch],
                    "rcp": [row.rcp],
                    "gcm": [row.gcm],
                    "model": [row.iam],
                    "ssp": [row.ssp],
                }
            )

        except Exception as e:
            logger.error(f"Error in file {row.path}: {e}")
            pass

        damages_arr.to_netcdf(os.path.join(row.path, f"{seed}.nc4"))

    return None


def read_energy_files_parallel(input_path, save_path, ec_cls, **kwargs):
    """Concatenate energy results from CSV to NetCDF by batches using
    multiprocessing

    This function takes all CSV files per batch and maps the
    ``read_energy_files`` function to all the files within a batch. The files
    will be saved in the same path as the CSV files but in NetCDF format.

    Once saved, the files will be read again, using a Dask ``Client`` and
    chunked to be finally saved as files per batch. Before saving, the function
    will calculate damages per capita using SSP populations for each scenario
    with a ``EconVars`` class and then calculate 2019 USD damages

    Parameters
    ----------
    input_path : str
        Path to root folder organized by batch containing all projection system
        files
    save_path : str
        Path to saved damages by batch files
    ec_cls : ``dscim.menu.simple_storage.EconVars``
    **kwargs
        Other elements too the ``read_energy_files`` damages

    Returns
    ------
        None
    """

    # Start reading and save to NetCDF
    for i in range(0, 15):
        logger.info(f"Processing damages in batch: {i}")
        # Read files available
        df = _parse_projection_filesys(
            input_path=input_path, query=f"exists==True&batch=='batch{i}'"
        )
        # Calculate the chunk size as an integer
        num_processes = multiprocessing.cpu_count()
        chunk_size = int(df.shape[0] / num_processes)
        logger.info(
            f"Starting multiprocessing in {num_processes} cores with {chunk_size} rows each"
        )
        chunks = [
            df.iloc[i : i + chunk_size, :] for i in range(0, df.shape[0], chunk_size)
        ]

        with multiprocessing.Pool(processes=num_processes) as p:
            result = p.map(partial(read_energy_files, **kwargs), chunks)

    return result


def calculate_energy_impacts(input_path, file_prefix, variable):
    """Calculate impacts for labor results for individual modeling unit.

    Read in individual damages files from the labor projection system output
    and re-index to add region dimension. This is needed to adjust the
    projection file outcomes that do not have a region dimension

    Paramemters
    ----------
    input_path str
        Path to model/gcm/iam/rcp/ folder, usually from the
        `_parse_projection_filesys` function.
    file_prefix str
        Prefix of the MC output filenames
    variable str
        Variable to use within `xr.Dataset`

    Returns
    -------
        xr.Dataset object with per-capita monetary damages
    """

    damages_delta = xr.open_dataset(f"{input_path}/{file_prefix}_delta.nc4").sel(
        year=slice(2010, 2099)
    )
    damages_histclim = xr.open_dataset(f"{input_path}/{file_prefix}_histclim.nc4").sel(
        year=slice(2010, 2099)
    )

    # generate the histclim variable
    impacts = damages_histclim[variable].to_dataset(name=f"histclim_{variable}")
    # generate the delta variable
    impacts[f"delta_{variable}"] = damages_delta[variable]

    return impacts


def concatenate_energy_damages(
    input_path,
    save_path,
    ec_cls,
    file_prefix="TINV_clim_integration_total_energy",
    variable="rebased",
    format_file="netcdf",
    **kwargs,
):
    """Concatenate damages across batches and create a lazy array for future
    calculations.

    Using the `value_mortality_damages` function this function lazily loads all
    damages for SCC calculations and scale damage to per-capital damages and
    also scale labor inpacts to indicate increasing damages as positive, and
    gains from warming as negative.

    Parameters
    ----------
    input_path str
        Directory containing all raw projection output.
    ec_cls dscim.simple_storage.EconVars
        EconVars class with population and GDP data to rescale damages
    save_path str
        Path to save concatenated file in .zarr or .nc4 format
    file_prefix str
        Prefix of the MC output filenames
    variables list
        Variable names to extract from calculated damages
    format_file str
        Format to save file. Options are 'netcdf' or 'zarr'
    **kwargs
        Other options passed to the `value_mortality_damages` function
        and the `_parse_projection_filesys`: query="rcp=='rcp45'"
    """

    # Load df with paths
    df = _parse_projection_filesys(
        input_path=input_path, query=kwargs.get("query", "exists==True")
    )

    # Process files by batches and save as .zarr files
    for i, g in df.groupby("batch"):
        logger.info(f"Processing damages in batch: {i}")
        list_damages_batch = []
        for idx, row in g.iterrows():
            try:
                ds = calculate_energy_impacts(
                    input_path=row.path,
                    file_prefix=file_prefix,
                    variable=variable,
                )
                list_damages_batch.append(ds)

            except Exception as e:
                logger.error(f"Error in batch{i}: {e}")
                pass

        # Concatenate file within batch
        conversion_value = 1.273526
        concat_ds = xr.combine_by_coords(list_damages_batch)
        for v in [f"histclim_{variable}", f"delta_{variable}"]:
            concat_ds[v] = (concat_ds[v] / ec_cls.econ_vars.pop) * conversion_value

        # Save file
        file_name = f"{variable}_{i}"
        path_to_file = os.path.join(save_path, file_name)

        # convert to float32
        concat_ds = concat_ds.astype(np.float32)
        logger.info(f"Concatenating and processing {i}")

        if format_file == "zarr":
            to_store = concat_ds.copy()
            for var in to_store.variables:
                to_store[var].encoding.clear()

            to_store.to_zarr(f"{path_to_file}.zarr", mode="w", consolidated=True)
        elif format_file == "netcdf":
            concat_ds.to_netcdf(f"{path_to_file}.nc4")

    return concat_ds


def calculate_energy_batch_damages(batch, ec, input_path, save_path):
    print(f"Processing batch={batch} damages in {os.getpid()}")
    concatenate_energy_damages(
        input_path=input_path,
        file_prefix="TINV_clim_integration_total_energy",
        save_path=save_path,
        ec_cls=ec,
        variable="rebased",
        format_file="zarr",
        query=f"exists==True&batch=='batch{batch}'",
    )
    print("Saved!")


def calculate_energy_damages(
    re_calculate=True,
    path_econ="/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr",
    input_path="/shares/gcp/outputs/energy_pixel_interaction/impacts-blueghost/integration_resampled",
    save_path="/shares/gcp/integration/float32/input_data_histclim/energy_data/replication_2022aug/",
):
    ec = EconVars(path_econ)

    if re_calculate:
        read_energy_files_parallel(
            input_path=input_path,
            save_path=save_path,
            ec_cls=ec,
            seed="TINV_clim_integration_total_energy_delta",
        )
        read_energy_files_parallel(
            input_path=input_path,
            save_path=save_path,
            ec_cls=ec,
            seed="TINV_clim_integration_total_energy_histclim",
        )

    # process in 3 rounds to limit memory usage
    for i in range(0, 3):
        partial_func = partial(
            calculate_energy_batch_damages,
            input_path=input_path,
            save_path=save_path,
            ec=ec,
        )
        print("Processing batches:")
        print(list(range(i * 5, i * 5 + 5)))
        p_umap(partial_func, list(range(i * 5, i * 5 + 5)))


def prep_mortality_damages(
    gcms,
    paths,
    vars,
    outpath,
    path_econ="/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr",
):

    print(
        "This function only works on mortality_v4 and mortality_v5 damages from the mortality repo's valuation. Earlier versions of mortality contain different variable definitions (per capita, not per capita, with or without histclim subtracted off."
    )

    ec = EconVars(path_econ=path_econ)

    # longest-string gcm has to be processed first so the coordinate is the right str length
    gcms = sorted(gcms, key=len, reverse=True)

    for i, gcm in enumerate(gcms):
        print(gcm, i, "/", len(gcms))

        def prep(ds, gcm=gcm):
            return ds.sel(gcm=gcm).drop("gcm")

        data = {}
        for var, name in vars.items():
            data[var] = xr.open_mfdataset(paths[var], preprocess=prep, parallel=True)[
                name
            ]

        damages = xr.Dataset(
            {
                "delta": (
                    data["delta_deaths"] - data["histclim_deaths"] + data["delta_costs"]
                )
                / ec.econ_vars.pop.load(),
                "histclim": data["histclim_deaths"] / ec.econ_vars.pop.load(),
            }
        ).expand_dims({"gcm": [gcm]})

        damages = damages.chunk(
            {"batch": 15, "ssp": 1, "model": 1, "rcp": 1, "gcm": 1, "year": 10}
        )
        damages.coords.update({"batch": [f"batch{i}" for i in damages.batch.values]})

        # convert to EPA VSL
        damages = damages * 0.90681089

        if i == 0:
            damages.to_zarr(
                outpath,
                consolidated=True,
                mode="w",
            )
        else:
            damages.to_zarr(
                outpath,
                consolidated=True,
                append_dim="gcm",
            )

        for v in data.values():
            v.close()
        damages.close()


def coastal_inputs(
    version,
    adapt_type,
    vsl_valuation,
    path,
):

    try:
        d = xr.open_zarr(f"{path}/coastal_damages_{version}.zarr")
    except GroupNotFoundError:
        print(f"Zarr not found: {path}/coastal_damages_{version}.zarr")
        exit()

    d = d.sel(adapt_type=adapt_type, vsl_valuation=vsl_valuation, drop=True)

    d.to_zarr(
        f"{path}/coastal_damages_{version}-{adapt_type}-{vsl_valuation}.zarr",
        consolidated=True,
        mode="w",
    )
