import os
import numpy as np
import xarray as xr
import pandas as pd
import pytest
import logging
import shutil
from itertools import chain, repeat
from dscim.menu.simple_storage import EconVars
from dscim.preprocessing.input_damages import (
    _parse_projection_filesys,
    concatenate_damage_output,
    calculate_labor_impacts,
    concatenate_labor_damages,
    calculate_labor_batch_damages,
    calculate_labor_damages,
    compute_ag_damages,
    read_energy_files,
    read_energy_files_parallel,
    calculate_energy_impacts,
    concatenate_energy_damages,
    calculate_energy_batch_damages,
    calculate_energy_damages,
    prep_mortality_damages,
    coastal_inputs,
)

logger = logging.getLogger(__name__)


def test_parse_projection_filesys(tmp_path):
    """
    Test that parse_projection_filesys correctly retrieves projection system output structure
    """
    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["high", "low"]
    ssp = [f"SSP{n}" for n in range(2, 4)]
    batch = ["batch6", "batch9"]

    for b in batch:
        for r in rcp:
            for g in gcm:
                for m in model:
                    for s in ssp:
                        os.makedirs(os.path.join(tmp_path, b, r, g, m, s))

    out_expected = {
        "batch": list(chain(repeat("batch6", 16), repeat("batch9", 16))),
        "rcp": list(chain(repeat("rcp45", 8), repeat("rcp85", 8))) * 2,
        "gcm": list(chain(repeat("ACCESS1-0", 4), repeat("GFDL-CM3", 4))) * 4,
        "model": list(chain(repeat("high", 2), repeat("low", 2))) * 8,
        "ssp": ["SSP2", "SSP3"] * 16,
        "path": [
            os.path.join(tmp_path, b, r, g, m, s)
            for b in ["batch6", "batch9"]
            for r in rcp
            for g in gcm
            for m in model
            for s in ssp
        ],
        "exists": [True] * 32,
        "iam": list(chain(repeat("OECD Env-Growth", 2), repeat("IIASA GDP", 2))) * 8,
    }
    df_out_expected = pd.DataFrame(out_expected)

    df_out_actual = _parse_projection_filesys(input_path=tmp_path)
    df_out_actual = df_out_actual.sort_values(
        by=["batch", "rcp", "gcm", "model", "ssp"]
    )
    df_out_actual.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(df_out_expected, df_out_actual)


def test_concatenate_damage_output(tmp_path):
    """
    Test that concatenate_damage_output correctly concatenates damages across batches and saves to a single zarr file
    """
    d = os.path.join(tmp_path, "concatenate_in")
    if not os.path.exists(d):
        os.makedirs(d)

    for b in ["batch" + str(i) for i in range(0, 15)]:
        ds_in = xr.Dataset(
            {
                "delta_rebased": (
                    ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                    np.full((2, 2, 2, 2, 1, 2, 2), 1).astype(object),
                ),
                "histclim_rebased": (
                    ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                    np.full((2, 2, 2, 2, 1, 2, 2), 2),
                ),
            },
            coords={
                "batch": (["batch"], [b]),
                "gcm": (["gcm"], np.array(["ACCESS1-0", "BNU-ESM"], dtype=object)),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "rcp": (["rcp"], ["rcp45", "rcp85"]),
                "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
                "ssp": (["ssp"], ["SSP2", "SSP3"]),
                "year": (["year"], [2020, 2099]),
            },
        )

        infile = os.path.join(d, f"test_insuffix_{b}.zarr")

        ds_in.to_zarr(infile)

    ds_out_expected = xr.Dataset(
        {
            "delta_rebased": (
                ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                np.full((2, 2, 2, 2, 15, 2, 2), 1),
            ),
            "histclim_rebased": (
                ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                np.full((2, 2, 2, 2, 15, 2, 2), 2),
            ),
        },
        coords={
            "batch": (["batch"], ["batch" + str(i) for i in range(0, 15)]),
            "gcm": (["gcm"], ["ACCESS1-0", "BNU-ESM"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "ssp": (["ssp"], ["SSP2", "SSP3"]),
            "year": (["year"], [2020, 2099]),
        },
    )

    concatenate_damage_output(
        damage_dir=d,
        basename="test_insuffix",
        save_path=os.path.join(d, "concatenate.zarr"),
    )
    ds_out_actual = xr.open_zarr(os.path.join(d, "concatenate.zarr")).sel(
        batch=["batch" + str(i) for i in range(0, 15)]
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


@pytest.fixture
def labor_in_val_fixture(tmp_path):
    """
    Create fake labor input data for tests and save to netcdf
    """
    ds_in_val = xr.Dataset(
        {
            "regions": (["region"], np.array(["ZWE.test_region", "USA.test_region"])),
            "rebased": (["year", "region"], np.full((4, 2), 2)),
        },
        coords={
            "year": (["year"], [2009, 2010, 2099, 2100]),
        },
    )

    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP3"]
    batch = ["batch" + str(i) for i in range(0, 15)]

    for b in batch:
        for r in rcp:
            for g in gcm:
                for m in model:
                    for s in ssp:
                        d = os.path.join(tmp_path, "labor_in", b, r, g, m, s)
                        if not os.path.exists(d):
                            os.makedirs(d)
                        infile_val = os.path.join(
                            d, "uninteracted_main_model-wage-levels.nc4"
                        )
                        ds_in_val.to_netcdf(infile_val)


@pytest.fixture
def labor_in_histclim_fixture(tmp_path):
    """
    Create fake labor input data for tests and save to netcdf
    """
    ds_in_histclim = xr.Dataset(
        {
            "regions": (["region"], np.array(["ZWE.test_region", "USA.test_region"])),
            "rebased": (["year", "region"], np.full((4, 2), 2)),
        },
        coords={
            "year": (["year"], [2009, 2010, 2099, 2100]),
        },
    )

    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP3"]
    batch = ["batch" + str(i) for i in range(0, 15)]

    for b in batch:
        for r in rcp:
            for g in gcm:
                for m in model:
                    for s in ssp:
                        d = os.path.join(tmp_path, "labor_in", b, r, g, m, s)
                        if not os.path.exists(d):
                            os.makedirs(d)
                        infile_histclim = os.path.join(
                            d, "uninteracted_main_model-histclim-wage-levels.nc4"
                        )
                        ds_in_histclim.to_netcdf(infile_histclim)


@pytest.fixture
def econvars_fixture(tmp_path):
    """
    Create fake socioeconomics input data for tests and save to zarr
    """
    econvars = xr.Dataset(
        {
            "gdp": (["ssp", "region", "model", "year"], np.full((2, 2, 2, 2), 1)),
            "pop": (["ssp", "region", "model", "year"], np.full((2, 2, 2, 2), 1)),
        },
        coords={
            "ssp": (["ssp"], ["SSP2", "SSP3"]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "year": (["year"], [2010, 2099]),
        },
    )

    d = os.path.join(tmp_path, "econvars_for_test", "econvars_for_test.zarr")
    econvars.to_zarr(d)
    econvars_for_test = EconVars(path_econ=d)

    return econvars_for_test


def test_calculate_labor_impacts(
    tmp_path,
    labor_in_val_fixture,
    labor_in_histclim_fixture,
    file_prefix="uninteracted_main_model",
    variable="rebased",
    val_type="wage-levels",
):
    """
    Test that calculate_labor_impacts correctly calculates the impacts for labor results
    """
    ds_out_expected = xr.Dataset(
        {
            "histclim_rebased": (["year", "region"], np.full((2, 2), 2)),
            "delta_rebased": (["year", "region"], np.full((2, 2), 0)),
        },
        coords={
            "year": (["year"], [2010, 2099]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
        },
    )

    d = os.path.join(
        tmp_path, "labor_in", "batch6", "rcp45", "ACCESS1-0", "high", "SSP3"
    )

    ds_out_actual = calculate_labor_impacts(d, file_prefix, variable, val_type)
    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_concatenate_labor_damages(
    tmp_path,
    econvars_fixture,
    labor_in_val_fixture,
    labor_in_histclim_fixture,
):
    """
    Test that concatenate_labor_damages correctly concatenates separate labor damages by batches across SSP-RCP-GCM-IAMs and saves to separate netcdf file by batches
    """
    concatenate_labor_damages(
        input_path=os.path.join(tmp_path, "labor_in"),
        save_path=tmp_path,
        ec_cls=econvars_fixture,
    )

    batch = ["batch" + str(i) for i in range(0, 15)]

    for b in batch:
        ds_out_expected = xr.Dataset(
            {
                "histclim_rebased": (
                    ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                    np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 2 * -1 * 1.273526)),
                ),
                "delta_rebased": (
                    ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                    np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 0)),
                ),
            },
            coords={
                "year": (["year"], [2010, 2099]),
                "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
                "ssp": (["ssp"], ["SSP3"]),
                "rcp": (["rcp"], ["rcp45", "rcp85"]),
                "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "batch": (["batch"], [b]),
            },
        )

        xr.testing.assert_equal(
            ds_out_expected,
            xr.open_dataset(os.path.join(tmp_path, f"rebased_wage-levels_{b}.nc4")),
        )


def test_error_concatenate_labor_damages(
    caplog,
    econvars_fixture,
    tmp_path,
    labor_in_val_fixture,
    labor_in_histclim_fixture,
):
    """
    Test that concatenate_labor_damages returns an exception when any batch cannot be processed succesfully
    """
    os.makedirs(
        os.path.join(tmp_path, "labor_in", "batch6", "rcp45", "CCSM4", "high", "SSP3")
    )
    concatenate_labor_damages(
        input_path=os.path.join(tmp_path, "labor_in"),
        save_path=tmp_path,
        ec_cls=econvars_fixture,
    )
    assert "Error in batchbatch6" in caplog.text

    shutil.rmtree(
        os.path.join(tmp_path, "labor_in", "batch6", "rcp45", "CCSM4", "high", "SSP3")
    )


def test_calculate_labor_batch_damages(
    tmp_path,
    econvars_fixture,
    labor_in_val_fixture,
    labor_in_histclim_fixture,
):
    """
    Test that calculate_labor_batch_damages correctly concatenates labor damages for a single batch across SSP-RCP-GCM-IAMs and saves to zarr file
    """
    ds_out_expected = xr.Dataset(
        {
            "histclim_rebased": (
                ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 2 * -1 * 1.273526)),
            ),
            "delta_rebased": (
                ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 0)),
            ),
        },
        coords={
            "year": (["year"], [2010, 2099]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "ssp": (["ssp"], ["SSP3"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "batch": (["batch"], ["batch6"]),
        },
    )

    calculate_labor_batch_damages(
        batch=6,
        ec=econvars_fixture,
        input_path=os.path.join(tmp_path, "labor_in"),
        save_path=tmp_path,
    )

    ds_out_actual = xr.open_zarr(
        os.path.join(tmp_path, "rebased_wage-levels_batch6.zarr")
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_calculate_labor_damages(
    tmp_path,
    labor_in_val_fixture,
    labor_in_histclim_fixture,
    econvars_fixture,
):
    """
    Test that calculate_labor_damages correctly concatenates separate labor damages by batches across SSP-RCP-GCM-IAMs using multiprocessing and saves to separate zarr file by batches
    """
    calculate_labor_damages(
        path_econ=os.path.join(tmp_path, "econvars_for_test", "econvars_for_test.zarr"),
        input_path=os.path.join(tmp_path, "labor_in"),
        save_path=tmp_path,
    )

    for i in range(0, 15):
        ds_out_expected = xr.Dataset(
            {
                "histclim_rebased": (
                    ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                    np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 2 * -1 * 1.273526)),
                ),
                "delta_rebased": (
                    ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                    np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 0)),
                ),
            },
            coords={
                "year": (["year"], [2010, 2099]),
                "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
                "ssp": (["ssp"], ["SSP3"]),
                "rcp": (["rcp"], ["rcp45", "rcp85"]),
                "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "batch": (["batch"], ["batch" + str(i)]),
            },
        )

        ds_out_actual = xr.open_zarr(
            os.path.join(tmp_path, f"rebased_wage-levels_batch{i}.zarr")
        )

        xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_compute_ag_damages(
    tmp_path,
    econvars_fixture,
):
    """
    Test that compute_ag_damages correctly reshapes ag estimate runs for use in integration system and saves to zarr file
    """
    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP2", "SSP3"]
    batch = ["batch3", "batch6", "batch9"]

    for r in rcp:
        for g in gcm:
            for m in model:
                for s in ssp:
                    for b in batch:
                        d = os.path.join(tmp_path, "ag_in", b, r, g, m, s)
                        if not os.path.exists(d):
                            os.makedirs(d)
                        infile = os.path.join(d, "disaggregated_damages.nc4")

                        ds_in = xr.Dataset(
                            {
                                "wc_no_reallocation": (
                                    [
                                        "gcm",
                                        "model",
                                        "rcp",
                                        "ssp",
                                        "batch",
                                        "region",
                                        "year",
                                        "variable",
                                        "demand_topcode",
                                        "Es_Ed",
                                    ],
                                    np.full((1, 1, 1, 1, 1, 2, 2, 1, 1, 1), 2),
                                ),
                            },
                            coords={
                                "region": (
                                    ["region"],
                                    ["ZWE.test_region", "USA.test_region"],
                                ),
                                "year": (["year"], [2010, 2099]),
                                "ssp": (["ssp"], [s]),
                                "iam": (["iam"], [m]),
                                "continent": (["region"], ["Africa", "Americas"]),
                                "variable": (["variable"], ["gdp"]),
                                "demand_topcode": (["demand_topcode"], ["agshare_10"]),
                                "Es_Ed": (["Es_Ed"], ["0.1_-0.04"]),
                                "gcm": (["gcm"], [g]),
                                "rcp": (["rcp"], [r]),
                                "batch": (["batch"], [b]),
                                "model": (["model"], [m]),
                                "market_level": (["market_level"], ["continent"]),
                            },
                        )

                        ds_in_save = ds_in.sel(iam=m, market_level="continent")

                        ds_in_save.to_netcdf(infile)

    ds_out_expected = xr.Dataset(
        {
            "delta": (
                ["gcm", "model", "rcp", "ssp", "batch", "region", "year"],
                np.float32(np.full((2, 2, 2, 2, 2, 2, 2), 2 * -1 * 1.273526)),
            ),
        },
        coords={
            "batch": (["batch"], ["batch6", "batch9"]),
            "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "ssp": (["ssp"], ["SSP2", "SSP3"]),
            "year": (["year"], [2010, 2099]),
        },
    )

    compute_ag_damages(
        input_path=os.path.join(tmp_path, "ag_in"),
        pop=econvars_fixture.econ_vars.pop,
        topcode="agshare_10",
        integration=True,
        varname="delta",
        save_path=os.path.join(tmp_path, "ag_in", "agriculture_test_output.zarr"),
        scalar=1,
        batches=[6, 9],
    )

    ds_out_actual = xr.open_zarr(
        os.path.join(tmp_path, "ag_in", "agriculture_test_output.zarr")
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


@pytest.fixture
def energy_in_csv_fixture(tmp_path):
    """
    Create fake energy input data for tests and save to csv
    """
    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP2", "SSP3"]
    batch = ["batch" + str(i) for i in range(0, 15)]

    for var in ["delta", "histclim"]:
        for r in rcp:
            for g in gcm:
                for m in model:
                    for s in ssp:
                        for b in batch:
                            d = os.path.join(tmp_path, "energy_in_csv", b, r, g, m, s)
                            if not os.path.exists(d):
                                os.makedirs(d)
                            infile = os.path.join(
                                d, f"TINV_clim_integration_total_energy_{var}.csv"
                            )
                            df = {
                                "region": [
                                    "ZWE.test_region",
                                    "ZWE.test_region",
                                    "USA.test_region",
                                    "USA.test_region",
                                ],
                                "year": [2010, 2099, 2010, 2099],
                                "rebased": [2, 2, 2, 2],
                            }
                            df_in = pd.DataFrame(data=df)
                            df_in.to_csv(infile, index=False)


def test_read_energy_files(
    tmp_path,
    energy_in_csv_fixture,
):
    """
    Test that read_energy_files correctly reads energy csv files, transforms them to Xarray object, and saves to netcdf file
    """
    read_energy_files(
        df=_parse_projection_filesys(
            input_path=os.path.join(tmp_path, "energy_in_csv")
        ),
        seed="TINV_clim_integration_total_energy_delta",
    )

    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP2", "SSP3"]
    batch = ["batch" + str(i) for i in range(0, 15)]

    model_trans = {
        "low": "IIASA GDP",
        "high": "OECD Env-Growth",
    }

    for r in rcp:
        for g in gcm:
            for m in model:
                for s in ssp:
                    for b in batch:
                        d = os.path.join(tmp_path, "energy_in_csv", b, r, g, m, s)
                        d_dir = os.path.join(
                            d, "TINV_clim_integration_total_energy_delta.nc4"
                        )
                        ds_out_actual = xr.open_dataset(d_dir)

                        ds_out_expected = xr.Dataset(
                            {
                                "rebased": (
                                    [
                                        "batch",
                                        "rcp",
                                        "gcm",
                                        "model",
                                        "ssp",
                                        "region",
                                        "year",
                                    ],
                                    np.full((1, 1, 1, 1, 1, 2, 2), 2),
                                ),
                            },
                            coords={
                                "batch": (["batch"], [b]),
                                "rcp": (["rcp"], [r]),
                                "gcm": (["gcm"], [g]),
                                "model": (["model"], [model_trans[m]]),
                                "ssp": (["ssp"], [s]),
                                "region": (
                                    ["region"],
                                    ["USA.test_region", "ZWE.test_region"],
                                ),
                                "year": (["year"], [2010, 2099]),
                            },
                        )

                        xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_error_read_energy_files(
    caplog,
    tmp_path,
    energy_in_csv_fixture,
):
    """
    Test that read_energy_files correctly returns an exception when any file cannot be processed successfully
    """
    os.makedirs(
        os.path.join(
            tmp_path, "energy_in_csv", "batch6", "rcp45", "CCSM4", "high", "SSP4"
        )
    )
    read_energy_files(
        df=_parse_projection_filesys(
            input_path=os.path.join(tmp_path, "energy_in_csv")
        ),
        seed="TINV_clim_integration_total_energy_delta",
    )
    assert "Error in file" in caplog.text

    shutil.rmtree(
        os.path.join(
            tmp_path, "energy_in_csv", "batch6", "rcp45", "CCSM4", "high", "SSP4"
        )
    )


def test_read_energy_files_parallel(
    tmp_path,
    energy_in_csv_fixture,
):
    """
    Test that read_energy_files_parallel correctly concatenates energy results from csv to netcdf by batches using multiprocessing
    """
    read_energy_files_parallel(
        input_path=os.path.join(tmp_path, "energy_in_csv"),
        seed="TINV_clim_integration_total_energy_delta",
    )

    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP2", "SSP3"]
    batch = ["batch" + str(i) for i in range(0, 15)]

    model_trans = {
        "low": "IIASA GDP",
        "high": "OECD Env-Growth",
    }

    for r in rcp:
        for g in gcm:
            for m in model:
                for s in ssp:
                    for b in batch:
                        d = os.path.join(tmp_path, "energy_in_csv", b, r, g, m, s)
                        d_dir = os.path.join(
                            d, "TINV_clim_integration_total_energy_delta.nc4"
                        )
                        ds_out_actual = xr.open_dataset(d_dir)

                        ds_out_expected = xr.Dataset(
                            {
                                "rebased": (
                                    [
                                        "batch",
                                        "rcp",
                                        "gcm",
                                        "model",
                                        "ssp",
                                        "region",
                                        "year",
                                    ],
                                    np.full((1, 1, 1, 1, 1, 2, 2), 2),
                                ),
                            },
                            coords={
                                "batch": (["batch"], [b]),
                                "rcp": (["rcp"], [r]),
                                "gcm": (["gcm"], [g]),
                                "model": (["model"], [model_trans[m]]),
                                "ssp": (["ssp"], [s]),
                                "region": (
                                    ["region"],
                                    ["USA.test_region", "ZWE.test_region"],
                                ),
                                "year": (["year"], [2010, 2099]),
                            },
                        )

                        xr.testing.assert_equal(ds_out_expected, ds_out_actual)


@pytest.fixture
def energy_in_netcdf_fixture(tmp_path):
    """
    Create fake energy input data for tests and save to netcdf
    """
    model_trans = {
        "low": "IIASA GDP",
        "high": "OECD Env-Growth",
    }
    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP2", "SSP3"]
    batch = ["batch" + str(i) for i in range(0, 15)]

    for var in ["delta", "histclim"]:
        for r in rcp:
            for g in gcm:
                for m in model:
                    for s in ssp:
                        for b in batch:
                            d = os.path.join(
                                tmp_path, "energy_in_netcdf", b, r, g, m, s
                            )
                            if not os.path.exists(d):
                                os.makedirs(d)
                            infile = os.path.join(
                                d, f"TINV_clim_integration_total_energy_{var}.nc4"
                            )

                            ds_in = xr.Dataset(
                                {
                                    "rebased": (
                                        [
                                            "batch",
                                            "rcp",
                                            "gcm",
                                            "model",
                                            "ssp",
                                            "region",
                                            "year",
                                        ],
                                        np.full((1, 1, 1, 1, 1, 2, 2), 2).astype(
                                            object
                                        ),
                                    ),
                                },
                                coords={
                                    "batch": (["batch"], [b]),
                                    "rcp": (["rcp"], [r]),
                                    "gcm": (["gcm"], [g]),
                                    "model": (["model"], [model_trans[m]]),
                                    "ssp": (["ssp"], [s]),
                                    "region": (
                                        ["region"],
                                        ["USA.test_region", "ZWE.test_region"],
                                    ),
                                    "year": (["year"], [2010, 2099]),
                                },
                            )

                            ds_in.to_netcdf(infile)


def test_calculate_energy_impacts(
    tmp_path,
    energy_in_netcdf_fixture,
):
    """
    Test that calculate_energy_impacts correctly calculates impacts for energy results for individual modeling unit
    """
    ds_out_expected = xr.Dataset(
        {
            "histclim_rebased": (
                ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                np.full((1, 1, 1, 1, 1, 2, 2), 2),
            ),
            "delta_rebased": (
                ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                np.full((1, 1, 1, 1, 1, 2, 2), 2),
            ),
        },
        coords={
            "batch": (["batch"], ["batch6"]),
            "rcp": (["rcp"], ["rcp45"]),
            "gcm": (["gcm"], ["ACCESS1-0"]),
            "model": (["model"], ["IIASA GDP"]),
            "ssp": (["ssp"], ["SSP3"]),
            "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
            "year": (["year"], [2010, 2099]),
        },
    )

    d = os.path.join(
        tmp_path, "energy_in_netcdf", "batch6", "rcp45", "ACCESS1-0", "low", "SSP3"
    )

    ds_out_actual = calculate_energy_impacts(
        input_path=d,
        file_prefix="TINV_clim_integration_total_energy",
        variable="rebased",
    )
    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_concatenate_energy_damages(
    tmp_path,
    econvars_fixture,
    energy_in_netcdf_fixture,
):
    """
    Test that concatenate_energy_damages correctly concatenates separate energy damages by batches across SSP-RCP-GCM-IAMs and saves to separate netcdf file by batches
    """
    concatenate_energy_damages(
        input_path=os.path.join(tmp_path, "energy_in_netcdf"),
        save_path=tmp_path,
        ec_cls=econvars_fixture,
    )

    batch = ["batch" + str(i) for i in range(0, 15)]

    for b in batch:
        ds_out_expected = xr.Dataset(
            {
                "histclim_rebased": (
                    ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                    np.float32(np.full((1, 2, 2, 2, 2, 2, 2), 2 * 1.273526)),
                ),
                "delta_rebased": (
                    ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                    np.float32(np.full((1, 2, 2, 2, 2, 2, 2), 2 * 1.273526)),
                ),
            },
            coords={
                "batch": (["batch"], [b]),
                "rcp": (["rcp"], ["rcp45", "rcp85"]),
                "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "ssp": (["ssp"], ["SSP2", "SSP3"]),
                "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
                "year": (["year"], [2010, 2099]),
            },
        )

        xr.testing.assert_equal(
            ds_out_expected,
            xr.open_dataset(os.path.join(tmp_path, f"rebased_{b}.nc4")),
        )


def test_error_concatenate_energy_damages(
    caplog,
    econvars_fixture,
    tmp_path,
    energy_in_netcdf_fixture,
):
    """
    Test that concatenate_energy_damages returns an exception when any batch cannot be processed succesfully
    """
    os.makedirs(
        os.path.join(
            tmp_path, "energy_in_netcdf", "batch6", "rcp45", "CCSM4", "high", "SSP3"
        )
    )
    concatenate_energy_damages(
        input_path=os.path.join(tmp_path, "energy_in_netcdf"),
        save_path=tmp_path,
        ec_cls=econvars_fixture,
    )
    assert "Error in batchbatch6" in caplog.text

    os.rmdir(
        os.path.join(
            tmp_path, "energy_in_netcdf", "batch6", "rcp45", "CCSM4", "high", "SSP3"
        )
    )


def test_calculate_energy_batch_damages(
    tmp_path,
    econvars_fixture,
    energy_in_netcdf_fixture,
):
    """
    Test that calculate_energy_batch_damages correctly concatenates energy damages for a single batch across SSP-RCP-GCM-IAMs and saves to zarr file
    """
    ds_out_expected = xr.Dataset(
        {
            "histclim_rebased": (
                ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                np.float32(np.full((1, 2, 2, 2, 2, 2, 2), 2 * 1.273526)),
            ),
            "delta_rebased": (
                ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                np.float32(np.full((1, 2, 2, 2, 2, 2, 2), 2 * 1.273526)),
            ),
        },
        coords={
            "batch": (["batch"], ["batch6"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "ssp": (["ssp"], ["SSP2", "SSP3"]),
            "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
            "year": (["year"], [2010, 2099]),
        },
    )

    calculate_energy_batch_damages(
        batch=6,
        ec=econvars_fixture,
        input_path=os.path.join(tmp_path, "energy_in_netcdf"),
        save_path=tmp_path,
    )

    ds_out_actual = xr.open_zarr(os.path.join(tmp_path, "rebased_batch6.zarr"))

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_calculate_energy_damages(
    tmp_path,
    econvars_fixture,
    energy_in_csv_fixture,
):
    """
    Test that calculate_energy_damages correctly concatenates separate energy damages by batches across SSP-RCP-GCM-IAMs using multiprocessing and saves to separate zarr file by batches
    """
    calculate_energy_damages(
        re_calculate=True,
        path_econ=os.path.join(tmp_path, "econvars_for_test", "econvars_for_test.zarr"),
        input_path=os.path.join(tmp_path, "energy_in_csv"),
        save_path=os.path.join(tmp_path, "energy_out"),
    )
    batch = ["batch" + str(i) for i in range(0, 15)]
    for b in batch:
        ds_out_expected = xr.Dataset(
            {
                "delta_rebased": (
                    ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                    np.float32(np.full((1, 2, 2, 2, 2, 2, 2), 2 * 1.273526)),
                ),
                "histclim_rebased": (
                    ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                    np.float32(np.full((1, 2, 2, 2, 2, 2, 2), 2 * 1.273526)),
                ),
            },
            coords={
                "batch": (["batch"], [b]),
                "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "rcp": (["rcp"], ["rcp45", "rcp85"]),
                "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
                "ssp": (["ssp"], ["SSP2", "SSP3"]),
                "year": (["year"], [2010, 2099]),
            },
        )

        ds_out_actual = xr.open_zarr(
            os.path.join(tmp_path, f"energy_out/rebased_{b}.zarr")
        )

        xr.testing.assert_equal(ds_out_expected, ds_out_actual)


@pytest.mark.parametrize("version_test", [0, 1, 4, 5])
def test_prep_mortality_damages(
    tmp_path,
    version_test,
    econvars_fixture,
):
    """
    Test that prep_mortality_damages correctly reshapes different versions of mortality estimate runs for use in integration system and saves to zarr file
    """
    for b in ["6", "9"]:
        for e in ["1.0", "1.34"]:
            ds_in = xr.Dataset(
                {
                    "monetized_costs": (
                        [
                            "gcm",
                            "batch",
                            "ssp",
                            "rcp",
                            "model",
                            "year",
                            "region",
                            "scaling",
                            "valuation",
                        ],
                        np.full((2, 1, 2, 2, 2, 2, 2, 4, 2), 0),
                    ),
                    "monetized_deaths": (
                        [
                            "gcm",
                            "batch",
                            "ssp",
                            "rcp",
                            "model",
                            "year",
                            "region",
                            "scaling",
                            "valuation",
                        ],
                        np.full((2, 1, 2, 2, 2, 2, 2, 4, 2), 1),
                    ),
                    "monetized_histclim_deaths": (
                        [
                            "gcm",
                            "batch",
                            "ssp",
                            "rcp",
                            "model",
                            "year",
                            "region",
                            "scaling",
                            "valuation",
                        ],
                        np.full((2, 1, 2, 2, 2, 2, 2, 4, 2), 2),
                    ),
                },
                coords={
                    "batch": (["batch"], [b]),
                    "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
                    "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                    "rcp": (["rcp"], ["rcp45", "rcp85"]),
                    "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
                    "scaling": (
                        ["scaling"],
                        ["epa_scaled", "epa_iso_scaled", "epa_popavg", "epa_row"],
                    ),
                    "ssp": (["ssp"], ["SSP2", "SSP3"]),
                    "valuation": (["valuation"], ["vsl", "vly"]),
                    "year": (["year"], [2010, 2099]),
                },
            )

            d = os.path.join(tmp_path, "mortality_in")
            if not os.path.exists(d):
                os.makedirs(d)
            infile = os.path.join(d, f"mortality_damages_batch{b}_eta{e}.zarr")

            ds_in.to_zarr(infile)

    prep_mortality_damages(
        gcms=["ACCESS1-0", "GFDL-CM3"],
        paths=str(
            os.path.join(
                tmp_path, f"mortality_in/mortality_damages_batch{b}_eta{e}.zarr"
            )
        ),
        vars={
            "delta_costs": "monetized_costs",
            "delta_deaths": "monetized_deaths",
            "histclim_deaths": "monetized_histclim_deaths",
        },
        outpath=os.path.join(tmp_path, "mortality_out"),
        mortality_version=version_test,
        path_econ=os.path.join(tmp_path, "econvars_for_test", "econvars_for_test.zarr"),
    )

    ds_out_actual = xr.open_zarr(
        os.path.join(
            tmp_path,
            "mortality_out",
            f"impacts-darwin-montecarlo-damages-v{version_test}.zarr",
        )
    )

    ds_out_expected = xr.Dataset(
        {
            "delta": (
                ["gcm", "batch", "ssp", "rcp", "model", "year", "region", "eta"],
                np.float32(np.full((2, 2, 2, 2, 2, 2, 2, 2), -0.90681089)),
            ),
            "histclim": (
                ["gcm", "batch", "ssp", "rcp", "model", "year", "region", "eta"],
                np.float32(np.full((2, 2, 2, 2, 2, 2, 2, 2), 2 * 0.90681089)),
            ),
        },
        coords={
            "batch": (["batch"], ["batch6", "batch9"]),
            "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
            "ssp": (["ssp"], ["SSP2", "SSP3"]),
            "year": (["year"], [2010, 2099]),
            "eta": (["eta"], [1.0, 1.34]),
        },
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_error_prep_mortality_damages(tmp_path):
    """
    Test that prep_mortality_damages complains when invalid mortality version is passed
    """
    with pytest.raises(ValueError) as excinfo:
        prep_mortality_damages(
            gcms=["ACCESS1-0", "GFDL-CM3"],
            paths=str(
                os.path.join(
                    tmp_path,
                    f"mortality_in/mortality_damages_batch{0}_eta{1}.zarr.zarr",
                )
            ),
            vars={
                "delta_costs": "monetized_costs",
                "delta_deaths": "monetized_deaths",
                "histclim_deaths": "monetized_histclim_deaths",
            },
            outpath=os.path.join(tmp_path, "mortality_out"),
            mortality_version=6,
            path_econ=os.path.join(
                tmp_path, "econvars_for_test", "econvars_for_test.zarr"
            ),
            etas=[1.0, 1.34],
        )
    assert "Mortality version not valid: " in str(excinfo.value)


@pytest.mark.parametrize("version_test", ["v0.21", "v0.20"])
def test_coastal_inputs(
    tmp_path,
    version_test,
):
    """
    Test that coastal_inputs correctly reshapes different versions of coastal results for use in integration system and saves to zarr file (v0.21 and v0.22 have exactly the same structure, so testing either one should be sufficient)
    """
    if version_test == "v0.21":
        ds_in = xr.Dataset(
            {
                "delta": (
                    [
                        "vsl_valuation",
                        "region",
                        "year",
                        "batch",
                        "slr",
                        "model",
                        "ssp",
                        "adapt_type",
                    ],
                    np.full((3, 2, 2, 2, 2, 2, 1, 3), 0),
                ),
                "histclim": (
                    [
                        "vsl_valuation",
                        "region",
                        "year",
                        "batch",
                        "slr",
                        "model",
                        "ssp",
                        "adapt_type",
                    ],
                    np.full((3, 2, 2, 2, 2, 2, 1, 3), 1),
                ),
            },
            coords={
                "adapt_type": (["adapt_type"], ["optimal", "noAdapt", "meanAdapt"]),
                "batch": (["batch"], ["batch3", "batch6"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
                "slr": (["slr"], [0, 9]),
                "ssp": (["ssp"], ["SSP3"]),
                "vsl_valuation": (["vsl_valuation"], ["iso", "row", "global"]),
                "year": (["year"], [2020, 2090]),
            },
        )

    else:
        ds_in = xr.Dataset(
            {
                "delta": (
                    [
                        "region",
                        "year",
                        "batch",
                        "slr",
                        "model",
                        "ssp",
                        "adapt_type",
                    ],
                    np.full((2, 2, 2, 2, 2, 1, 3), 0),
                ),
                "histclim": (
                    [
                        "region",
                        "year",
                        "batch",
                        "slr",
                        "model",
                        "ssp",
                        "adapt_type",
                    ],
                    np.full((2, 2, 2, 2, 2, 1, 3), 1),
                ),
            },
            coords={
                "adapt_type": (["adapt_type"], ["optimal", "noAdapt", "meanAdapt"]),
                "batch": (["batch"], ["batch3", "batch6"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
                "slr": (["slr"], [0, 9]),
                "ssp": (["ssp"], ["SSP3"]),
                "year": (["year"], [2020, 2090]),
            },
        )

    d = os.path.join(tmp_path, "coastal_in")
    if not os.path.exists(d):
        os.makedirs(d)
    infile = os.path.join(d, f"coastal_damages_{version_test}.zarr")

    ds_in.to_zarr(infile)

    if version_test == "v0.21":
        coastal_inputs(
            version=version_test,
            vsl_valuation="iso",
            adapt_type="optimal",
            path=os.path.join(tmp_path, "coastal_in"),
        )

        ds_out_actual = xr.open_zarr(
            os.path.join(
                tmp_path, "coastal_in", "coastal_damages_v0.21-optimal-iso.zarr"
            )
        )

    else:
        coastal_inputs(
            version=version_test,
            adapt_type="optimal",
            path=os.path.join(tmp_path, "coastal_in"),
        )
        ds_out_actual = xr.open_zarr(
            os.path.join(tmp_path, "coastal_in", "coastal_damages_v0.20-optimal.zarr")
        )

    ds_out_expected = xr.Dataset(
        {
            "delta": (
                ["region", "year", "batch", "slr", "model", "ssp"],
                np.full((2, 2, 2, 2, 2, 1), 0),
            ),
            "histclim": (
                ["region", "year", "batch", "slr", "model", "ssp"],
                np.full((2, 2, 2, 2, 2, 1), 1),
            ),
        },
        coords={
            "batch": (["batch"], ["batch3", "batch6"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
            "slr": (["slr"], [0, 9]),
            "ssp": (["ssp"], ["SSP3"]),
            "year": (["year"], [2020, 2090]),
        },
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_error_coastal_inputs(
    tmp_path,
    caplog,
):
    """
    Test that coastal_inputs complains when vsl_valuation is a coordinate in the coastal input file but is set to None
    """
    ds_in = xr.Dataset(
        {
            "delta": (
                [
                    "vsl_valuation",
                    "region",
                    "year",
                    "batch",
                    "slr",
                    "model",
                    "ssp",
                    "adapt_type",
                ],
                np.full((3, 2, 2, 2, 2, 2, 1, 3), 0),
            ),
            "histclim": (
                [
                    "vsl_valuation",
                    "region",
                    "year",
                    "batch",
                    "slr",
                    "model",
                    "ssp",
                    "adapt_type",
                ],
                np.full((3, 2, 2, 2, 2, 2, 1, 3), 1),
            ),
        },
        coords={
            "adapt_type": (["adapt_type"], ["optimal", "noAdapt", "meanAdapt"]),
            "batch": (["batch"], ["batch3", "batch6"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
            "slr": (["slr"], [0, 9]),
            "ssp": (["ssp"], ["SSP3"]),
            "vsl_valuation": (["vsl_valuation"], ["iso", "row", "global"]),
            "year": (["year"], [2020, 2090]),
        },
    )

    d = os.path.join(tmp_path, "coastal_in")
    if not os.path.exists(d):
        os.makedirs(d)
    infile = os.path.join(d, "coastal_damages_v0.22.zarr")

    ds_in.to_zarr(infile)

    with pytest.raises(ValueError) as excinfo:
        coastal_inputs(
            version="v0.22",
            adapt_type="optimal",
            path=os.path.join(tmp_path, "coastal_in"),
        )
    assert (
        str(excinfo.value)
        == "vsl_valuation is a coordinate in the input dataset but is set to None. Please provide a value for vsl_valuation by which to subset the input dataset."
    )
