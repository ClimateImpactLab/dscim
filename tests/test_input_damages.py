import os
import numpy as np
import xarray as xr
import pandas as pd
import pytest
import logging
from itertools import chain, repeat
from dscim.menu.simple_storage import EconVars
from dscim.preprocessing.input_damages import (
    _parse_projection_filesys,
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
    coastal_inputs,
)

logger = logging.getLogger(__name__)


def test_parse_projection_filesys(tmp_path):
    rcp = ["rcp85", "rcp45"]
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
        "batch": list(chain(repeat("batch9", 16), repeat("batch6", 16))),
        "rcp": list(chain(repeat("rcp85", 8), repeat("rcp45", 8))) * 2,
        "gcm": list(chain(repeat("ACCESS1-0", 4), repeat("GFDL-CM3", 4))) * 4,
        "model": list(chain(repeat("high", 2), repeat("low", 2))) * 8,
        "ssp": ["SSP2", "SSP3"] * 16,
        "path": [
            os.path.join(tmp_path, b, r, g, m, s)
            for b in ["batch9", "batch6"]
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
    df_out_actual.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(df_out_expected, df_out_actual)


@pytest.fixture
def labor_in_val(tmp_path):
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
def labor_in_histclim(tmp_path):
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
def econvars(tmp_path):
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
    labor_in_val,
    labor_in_histclim,
    file_prefix="uninteracted_main_model",
    variable="rebased",
    val_type="wage-levels",
):
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


@pytest.mark.parametrize("out_format", ["return", "save"])
def test_concatenate_labor_damages(
    tmp_path,
    econvars,
    labor_in_val,
    labor_in_histclim,
    out_format,
):
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
            "batch": (["batch"], ["batch9"]),
        },
    )

    ds_out_actual = concatenate_labor_damages(
        input_path=os.path.join(tmp_path, "labor_in"),
        save_path=tmp_path,
        ec_cls=econvars,
    )

    if out_format == "return":
        xr.testing.assert_equal(ds_out_expected, ds_out_actual)

    elif out_format == "save":
        xr.testing.assert_equal(
            ds_out_expected,
            xr.open_dataset(os.path.join(tmp_path, "rebased_wage-levels_batch9.nc4")),
        )


def test_error_concatenate_labor_damages(
    caplog,
    econvars,
    tmp_path,
    labor_in_val,
    labor_in_histclim,
):
    os.makedirs(
        os.path.join(tmp_path, "labor_in", "batch6", "rcp45", "CCSM4", "high", "SSP3")
    )
    concatenate_labor_damages(
        input_path=os.path.join(tmp_path, "labor_in"),
        save_path=tmp_path,
        ec_cls=econvars,
    )
    assert "Error in batchbatch6" in caplog.text

    os.rmdir(
        os.path.join(tmp_path, "labor_in", "batch6", "rcp45", "CCSM4", "high", "SSP3")
    )


def test_calculate_labor_batch_damages(
    tmp_path,
    econvars,
    labor_in_val,
    labor_in_histclim,
):
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
        ec=econvars,
        input_path=os.path.join(tmp_path, "labor_in"),
        save_path=tmp_path,
    )

    ds_out_actual = xr.open_zarr(
        os.path.join(tmp_path, "rebased_wage-levels_batch6.zarr")
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_calculate_labor_damages(
    tmp_path,
    labor_in_val,
    labor_in_histclim,
    econvars,
):
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
    econvars,
):
    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP2", "SSP3"]
    batch = ["batch6", "batch9"]

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
                np.full((2, 2, 2, 2, 2, 2, 2), 2 * -1 * 1.273526),
            ),
        },
        coords={
            "batch": (["batch"], ["batch6", "batch9"]),
            "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "ssp": (["ssp"], ["SSP2", "SSP3"]),
            "year": ([2010, 2099]),
        },
    )

    compute_ag_damages(
        input_path=os.path.join(tmp_path, "ag_in"),
        pop=econvars.econ_vars.pop,
        topcode="agshare_10",
        integration=True,
        varname="delta",
        save_path=os.path.join(tmp_path, "ag_in", "agriculture_test_output.zarr"),
        scalar=1,
    )

    ds_out_actual = xr.open_zarr(
        os.path.join(tmp_path, "ag_in", "agriculture_test_output.zarr")
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


@pytest.fixture
def energy_in_csv(tmp_path):
    rcp = ["rcp45", "rcp85"]
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
    ]
    model = ["low", "high"]
    ssp = ["SSP3"]
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
    energy_in_csv,
):
    read_energy_files(
        df=_parse_projection_filesys(
            input_path=os.path.join(tmp_path, "energy_in_csv")
        ),
        seed="TINV_clim_integration_total_energy_delta",
    )

    rcp = ["rcp45", "rcp85"]
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
    ]
    model = ["low", "high"]
    ssp = ["SSP3"]
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


def test_read_energy_files_parallel(
    tmp_path,
    energy_in_csv,
):
    read_energy_files_parallel(
        input_path=os.path.join(tmp_path, "energy_in_csv"),
        seed="TINV_clim_integration_total_energy_delta",
    )

    rcp = ["rcp45", "rcp85"]
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
    ]
    model = ["low", "high"]
    ssp = ["SSP3"]
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
def energy_in_netcdf(tmp_path):
    model_trans = {
        "low": "IIASA GDP",
        "high": "OECD Env-Growth",
    }
    rcp = ["rcp45", "rcp85"]
    gcm = ["ACCESS1-0", "GFDL-CM3"]
    model = ["low", "high"]
    ssp = ["SSP3"]
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

                            ds_in.to_netcdf(infile)


def test_calculate_energy_impacts(
    tmp_path,
    energy_in_netcdf,
):
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


@pytest.mark.parametrize("out_format", ["return", "save"])
def test_concatenate_energy_damages(
    tmp_path,
    econvars,
    energy_in_netcdf,
    out_format,
):
    ds_out_expected = xr.Dataset(
        {
            "histclim_rebased": (
                ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 2 * 1.273526)),
            ),
            "delta_rebased": (
                ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 2 * 1.273526)),
            ),
        },
        coords={
            "batch": (["batch"], ["batch9"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "ssp": (["ssp"], ["SSP3"]),
            "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
            "year": (["year"], [2010, 2099]),
        },
    )

    ds_out_actual = concatenate_energy_damages(
        input_path=os.path.join(tmp_path, "energy_in_netcdf"),
        save_path=tmp_path,
        ec_cls=econvars,
    )

    if out_format == "return":
        xr.testing.assert_equal(ds_out_expected, ds_out_actual)

    elif out_format == "save":
        xr.testing.assert_equal(
            ds_out_expected,
            xr.open_dataset(os.path.join(tmp_path, "rebased_batch9.nc4")),
        )


def test_error_concatenate_energy_damages(
    caplog,
    econvars,
    tmp_path,
    energy_in_netcdf,
):
    os.makedirs(
        os.path.join(
            tmp_path, "energy_in_netcdf", "batch6", "rcp45", "CCSM4", "high", "SSP3"
        )
    )
    concatenate_energy_damages(
        input_path=os.path.join(tmp_path, "energy_in_netcdf"),
        save_path=tmp_path,
        ec_cls=econvars,
    )
    assert "Error in batchbatch6" in caplog.text

    os.rmdir(
        os.path.join(
            tmp_path, "energy_in_netcdf", "batch6", "rcp45", "CCSM4", "high", "SSP3"
        )
    )


def test_calculate_energy_batch_damages(
    tmp_path,
    econvars,
    energy_in_netcdf,
):
    ds_out_expected = xr.Dataset(
        {
            "histclim_rebased": (
                ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 2 * 1.273526)),
            ),
            "delta_rebased": (
                ["batch", "rcp", "gcm", "model", "ssp", "region", "year"],
                np.float32(np.full((1, 2, 2, 2, 1, 2, 2), 2 * 1.273526)),
            ),
        },
        coords={
            "batch": (["batch"], ["batch6"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "gcm": (["gcm"], ["ACCESS1-0", "GFDL-CM3"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "ssp": (["ssp"], ["SSP3"]),
            "region": (["region"], ["USA.test_region", "ZWE.test_region"]),
            "year": (["year"], [2010, 2099]),
        },
    )

    calculate_energy_batch_damages(
        batch=6,
        ec=econvars,
        input_path=os.path.join(tmp_path, "energy_in_netcdf"),
        save_path=tmp_path,
    )

    ds_out_actual = xr.open_zarr(os.path.join(tmp_path, "rebased_batch6.zarr"))

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


@pytest.mark.parametrize("version_test", ["v0.21", "v0.20"])
def test_coastal_inputs(tmp_path, version_test):
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
                "year": ([2020, 2090]),
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
                "year": ([2020, 2090]),
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
            "year": ([2020, 2090]),
        },
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_error_coastal_inputs(tmp_path, caplog):
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
            "year": ([2020, 2090]),
        },
    )

    d = os.path.join(tmp_path, "coastal_in")
    if not os.path.exists(d):
        os.makedirs(d)
    infile = os.path.join(d, f"coastal_damages_v0.22.zarr")

    ds_in.to_zarr(infile)

    coastal_inputs(
        version="v0.22",
        adapt_type="optimal",
        path=os.path.join(tmp_path, "coastal_in"),
    )

    assert "ValueError" in caplog.text
