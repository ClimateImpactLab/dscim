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
            "gdp": (["ssp", "region", "model", "year"], np.full((1, 2, 2, 2), 1)),
            "pop": (["ssp", "region", "model", "year"], np.full((1, 2, 2, 2), 1)),
        },
        coords={
            "ssp": (["ssp"], ["SSP3"]),
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
