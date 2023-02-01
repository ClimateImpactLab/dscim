import os
import numpy as np
import xarray as xr
import pandas as pd
from itertools import chain, repeat
from dscim.preprocessing.input_damages import (
    _parse_projection_filesys,
    calculate_labor_impacts,
    concatenate_labor_damages,
    calculate_labor_batch_damages,
)


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
        "batch": list(chain(repeat("batch6", 16), repeat("batch9", 16))),
        "rcp": list(chain(repeat("rcp85", 8), repeat("rcp45", 8))) * 2,
        "gcm": list(chain(repeat("ACCESS1-0", 4), repeat("GFDL-CM3", 4))) * 4,
        "model": list(chain(repeat("high", 2), repeat("low", 2))) * 8,
        "ssp": ["SSP2", "SSP3"] * 16,
        "path": [
            os.path.join(tmp_path, b, r, g, m, s)
            for b in batch
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


def test_calculate_labor_impacts(
    tmp_path,
    file_prefix="test_file_prefix",
    variable="test_variable",
    val_type="test_val_type",
):
    d = os.path.join(tmp_path, "labor_impacts")
    os.makedirs(d)
    infile_val = os.path.join(d, f"{file_prefix}-{val_type}.nc4")
    infile_histclim = os.path.join(d, f"{file_prefix}-histclim-{val_type}.nc4")

    ds_in_val = xr.Dataset(
        {
            "regions": (["region"], np.array(["ZWE.test_region", "USA.test_region"])),
            "test_variable": (
                ["year", "region"],
                np.array([[10, 9], [8, 7], [6, 5], [4, 3]]),
            ),
        },
        coords={
            "year": (["year"], [2009, 2010, 2099, 2100]),
        },
    )

    ds_in_histclim = xr.Dataset(
        {
            "regions": (["region"], np.array(["ZWE.test_region", "USA.test_region"])),
            "test_variable": (
                ["year", "region"],
                np.array([[3, 4], [5, 6], [7, 8], [9, 10]]),
            ),
        },
        coords={
            "year": (["year"], [2009, 2010, 2099, 2100]),
        },
    )

    ds_out_expected = xr.Dataset(
        {
            "histclim_test_variable": (["year", "region"], np.array([[5, 6], [7, 8]])),
            "delta_test_variable": (["year", "region"], np.array([[3, 1], [-1, -3]])),
        },
        coords={
            "year": (["year"], [2010, 2099]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
        },
    )

    ds_in_val.to_netcdf(infile_val)
    ds_in_histclim.to_netcdf(infile_histclim)

    ds_out_actual = calculate_labor_impacts(d, file_prefix, variable, val_type)
    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_concatenate_labor_damages(
    tmp_path,
    econ,
    file_prefix="test_file_prefix",
    variable="test_variable",
    val_type="test_val_type",
):
    rcp = "rcp85"
    gcm = "ACCESS1-0"
    model = "high"
    ssp = "SSP3"
    batch = "batch6"

    d = os.path.join(tmp_path, "concatenate_labor_damages", batch, rcp, gcm, model, ssp)
    os.makedirs(d)
    infile_val = os.path.join(d, f"{file_prefix}-{val_type}.nc4")
    infile_histclim = os.path.join(d, f"{file_prefix}-histclim-{val_type}.nc4")

    ds_in_val = xr.Dataset(
        {
            "regions": (["region"], np.array(["IND.21.317.1249", "CAN.2.33.913"])),
            "test_variable": (["year", "region"], np.full((4, 2), 3)),
        },
        coords={
            "year": (["year"], [2009, 2010, 2099, 2100]),
        },
    )

    ds_in_histclim = xr.Dataset(
        {
            "regions": (["region"], np.array(["IND.21.317.1249", "CAN.2.33.913"])),
            "test_variable": (["year", "region"], np.full((4, 2), 1)),
        },
        coords={
            "year": (["year"], [2009, 2010, 2099, 2100]),
        },
    )
    ds_in_val.to_netcdf(infile_val)
    ds_in_histclim.to_netcdf(infile_histclim)

    ds_out_expected = xr.Dataset(
        {
            "histclim_test_variable": (
                ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                np.array(
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            np.float32(
                                                list(
                                                    1
                                                    / econ.econ_vars.pop.sel(
                                                        year=[2010],
                                                        model="OECD Env-Growth",
                                                        ssp="SSP3",
                                                        region=[
                                                            "IND.21.317.1249",
                                                            "CAN.2.33.913",
                                                        ],
                                                    ).values.flatten()
                                                    * -1
                                                    * 1.273526
                                                )
                                            ),
                                            np.float32(
                                                list(
                                                    1
                                                    / econ.econ_vars.pop.sel(
                                                        year=[2099],
                                                        model="OECD Env-Growth",
                                                        ssp="SSP3",
                                                        region=[
                                                            "IND.21.317.1249",
                                                            "CAN.2.33.913",
                                                        ],
                                                    ).values.flatten()
                                                    * -1
                                                    * 1.273526
                                                )
                                            ),
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ),
            ),
            "delta_test_variable": (
                ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                np.array(
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            np.float32(
                                                list(
                                                    2
                                                    / econ.econ_vars.pop.sel(
                                                        year=[2010],
                                                        model="OECD Env-Growth",
                                                        ssp="SSP3",
                                                        region=[
                                                            "IND.21.317.1249",
                                                            "CAN.2.33.913",
                                                        ],
                                                    ).values.flatten()
                                                    * -1
                                                    * 1.273526
                                                )
                                            ),
                                            np.float32(
                                                list(
                                                    2
                                                    / econ.econ_vars.pop.sel(
                                                        year=[2099],
                                                        model="OECD Env-Growth",
                                                        ssp="SSP3",
                                                        region=[
                                                            "IND.21.317.1249",
                                                            "CAN.2.33.913",
                                                        ],
                                                    ).values.flatten()
                                                    * -1
                                                    * 1.273526
                                                )
                                            ),
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ),
            ),
        },
        coords={
            "year": (["year"], [2010, 2099]),
            "region": (["region"], ["IND.21.317.1249", "CAN.2.33.913"]),
            "ssp": (["ssp"], ["SSP3"]),
            "rcp": (["rcp"], ["rcp85"]),
            "gcm": (["gcm"], ["ACCESS1-0"]),
            "model": (["model"], ["OECD Env-Growth"]),
            "batch": (["batch"], ["batch6"]),
        },
    )

    ds_out_actual = concatenate_labor_damages(
        input_path=os.path.join(tmp_path, "concatenate_labor_damages"),
        save_path=tmp_path,
        ec_cls=econ,
        file_prefix=file_prefix,
        variable=variable,
        val_type=val_type,
    )
    xr.testing.assert_equal(ds_out_expected, ds_out_actual)


def test_calculate_labor_batch_damages(
    tmp_path,
    econ,
    file_prefix="uninteracted_main_model",
    variable="rebased",
    val_type="wage-levels",
):
    rcp = "rcp85"
    gcm = "ACCESS1-0"
    model = "high"
    ssp = "SSP3"
    batch = ["batch6", "batch9"]

    for b in batch:
        i = 1

        d = os.path.join(
            tmp_path, "calculate_labor_batch_damages", b, rcp, gcm, model, ssp
        )
        os.makedirs(d)
        infile_val = os.path.join(d, f"{file_prefix}-{val_type}.nc4")
        infile_histclim = os.path.join(d, f"{file_prefix}-histclim-{val_type}.nc4")

        ds_in_val = xr.Dataset(
            {
                "regions": (["region"], np.array(["IND.21.317.1249", "CAN.2.33.913"])),
                "rebased": (["year", "region"], np.full((4, 2), i + 2)),
            },
            coords={
                "year": (["year"], [2009, 2010, 2099, 2100]),
            },
        )

        ds_in_histclim = xr.Dataset(
            {
                "regions": (["region"], np.array(["IND.21.317.1249", "CAN.2.33.913"])),
                "rebased": (["year", "region"], np.full((4, 2), i)),
            },
            coords={
                "year": (["year"], [2009, 2010, 2099, 2100]),
            },
        )

        ds_in_val.to_netcdf(infile_val)
        ds_in_histclim.to_netcdf(infile_histclim)

        i = i + 1

    ds_out_expected = xr.Dataset(
        {
            "histclim_rebased": (
                ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                np.array(
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            np.float32(
                                                list(
                                                    1
                                                    / econ.econ_vars.pop.sel(
                                                        year=[2010],
                                                        model="OECD Env-Growth",
                                                        ssp="SSP3",
                                                        region=[
                                                            "IND.21.317.1249",
                                                            "CAN.2.33.913",
                                                        ],
                                                    ).values.flatten()
                                                    * -1
                                                    * 1.273526
                                                )
                                            ),
                                            np.float32(
                                                list(
                                                    1
                                                    / econ.econ_vars.pop.sel(
                                                        year=[2099],
                                                        model="OECD Env-Growth",
                                                        ssp="SSP3",
                                                        region=[
                                                            "IND.21.317.1249",
                                                            "CAN.2.33.913",
                                                        ],
                                                    ).values.flatten()
                                                    * -1
                                                    * 1.273526
                                                )
                                            ),
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ),
            ),
            "delta_rebased": (
                ["ssp", "rcp", "model", "gcm", "batch", "year", "region"],
                np.array(
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            np.float32(
                                                list(
                                                    2
                                                    / econ.econ_vars.pop.sel(
                                                        year=[2010],
                                                        model="OECD Env-Growth",
                                                        ssp="SSP3",
                                                        region=[
                                                            "IND.21.317.1249",
                                                            "CAN.2.33.913",
                                                        ],
                                                    ).values.flatten()
                                                    * -1
                                                    * 1.273526
                                                )
                                            ),
                                            np.float32(
                                                list(
                                                    2
                                                    / econ.econ_vars.pop.sel(
                                                        year=[2099],
                                                        model="OECD Env-Growth",
                                                        ssp="SSP3",
                                                        region=[
                                                            "IND.21.317.1249",
                                                            "CAN.2.33.913",
                                                        ],
                                                    ).values.flatten()
                                                    * -1
                                                    * 1.273526
                                                )
                                            ),
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ),
            ),
        },
        coords={
            "year": (["year"], [2010, 2099]),
            "region": (["region"], ["IND.21.317.1249", "CAN.2.33.913"]),
            "ssp": (["ssp"], ["SSP3"]),
            "rcp": (["rcp"], ["rcp85"]),
            "gcm": (["gcm"], ["ACCESS1-0"]),
            "model": (["model"], ["OECD Env-Growth"]),
            "batch": (["batch"], ["batch6"]),
        },
    )

    calculate_labor_batch_damages(
        batch=6,
        ec=econ,
        input_path=os.path.join(tmp_path, "calculate_labor_batch_damages"),
        save_path=os.path.join(tmp_path, "calculate_labor_batch_damages"),
    )

    ds_out_actual = xr.open_zarr(
        os.path.join(
            tmp_path, "calculate_labor_batch_damages/rebased_wage-levels_batch6.zarr"
        )
    )

    xr.testing.assert_equal(ds_out_expected, ds_out_actual)
