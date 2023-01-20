import xarray as xr
import numpy as np
import math
import pytest
import copy
import dscim.utils.utils as estimations
from dscim.menu.risk_aversion import RiskAversionRecipe
from dscim.preprocessing.preprocessing import subset_USA_ssp_econ, subset_USA_reduced_damages, sum_AMEL
from pathlib import Path
import yaml


def test_subset_USA_ssp_econ(tmp_path):
    """
    Test that subset_USA_ssp_econ returns a Zarr file with only regions containing USA
    """

    d = tmp_path / "USA_econ"
    d.mkdir()
    infile = d / "global_ssp_econ.zarr"
    outfile = d / "USA_ssp_econ.zarr"
    ds_in = xr.Dataset(
        {
            "gdp": (["ssp", "region", "model", "year"], np.ones((1, 2, 2, 3))),
            "gdppc": (["ssp", "region", "model", "year"], np.ones((1, 2, 2, 3))),
            "pop": (["ssp", "region", "model", "year"], np.ones((1, 2, 2, 3))),
        },
        coords={
            "ssp": (["ssp"], ["SSP3"]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "year": (["year"], [2021, 2022, 2023]),
        },
    )
    
    ds_out_expected = xr.Dataset(
        {
            "gdp": (["ssp", "region", "model", "year"], np.ones((1, 1, 2, 3))),
            "gdppc": (["ssp", "region", "model", "year"], np.ones((1, 1, 2, 3))),
            "pop": (["ssp", "region", "model", "year"], np.ones((1, 1, 2, 3))),
        },
        coords={
            "ssp": (["ssp"], ["SSP3"]),
            "region": (["region"], ["USA.test_region"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "year": (["year"], [2021, 2022, 2023]),
        },
    )
    ds_in.to_zarr(infile)
    
    subset_USA_ssp_econ(infile, outfile)
    ds_out_actual = xr.open_zarr(outfile)
    
    xr.testing.assert_equal(ds_out_expected, ds_out_actual)
    
@pytest.mark.parametrize("recipe", ["adding_up", "risk_aversion"])
def test_subset_USA_reduced_damages(tmp_path, recipe):
    """
    Test that subset_USA_reduced_damages returns a Zarr file with only regions containing USA
    """
    eta = 10
    sector = "dummy_sector"
    reduction = "cc"
    
    d = tmp_path / "USA_econ"
    d.mkdir()
    if recipe == "adding_up":
        infile = d / f"{sector}/{recipe}_{reduction}.zarr"
        outfile = d / f"{sector}_USA/{recipe}_{reduction}.zarr"
    else:
        infile = d / f"{sector}/{recipe}_{reduction}_eta{eta}.zarr"
        outfile = d / f"{sector}_USA/{recipe}_{reduction}_eta{eta}.zarr"        

    ds_in = xr.Dataset(
        {
            reduction: (["ssp", "region", "model", "year", "gcm", "rcp"], np.ones((2, 2, 2, 3, 2, 2))),
        },
        coords={
            "ssp": (["ssp"], ["SSP3", "SSP4"]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "year": (["year"], [2021, 2022, 2023]),
            "gcm": (["gcm"], ["Jonahs_GCM", "surrogate_GCM"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"])
        },
    )
    
    ds_out_expected = xr.Dataset(
        {
            reduction: (["ssp", "region", "model", "year", "gcm", "rcp"], np.ones((2, 1, 2, 3, 2, 2))),
        },
        coords={
            "ssp": (["ssp"], ["SSP3", "SSP4"]),
            "region": (["region"], ["USA.test_region"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "year": (["year"], [2021, 2022, 2023]),
            "gcm": (["gcm"], ["Jonahs_GCM", "surrogate_GCM"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"])
        },
    )
    
    
    ds_in.to_zarr(infile,
                consolidated=True,
                mode="w"
                 )
    
    subset_USA_reduced_damages(sector,reduction,recipe,eta,d)
    ds_out_actual = xr.open_zarr(outfile)
    
    xr.testing.assert_equal(ds_out_expected, ds_out_actual)

    
def test_sum_AMEL(tmp_path):
    
    d = tmp_path / "AMEL"
    d.mkdir()
    dummy_AMEL_dir = d / "dummy_AMEL"
    dummy_AMEL_dir.mkdir()
    dummy_sector1_dir = d / "dummy_sector1"
    dummy_sector1_dir.mkdir()
    dummy_sector2_dir = d / "dummy_sector2"
    dummy_sector2_dir.mkdir()
    
    sectors = ["dummy_sector1","dummy_sector2"]
    
    config_data = dict(
        sectors= dict(
            dummy_AMEL = dict(
                sector_path = str(dummy_AMEL_dir / "dummy_AMEL.zarr")
            ),
            dummy_sector1 = dict(
                sector_path = str(dummy_sector1_dir / "dummy_sector1.zarr"),
                delta = "delta_dummy1",
                histclim = "histclim_dummy1"
            ),
            dummy_sector2 = dict(
                sector_path = str(dummy_sector2_dir / "dummy_sector2.zarr"),
                delta = "delta_dummy2",
                histclim = "histclim_dummy2"
            ),
        ),
    )

    config_in = d / "config.yml"
    
    with open(config_in, 'w') as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    
    damages_ds_1 = xr.Dataset(
        {
            "delta_dummy1": (["gcm", "model", "rcp", "ssp", "batch", "region", "year"], np.ones((2, 2, 2, 2, 2, 2, 2))),
            "histclim_dummy1": (["gcm", "model", "rcp", "ssp", "batch", "region", "year"], np.ones((2, 2, 2, 2, 2, 2, 2))),
        },
        coords={
            "gcm": (["gcm"], ["ABCD1", "surrogate_GCM"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "ssp": (["ssp"], ["SSP3", "SSP4"]),
            "batch": (["batch"], [0, 1]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "year": (["year"], [2022, 2023]),
        },
    )
    
    damages_ds_1.to_zarr(dummy_sector1_dir / "dummy_sector1.zarr",
                consolidated=True,
                mode="w")

    damages_ds_2 = xr.Dataset(
        {
            "delta_dummy2": (["gcm", "model", "rcp", "ssp", "batch", "region", "year"], np.ones((2, 2, 2, 2, 2, 2, 2))),
            "histclim_dummy2": (["gcm", "model", "rcp", "ssp", "batch", "region", "year"], np.ones((2, 2, 2, 2, 2, 2, 2))),
        },
        coords={
            "gcm": (["gcm"], ["ABCD1", "surrogate_GCM"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "ssp": (["ssp"], ["SSP3", "SSP4"]),
            "batch": (["batch"], [0, 1]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "year": (["year"], [2022, 2023]),
        },
    ) + 1
    
    damages_ds_2.to_zarr(dummy_sector2_dir / "dummy_sector2.zarr",
            consolidated=True,
            mode="w")
    
    damages_out_expected = (xr.Dataset(
        {
            "summed_delta": (["gcm", "model", "rcp", "ssp", "batch", "region", "year"], np.ones((2, 2, 2, 2, 2, 2, 2))),
            "summed_histclim": (["gcm", "model", "rcp", "ssp", "batch", "region", "year"], np.ones((2, 2, 2, 2, 2, 2, 2))),
        },
        coords={
            "gcm": (["gcm"], ["ABCD1", "surrogate_GCM"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "rcp": (["rcp"], ["rcp45", "rcp85"]),
            "ssp": (["ssp"], ["SSP3", "SSP4"]),
            "batch": (["batch"], [0, 1]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "year": (["year"], [2022, 2023]),
        },) + 2).astype(np.float32).chunk({
        "batch": 15,
        "ssp": 1,
        "model": 1,
        "rcp": 1,
        "gcm": 1,
        "year": 10,
        "region": 24378,
    })
    
    sum_AMEL(sectors, config_in, "dummy_AMEL")
    xr.testing.assert_equal(xr.open_zarr(dummy_AMEL_dir / "dummy_AMEL.zarr"), damages_out_expected)