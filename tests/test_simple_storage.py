import numpy as np
import xarray as xr
import pytest
import os
from dscim.menu.simple_storage import StackedDamages, EconVars


@pytest.fixture
def stacked_damages(econ, climate):
    datadir = os.path.join(os.path.dirname(__file__), "data")
    stacked_damages = StackedDamages(
        sector_path=[{"dummy_sector": os.path.join(datadir, "damages")}],
        save_path=None,
        econ_vars=econ,
        climate_vars=climate,
        eta=1.421158116,
        gdppc_bottom_code=234.235646874999,
        subset_dict={
            "ssp": ["SSP2", "SSP3", "SSP4"],
            "region": [
                "IND.21.317.1249",
                "CAN.2.33.913",
                "USA.14.608",
                "EGY.11",
                "SDN.4.11.50.164",
                "NGA.25.510",
                "SAU.7",
                "RUS.16.430.430",
                "SOM.2.5",
            ],
        },
        ce_path=os.path.join(os.path.dirname(__file__), "data/CEs"),
    )

    yield stacked_damages


def test_adding_up_damages(stacked_damages):

    """
    checks that does the only single thing it's supposed to be doing (no parameters)
    """

    xr.testing.assert_equal(
        stacked_damages.adding_up_damages,
        (
            (
                xr.open_zarr(f"{stacked_damages.ce_path}/adding_up_no_cc.zarr").no_cc
                - xr.open_zarr(f"{stacked_damages.ce_path}/adding_up_cc.zarr").cc
            )
            * stacked_damages.pop
        ).sum("region"),
    )


def test_econvars_netcdf(tmp_path):
    """
    Test that EconVars instances give "gdp", "pop" from NetCDF file
    """
    # Set up input data in temporary directory because EconVars needs to read
    # from file on directory.
    d = tmp_path / "econvars"
    d.mkdir()
    infile_path = d / "data.nc"
    ds_in = xr.Dataset(
        {
        "pop": (["region", "runid", "year"], np.ones((1, 2, 3))),
        "gdp": (["region", "runid", "year"], np.ones((1, 2, 3))),
        },
        coords={
            "region": (["region"], ["a"]),
            "runid": (["runid"], [1, 2]),
            "year": (["year"], [5, 6, 7]),
        }
    )
    ds_in.to_netcdf(infile_path)

    evs = EconVars(path_econ=str(infile_path))
    actual = evs.econ_vars

    xr.testing.assert_equal(actual, ds_in)


def test_econvars_zarr(tmp_path):
    """
    Test that EconVars instances give "gdp", "pop" from Zarr store
    """
    # Set up input data in temporary directory because EconVars needs to read
    # from Zarr Store on disk.
    d = tmp_path / "econvars"
    d.mkdir()
    infile_path = d / "data.zarr"
    ds_in = xr.Dataset(
        {
            "pop": (["region", "runid", "year"], np.ones((1, 2, 3))),
            "gdp": (["region", "runid", "year"], np.ones((1, 2, 3))),
        },
        coords={
            "region": (["region"], ["a"]),
            "runid": (["runid"], [1, 2]),
            "year": (["year"], [5, 6, 7]),
        }
    )
    ds_in.to_zarr(infile_path, consolidated=True)

    evs = EconVars(path_econ=str(infile_path))
    actual = evs.econ_vars

    xr.testing.assert_equal(actual, ds_in)


