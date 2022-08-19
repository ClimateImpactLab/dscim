import numpy as np
import xarray as xr
import pytest
import os
from dscim.menu.simple_storage import StackedDamages


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
