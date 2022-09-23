import numpy as np
import xarray as xr
import pytest
import os
from dscim.menu.simple_storage import StackedDamages, EconVars, Climate


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
        },
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
        },
    )
    ds_in.to_zarr(infile_path, consolidated=True)

    evs = EconVars(path_econ=str(infile_path))
    actual = evs.econ_vars

    xr.testing.assert_equal(actual, ds_in)


def test_climate_conversion(tmp_path):
    """
    Test that Climate instances give "conversion" from a pulse conversion NetCDF file
    """
    # Set up input data in temporary directory because Climate needs to read
    # from a magical NetCDF on disk.
    d = tmp_path / "climate"
    d.mkdir()
    infile_path = d / "conversion.nc"
    gases = ["CO2_Fossil", "CH4", "N2O"]
    test_numbers = [1.0] * len(gases)
    # This struction might appear weird, but matches that used for initial EPA estimates.
    ds_in = xr.Dataset(
        {
            "__xarray_dataarray_variable__": (["gas"], test_numbers),
        },
        coords={
            "gas": (["gas"], gases),
        },
    )
    ds_in.to_netcdf(infile_path)

    expected = xr.DataArray(test_numbers, coords=[gases], dims=["gas"], name="gas")

    clim = Climate(
        gmst_path="",
        gmsl_path="",
        gmst_fair_path="bacon",
        gmsl_fair_path="bacon",
        pulse_year=1,
        damages_pulse_conversion_path=str(infile_path),
        ecs_mask_path=None,
        emission_scenarios=None,
        gases=gases,
    )
    actual = clim.conversion

    xr.testing.assert_equal(actual, expected)


def test_climate_gmsl_anomalies(tmp_path):
    """
    Test that Climate instances give "gmsl_anomalies" from a GMSL FAIR Zarr Store path
    """
    # Set up input data in temporary directory because Climate needs to read
    # from a magical Zarr Store.
    d = tmp_path / "climate"
    d.mkdir()
    infile_path = d / "gmsl_fair.zarr"
    gases = ["CO2_Fossil", "CH4", "N2O"]
    # This struction might appear weird, but matches that used for initial EPA runs.
    ds_in = xr.Dataset(
        {
            "gmsl": (
                ["runtype", "pulse_year", "simulation", "runid", "year", "gas"],
                np.ones((2, 1, 1, 2, 1, 3)),
            ),
        },
        coords={
            "gas": (["gas"], gases),
            "pulse_year": (["pulse_year"], [2020]),
            "runid": (["runid"], [1, 2]),
            "runtype": (["runtype"], ["control", "pulse"]),
            "simulation": (["simulation"], [1]),
            "year": (["year"], [2020]),
        },
    )
    ds_in.to_zarr(infile_path)

    expected = xr.Dataset(
        {
            "pulse_gmsl": (
                ["pulse_year", "simulation", "runid", "year", "gas"],
                np.ones((1, 1, 2, 1, 3)),
            ),
            "control_gmsl": (
                ["pulse_year", "simulation", "runid", "year", "gas"],
                np.ones((1, 1, 2, 1, 3)),
            ),
        },
        coords={
            "gas": (["gas"], gases),
            "pulse_year": (["pulse_year"], [2020]),
            "runid": (["runid"], [1, 2]),
            "simulation": (["simulation"], [1]),
            "year": (["year"], [2020]),
        },
    )

    clim = Climate(
        gmst_path="",
        gmsl_path="",
        gmst_fair_path="bacon",
        gmsl_fair_path=str(infile_path),
        pulse_year=2020,
        damages_pulse_conversion_path="bacon",
        ecs_mask_path=None,
        emission_scenarios=None,
        gases=gases,
    )
    actual = clim.gmsl_anomalies

    xr.testing.assert_equal(actual, expected)


def test_climate_gmst_anomalies(tmp_path):
    """
    Test that Climate instances give "gmst_anomalies" from a GMST FAIR NetCDF file path
    """
    # Set up input data in temporary directory because Climate needs to read
    # from a magical NetCDF file.
    d = tmp_path / "climate"
    d.mkdir()
    infile_path = d / "gmst_fair.nc"
    gases = ["CH4", "CO2_Fossil", "N2O"]
    # This struction might appear weird, but matches that used for initial EPA runs.
    # Note we're making the pulse slightly higher here.
    # We're also making "year" close to "pulse_year" so we get overlap and some
    # clipping, which seems to be part of what this property/method does.
    ds_in = xr.Dataset(
        {
            "control_temperature": (
                ["gas", "pulse_year", "runid", "year"],
                np.ones((3, 1, 2, 3)),
            ),
            "pulse_temperature": (
                ["gas", "pulse_year", "runid", "year"],
                np.ones((3, 1, 2, 3)) * 2,
            ),
        },
        coords={
            "gas": (["gas"], gases),
            "pulse_year": (["pulse_year"], [2002]),
            "runid": (["runid"], [1, 2]),
            "year": (["year"], [2000, 2001, 2002]),
        },
    )
    ds_in.to_netcdf(infile_path)

    expected = xr.Dataset(
        {
            "control_temperature": (
                ["gas", "pulse_year", "runid", "year"],
                np.zeros((3, 1, 2, 1)),
            ),
            "pulse_temperature": (
                ["gas", "pulse_year", "runid", "year"],
                np.zeros((3, 1, 2, 1)),
            ),
        },
        coords={
            "gas": (["gas"], gases),
            "pulse_year": (["pulse_year"], [2002]),
            "runid": (["runid"], [1, 2]),
            "year": (["year"], [2002]),
        },
    ).chunk({"year": 11})

    clim = Climate(
        gmst_path="",
        gmsl_path="",
        gmsl_fair_path="",
        gmst_fair_path=str(infile_path),
        pulse_year=2002,
        damages_pulse_conversion_path="bacon",
        ecs_mask_path="",
        emission_scenarios=None,
        gases=gases,
    )
    actual = clim.gmst_anomalies

    # Should be lazy dask arrays due to rechunking.
    assert isinstance(
        actual["control_temperature"].data, type(expected["control_temperature"].data)
    )
    assert isinstance(
        actual["pulse_temperature"].data, type(expected["pulse_temperature"].data)
    )
    # Check again, once Dataset is actually computed.
    xr.testing.assert_allclose(actual.compute(), expected.compute())


@pytest.mark.parametrize(
    "test_path,expected",
    [
        pytest.param("a string", ["temperature", "gmsl"], id="GMSL FAIR path not None"),
        pytest.param(None, ["temperature"], id="GMSL FAIR path is None"),
    ],
)
def test_climate_anomaly_vars(test_path, expected):
    """
    Test that Climate instances give "anomaly_vars" given different self.gmsl_fair_path types
    """
    clim = Climate(
        gmst_path="bacon",
        gmsl_path="bacon",
        gmst_fair_path="bacon",
        pulse_year=0,
        damages_pulse_conversion_path="bacon",
        gmsl_fair_path=str(test_path),
    )
    actual = clim.anomaly_vars
    assert actual == expected


def test_climate_anomalies(tmp_path):
    """
    Test that Climate instances give "anomalies" from good input

    This is by no means a comprehensive tests of all the expected behaviors
    here. Just the minimum, matching using similar parameters to what was
    used for inital EPA analysis.
    """
    # Lots and lots of magically formatted setup data required to be on disk so
    # bear with me here. I'm not crazy about all this setup being in the
    # tests.
    # Set up input data in temporary directory because Climate needs to read
    # from a magical NetCDF file.
    d = tmp_path / "climate"
    d.mkdir()
    gmst_infile_path = d / "gmst_fair2.nc"
    gases_gmst = ["CH4", "CO2_Fossil", "N2O"]
    # This struction might appear weird, but matches that used for initial EPA runs.
    # Note we're making the pulse slightly higher here.
    # We're also making "year" close to "pulse_year" so we get overlap and some
    # clipping, which seems to be part of what this property/method does.
    gmst_in = xr.Dataset(
        {
            "control_temperature": (
                ["gas", "pulse_year", "runid", "year"],
                np.ones((3, 1, 2, 3)),
            ),
            "pulse_temperature": (
                ["gas", "pulse_year", "runid", "year"],
                np.ones((3, 1, 2, 3)) * 2,
            ),
        },
        coords={
            "gas": (["gas"], gases_gmst),
            "pulse_year": (["pulse_year"], [2002]),
            "runid": (["runid"], [1, 2]),
            "year": (["year"], [2000, 2001, 2002]),
        },
    )
    gmst_in.to_netcdf(gmst_infile_path)

    gmsl_infile_path = d / "gmsl_fair.zarr"
    gases_gmsl = ["CO2_Fossil", "CH4", "N2O"]
    gmsl_infile_path = d / "gmsl_fair2.zarr"
    # As usual, this struction might appear weird, but matches that used for initial EPA runs.
    gmsl_in = xr.Dataset(
        {
            "gmsl": (
                ["runtype", "pulse_year", "simulation", "runid", "year", "gas"],
                np.ones((2, 1, 1, 2, 3, 3)),
            ),
        },
        coords={
            "gas": (["gas"], gases_gmsl),
            "pulse_year": (["pulse_year"], [2002]),
            "runid": (["runid"], [1, 2]),
            "runtype": (["runtype"], ["control", "pulse"]),
            "simulation": (["simulation"], [1]),
            "year": (["year"], [2000, 2001, 2002]),
        },
    )
    gmsl_in.to_zarr(gmsl_infile_path)

    expected = xr.Dataset(
        {
            "control_temperature": (["gas", "runid", "year"], np.zeros((3, 2, 1))),
            "pulse_temperature": (["gas", "runid", "year"], np.zeros((3, 2, 1))),
            "control_gmsl": (
                ["simulation", "runid", "year", "gas"],
                np.ones((1, 2, 1, 3)),
            ),
            "pulse_gmsl": (
                ["simulation", "runid", "year", "gas"],
                np.ones((1, 2, 1, 3)),
            ),
        },
        coords={
            "year": (["year"], [2002]),
            "gas": (["gas"], gases_gmst),
            "simulation": (["simulation"], [1]),
            "runid": (["runid"], [1, 2]),
        },
    ).chunk({"year": 11})

    clim = Climate(
        gmst_path="",
        gmsl_path="",
        gmsl_fair_path=str(gmsl_infile_path),
        gmst_fair_path=str(gmst_infile_path),
        pulse_year=2002,
        damages_pulse_conversion_path="bacon",
        ecs_mask_path=None,
        emission_scenarios=None,
        gases=gases_gmst,
    )
    actual = clim.anomalies

    # Should be lazy dask arrays due to rechunking.
    assert isinstance(
        actual["control_temperature"].data, type(expected["control_temperature"].data)
    )
    assert isinstance(
        actual["pulse_temperature"].data, type(expected["pulse_temperature"].data)
    )
    assert isinstance(actual["control_gmsl"].data, type(expected["control_gmsl"].data))
    assert isinstance(actual["pulse_gmsl"].data, type(expected["pulse_gmsl"].data))
    # Check again, once Dataset is actually computed.
    xr.testing.assert_allclose(actual.compute(), expected.compute())
