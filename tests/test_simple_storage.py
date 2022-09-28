from dataclasses import dataclass
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
        gmsl_fair_path=test_path,
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


def test_stackeddamages_cut():
    """
    Test basic StackedDamages.cut on pulse_year
    """
    # Lots of setup for this on.
    # First, create the DataArray we want to .cut(). We want some values
    # before the pulse year and some values after the end_year so we can
    # check they're cut.
    time = np.arange(2045, 2080)
    x_fake = np.arange(len(time))
    input_xr = xr.DataArray(x_fake, coords=[time], dims=["year"], name="foobar")
    expected = input_xr.sel(year=slice(2048, 2075))
    # Setup input data to instantiate a minimalist StackedDamages
    # with enough fake data that we can test basic behavior.

    @dataclass
    class MockClimate:
        pulse_year: int

    fake_climate = MockClimate(pulse_year=2050)

    damages = StackedDamages(
        sector_path="",
        save_path="",
        econ_vars="FakeEconVars",
        climate_vars=fake_climate,
        eta=0,
        gdppc_bottom_code="???",
        subset_dict={},
    )

    actual = damages.cut(input_xr, end_year=2075)

    xr.testing.assert_equal(actual, expected)


def test_stackeddamages_cut_subset_dict():
    """
    Test StackedDamages.cut on self.climate.pulse_year and a self.subset_dict
    """
    # Lots of setup for this on.
    # First, create the DataArray we want to .cut(). We want some values
    # before the pulse year and some values after the end_year so we can
    # check they're cut. We also want an extra coordinate (dim0) that will
    # get subset by `subset_dict` once we instantiate StackedDamages.
    time = np.arange(2045, 2080)
    x_fake = np.arange(len(time))
    input_xr = xr.DataArray(
        [x_fake, x_fake],
        coords=[["a", "b"], time],
        dims=["dim0", "year"],
        name="fakedata",
    )
    expected = input_xr.sel(dim0=["b"], year=slice(2048, 2075))
    # Setup input data to instantiate a minimalist StackedDamages
    # with enough fake data that we can test basic behavior.

    @dataclass
    class MockClimate:
        pulse_year: int

    fake_climate = MockClimate(pulse_year=2050)

    damages = StackedDamages(
        sector_path="",
        save_path="",
        econ_vars="FakeEconVars",
        climate_vars=fake_climate,
        eta=0,
        gdppc_bottom_code="???",
        subset_dict={"dim0": ["b"]},
    )

    actual = damages.cut(input_xr, end_year=2075)

    xr.testing.assert_equal(actual, expected)


def test_stackeddamages_cut_econ_vars_2099():
    """
    Test StackedDamages.cut_econ_vars clips on pulse_year without 2300 year
    """
    # Lots of setup for this on.
    # First, create the input Dataset. We want some values before the pulse
    # year and some values after,
    # but not 2300, because this seems to trigger a particular behavior.
    time = np.arange(2045, 2110)
    x_fake = np.ones(len(time))
    input_xr = xr.DataArray(
        x_fake, coords=[time], dims=["year"], name="foobar"
    ).to_dataset()
    expected = input_xr.sel(year=slice(2048, 2099))
    # Setup input data to instantiate a minimalist StackedDamages
    # with enough fake data that we can test basic behavior.

    @dataclass
    class MockClimate:
        pulse_year: int

    @dataclass
    class MockEconVars:
        econ_vars: xr.Dataset

    fake_climate = MockClimate(pulse_year=2050)
    fake_econvars = MockEconVars(econ_vars=input_xr)

    damages = StackedDamages(
        sector_path="",
        save_path="",
        econ_vars=fake_econvars,
        climate_vars=fake_climate,
        eta=0,
        gdppc_bottom_code=0.00,
        subset_dict={},
    )

    actual = damages.cut_econ_vars

    xr.testing.assert_equal(actual, expected)


def test_stackeddamages_cut_econ_vars_2300():
    """
    Test StackedDamages.cut_econ_vars clips on pulse_year *with* year 2300
    """
    # Lots of setup for this on.
    # First, create the input Dataset. We want some values before the pulse
    # year and some values after,
    # INCLUDING 2300 here, because this triggers a particular behavior.
    time = np.arange(2245, 2310)
    x_fake = np.ones(len(time))
    input_xr = xr.DataArray(
        x_fake, coords=[time], dims=["year"], name="foobar"
    ).to_dataset()
    expected = input_xr.sel(year=slice(2248, 2300))
    # Setup input data to instantiate a minimalist StackedDamages
    # with enough fake data that we can test basic behavior.

    @dataclass
    class MockClimate:
        pulse_year: int

    @dataclass
    class MockEconVars:
        econ_vars: xr.Dataset

    fake_climate = MockClimate(pulse_year=2250)
    fake_econvars = MockEconVars(econ_vars=input_xr)

    damages = StackedDamages(
        sector_path="",
        save_path="",
        econ_vars=fake_econvars,
        climate_vars=fake_climate,
        eta=0,
        gdppc_bottom_code=0.00,
        subset_dict={},
    )

    actual = damages.cut_econ_vars

    xr.testing.assert_equal(actual, expected)


def test_stackeddamages_gdp():
    """
    Test StackedDamages.gdp returns something "cut".
    """
    # Lots of setup for this on.
    # First, create the input Dataset. We want some values before the pulse
    # year and some values after.
    time = np.arange(2045, 2110)
    x_fake = np.ones(len(time))
    input_xr = xr.DataArray(
        x_fake, coords=[time], dims=["year"], name="gdp"
    ).to_dataset()
    expected = input_xr.sel(year=slice(2048, 2099))["gdp"]
    # Setup input data to instantiate a minimalist StackedDamages
    # with enough fake data that we can test basic behavior.

    @dataclass
    class MockClimate:
        pulse_year: int

    @dataclass
    class MockEconVars:
        econ_vars: xr.Dataset

    fake_climate = MockClimate(pulse_year=2050)
    fake_econvars = MockEconVars(econ_vars=input_xr)

    damages = StackedDamages(
        sector_path="",
        save_path="",
        econ_vars=fake_econvars,
        climate_vars=fake_climate,
        eta=0,
        gdppc_bottom_code=0.00,
        subset_dict={},
    )

    actual = damages.gdp

    xr.testing.assert_equal(actual, expected)


def test_stackeddamages_pop():
    """
    Test StackedDamages.pop returns something and it should be "cut".
    """
    # Lots of setup for this on.
    # First, create the input Dataset. We want some values before the pulse
    # year and some values after.
    time = np.arange(2045, 2110)
    x_fake = np.ones(len(time))
    input_xr = xr.DataArray(
        x_fake, coords=[time], dims=["year"], name="pop"
    ).to_dataset()
    expected = input_xr.sel(year=slice(2048, 2099))["pop"]
    # Setup input data to instantiate a minimalist StackedDamages
    # with enough fake data that we can test basic behavior.

    @dataclass
    class MockClimate:
        pulse_year: int

    @dataclass
    class MockEconVars:
        econ_vars: xr.Dataset

    fake_climate = MockClimate(pulse_year=2050)
    fake_econvars = MockEconVars(econ_vars=input_xr)

    damages = StackedDamages(
        sector_path="",
        save_path="",
        econ_vars=fake_econvars,
        climate_vars=fake_climate,
        eta=0,
        gdppc_bottom_code=0.00,
        subset_dict={},
    )

    actual = damages.pop

    xr.testing.assert_equal(actual, expected)


def test_stackeddamages_gdppc():
    """
    Test StackedDamages.gdppc returns "cut" gdp/pop.
    """
    # Lots of setup for this on.
    # First, create the input Dataset. We want some values before the pulse
    # year and some values after.
    time = np.arange(2045, 2110)
    x_fake = np.ones(len(time))
    input_xr = xr.Dataset(
        {
            "gdp": (["year"], x_fake * 2.0),
            "pop": (["year"], x_fake * 0.5),
        },
        coords={
            "year": (["year"], time),
        },
    )
    expected = (input_xr["gdp"] / input_xr["pop"]).sel(year=slice(2048, 2099))
    # Setup input data to instantiate a minimalist StackedDamages
    # with enough fake data that we can test basic behavior.

    @dataclass
    class MockClimate:
        pulse_year: int

    @dataclass
    class MockEconVars:
        econ_vars: xr.Dataset

    fake_climate = MockClimate(pulse_year=2050)
    fake_econvars = MockEconVars(econ_vars=input_xr)

    damages = StackedDamages(
        sector_path="",
        save_path="",
        econ_vars=fake_econvars,
        climate_vars=fake_climate,
        eta=0,
        gdppc_bottom_code=0.00,
        subset_dict={},
    )

    actual = damages.gdppc

    xr.testing.assert_equal(actual, expected)


def test_stackeddamages_gdppc_with_bottom_code():
    """
    Test StackedDamages.gdppc uses gdppc_bottom_code instead of gdp/pop.
    """
    # Lots of setup for this on.
    # First, create the input Dataset. We want some values before the pulse
    # year and some values after.
    time = np.arange(2045, 2110)
    x_fake = np.ones(len(time))
    gdppc_bottom_code = 5.0
    input_xr = xr.Dataset(
        {
            "gdp": (["year"], x_fake * 2.0),
            "pop": (["year"], x_fake * 0.5),
        },
        coords={
            "year": (["year"], time),
        },
    )
    expected = xr.DataArray(
        x_fake * gdppc_bottom_code,
        coords=[time],
        dims=["year"],
    ).sel(year=slice(2048, 2099))

    # Setup input data to instantiate a minimalist StackedDamages
    # with enough fake data that we can test basic behavior.

    @dataclass
    class MockClimate:
        pulse_year: int

    @dataclass
    class MockEconVars:
        econ_vars: xr.Dataset

    fake_climate = MockClimate(pulse_year=2050)
    fake_econvars = MockEconVars(econ_vars=input_xr)

    damages = StackedDamages(
        sector_path="",
        save_path="",
        econ_vars=fake_econvars,
        climate_vars=fake_climate,
        eta=0,
        gdppc_bottom_code=gdppc_bottom_code,
        subset_dict={},
    )

    actual = damages.gdppc

    xr.testing.assert_equal(actual, expected)
