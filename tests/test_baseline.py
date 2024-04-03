import xarray as xr
from pandas.testing import assert_frame_equal
from xarray.testing import assert_allclose
import pytest
import pandas
from . import open_zipped_results
from dscim.menu.baseline import Baseline


@pytest.mark.parametrize("menu_class", [Baseline], indirect=True)
def test_adding_up_points(menu_instance, discount_types):
    path = f"adding_up_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_points.csv"
    expected = open_zipped_results(path)
    actual = menu_instance.damage_function_points
    assert_frame_equal(
        expected.sort_values(by=list(expected.columns)).reset_index(drop = True),
        actual.sort_values(by=list(actual.columns)).reset_index(drop = True),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("menu_class", [Baseline], indirect=True)
def test_adding_up_coefficients(menu_instance, discount_types):
    path = f"adding_up_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_coefficients.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.damage_function_coefficients

    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("menu_class", [Baseline], indirect=True)
def test_adding_up_fit(menu_instance, discount_types):
    path = f"adding_up_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_fit.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.damage_function_fit

    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
    )


@pytest.mark.parametrize("menu_class", [Baseline], indirect=True)
def test_adding_global_consumption(menu_instance, discount_types):
    path = f"adding_up_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_global_consumption.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.global_consumption.squeeze()
    # Small format hack from I/O
    if isinstance(expected, xr.Dataset):
        expected = expected.to_array().squeeze().drop("variable")

    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
    )


@pytest.mark.parametrize("menu_class", [Baseline], indirect=True)
def test_adding_scc(menu_instance, discount_types):
    path = f"adding_up_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_scc.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.calculate_scc.squeeze()

    # Small format hack from I/O
    if isinstance(expected, xr.Dataset):
        expected = expected.to_array().squeeze().drop("variable")

    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
@pytest.mark.parametrize("menu_class", [Baseline], indirect=True)
def test_global_damages_calculation(menu_instance):
    assert isinstance(menu_instance.global_damages_calculation(), pandas.DataFrame)
