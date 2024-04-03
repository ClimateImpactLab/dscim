import pandas
import xarray as xr
import numpy as np
from pandas.testing import assert_frame_equal
from xarray.testing import assert_allclose
import pytest

from . import open_zipped_results
from dscim.menu.risk_aversion import RiskAversionRecipe


@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_risk_aversion_points(menu_instance, discount_types):
    path = f"risk_aversion_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_points.csv"
    expected = open_zipped_results(path)
    actual = menu_instance.damage_function_points
    assert_frame_equal(
        expected.sort_values(by=list(expected.columns)),
        actual.sort_values(by=list(actual.columns)),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_risk_aversion_coefficients(menu_instance, discount_types):
    path = f"risk_aversion_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_coefficients.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.damage_function_coefficients
    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
    )


@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_risk_aversion_fit(menu_instance, discount_types):
    path = f"risk_aversion_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_fit.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.damage_function_fit
    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
    )


@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_risk_global_consumption(menu_instance, discount_types):
    path = f"risk_aversion_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_global_consumption.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.global_consumption.squeeze()
    # Small format hack from I/O
    if isinstance(expected, xr.Dataset):
        expected = expected.to_array().squeeze().drop("variable")

    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
    )


@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_risk_scc(menu_instance, discount_types):
    path = f"risk_aversion_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_scc.nc4"
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
@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_global_damages_calculation(menu_instance):
    global_damages = menu_instance.global_damages_calculation()
    assert (
        isinstance(global_damages, pandas.DataFrame)
        and "region" not in global_damages.columns
    )
