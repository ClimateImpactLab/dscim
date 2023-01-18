import pandas as pd
import xarray as xr
import numpy as np
import pytest
import statsmodels.formula.api as smf
from dscim.utils.utils import c_equivalence, model_outputs, extrapolate


def test_extrapolate():
    """
    common input :
        - xr_array : containing 100 values sampled from 4-9 uniform distribution and with time index from 1 to 100
        - start_year : 99
        - end_year : 100

    input cases variation :
        - method :
            - 'linear'
            - 'log_linear'
            - 'elog'
            - 'squared'
            - 'growth_constant'
                - cap = None
                - cap = 0.01
            - 'does not exist' (throws Error)
        - var :
            - 'value'
            - None (throws Error with method "linear" and xr_array not an xr.Dataset.)
        - interp_year :
            - 101
            - 102
    """

    # template time indexed data
    dims = ["year"]
    rng = np.random.default_rng(12345)
    values = rng.uniform(
        4, 9, 100
    )  # looks like the log of gdp per cap of a country randomly experience poverty, development and wealth
    xr_array = xr.DataArray(
        data=np.array(values), dims=dims, coords={"year": list(range(1, 101))}
    )
    xr_dataset = xr.Dataset({"value": xr_array})
    start_year = 99
    end_year = 100
    interp_year = 101
    first = values[start_year - 1]  # for later
    second = values[end_year - 1]
    cap = 0.01  # with above rng the growth rate between two last periods is large -- cap it at 1

    with pytest.raises(ValueError):
        extrapolate(xr_dataset, start_year, end_year, interp_year, "linear")  # var None
    with pytest.raises(ValueError):
        extrapolate(
            xr_array, start_year, end_year, interp_year, "does not exist"
        )  # invalid method

    # used many times
    standard = extrapolate(
        xr_dataset, start_year, end_year, interp_year, method="linear", var="value"
    )
    # basis function

    def linear_extrapolation(x, x_1, x_2, y_1, y_2):
        return y_1 + (x - x_1) * (y_2 - y_1) / (x_2 - x_1)

    # run numerical assertions
    assert (
        isinstance(standard, xr.DataArray)
        and list(standard.coords) == ["year"]
        and standard.coords["year"].values.shape == (1,)
        and standard.coords["year"].values[0] == 101
    )
    np.testing.assert_almost_equal(
        standard, linear_extrapolation(101, 99, 100, first, second)
    )
    np.testing.assert_almost_equal(
        extrapolate(
            xr_array, start_year, end_year, interp_year, method="growth_constant"
        ),
        ((second - first) / first + 1) * second,
    )
    np.testing.assert_almost_equal(
        extrapolate(
            xr_array,
            start_year,
            end_year,
            interp_year,
            method="growth_constant",
            cap=cap,
        ),
        second * (1 + cap),
    )
    np.testing.assert_almost_equal(
        extrapolate(xr_array, start_year, end_year, interp_year, method="linear_log"),
        np.exp(linear_extrapolation(101, 99, 100, np.log(first), np.log(second))),
    )
    np.testing.assert_almost_equal(
        extrapolate(xr_array, start_year, end_year, interp_year, method="elog"),
        np.log(
            linear_extrapolation(
                101, 99, 100, np.exp(first / 1e14), np.exp(second / 1e14)
            )
        )
        * 1e14,
    )
    np.testing.assert_almost_equal(
        extrapolate(xr_array, start_year, end_year, interp_year, method="squared"),
        np.sqrt(
            linear_extrapolation(101, 99, 100, np.square(first), np.square(second))
        ),
    )
    # check interp_year param is useful
    standard = extrapolate(
        xr_dataset, start_year, end_year, interp_year + 1, "linear", "value"
    )
    assert (
        isinstance(standard, xr.DataArray)
        and list(standard.coords) == ["year"]
        and standard.coords["year"].values.shape == (2,)
        and standard.coords["year"].values[1] == 102
    )


def test_c_equivalence():
    """
    input cases covered :
    etas :
        - 0
        - 1 : division by zero error
        - 10
    weights :
        - None
        - not None
    array :
        - only positive values
        - with some negative values : ValueError
        - not xarray.DataArray or xarray.Dataset
    """

    dims = ["dim1", "dim2"]
    array = xr.DataArray(
        data=np.array([[5.0, 10.0], [3.0, 2.0]]),
        dims=dims,
        coords=np.array([[1, 2], [3, 4]]),
    )
    weights = xr.DataArray(
        data=np.array([[0.4, 0.2], [0.1, 0.3]]),
        dims=dims,
        coords=np.array([[1, 2], [3, 4]]),
    )  # sum of weights equals 1

    assert (
        c_equivalence(array, dims, eta=0, weights=None, func_args=None, func=None)
        == array.values.mean()
    )  # most things cancel out with eta=0
    assert c_equivalence(
        array, dims, eta=1, weights=None, func_args=None, func=None
    ) == (np.exp(np.log(array).values.mean()))
    assert c_equivalence(
        array, dims, eta=10, weights=None, func_args=None, func=None
    ) == (np.divide(array ** (1 - 10), 1 - 10).values.mean() * (1 - 10)) ** (
        1 / (1 - 10)
    )
    assert c_equivalence(
        array, dims, eta=10, weights=weights, func_args=None, func=None
    ) == ((np.divide(array ** (1 - 10), 1 - 10) * weights).values.sum() * (1 - 10)) ** (
        1 / (1 - 10)
    )
    with pytest.raises(ValueError):
        c_equivalence(-array, dims, eta=10, weights=None, func_args=None, func=None)
    with pytest.raises(TypeError):
        c_equivalence(
            "I am not an xarray object",
            dims,
            eta=10,
            weights=None,
            func_args=None,
            func=None,
        )


def run_model_outputs(conf):
    """
    helper function for test_model_outputs
    """
    out = model_outputs(
        damage_function=conf["damage_function"],
        extrapolation_type=conf["extrapolation_type"],
        formula=conf["formula"],
        year_range=conf["year_range"],
        year_start_pred=conf["year_start_pre"],
        year_end_pred=conf["year_end_pre"],
        quantiles=conf["quantiles"],
        global_c=conf["global_c"],
        type_estimation=conf["type_estimation"],
        extrap_formula="",  # deprecated when this test was written
        extrap_year="",  # same
    )

    return out


def test_model_outputs():

    """
    input cases variation :
    formula :
        - 'damage ~ gmsl'
        - 'damage ~ anomaly'
        Note : damage_function needs to contain a column with the explanatory variable name
    """
    conf = {}

    # preparing common args
    rng = np.random.default_rng(12345)
    year_range = range(2010, 2100)
    conf["year_range"] = year_range
    sample_years = list(year_range)
    clim = rng.standard_normal(len(sample_years))
    eps = rng.standard_normal(len(sample_years))
    damage = clim + eps  # fake stochastic damage function data that's linear in clim
    conf["extrapolation_type"] = "global_c_ratio"
    year_start_pre = 2100
    year_end_pre = 2300
    conf["year_start_pre"] = year_start_pre
    conf["year_end_pre"] = year_end_pre
    conf["quantiles"] = [0.5]
    dims = ["year"]
    pred_years = list(range(year_start_pre - 1, year_end_pre + 1))
    global_c = xr.DataArray(
        data=np.array(rng.standard_normal(len(pred_years))),
        dims=dims,
        coords={"year": pred_years},
    )
    conf["global_c"] = global_c

    # test cases
    for explanatory_var in ["anomaly", "gmsl"]:
        conf["type_estimation"] = "ols"
        formula = "damage ~ " + explanatory_var
        conf["formula"] = formula
        df = pd.DataFrame(
            {"year": sample_years, explanatory_var: clim, "damage": damage}
        )
        conf["damage_function"] = df
        out = run_model_outputs(conf)
        # check format
        assert isinstance(out, dict) and list(out.keys()) == ["parameters", "preds"]
        expected_coeff = (
            smf.ols(formula=formula, data=df[df.year >= 2020 - 2][df.year <= 2020 + 2])
            .fit()
            .params[explanatory_var]
        )  # 5 years rolling window (-2, T, +2)
        actual_coeff = out["parameters"].sel(year=2020)[explanatory_var].item()
        # check result
        assert expected_coeff == actual_coeff

        expected_pred = (
            smf.ols(formula=formula, data=df[df.year >= 2020 - 2][df.year <= 2020 + 2])
            .fit()
            .predict(exog=pd.DataFrame({explanatory_var: [3]}))
        )
        actual_pred = (
            out["preds"].sel({"year": 2020, explanatory_var: 3})["y_hat"].item()
        )

        assert np.round(expected_pred[0], 6) == np.round(actual_pred, 6)
