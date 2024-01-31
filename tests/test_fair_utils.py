import numpy as np
import pandas as pd
import xarray as xr
import pytest
import statsmodels.formula.api as smf
from dscim.utils.utils import modeler
from dscim.utils.utils import get_weights
from dscim.utils.utils import quantile_weight_quantilereg


def test_modeler():
    """
    input cases covered :
    type_estimation :
        - "ols"
        - "quantreg"
    quantiles :
        - [0.5]
    """

    form = "damage ~ temp + np.power(temp, 2)"
    rng = np.random.default_rng(12345)
    temp = rng.standard_normal(1000)
    eps = rng.standard_normal(1000)
    exo = pd.DataFrame({"temp": [0.5]})
    damage = temp + eps
    df = pd.DataFrame({"damage": damage, "temp": temp})
    expected_ols = smf.ols(formula=form, data=df).fit()
    expected_quant = smf.quantreg(formula=form, data=df).fit(q=0.5)

    actual_ols = modeler(df, form, "ols", exo)
    actual_quant = modeler(df, form, "quantreg", exo, [0.5])

    # checking returned values matches spec
    assert all(isinstance(x, tuple) for x in [actual_ols, actual_quant])
    assert all(len(x) == 2 for x in [actual_ols, actual_quant])

    # checking values
    assert expected_ols.params["temp"] == actual_ols[0]["temp"][0]
    assert expected_quant.params["temp"] == actual_quant[0]["temp"][0]
    assert expected_ols.predict(exog=exo)[0] == actual_ols[1]["y_hat"][0]
    assert expected_quant.predict(exog=exo)[0] == actual_quant[1]["y_hat"][0]


def test_get_weights():
    """
    input cases covered :
    quantiles :
        any > 1 -> RunTimeError
        [0.5, 0.6, 0.7] -> [0.55, 0.1, 0.35]
        [1] -> [1]
        [0] -> [1]
    """

    with pytest.raises(RuntimeError):
        get_weights([1.1])
    np.testing.assert_allclose(
        get_weights([0.5, 0.6, 0.7]), np.array([0.55, 0.1, 0.35])
    )
    np.testing.assert_allclose(get_weights([1]), np.array([1]))
    np.testing.assert_allclose(get_weights([0]), np.array([1]))

def test_quantile_weight_quantilereg_dim():
    """
    input cases covered :
    fair dim is included in uncollapsed_sccs
    """
    ds_in = xr.Dataset(
            {
                'uncollapsed_sccs': (
                    ["discount_type", "model", "ssp", "rcp", "simulation", "gas", "q", "weitzman_parameter", "fair_aggregation"],
                    np.ones((1, 2, 2, 2, 5, 1, 3, 1, 1)),
                ),
            },
            coords={
                "discount_type": (["discount_type"], ["Euler_Ramsey"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "ssp": (["ssp"], ["SSP3", "SSP4"]),
                "rcp": (["rcp"], ["rcp245", "rcp585"]),
                "simulation": (["simulation"], [0,1,2,3,4]),
                "gas": (["gas"], ["CO2_Fossil"]),
                "q": (["q"], [0.01, 0.5, 0.99]),
                "weitzman_parameter": (["weitzman_parameter"], ["0.1"]),
                "fair_aggregation": (["fair_aggregation"], ["uncollapsed"]),
            },
        )
    
    ds_out = quantile_weight_quantilereg(ds_in.uncollapsed_sccs, ["simulation"], quantiles=[0.01,0.5,0.99])
    
    ds_out_expected = xr.Dataset(
            {
                'uncollapsed_sccs': (
                    ["discount_type", "model", "ssp", "rcp", "simulation", "gas", "q", "weitzman_parameter", "fair_aggregation"],
                    np.ones((1, 2, 2, 2, 5, 1, 3, 1, 1)),
                ),
            },
            coords={
                "discount_type": (["discount_type"], ["Euler_Ramsey"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "ssp": (["ssp"], ["SSP3", "SSP4"]),
                "rcp": (["rcp"], ["rcp245", "rcp585"]),
                "gas": (["gas"], ["CO2_Fossil"]),
                "weitzman_parameter": (["weitzman_parameter"], ["0.1"]),
                "fair_aggregation": (["fair_aggregation"], ["uncollapsed"]),
                "quantile": (["quantile"], [0.01,0.5,0.99])
            },
        )
    
    np.testing.assert_equal(ds_out,ds_out_expected)
    
def test_quantile_weight_quantilereg_nodim():
    """
    input cases covered :
    fair dim is included in not included in uncollapsed_sccs
    """
    ds_in = xr.Dataset(
            {
                'uncollapsed_sccs': (
                    ["discount_type", "model", "ssp", "rcp", "gas", "q", "weitzman_parameter", "fair_aggregation"],
                    np.ones((1, 2, 2, 2, 1, 3, 1, 1)),
                ),
            },
            coords={
                "discount_type": (["discount_type"], ["Euler_Ramsey"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "ssp": (["ssp"], ["SSP3", "SSP4"]),
                "rcp": (["rcp"], ["rcp245", "rcp585"]),
                "gas": (["gas"], ["CO2_Fossil"]),
                "q": (["q"], [0.01, 0.5, 0.99]),
                "weitzman_parameter": (["weitzman_parameter"], ["0.1"]),
                "fair_aggregation": (["fair_aggregation"], ["uncollapsed"]),
            },
        )
    
    ds_out = quantile_weight_quantilereg(ds_in.uncollapsed_sccs, ["simulation"], quantiles=[0.01,0.5,0.99])
    
    ds_out_expected = xr.Dataset(
            {
                'uncollapsed_sccs': (
                    ["discount_type", "model", "ssp", "rcp", "simulation", "gas", "q", "weitzman_parameter", "fair_aggregation"],
                    np.ones((1, 2, 2, 2, 5, 1, 3, 1, 1)),
                ),
            },
            coords={
                "discount_type": (["discount_type"], ["Euler_Ramsey"]),
                "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
                "ssp": (["ssp"], ["SSP3", "SSP4"]),
                "rcp": (["rcp"], ["rcp245", "rcp585"]),
                "gas": (["gas"], ["CO2_Fossil"]),
                "weitzman_parameter": (["weitzman_parameter"], ["0.1"]),
                "fair_aggregation": (["fair_aggregation"], ["uncollapsed"]),
                "quantile": (["quantile"], [0.01,0.5,0.99])
            },
        )
    
    np.testing.assert_equal(ds_out,ds_out_expected)