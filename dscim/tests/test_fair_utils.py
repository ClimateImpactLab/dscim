import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from dscim.utils.utils import modeler
from dscim.utils.utils import get_weights


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
