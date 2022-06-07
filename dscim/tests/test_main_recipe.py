import xarray as xr
import numpy as np
import math
import pytest
import copy
import dscim.utils.utils as estimations
from dscim.menu.risk_aversion import RiskAversionRecipe


@pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_calculate_discount_factors(menu_instance):
    """
    input cases :

    in all cases, t is set to [0,1,2,3].

    - eta = 1, consumption = [6, 5.9, 5.8, 5.7], rho = 0.01 => factors should be increasing
    - eta = 1, consumption =  [6, 5.9, 5.8, 5.7], rho = 0.015 => factors increase but less than with rho = 0.01
    - eta = 1, consumption = [6, 5.9, 5.8, 5.7], rho = 0.02 => factors decrease

    - eta = 1, consumption = [6, 6 , 6, 6], rho = 0.01 => factors decrease
    - eta = 1, consumption = [6, 6.1 , 6.2, 6.3], rho = 0.01 => factors decrease and more than with constant consumption
    - eta = 1, consumption = [6, 6 , 6.2, 6.3], rho = 0 => first rate should be 1
    - eta = 2, consumption = [6, 6 , 6, 6], rho = 0.01 => factors decrease and should be same as with eta = 1
    - eta = 2, consumption = [6, 6.1 , 6.2, 6.3], rho = 0.01 => factors decrease and even more than with eta = 1

    """

    cp_menu_instance = copy.deepcopy(
        menu_instance
    )  # because we'll modify the fields of an object shared across tests
    cp_menu_instance.climate.pulse_year = 0
    cp_menu_instance.ext_end_year = 3
    years = [
        range(cp_menu_instance.climate.pulse_year, cp_menu_instance.ext_end_year + 1)
    ]

    def do_calculation(eta, consumption, rho, years=years):
        cp_menu_instance.eta = eta
        cp_menu_instance.rho = rho
        return cp_menu_instance.calculate_discount_factors(
            xr.DataArray(consumption, dims=["year"], coords=years)
        )

    low_rho = do_calculation(1, [6, 5.9, 5.8, 5.7], 0.01)
    assert all([x > 0 for x in low_rho.diff("year").values])
    mid_rho = do_calculation(1, [6, 5.9, 5.8, 5.7], 0.015)
    diff = mid_rho.diff("year") - low_rho.diff("year")
    assert all([x > 0 for x in mid_rho.diff("year").values]) and all(
        [x < 0 for x in diff.values]
    )
    high_rho = do_calculation(1, [6, 5.9, 5.8, 5.7], 0.02)
    assert all([x < 0 for x in high_rho.diff("year").values])
    constant_cons = do_calculation(1, [6, 6, 6, 6], 0.01)
    assert all([x < 0 for x in constant_cons.diff("year").values])
    increasing_cons = do_calculation(1, [6, 6.1, 6.2, 6.3], 0.01)
    diff = increasing_cons.diff("year") - constant_cons.diff("year")
    assert all([x < 0 for x in increasing_cons.diff("year").values]) and all(
        [x < 0 for x in diff.values]
    )
    zero_rho = do_calculation(1, [6, 6, 6.2, 6.3], 0)
    assert all(x == 1 for x in zero_rho[0:1]) and all(x < 1 for x in zero_rho[2:])
    high_eta_constant_cons = do_calculation(2, [6, 6, 6, 6], 0.01)
    diff = high_eta_constant_cons.diff("year") - constant_cons.diff("year")
    assert all([x < 0 for x in high_eta_constant_cons.diff("year").values]) and all(
        [x == 0 for x in diff.values]
    )
    high_eta_increasing_cons = do_calculation(2, [6, 6.1, 6.2, 6.3], 0.01)
    diff = high_eta_increasing_cons.diff("year") - increasing_cons.diff("year")
    assert all([x < 0 for x in high_eta_increasing_cons.diff("year").values]) and all(
        [x < 0 for x in diff.values]
    )


@pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_calculate_stream_discount_factors(menu_instance):

    """
    `   input cases

        discounting_type :
            'gwr_gwr'
            'naive'
            'euler'
                fair_aggregation field :
                    ['ce', 'median', 'mean']
                    ['uncollapsed']
    """

    # gwr_gwr
    xr.testing.assert_allclose(
        menu_instance.calculate_discount_factors(
            menu_instance.global_consumption_per_capita("naive_ramsey")
        )
        .mean(dim=["ssp", "model"])
        .expand_dims({"fair_aggregation": menu_instance.fair_aggregation}),
        menu_instance.calculate_stream_discount_factors(
            "gwr_gwr", menu_instance.fair_aggregation
        ),
    )

    # naive
    xr.testing.assert_allclose(
        menu_instance.calculate_discount_factors(
            menu_instance.global_consumption_per_capita(menu_instance.discounting_type)
        ).expand_dims({"fair_aggregation": menu_instance.fair_aggregation}),
        menu_instance.calculate_stream_discount_factors(
            "naive", menu_instance.fair_aggregation
        ),
    )

    # euler
    euler = menu_instance.calculate_stream_discount_factors(
        "euler", menu_instance.fair_aggregation
    )
    pop = menu_instance.collapsed_pop.sum("region")
    pop = pop.reindex(
        year=range(pop.year.min().values, menu_instance.ext_end_year + 1),
        method="ffill",
    )

    # ce
    xr.testing.assert_allclose(
        menu_instance.calculate_discount_factors(menu_instance.ce_fair_no_pulse / pop),
        euler.sel(fair_aggregation="ce", drop=True),
    )

    # mean
    xr.testing.assert_allclose(
        menu_instance.calculate_discount_factors(
            menu_instance.global_consumption_no_pulse.mean(dim="simulation") / pop
        ),
        euler.sel(fair_aggregation="mean", drop=True),
    )

    # median
    median = menu_instance.calculate_discount_factors(
        (
            menu_instance.global_consumption
            - estimations.compute_damages(
                menu_instance.climate.fair_median_params_control,
                betas=menu_instance.damage_function_coefficients,
                formula=menu_instance.formula,
            )
        ).expand_dims(
            {"weitzman_parameter": [str(i) for i in menu_instance.weitzman_parameter]}
        )
        / pop
    )
    xr.testing.assert_allclose(
        median, euler.sel(fair_aggregation="median_params", drop=True)
    )

    # uncollapsed
    cp_menu_instance = copy.deepcopy(menu_instance)  # because will modify fields
    cp_menu_instance.fair_aggregation = ["uncollapsed"]
    euler = cp_menu_instance.calculate_stream_discount_factors(
        "euler", cp_menu_instance.fair_aggregation
    )
    xr.testing.assert_allclose(
        cp_menu_instance.calculate_discount_factors(
            cp_menu_instance.global_consumption_no_pulse / pop
        ),
        euler.sel(fair_aggregation="uncollapsed", drop=True),
    )


@pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_discounted_damages(menu_instance):

    """
    using as template damages [1, 1, 1, 1]

    input cases :
        discrate='constant' -> implied rates of discounted damages should all be 1/(1+r)
        discrate='whatever' -> implied rates should at least not be all equal. Since 'non-constant' rates are required,
        (redirecting, implicitly, to the discounting type of the class instance), the function shouldn't return
        discounted values with implied rates that are all almost constant, comparing successive periods (this is only true
        with constant damages, which is the case here).
    """

    dims = ["year", "fair_aggregation"]
    template_damages = xr.DataArray(
        data=np.ones([70, 1]),
        dims=dims,
        coords={"year": list(range(2020, 2090)), "fair_aggregation": ["mean"]},
    )

    def factors(npvector):
        "implied 'discounting factors'"
        return npvector[1:] / npvector[:-1]

    # checking the non constant case
    notconstant = (
        menu_instance.discounted_damages(template_damages, "whatever")
        .sel(
            model="IIASA GDP",
            ssp="SSP2",
            gas="CO2_Fossil",
            fair_aggregation="mean",
            drop=True,
        )
        .squeeze()
        .values
    )
    assert not all([math.isclose(x, 0) for x in np.diff(factors(notconstant))])
    assert notconstant[0] == 1 / (1 + menu_instance.rho)

    # checking the constant case
    template_damages = xr.DataArray(
        data=np.ones([4, 1]),
        dims=dims,
        coords={"year": list(range(2020, 2024)), "fair_aggregation": ["mean"]},
    )
    constant = menu_instance.discounted_damages(template_damages, "constant")
    for r in menu_instance.CONST_DISC_RATES:
        constant_values = constant.sel(discrate=r, fair_aggregation="mean").values
        np.testing.assert_allclose(
            factors(constant_values),
            np.array([1 / (1 + r), 1 / (1 + r), 1 / (1 + r)]),
        )
        assert constant_values[0] == 1


@pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_weitzman_min(menu_instance):

    """
    testing four behaviors with a template of consumption values including negative values :

        for an arbitrary positive weitzman parameter :

        1. verify all values affected by the transformation are strictly below the value of the parameter
        2. verify there are no negative values among the transformed values (it's the case with CRRA utility)

        for a larger weitzman parameter :

        3. verify that the speed of convergence of consumption to zero is lower than the former case

        4. when parameter <= 1 it is interpreted correctly as a share of each no_cc_consumption value, one-to-one

    """

    fast = np.array(float(10))
    slow = np.array(float(20))  # higher weitzman parameter
    as_share = np.array(0.5)
    all_cons = np.array(float(1))
    initial_values = np.array(
        list(np.linspace(-50, -1, 50)) + list(np.linspace(1, 100, 100))
    )
    censored_values = menu_instance.weitzman_min(None, initial_values, fast)
    affected = np.where(initial_values < fast)
    assert all([x < fast for x in censored_values[affected]])
    assert all([x > 0 for x in censored_values])
    censored_values_slower = menu_instance.weitzman_min(None, initial_values, slow)
    index = int(np.where(censored_values >= fast)[0][0])
    diff = censored_values_slower[0:index] - censored_values[0:index]
    assert all(x > 0 for x in diff)
    as_share = menu_instance.weitzman_min(
        [20, 50, 100], [5, 30, 60], as_share
    )  # the first value only should be censored
    assert as_share[0] != 5 and [as_share[1], as_share[2]] == [30, 60]
    all_cons = menu_instance.weitzman_min(
        [20, 50, 100], [5, 30, 60], all_cons
    )  # all values should be changed
    assert all([x != 0 for x in all_cons - [5, 30, 60]])


@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_damage_function_calculation(menu_instance):

    """
    The dimensions of the returned arrays depend therefore on the discounting type :
    - params should be (discount_type: 1, ssp: *, model: *, year: 281)
    - preds should be (discount_type: 1, ssp: * , model: *, anomaly: 100, year: 281)
    where the star depends on discounting type. The model fit may be either : (1) ssp specific, (2) ssp-iam specific, (3) unique across ssp-iam.
    We therefore simply test the shape of the returned arrays for each of these three cases.
    """

    params = menu_instance.damage_function["params"]
    preds = menu_instance.damage_function["preds"]
    params_dims = (
        params.dims["discount_type"],
        params.dims["ssp"],
        params.dims["model"],
        params.dims["year"],
    )
    preds_dims = (
        preds.dims["discount_type"],
        preds.dims["ssp"],
        preds.dims["model"],
        preds.dims["anomaly"],
        preds.dims["year"],
    )

    if (menu_instance.discounting_type == "constant") or (
        "ramsey" in menu_instance.discounting_type
    ):
        assert params_dims == (1, 3, 2, 281)
        assert preds_dims == (1, 3, 2, 100, 281)
    elif menu_instance.discounting_type == "constant_model_collapsed":
        assert params_dims == (1, 3, 1, 281)
        assert preds_dims == (1, 3, 1, 100, 281)
    elif ("gwr" in menu_instance.discounting_type) or (
        menu_instance.discounting_type == "rawlsian"
    ):
        assert params_dims == (1, 1, 1, 281)
        assert preds_dims == (1, 1, 1, 100, 281)
    else:
        raise ValueError(
            "need to add test for damage_function_calculation and discounting type "
            + str(menu_instance.discounting_type)
        )


@pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_marginal_damages(menu_instance):

    """
    simply check it has a fair aggregation types dimension and coords of that dimension match menu_instance.fair_aggregation
    """

    marginal = menu_instance.marginal_damages
    assert isinstance(marginal, xr.DataArray) and "fair_aggregation" in marginal.coords
    np.testing.assert_equal(
        marginal.coords["fair_aggregation"].values,
        np.array(menu_instance.fair_aggregation),
    )


@pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
def test_collapse_pop(menu_instance):

    if (menu_instance.discounting_type == "constant") or (
        "ramsey" in menu_instance.discounting_type
    ):
        assert menu_instance.pop.equals(menu_instance.collapsed_pop)
    elif menu_instance.discounting_type == "constant_model_collapsed":
        assert "model" not in menu_instance.collapsed_pop.coords
    elif ("gwr" in menu_instance.discounting_type) or (
        menu_instance.discounting_type == "rawlsian"
    ):
        assert all(
            [x not in menu_instance.collapsed_pop.coords for x in ["ssp", "model"]]
        )
    else:
        raise ValueError(
            "need to add test for collapsed_pop and discounting type "
            + str(menu_instance.discounting_type)
        )


def test_global_consumption_calculation(menu_instance):

    global_cons = menu_instance.global_consumption_calculation(
        menu_instance.discounting_type
    )
    if (menu_instance.discounting_type == "constant") or (
        "ramsey" in menu_instance.discounting_type
    ):
        assert "region" not in global_cons.coords
    elif menu_instance.discounting_type == "constant_model_collapsed":
        assert all([x not in global_cons.coords for x in ["region", "model"]])
    elif "gwr" in menu_instance.discounting_type:
        assert all([x not in global_cons.coords for x in ["region", "model", "ssp"]])
    else:
        raise ValueError(
            f"{menu_instance.discounting_type} is not yet tested in test_global_consumption_calculation"
        )
