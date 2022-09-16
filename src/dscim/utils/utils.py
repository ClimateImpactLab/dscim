import os
import warnings
import dask
import logging
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import impactlab_tools.utils.weighting
import dask.dataframe as dd
from itertools import product

logger = logging.getLogger(__name__)


def modeler(df, formula, type_estimation, exog, quantiles=None):
    """Wrapper function for statsmodels functions (OLS and quantiles)

    For the quantiles, the function will fit for all quantiles between 0 and 1
    with a 0.25 step.

    Parameters:
    ---------
    df: pd.DataFrame
        A dataframe of prepared damage function. Since formulas are hard to
        get, the function expects standarized column names: `damages` must
        contain all damages coming from the damage function, and `temp` must
        be the name of temperature anomalies from the models.
    formula: str
        R-like formula to start fitting. (i.e. damages ~ temp +
        np.power(temp,2)) defines a quadratic relationship between temperature
        and damages.
    type_estimation: str
        Model to use . 'ols' and 'quantreg' are the one availables. Any other
        passed option will yield a `NonImplementedError`
    exog: pd.DataFrame
        predictors used to generate predicted damage function fit, the name of the variable should be the same
        as that used in the model formula.
    quantiles: list of floats between 0 and 1
        List of quantile regressions to be run.

    Returns
    ------
    tuple
        pd.DataFrame with coefficients by year
        pd.DataFrame with predictions
    """

    if type_estimation == "ols":
        mod = smf.ols(formula=formula, data=df).fit()
        params = pd.DataFrame(mod.params).T
        y_hat = pd.DataFrame(dict(exog, y_hat=mod.predict(exog=exog)))

    elif type_estimation == "quantreg":
        # silence the 'max recursions' warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # set up the model
            mod = smf.quantreg(formula=formula, data=df)
            q_params, q_y_hat = [], []

            # calculate each quantile regression
            # save parameters and predictions
            for quant in quantiles:
                quant_mod = mod.fit(q=quant)
                params_quantile = pd.DataFrame(quant_mod.params).T
                params_quantile = params_quantile.assign(q=quant)
                q_params.append(params_quantile)

                fitted = pd.DataFrame(dict(exog, y_hat=quant_mod.predict(exog=exog)))
                q_y_hat.append(fitted.assign(q=quant))

            params = pd.concat(q_params)
            y_hat = pd.DataFrame(pd.concat(q_y_hat))

    else:
        raise NotImplementedError(f"{type_estimation} is not valid option!")

    return params, y_hat


def get_weights(quantiles):
    """
    Compute the weights of each quantile regression.

    Parameters :
    ------------
    quantiles : array_like
        representing a sequence of quantile values, must be between 0-1, otherwise throws as RunTimeError.

    Returns :
    --------
    vector of weights represented by an np.ndarray
    """
    quantiles = np.array(quantiles)
    # find midpoints between quantiles
    if np.any(np.logical_or(quantiles > 1, quantiles < 0)):
        raise RuntimeError("quantiles must be between 0-1")
    bounds = np.array([0] + ((quantiles[:-1] + quantiles[1:]) / 2).tolist() + [1])
    weights = np.diff(bounds)
    return weights


def quantile_weight_quantilereg(array, quantiles=None):
    """Produce quantile weights of the quantile regression damages.

    Parameters
    ----------
    qr_quantiles: the quantile regression quantiles for damages. (Must be 0-1!)
    quantiles : sequence or None, optional
        Quantiles to compute. Default is (0.01, 0.05, 0.167, 0.25, 0.5, 0.75,
        0.833, 0.95, 0.99).
    """
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.167, 0.25, 0.5, 0.75, 0.833, 0.95, 0.99]

    qr_quantiles = array.q.values
    weights = xr.DataArray(get_weights(qr_quantiles), dims=["q"], coords=[qr_quantiles])

    ds_stacked = array.stack(obs=("simulation", "q"))
    weights_by_obs = weights.sel(q=ds_stacked.obs.q)
    dim = "obs"

    # these are quantiles of the statistical or full uncertainty, weighted by the quantile regression quantiles
    ds_quantiles = impactlab_tools.utils.weighting.weighted_quantile_xr(
        ds_stacked, quantiles, sample_weight=weights_by_obs.values, dim=dim
    )

    return ds_quantiles


def extrapolate(
    xr_array, start_year, end_year, interp_year, method, var=None, cap=None
):
    """Extrapolate values from xarray object indexed in time

    Extrapolate values (of a specific type, see ``xr_array``) in xr.DataArray. This function uses the ``interp``
    option in xarray objects. The extrapolation can either be linear or
    constant by using the range of years between ``start_year`` and
    ``end_year``.

    Parameters:
    ----------
    xr_array : xr.Dataset or xr.DataArray
        Array with a ``'year'`` dimension. There can't be negative values, nor in the initial,
        neither in the extrapolated array.
    start_year : int
        Extrapolation subsample start time.
        TODO : It is currently used to subsample the data before extrapolation happens. Should be removed and the last two data points be kept in all cases.
    end_year : int
        Extrapolation subsample end time
    interp_year : int
        End of extrapolation. The function will calculate values from the
        range given by the ``end_year`` and the ``interp_year``.
    var : str or None
        If str, it is required that ``xr_array`` is a dataset, and ``var`` defines the variable within
        the dataset. Default value is ``None``.
    method : str
        Interpolation method. Valid values are "linear", "linear_log", "elog", "squared", "growth_constant", otherwise a
        ValueError is raised. ``xr_array`` should be of type ``xr.Array`` if ``method`` is equal to "growth_constant".
    cap: xr.DataArray
        Used only if ``method`` is equal to "growth_constant". An Array with identical dimensions to ``xr_array``, which will
        serve as a cap for end of century growth rates of ``xr_array``.
        If ``None`` (the default value) is passed, no cap will be imposed.

    Returns
    -------
        An interpolated ``xarray`` object with additional years.
    """

    if isinstance(xr_array, xr.Dataset) and var is None:
        raise ValueError("Need to pass a var value to subset the xr.Dataset")

    if method == "linear":
        if var is not None:
            return (
                xr_array[var]
                .sel({"year": slice(start_year, end_year)})
                .interp(
                    year=np.arange(end_year + 1, interp_year + 1),
                    method="linear",
                    kwargs={"fill_value": "extrapolate"},
                )
            )
        else:
            return xr_array.sel(year=slice(start_year, end_year)).interp(
                year=np.arange(end_year + 1, interp_year + 1),
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )

    elif method == "linear_log":
        log_array = np.log(xr_array).sel(year=slice(start_year, end_year))
        log_interp = log_array.interp(
            year=np.arange(end_year + 1, interp_year + 1),
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
        return np.exp(log_interp)

    elif method == "elog":
        factor = 1e14
        exp_array = np.exp(xr_array / factor).sel(year=slice(start_year, end_year))
        exp_interp = exp_array.interp(
            year=np.arange(end_year + 1, interp_year + 1),
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
        return np.log(exp_interp) * factor

    elif method == "squared":
        sqr_array = (xr_array**2).sel(year=slice(start_year, end_year))
        sqr_interp = sqr_array.interp(
            year=np.arange(end_year + 1, interp_year + 1),
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
        return np.sqrt(sqr_interp)

    elif method == "growth_constant":
        # Calculate growth rates
        growth_rates = xr_array.diff("year") / xr_array.shift(year=1)

        # cap growth in final year
        if cap is not None:
            growth_rates = xr.where(
                (growth_rates.year == end_year) & (growth_rates > cap),
                cap,
                growth_rates,
            )
        else:
            print("End-of-century growth rates are not capped.")

        # hold last year constant for extrapolated array
        growth_rates_ext = growth_rates.reindex(
            year=range(end_year + 1, interp_year + 1), method="ffill"
        )

        # Start growth rates from zero from the first year
        # Calculate new array using fixed growth rate
        growth_rates_cum = np.multiply.accumulate(
            (growth_rates_ext.values + 1), growth_rates_ext.dims.index("year")
        )
        growth_rates_cum_xarr = xr.DataArray(
            growth_rates_cum, coords=growth_rates_ext.coords
        )

        return growth_rates_cum_xarr * xr_array.sel(year=end_year)

    else:
        raise ValueError(f"{method} is not a valid interpolation method.")


def power(a, b):
    """Power function that doesn't return NaN values for negative fractional exponents"""
    return np.sign(a) * (np.abs(a)) ** b


def c_equivalence(array, dims, eta, weights=None, func_args=None, func=None):
    """Calculate certainty equivalent assuming a CRRA utility
    function.

    Parameters
    ----------
    array :  xr.Dataset or xr.DataArray
        A `xarray` object containing non-negative consumption numbers.
    dims : list
        List of dimensions to use in aggregation.
    eta : int or float
        the parameter determining the curvature of the CRRA utility function. Must be different from 1.
    func : function
        If user desired a different utility function to the CRRA, this
        argument can take a `lambda` function where `array` is the main
        variable.
    weights :  xr.DataArray
        Array with weights for aggregation calculation. `None` by default.
    func_args : dict
        Other arguments passed to `func`

    Returns
    -------
        xarray.Dataset with CEs over the desired dims
    """

    if isinstance(array, xr.DataArray):
        negative_values = array.where(lambda x: x < 0, drop=True).size
    elif isinstance(array, xr.Dataset):
        negative_values = (
            array[list(array.variables)[0]].where(lambda x: x < 0, drop=True).size
        )
    else:
        raise TypeError("array should be either an xarray.DataArray or xarray.Dataset")

    if negative_values != 0:
        raise ValueError(
            "Negative values were passed to the certainty equivalence function"
        )

    if func is None:
        exp = 1 / (1 - eta)
        utility = np.true_divide(array ** (1 - eta), (1 - eta))

        if weights is None:
            exp_utility = utility.mean(dim=dims)
        else:
            exp_utility = utility.weighted(weights).mean(dim=dims)

        ce_array = (exp_utility * (1 - eta)) ** exp

    else:
        try:
            ce_array = array.map_partitions(func, **func_args)
        except Exception as exception:
            raise ValueError(f"Custom function not conforming with array: {exception}")

    return ce_array


def model_outputs(
    damage_function,
    extrapolation_type,
    formula,
    extrap_formula,
    year_range,
    extrap_year,
    year_start_pred,
    year_end_pred,
    quantiles,
    global_c=None,
    base_year=2010,
    min_anomaly=0,
    max_anomaly=20,
    step_anomaly=0.2,
    min_gmsl=0,
    max_gmsl=300,
    step_gmsl=3,
    type_estimation="ols",
):
    """Estimate damage function coefficients and predictions using
    passed formula.

    This model is estimated for each year in our timeframe (2010 to 2099) and
    using a rolling window across time (5-yr by default).

    Damage functions are extrapolated for each year between 2099 and 2300.

    Parameters
    ----------
    damage_function: pandas.DataFrame
        A global damage function.
    type_estimation: str
        Type of model use for damage function fitting: `ols`, `quantreg`
    extrapolation_type : str
        Type of extrapolation: `global_c_ratio`, `time_trends`
    global_c : xr.DataArray
        Array with global consumption extrapolated to 2300. This is only used
        when ``extrapolation_type`` is ``global_c_ratio``.
    fix_global_c : int
        Year to fix damages and use as base to the ``global_c_ratio``
        extrapolation. Default value is 2099.
    base_year: int
        Base year for `time_trends` rebasing.
    interp_year: int
        Year pre-2100 to fit damages used for linear extrapolation.
        i.e. If `interp_year=2085`, data from years 2085 to 2099 will be
        used to extrapolate data to years post-2100. 2085 is the default.
        Only used in `time_trends` rebasing.
    year_start_pred: int
        Start of extrapolation
    year_end_pred: int
        End of extrapolation
    year_range: sequence, lst, tuple, range
        Range of years to estimate over. Default is 2010 to 2100

    Returns
    ------

    dict
        dict with two keys, `params` and `preds`. Each value is a Pandas
        DataFrame with yearly damage functions (coefficients and predictions
        respectively).
    """

    # set year of prediction for global C extrapolation
    fix_global_c = year_start_pred - 1

    # set exogenous variables for predictions
    gmsl = np.arange(min_gmsl, max_gmsl, step_gmsl)
    temps = np.arange(min_anomaly, max_anomaly, step_anomaly)
    extrap_years = range(year_start_pred, year_end_pred + 1)

    if ("anomaly" in damage_function.columns) and ("gmsl" in damage_function.columns):
        exog_X = pd.DataFrame(product(temps, gmsl))
        exog = dict(anomaly=exog_X.values[:, 0], gmsl=exog_X.values[:, 1])

        extrap_X = pd.DataFrame(product(temps, extrap_years, gmsl))
        extrap_exog = dict(
            anomaly=extrap_X.values[:, 0],
            year_rebase=extrap_X.values[:, 1] - base_year,
            gmsl=extrap_X.values[:, 2],
        )
    elif "anomaly" in damage_function.columns:
        exog = dict(anomaly=temps)

        extrap_X = pd.DataFrame(product(temps, extrap_years))
        extrap_exog = dict(
            anomaly=extrap_X.values[:, 0], year_rebase=extrap_X.values[:, 1] - base_year
        )
    elif "gmsl" in damage_function.columns:
        exog = dict(gmsl=gmsl)

        extrap_X = pd.DataFrame(product(gmsl, extrap_years))
        extrap_exog = dict(
            gmsl=extrap_X.values[:, 0], year_rebase=extrap_X.values[:, 1] - base_year
        )
    else:
        print("Independent variables not found.")

    # Rolling window estimation (5-yr)
    list_params, list_y_hats = [], []

    for year in year_range:
        time_window = range(year - 2, year + 3)
        df = damage_function[damage_function.year.isin(time_window)]
        params, y_hat = modeler(
            df=df,
            formula=formula,
            type_estimation=type_estimation,
            exog=exog,
            quantiles=quantiles,
        )

        params, y_hat = params.assign(year=year), y_hat.assign(year=year)
        list_params.append(params)
        list_y_hats.append(y_hat)

    # Concatenate results
    param_df = pd.concat(list_params)
    y_hat_df = pd.concat(list_y_hats)

    if extrapolation_type == "time_trends":

        raise NotImplementedError(
            "This has not been tested since adding quantregs option."
        )

        # Linear-extrapolation for post-2100 years
        df_extrap = damage_function[damage_function.year >= extrap_year]
        df_extrap = df_extrap.assign(year_rebase=df_extrap.year - base_year)

        extrap_params, extrap_y_hat = modeler(
            df=df_extrap,
            formula=extrap_formula,
            type_estimation=type_estimation,
            exog=extrap_exog,
        )
        extrap_y_hat = extrap_y_hat.drop(columns="year_rebase")
        extrap_y_hat["year"] = extrap_X.values[:, 1]

        # Reformulating parameters
        extrapolation_year = []
        for year in extrap_years:

            params_df = extrap_params.copy(deep=True)
            unint_terms = int(len(params_df.columns) / 2)

            # sum uninteracted terms with matching (interacted term * rebased year)
            for i in range(0, unint_terms):
                params_df.iloc[:, i] = params_df.iloc[:, i] + params_df.iloc[
                    :, (unint_terms + i)
                ] * (year - base_year)

            params_df = params_df.iloc[:, range(0, unint_terms)]

            params_df = params_df.assign(year=year)
            extrapolation_year.append(params_df)

        extrapolation_results = pd.concat(extrapolation_year)

        parameters = pd.concat([param_df, extrapolation_results]).to_xarray()
        preds = pd.concat([y_hat_df, extrap_y_hat]).to_xarray()

    elif extrapolation_type == "global_c_ratio":

        # convert to xarray immediately
        index = ["year", "q"] if type_estimation == "quantreg" else ["year"]
        y_hat_df = y_hat_df.set_index(
            [i for i in y_hat_df.columns if "y_hat" not in i]
        ).to_xarray()
        param_df = param_df.set_index(index).to_xarray()

        # Calculate global consumption ratios to fixed year
        global_c_factors = (
            global_c.sel(year=slice(fix_global_c + 1, None))
            / global_c.sel(year=fix_global_c)
        ).squeeze()

        # Extrapolate by multiplying fixed params by ratios
        extrap_preds = y_hat_df.sel(year=fix_global_c) * global_c_factors
        extrap_params = param_df.sel(year=fix_global_c) * global_c_factors

        # concatenate extrapolation and pre-2100
        preds = xr.concat([y_hat_df, extrap_preds], dim="year")
        parameters = xr.concat([param_df, extrap_params], dim="year")

        # For the local case we don't care about the time dimension, the
        # extrapolation will index on the last year available of damages (2099)
        # by design (changed using the fix_global_c parameter) and will use
        # those damages to extrapolate to the last year of interest (2300)
        # which comes from the global_c object.
        if global_c_factors.year.dims == ():
            raise NotImplementedError("WATCH OUT! Local is scrapped.")

    # Return all results
    res = {
        "parameters": parameters,
        "preds": preds,
    }

    return res


def compute_damages(anomaly, betas, formula):
    """Calculate damages using FAIR anomalies (either control or pulse).

    Parameters
    ----------
    anomaly: xr.DataArray or xr.Dataset
        `xarray` object containing FAIR temperature anomaly. These can be
        either from control or pulse runs.
    betas: xr.DataSet
        An `xarray` object containing the fit coefficients from the yearly damage
        functions
    formula : str
        The formula used to calculate damages

    Returns
    -------
        xr.DataArray with damages
    """

    # Parse formula to calculate damages
    if (
        formula
        == "damages ~ -1 + anomaly * gmsl + anomaly * np.power(gmsl, 2) + gmsl * np.power(anomaly, 2) + np.power(anomaly, 2) * np.power(gmsl, 2)"
    ):
        betas, anomaly = xr.broadcast(betas, anomaly)
        betas = betas.transpose(*sorted(betas.dims)).sortby(list(betas.dims))
        anomaly = anomaly.transpose(*sorted(anomaly.dims)).sortby(list(anomaly.dims))
        damages_fair = (
            betas["anomaly"] * anomaly.temperature
            + betas["np.power(anomaly, 2)"] * np.power(anomaly.temperature, 2)
            + betas["gmsl"] * anomaly.gmsl
            + betas["np.power(gmsl, 2)"] * np.power(anomaly.gmsl, 2)
            + betas["anomaly:gmsl"] * anomaly.temperature * anomaly.gmsl
            + betas["anomaly:np.power(gmsl, 2)"]
            * anomaly.temperature
            * np.power(anomaly.gmsl, 2)
            + betas["gmsl:np.power(anomaly, 2)"]
            * anomaly.gmsl
            * np.power(anomaly.temperature, 2)
            + betas["np.power(anomaly, 2):np.power(gmsl, 2)"]
            * np.power(anomaly.temperature, 2)
            * np.power(anomaly.gmsl, 2)
        )

    elif (
        formula
        == "damages ~ -1 + anomaly:gmsl + anomaly:np.power(gmsl, 2) + gmsl:np.power(anomaly, 2) + np.power(anomaly, 2):np.power(gmsl, 2)"
    ):
        damages_fair = (
            betas["anomaly:gmsl"] * anomaly.temperature * anomaly.gmsl
            + betas["anomaly:np.power(gmsl, 2)"]
            * anomaly.temperature
            * np.power(anomaly.gmsl, 2)
            + betas["gmsl:np.power(anomaly, 2)"]
            * anomaly.gmsl
            * np.power(anomaly.temperature, 2)
            + betas["np.power(anomaly, 2):np.power(gmsl, 2)"]
            * np.power(anomaly.temperature, 2)
            * np.power(anomaly.gmsl, 2)
        )
    elif formula == "damages ~ -1 + gmsl:anomaly + gmsl:np.power(anomaly, 2)":
        damages_fair = betas[
            "gmsl:anomaly"
        ] * anomaly.temperature * anomaly.gmsl + betas[
            "gmsl:np.power(anomaly, 2)"
        ] * anomaly.gmsl * np.power(
            anomaly.temperature, 2
        )
    elif (
        formula
        == "damages ~ -1 + anomaly + np.power(anomaly, 2) + gmsl + np.power(gmsl, 2)"
    ):

        damages_fair = (
            betas["anomaly"] * anomaly.temperature
            + betas["np.power(anomaly, 2)"] * np.power(anomaly.temperature, 2)
            + betas["gmsl"] * anomaly.gmsl
            + betas["np.power(gmsl, 2)"] * np.power(anomaly.gmsl, 2)
        )
    elif (
        formula == "damages ~ anomaly + np.power(anomaly, 2) + gmsl + np.power(gmsl, 2)"
    ):

        damages_fair = (
            betas["Intercept"]
            + betas["anomaly"] * anomaly.temperature
            + betas["np.power(anomaly, 2)"] * np.power(anomaly.temperature, 2)
            + betas["gmsl"] * anomaly.gmsl
            + betas["np.power(gmsl, 2)"] * np.power(anomaly.gmsl, 2)
        )
    elif formula == "damages ~ -1 + gmsl + anomaly + np.power(anomaly, 2)":

        damages_fair = (
            betas["anomaly"] * anomaly.temperature
            + betas["np.power(anomaly, 2)"] * np.power(anomaly.temperature, 2)
            + betas["gmsl"] * anomaly.gmsl
        )

    elif formula == "damages ~ -1 + anomaly + np.power(anomaly, 2)":

        damages_fair = betas["anomaly"] * anomaly.temperature + betas[
            "np.power(anomaly, 2)"
        ] * np.power(anomaly.temperature, 2)

    elif formula == "damages ~ anomaly + np.power(anomaly, 2)":

        damages_fair = (
            betas["Intercept"]
            + betas["anomaly"] * anomaly.temperature
            + betas["np.power(anomaly, 2)"] * np.power(anomaly.temperature, 2)
        )
    elif formula == "damages ~ gmsl + np.power(gmsl, 2)":

        damages_fair = (
            betas["Intercept"]
            + betas["gmsl"] * anomaly.gmsl
            + betas["np.power(gmsl, 2)"] * np.power(anomaly.gmsl, 2)
        )
    elif formula == "damages ~ -1 + gmsl + np.power(gmsl, 2)":

        damages_fair = betas["gmsl"] * anomaly.gmsl + betas[
            "np.power(gmsl, 2)"
        ] * np.power(anomaly.gmsl, 2)
    elif formula == "damages ~ -1 + gmsl":

        damages_fair = betas["gmsl"] * anomaly.gmsl

    elif formula == "damages ~ -1 + np.power(anomaly, 2)":
        damages_fair = betas["np.power(anomaly, 2)"] * np.power(anomaly.temperature, 2)

    return damages_fair
