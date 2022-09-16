import os
import dask
import logging
import subprocess
from subprocess import CalledProcessError
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import xarray as xr
from dscim.descriptors import cachedproperty
from itertools import product
from dscim.menu.decorators import save
from dscim.menu.simple_storage import StackedDamages, EconVars
from dscim.utils.utils import (
    model_outputs,
    compute_damages,
    c_equivalence,
    power,
    quantile_weight_quantilereg,
    extrapolate,
)


class MainRecipe(StackedDamages, ABC):
    """Main class for DSCIM execution.

    Parameters
    ----------
    discounting_type : str
        Choice of discounting: ``euler_gwr``, ``euler_ramsey``, ``constant``, ``naive_ramsey``,
        ``naive_gwr``, ``gwr_gwr``.
    discrete_discounting: boolean
        Discounting is discrete if ``True``, else continuous (default is ``False``).
    fit_type : str
        Type of damage function estimation: ``'ols'``, ``'quantreg'``
    weitzman_parameter: list of float or None, optional
        If <= 1: The share of global consumption below which bottom coding is implemented.
        If > 1: Absolute dollar value of global consumption below which bottom.
        Default is [0.1, 0.5].
        coding is implemented.
    fair_aggregation : list of str or None, optional
        How to value climate uncertainty from FAIR: ``median``, ``mean``,
        ``ce``, ``median_params``. Default is ["ce", "mean", "gwr_mean",
        "median", "median_params"].
    rho : float
        Pure rate of time preference parameter
    fair_dims : list of str or None, optional
        List of dimensions over which the FAIR CE/mean/median options should be collapsed. Default value is ["simulation"], but lists such as ["simulation", "rcp", "ssp"] can be passed. Note: If dimensions other than 'simulation' are passed, 'median_params' fair aggregation cannot be passed.
    """

    NAME = ""
    CONST_DISC_RATES = [0.01, 0.015, 0.02, 0.025, 0.03, 0.05]
    DISCOUNT_TYPES = [
        "constant",
        "constant_model_collapsed",
        "naive_ramsey",
        "euler_ramsey",
        "naive_gwr",
        "gwr_gwr",
        "euler_gwr",
    ]
    FORMULAS = [
        "damages ~ -1 + np.power(anomaly, 2)",
        "damages ~ gmsl + np.power(gmsl, 2)",
        "damages ~ -1 + gmsl + np.power(gmsl, 2)",
        "damages ~ -1 + gmsl",
        "damages ~ anomaly + np.power(anomaly, 2)",
        "damages ~ -1 + anomaly + np.power(anomaly, 2)",
        "damages ~ -1 + gmsl + anomaly + np.power(anomaly, 2)",
        "damages ~ -1 + anomaly + np.power(anomaly, 2) + gmsl + np.power(gmsl, 2)",
        "damages ~ -1 + anomaly * gmsl + anomaly * np.power(gmsl, 2) + gmsl * np.power(anomaly, 2) + np.power(anomaly, 2) * np.power(gmsl, 2)",
        "damages ~ anomaly + np.power(anomaly, 2) + gmsl + np.power(gmsl, 2)",
        "damages ~ -1 + anomaly:gmsl + anomaly:np.power(gmsl, 2) + gmsl:np.power(anomaly, 2) + np.power(anomaly, 2):np.power(gmsl, 2)",
        "damages ~ -1 + gmsl:anomaly + gmsl:np.power(anomaly, 2)",
    ]

    def __init__(
        self,
        econ_vars,
        climate_vars,
        sector,
        formula,
        sector_path=None,
        save_path=None,
        rho=0.00461878399,
        eta=1.421158116,
        fit_type="ols",
        discounting_type=None,
        ext_method="global_c_ratio",
        ext_subset_start_year=2085,
        ext_subset_end_year=2099,
        ext_end_year=2300,
        subset_dict=None,
        ce_path=None,
        damage_function_path=None,
        clip_gmsl=False,
        gdppc_bottom_code=39.39265060424805,
        scc_quantiles=None,
        scenario_dimensions=None,
        weitzman_parameter=None,
        fair_aggregation=None,
        filename_suffix="",
        discrete_discounting=False,
        quantreg_quantiles=None,
        quantreg_weights=None,
        full_uncertainty_quantiles=None,
        extrap_formula=None,
        fair_dims=None,
        save_files=None,
        **kwargs,
    ):
        if scc_quantiles is None:
            scc_quantiles = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]

        if weitzman_parameter is None:
            weitzman_parameter = [0.1, 0.5]

        if fair_aggregation is None:
            fair_aggregation = ["ce", "mean", "gwr_mean", "median", "median_params"]

        if quantreg_quantiles is None:
            quantreg_quantiles = [
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
            ]

        if quantreg_weights is None:
            quantreg_weights = [
                0.075,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.075,
            ]

        if full_uncertainty_quantiles is None:
            full_uncertainty_quantiles = [
                0.01,
                0.05,
                0.17,
                0.25,
                0.5,
                0.75,
                0.83,
                0.95,
                0.99,
            ]

        if fair_dims is None:
            fair_dims = ["simulation"]

        if save_files is None:
            save_files = [
                "damage_function_points",
                "damage_function_coefficients",
                "damage_function_fit",
                "marginal_damages",
                "discount_factors",
                "uncollapsed_sccs",
                "scc",
                "uncollapsed_discount_factors",
                "uncollapsed_marginal_damages",
                "global_consumption",
                "global_consumption_no_pulse",
            ]

        super().__init__(
            sector_path=sector_path,
            save_path=save_path,
            econ_vars=econ_vars,
            climate_vars=climate_vars,
            gdppc_bottom_code=gdppc_bottom_code,
            eta=eta,
            subset_dict=subset_dict,
            ce_path=ce_path,
        )

        self.rho = rho
        self.eta = eta
        self.fit_type = fit_type
        self.fair_aggregation = fair_aggregation
        self.filename_suffix = filename_suffix
        self.weitzman_parameter = weitzman_parameter
        self.discrete_discounting = discrete_discounting
        self.discounting_type = discounting_type
        self.sector = sector
        self.save_path = save_path
        self.damage_function_path = damage_function_path
        self.ext_subset_start_year = ext_subset_start_year
        self.ext_subset_end_year = ext_subset_end_year
        self.ext_end_year = ext_end_year
        self.ext_method = ext_method
        self.clip_gmsl = clip_gmsl
        self.scenario_dimensions = scenario_dimensions
        self.scc_quantiles = scc_quantiles
        self.quantreg_quantiles = quantreg_quantiles
        self.quantreg_weights = quantreg_weights
        self.full_uncertainty_quantiles = full_uncertainty_quantiles
        self.formula = formula
        self.ce_path = ce_path
        self.extrap_formula = extrap_formula
        self.fair_dims = fair_dims
        self.save_files = save_files
        self.__dict__.update(**kwargs)
        self.kwargs = kwargs

        self.logger = logging.getLogger(__name__)

        if self.quantreg_quantiles is not None:
            assert len(self.quantreg_quantiles) == len(
                self.quantreg_weights
            ), "Length of quantreg quantiles does not match length of weights."

        assert (
            self.discounting_type in self.DISCOUNT_TYPES
        ), f"Discount type not implemented. Try one of {self.DISCOUNT_TYPES}."

        assert (
            self.formula in self.FORMULAS
        ), f"Formula not implemented. Try one of {self.FORMULAS}."

        # Set stream of discounts to None if discounting_type is 'constant'
        # 'constant_model_collapsed' should be here except that we allow
        # for a collapsed-model Ramsey rate to be calculated (for labour
        # and energy purposes)
        if self.discounting_type in ["constant", "constant_model_collapsed"]:
            self.stream_discount_factors = None

        # assert formulas for which clip_gmsl is implemented
        if self.clip_gmsl:
            assert self.formula in [
                "damages ~ -1 + anomaly + np.power(anomaly, 2) + gmsl + np.power(gmsl, 2)",
                "damages ~ -1 + gmsl + np.power(gmsl, 2)",
            ]

    def __repr__(self):
        return f"""
        Running {self.NAME}
        sector: {self.sector}
        discounting: {self.discounting_type}
        eta: {self.eta}
        rho: {self.rho}
        """

    def order_plate(self, course):
        """
        Execute menu option section and save results

        This method is a entry point to the class and allows the user to
        calculate different elements of a specific menu option. These elements
        will automatically be saved in the path defined in `save_path`.

        Parameters
        ----------
        course str
            Output to be calculated. Options are:
                - `damage_function`: Return and save all damage function
                  elements including damage function points, coefficients, and
                  fitted values.
                - `scc`: Return Social Cost of Carbon calculation. All elements
                from `damage_function` are saved and returned.

        Returns
        -------
        None. Saved all elements to `save_path`

        """

        self.logger.info(f"\n Executing {self.__repr__()}")

        def damage_function():
            self.logger.info("Processing damage functions ...")
            if self.damage_function_path is None:
                self.logger.info(
                    "Existing damage functions not found. Damage points will be loaded."
                )
                self.damage_function_points
            self.damage_function_coefficients
            try:
                self.damage_function_fit
            except FileNotFoundError:
                pass

        def scc():
            damage_function()
            self.global_consumption
            self.global_consumption_no_pulse
            self.logger.info("Processing SCC calculation ...")
            if self.fit_type == "quantreg":
                self.full_uncertainty_iqr
            else:
                if len(self.fair_aggregation) > 0:
                    self.stream_discount_factors
                    self.calculate_scc
                self.uncollapsed_sccs
                self.uncollapsed_marginal_damages
                self.uncollapsed_discount_factors

        course_dict = {"damage_function": damage_function, "scc": scc}

        try:
            course_dict[course]()
            self.logger.info(f"Results available: {self.save_path}")
        except KeyError as e:
            self.logger.error(f"{course} is not a valid option: {e}")
            raise e
        except Exception as e:
            self.logger.error("Error detected.")
            raise e

        return None

    def order_scc(self):
        """
        Execute menu option section and save results

        This method is a wrapper to `order_plate` that calls the "scc" course,
        which is the Social Cost of Carbon calculation. Elements involved in the calculation
        (`fair` and `damage_function`) will automatically be saved in the path
        defined in `save_path`.

        Parameters
        ----------
        None

        Returns
        -------
        xr.Dataset of SCCs

        """

        self.logger.info(f"\n Executing {self.__repr__()}")

        try:
            sccds = self.calculate_scc
            self.logger.info(f"Results available: {self.save_path}")
        except Exception as e:
            self.logger.error("Error detected.")
            raise e

        if ("rcp45" in sccds.rcp) or ("rcp85" in sccds.rcp):
            # leave the dataset alone if there are already rcp scenario names
            pass
        else:
            # rename the CMIP6 scenario names that start with "ssp*"
            sccds = sccds.sortby(sccds.rcp)

            rcpdt = {
                "ssp126": "RCP2.6",
                "ssp245": "RCP4.5",
                "ssp370": "RCP7.0",
                "ssp460": "RCP6.0",
                "ssp585": "RCP8.5",
            }
            rlst = []
            for rcp in sccds.rcp.values:
                rlst.append(rcpdt[rcp])
            sccds.coords["rcp"] = rlst
            sccds = sccds.sortby(sccds.rcp)

        return sccds.squeeze(drop=True)

    @property
    def output_attrs(self):
        """Return dict with class attributes for output metadata

        Returns
        ------
            A dict Class metadata
        """

        import dscim

        # find machine name
        machine_name = os.getenv("HOSTNAME")
        if machine_name is None:
            try:
                machine_name = os.uname()[1]
            except AttributeError:
                machine_name = "unknown"

        # find git commit hash
        try:
            label = subprocess.check_output(["git", "describe", "--always"]).strip()
        except CalledProcessError:
            label = "unknown"

        meta = {}
        for attr_dict in [
            vars(self),
            vars(vars(self)["climate"]),
            vars(vars(self)["econ_vars"]),
        ]:
            meta.update(
                {
                    k: v
                    for k, v in attr_dict.items()
                    if (type(v) not in [xr.DataArray, xr.Dataset, pd.DataFrame])
                    and k not in ["damage_function", "logger"]
                }
            )

        # update with git hash and machine name
        meta.update(dict(machine=machine_name, commit=label))

        # convert to strs
        meta = {k: v if type(v) in [int, float] else str(v) for k, v in meta.items()}

        return meta

    @cachedproperty
    def collapsed_pop(self):
        """Collapse population according to discount type."""
        if (self.discounting_type == "constant") or ("ramsey" in self.discounting_type):
            pop = self.pop
        elif self.discounting_type == "constant_model_collapsed":
            pop = self.pop.mean("model")
        elif "gwr" in self.discounting_type:
            pop = self.pop.mean(["model", "ssp"])
        return pop

    @abstractmethod
    def ce_cc_calculation(self):
        """Calculate CE damages depending on discount type"""

    @abstractmethod
    def ce_no_cc_calculation(self):
        """Calculate GDP CE depending on discount type."""

    @abstractmethod
    def calculated_damages(self):
        """Calculate damages (difference between CEs) for collapsing"""

    @abstractmethod
    def global_damages_calculation(self):
        """Calculate global collapsed damages for a desired discount type"""

    @property
    def ce_cc(self):
        """Certainty equivalent of consumption with climate change damages"""
        return self.ce_cc_calculation()

    @property
    def ce_no_cc(self):
        """Certainty equivalent of consumption without climate change damages"""
        return self.ce_no_cc_calculation()

    @cachedproperty
    @save(name="damage_function_points")
    def damage_function_points(self) -> pd.DataFrame:
        """Global damages by RCP/GCM or SLR

        Returns
        --------
            pd.DataFrame
        """
        df = self.global_damages_calculation()

        if "slr" in df.columns:
            df = df.merge(self.climate.gmsl, on=["year", "slr"])
        if "gcm" in df.columns:
            df = df.merge(self.climate.gmst, on=["year", "gcm", "rcp"])

        # removing illegal combinations from estimation
        if any([i in df.ssp.unique() for i in ["SSP1", "SSP5"]]):
            self.logger.info("Dropping illegal model combinations.")
            for var in [i for i in df.columns if i in ["anomaly", "gmsl"]]:
                df.loc[
                    ((df.ssp == "SSP1") & (df.rcp == "rcp85"))
                    | ((df.ssp == "SSP5") & (df.rcp == "rcp45")),
                    var,
                ] = np.nan

        # agriculture lacks ACCESS0-1/rcp85 combo
        if "agriculture" in self.sector:
            self.logger.info("Dropping illegal model combinations for agriculture.")
            df.loc[(df.gcm == "ACCESS1-0") & (df.rcp == "rcp85"), "anomaly"] = np.nan

        return df

    def damage_function_calculation(self, damage_function_points, global_consumption):
        """The damage function model fit may be : (1) ssp specific, (2) ssp-model specific, (3) unique across ssp-model.
        This depends on the type of discounting. In each case the input data passed to the fitting functions and the formatting of the returned
        output is different because dimensions are different. This function handles this and returns the model fit.

        Returns
        ------
        dict with two xr.Datasets, 'params' (model fit) and 'preds' (predictions from model fit), with dimensions depending
        on self.discounting_type.
        """

        yrs = range(self.climate.pulse_year, self.ext_subset_end_year + 1)

        params_list, preds_list = [], []

        if self.discounting_type == "constant_model_collapsed":
            for ssp in damage_function_points["ssp"].unique():

                # Subset dataframe to specific SSP
                fit_subset = damage_function_points[
                    damage_function_points["ssp"] == ssp
                ]

                global_c_subset = global_consumption.sel({"ssp": ssp})
                # Fit damage function curves using the data subset
                damage_function = model_outputs(
                    damage_function=fit_subset,
                    formula=self.formula,
                    extrap_formula=self.extrap_formula,
                    type_estimation=self.fit_type,
                    global_c=global_c_subset,
                    extrapolation_type=self.ext_method,
                    quantiles=self.quantreg_quantiles,
                    year_range=yrs,
                    extrap_year=self.ext_subset_start_year,
                    year_start_pred=self.ext_subset_end_year + 1,
                    year_end_pred=self.ext_end_year,
                )

                # Add variables
                params = damage_function["parameters"].expand_dims(
                    dict(
                        discount_type=[self.discounting_type],
                        ssp=[ssp],
                        model=[str(list(self.gdp.model.values))],
                    )
                )

                preds = damage_function["preds"].expand_dims(
                    dict(
                        discount_type=[self.discounting_type],
                        ssp=[ssp],
                        model=[str(list(self.gdp.model.values))],
                    )
                )

                params_list.append(params)
                preds_list.append(preds)

        elif (self.discounting_type == "constant") or (
            "ramsey" in self.discounting_type
        ):
            for ssp, model in list(
                product(
                    damage_function_points.ssp.unique(),
                    damage_function_points.model.unique(),
                )
            ):

                # Subset dataframe to specific SSP-IAM combination.
                fit_subset = damage_function_points[
                    (damage_function_points["ssp"] == ssp)
                    & (damage_function_points["model"] == model)
                ]

                global_c_subset = global_consumption.sel({"ssp": ssp, "model": model})

                # Fit damage function curves using the data subset
                damage_function = model_outputs(
                    damage_function=fit_subset,
                    formula=self.formula,
                    extrap_formula=self.extrap_formula,
                    type_estimation=self.fit_type,
                    global_c=global_c_subset,
                    extrapolation_type=self.ext_method,
                    quantiles=self.quantreg_quantiles,
                    year_range=yrs,
                    extrap_year=self.ext_subset_start_year,
                    year_start_pred=self.ext_subset_end_year + 1,
                    year_end_pred=self.ext_end_year,
                )

                # Add variables
                params = damage_function["parameters"].expand_dims(
                    dict(
                        discount_type=[self.discounting_type], ssp=[ssp], model=[model]
                    )
                )

                preds = damage_function["preds"].expand_dims(
                    dict(
                        discount_type=[self.discounting_type], ssp=[ssp], model=[model]
                    )
                )

                params_list.append(params)
                preds_list.append(preds)

        elif "gwr" in self.discounting_type:
            # Fit damage function across all SSP-IAM combinations, as expected
            # from the Weitzman-Ramsey discounting
            fit_subset = damage_function_points

            # Fit damage function curves using the data subset
            damage_function = model_outputs(
                damage_function=fit_subset,
                type_estimation=self.fit_type,
                formula=self.formula,
                extrap_formula=self.extrap_formula,
                global_c=global_consumption,
                extrapolation_type=self.ext_method,
                quantiles=self.quantreg_quantiles,
                year_range=yrs,
                extrap_year=self.ext_subset_start_year,
                year_start_pred=self.ext_subset_end_year + 1,
                year_end_pred=self.ext_end_year,
            )

            # Add variables
            params = damage_function["parameters"].expand_dims(
                dict(
                    discount_type=[self.discounting_type],
                    ssp=[str(list(self.gdp.ssp.values))],
                    model=[str(list(self.gdp.model.values))],
                )
            )

            preds = damage_function["preds"].expand_dims(
                dict(
                    discount_type=[self.discounting_type],
                    ssp=[str(list(self.gdp.ssp.values))],
                    model=[str(list(self.gdp.model.values))],
                )
            )

            params_list.append(params)
            preds_list.append(preds)

        return dict(
            params=xr.combine_by_coords(params_list),
            preds=xr.combine_by_coords(preds_list),
        )

    @cachedproperty
    def damage_function(self):
        """Calls damage function calculation method.

        This function calls the damage function calculation in
        model_outputs(). It calculates a damage function for each
        passed `scenario_dimension` based on subsets of
        self.damage_function_points and extrapolates this function
        using the specified method for all years post-end_ext_subset_year.

        Returns
        -------
        dict
            dict['params'] is a dataframe of betas for each year
            dict['preds'] is a dataframe of predicted y hat for each
                year and anomaly
        """
        if self.scenario_dimensions is None:

            # this only occurs for global discounting
            # with a single scenario passed
            damage_function = self.damage_function_calculation(
                damage_function_points=self.damage_function_points,
                global_consumption=self.global_consumption,
            )

        else:
            # cycle through the different combinations of the scenario dims
            subset = self.damage_function_points.groupby(self.scenario_dimensions)
            damage_function, dict_list = {}, []

            for name, dt in subset:

                # turn single-dim into a tuple to make indexing easier later
                if len(self.scenario_dimensions) == 1:
                    name = tuple([name])

                df = self.damage_function_calculation(
                    damage_function_points=dt,
                    global_consumption=self.global_consumption,
                )
                # assigning dimensions to each dataArray in the dictionary
                for key in df.keys():
                    df[key] = df[key].expand_dims(
                        {
                            var: [val]
                            for var, val in zip(self.scenario_dimensions, list(name))
                        }
                    )
                dict_list.append(df)

            # concatenate different scenarios into one big dataArray
            damage_function["params"] = xr.combine_by_coords(
                [x["params"] for x in dict_list]
            )

            damage_function["preds"] = xr.combine_by_coords(
                [x["preds"] for x in dict_list]
            )

        return damage_function

    @property
    @save(name="damage_function_coefficients")
    def damage_function_coefficients(self) -> xr.Dataset:
        """
        Load damage function coefficients if the coefficients are provided by the user.
        Otherwise, compute them.
        """
        if self.damage_function_path is not None:
            return xr.open_dataset(
                f"{self.damage_function_path}/{self.NAME}_{self.discounting_type}_eta{self.eta}_rho{self.rho}_damage_function_coefficients.nc4"
            )
        else:
            return self.damage_function["params"]

    @property
    @save(name="damage_function_fit")
    def damage_function_fit(self) -> xr.Dataset:
        """
        Load fitted damage function if the fit is provided by the user.
        Otherwise, compute them.
        """
        if self.damage_function_path is not None:
            return xr.open_dataset(
                f"{self.damage_function_path}/{self.NAME}_{self.discounting_type}_eta{self.eta}_rho{self.rho}_damage_function_fit.nc4"
            )
        else:
            return self.damage_function["preds"]

    @property
    def median_params_marginal_damages(self):
        """Calculate marginal damages due to a pulse using a FAIR simulation
        calculated with the median climate parameters.
        """

        fair_control = self.climate.fair_median_params_control
        fair_pulse = self.climate.fair_median_params_pulse

        if self.clip_gmsl:
            fair_control["gmsl"] = np.minimum(fair_control["gmsl"], self.gmsl_max)
            fair_pulse["gmsl"] = np.minimum(fair_pulse["gmsl"], self.gmsl_max)

        damages_pulse = compute_damages(
            fair_pulse,
            betas=self.damage_function_coefficients,
            formula=self.formula,
        )

        damages_no_pulse = compute_damages(
            fair_control,
            betas=self.damage_function_coefficients,
            formula=self.formula,
        )

        median_params_marginal_damages = damages_pulse - damages_no_pulse

        # collapse further if further collapsing dimensions are provided
        if len(self.fair_dims) > 1:
            median_params_marginal_damages = median_params_marginal_damages.mean(
                [i for i in self.fair_dims if i in median_params_marginal_damages.dims]
            )

        # add a Weitzman parameter dimension so this dataset can be concatenated
        # with the other FAIR aggregation results
        median_params_marginal_damages = median_params_marginal_damages.expand_dims(
            {"weitzman_parameter": [str(i) for i in self.weitzman_parameter]}
        )

        return median_params_marginal_damages

    @abstractmethod
    def global_consumption_calculation(self, disc_type):
        """Calculation of global consumption without climate change

        Returns
        -------
            xr.DataArray
        """

    def global_consumption_per_capita(self, disc_type):
        """Global consumption per capita

        Returns
        -------
            xr.DataArray
        """

        # Calculate global consumption per capita
        array_pc = self.global_consumption_calculation(
            disc_type
        ) / self.collapsed_pop.sum("region")

        if self.NAME == "equity":

            # equity recipe's growth is capped to
            # risk aversion recipe's growth rates
            extrapolated = extrapolate(
                xr_array=array_pc,
                start_year=self.ext_subset_start_year,
                end_year=self.ext_subset_end_year,
                interp_year=self.ext_end_year,
                method="growth_constant",
                cap=self.risk_aversion_growth_rates(),
            )

        else:
            extrapolated = extrapolate(
                xr_array=array_pc,
                start_year=self.ext_subset_start_year,
                end_year=self.ext_subset_end_year,
                interp_year=self.ext_end_year,
                method="growth_constant",
            )

        complete_array = xr.concat([array_pc, extrapolated], dim="year")

        return complete_array

    @cachedproperty
    @save(name="global_consumption")
    def global_consumption(self):
        """Global consumption without climate change"""

        # rff simulation means that GDP already exists out to 2300
        if 2300 in self.gdp.year:

            self.logger.debug("Global consumption found up to 2300.")
            global_cons = self.gdp.sum("region").rename("global_consumption")
        else:

            self.logger.info("Extrapolating global consumption.")

            # holding population constant
            # from 2100 to 2300 with 2099 values
            pop = self.collapsed_pop.sum("region")
            pop = pop.reindex(
                year=range(pop.year.min().values, self.ext_end_year + 1),
                method="ffill",
            )

            # Calculate global consumption back by
            global_cons = (
                self.global_consumption_per_capita(self.discounting_type) * pop
            )

        # Add dimension
        # @TODO: remove this line altogether
        global_cons = global_cons.expand_dims(
            {"discount_type": [self.discounting_type]}
        )

        return global_cons

    def weitzman_min(self, no_cc_consumption, cc_consumption, parameter):
        """
        Implements bottom coding that fixes marginal utility below a threshold
        to the marginal utility at that threshold. The threshold is the Weitzman
        parameter.

        Parameters
        ----------
        no_cc_consumption: xr.DataArray
            Consumption array of which the share will be used to calculate
            the absolute Weitzman parameter, only if parameter <= 1.
        consumption: xr.DataArray
            Consumption array to be bottom-coded.
        parameter: float
            A positive number representing the Weitzman parameter, below which marginal utility will be
            top coded; ie., 0.01 implies marginal utility is top coded to the
            value of marginal utility at 1% of no-climate change global consumption.
            If parameter > 1, it is assumed to be an absolute value.
            If parameter <= 1, it is assumed to be a share of future global consumption
            (without climate change).

        Returns
        -------
            xr.Dataset

        """
        # if parameter is share of consumption,
        # multiply by no-climate-change consumption
        if parameter <= 1:
            parameter = parameter * no_cc_consumption

        w_utility = parameter ** (1 - self.eta) / (1 - self.eta)
        bottom_utility = parameter ** (-self.eta) * (parameter - cc_consumption)
        bottom_coded_cons = power(
            ((1 - self.eta) * (w_utility - bottom_utility)), (1 / (1 - self.eta))
        )

        clipped_cons = xr.where(
            cc_consumption > parameter, cc_consumption, bottom_coded_cons
        )

        return clipped_cons

    @property
    def gmsl_max(self):
        """
        This function finds the GMSL value at which the damage function
        reaches its local maximum along the GMSL dimension.

        Returns
        -------
        xr.DataArray
            the array of GMSL values at which the local maximum is located, for all
            years, ssps, models, and if applicable, values of GMST
        """

        if self.formula in [
            "damages ~ -1 + anomaly + np.power(anomaly, 2) + gmsl + np.power(gmsl, 2)",
            "damages ~ -1 + gmsl + np.power(gmsl, 2)",
        ]:

            gmsl_max = -self.damage_function_coefficients["gmsl"] / (
                2 * self.damage_function_coefficients["np.power(gmsl, 2)"]
            )

            # if the damage function curves up, there is no local max.
            # Thus there will be no linearization.
            gmsl_max = xr.where(
                self.damage_function_coefficients["np.power(gmsl, 2)"] > 0,
                np.inf,
                gmsl_max,
            )

            # confirm that all max values are positive, which is expected
            # for our damage functions
            assert len(gmsl_max.where(gmsl_max < 0, drop=True)) == 0

        else:
            raise NotImplementedError(
                f"The first derivative w.r.t. gmsl for {self.formula} has not been implemented in the menu."
            )

        return gmsl_max

    @cachedproperty
    @save(name="global_consumption_no_pulse")
    def global_consumption_no_pulse(self):
        """Global consumption under FAIR control scenario."""

        fair_control = self.climate.fair_control

        if self.clip_gmsl:
            fair_control["gmsl"] = np.minimum(fair_control["gmsl"], self.gmsl_max)

        damages = compute_damages(
            fair_control,
            betas=self.damage_function_coefficients,
            formula=self.formula,
        )

        cc_cons = self.global_consumption - damages

        gc_no_pulse = []
        for wp in self.weitzman_parameter:

            gc = self.weitzman_min(
                no_cc_consumption=self.global_consumption,
                cc_consumption=cc_cons,
                parameter=wp,
            )
            gc = gc.assign_coords({"weitzman_parameter": str(wp)})

            gc_no_pulse.append(gc)

        return xr.concat(gc_no_pulse, dim="weitzman_parameter")

    @cachedproperty
    @save(name="global_consumption_pulse")
    def global_consumption_pulse(self):
        """Global consumption under FAIR pulse scenario."""

        fair_pulse = self.climate.fair_pulse

        if self.clip_gmsl:
            fair_pulse["gmsl"] = np.minimum(fair_pulse["gmsl"], self.gmsl_max)

        damages = compute_damages(
            fair_pulse,
            betas=self.damage_function_coefficients,
            formula=self.formula,
        )

        cc_cons = self.global_consumption - damages
        gc_no_pulse = []
        for wp in self.weitzman_parameter:
            gc = self.weitzman_min(
                no_cc_consumption=self.global_consumption,
                cc_consumption=cc_cons,
                parameter=wp,
            )
            gc = gc.assign_coords({"weitzman_parameter": str(wp)})
            gc_no_pulse.append(gc)
        return xr.concat(gc_no_pulse, dim="weitzman_parameter")

    @property
    @save(name="ce_fair_pulse")
    def ce_fair_pulse(self):
        """Certainty equivalent of global consumption under FAIR pulse scenario"""
        ce_array = self.ce(self.global_consumption_pulse, dims=self.fair_dims)

        return ce_array.rename("ce_fair_pulse")

    @property
    @save(name="ce_fair_no_pulse")
    def ce_fair_no_pulse(self):
        """Certainty equivalent of global consumption under FAIR control scenario"""

        ce_array = self.ce(self.global_consumption_no_pulse, dims=self.fair_dims)

        return ce_array.rename("ce_fair_no_pulse")

    @cachedproperty
    @save(name="marginal_damages")
    def marginal_damages(self):
        """Marginal damages due to additional pulse"""

        marginal_damages = []

        for agg in [i for i in self.fair_aggregation if i != "median"]:

            if agg == "ce":
                md = self.ce_fair_no_pulse - self.ce_fair_pulse
            elif agg in ["mean", "gwr_mean"]:
                md = self.global_consumption_no_pulse.mean(
                    self.fair_dims
                ) - self.global_consumption_pulse.mean(self.fair_dims)
            elif agg == "median_params":
                md = self.median_params_marginal_damages
            else:
                raise NotImplementedError(
                    (
                        f"{agg} is not available. Enter list including"
                        '["ce", "fair", "median", "median_params"]'
                    )
                )

            md = md.assign_coords({"fair_aggregation": agg}).expand_dims(
                "fair_aggregation"
            )

            # convert to the marginal damages from a single tonne
            md = md * self.climate.conversion
            marginal_damages.append(md)

        return xr.concat(marginal_damages, dim="fair_aggregation")

    @cachedproperty
    @save(name="scc")
    def calculate_scc(self):
        """Calculate range of FAIR-aggregated SCCs"""

        discounted_damages = self.discounted_damages(
            damages=self.marginal_damages, discrate=self.discounting_type
        )
        discounted_damages = discounted_damages.sum(dim="year").rename("scc")

        if "median" in self.fair_aggregation:
            median = self.uncollapsed_sccs.median(self.fair_dims)
            median["fair_aggregation"] = ["median"]
            discounted_damages = xr.merge([median.rename("scc"), discounted_damages])

        return discounted_damages

    def discounted_damages(self, damages, discrate):
        """Discount marginal damages. Distinguishes between constant discount rates method and non-constant discount rates.

        Parameters
        ----------
        damages : xr.DataArray or xr.Dataset
            Array of damages with a`discount_type` dimension to subset the damages.
        discrate : str
             Discounting type. Be aware that the constant rates are class-wide defined. If this str is either 'constant' or 'constant_model_collapsed', the predetermined constant discount rates are used, otherwise, the stream of (non-constant) discount factors from self.stream_discount_factors() is used.

        Returns
        -------
            xr.Dataset
        """

        if discrate in ["constant", "constant_model_collapsed"]:
            if self.discrete_discounting:
                discrate_damages = [
                    damages * (1 / (1 + r)) ** (damages.year - self.climate.pulse_year)
                    for r in self.CONST_DISC_RATES
                ]
            else:
                discrate_damages = [
                    damages * np.exp(-r * (damages.year - self.climate.pulse_year))
                    for r in self.CONST_DISC_RATES
                ]

            pvd_damages = xr.concat(
                discrate_damages,
                dim=pd.Index(self.CONST_DISC_RATES, name="discrate"),
            )
        else:
            factors = self.calculate_stream_discount_factors(
                discounting_type=self.discounting_type,
                fair_aggregation=damages.fair_aggregation,
            )
            pvd_damages = factors * damages

        return pvd_damages

    @cachedproperty
    @save(name="uncollapsed_sccs")
    def uncollapsed_sccs(self):
        """Calculate full distribution of SCCs without FAIR aggregation"""

        md = self.global_consumption_no_pulse - self.global_consumption_pulse

        # convert to the marginal damages from a single pulse
        md = md * self.climate.conversion

        md = md.expand_dims({"fair_aggregation": ["uncollapsed"]})

        sccs = self.discounted_damages(
            damages=md,
            discrate=self.discounting_type,
        ).sum(dim="year")

        return sccs

    # @cachedproperty
    # def quantiles_sccs(self):
    #     return self.uncollapsed_sccs.quantile(self.scc_quantiles, dim=self.fair_dims)

    @cachedproperty
    @save(name="full_uncertainty_iqr")
    def full_uncertainty_iqr(self):
        """Calculate the distribution of quantile-weighted SCCs produced from
        quantile regressions.
        """
        return quantile_weight_quantilereg(
            self.uncollapsed_sccs, quantiles=self.full_uncertainty_quantiles
        )

    def calculate_discount_factors(self, cons_pc):
        """Calculates the stream of discount factors based on the Euler equation that defines an optimal
        intertemporal consumption allocation. Rearranging that equation shows that an outcome that will occur at the
        period t will be converted into today's, period 0 value, the following way :

        Discrete discounting: discount_factor_t = [ 1/(1+rho)^t ] * [ U'(C(t)) / U'(C(0)) ]

        where rho is the pure rate of time preference, U() is the utility function, U'() the first derivative,
        C(0) and C(t) today's and the future consumption respectively. The higher rho, the higher the importance
        of the present relative to the future so the lower the discount factor, and, if the utility function is concave,
        the higher the growth of consumption, the lower the importance of the future consumption relative to today, and
        therefore again the lower the discount factor.

        Using a CRRA utility function and plugging the first derivative :

        discount_factor_t = [ 1/(1+rho)^t ] * [ C(0)^eta / C(t)^eta ]

        eta represents the degree of concavity of the utility function.

        With continuous discounting,
        discount_factor_t = Product_1^t [e^-(rho + eta * g),
            where g = ln(C(t)/C(t-1))

        rearranging yields rho_continuous = e^rho_discrete - 1

        Parameters
        ----------
        cons_pc : array
            Array of per capita consumption from pulse year to end of time period.

        Returns
        -------
        `xarray.DataArray` with discount factors computed following the last equation in the above description.
        """

        # subset to pulse year onward
        cons_pc = cons_pc.sel(year=slice(self.climate.pulse_year, self.ext_end_year))

        # calculate the time preference component of the discount factors for each period.
        if self.discrete_discounting:
            # plug the unique rho in an array
            rhos = xr.DataArray(self.rho, coords=[cons_pc.year])
        else:
            # plug the unique rho in an array, and compute e^rho - 1
            rhos = np.expm1(xr.DataArray(self.rho, coords=[cons_pc.year]))

        stream_rhos = np.divide(
            1, np.multiply.accumulate((rhos.values + 1), rhos.dims.index("year"))
        )

        # calculate the marginal utility component of the discount factors for each period.
        ratio = cons_pc.sel(year=self.climate.pulse_year) ** (self.eta) / cons_pc ** (
            self.eta
        )

        # the discount factor is the product of the two components.
        factors = stream_rhos * ratio

        return factors

    def calculate_stream_discount_factors(self, discounting_type, fair_aggregation):
        """Stream of discount factors
        Returns specified Ramsey or Weitzman-Ramsey discount factors.

        Parameters
        ----------

        discounting_type : str
            Type of discounting to implement. Typically,
            self.discounting_type is passed. However, for local Euler rates,
            this changes depending on the option.

        Returns
        -------
        `xarray.DataArray`
            Discount rates indexed by year and (if Ramsey) SSP/model
        """

        # holding population constant
        # from 2100 to 2300 with 2099 values
        full_pop = self.collapsed_pop.sum("region")
        full_pop = full_pop.reindex(
            year=range(full_pop.year.min().values, self.ext_end_year + 1),
            method="ffill",
        )

        # for aggregations other than uncollapsed,
        # we need to collapse over pop dimensions
        if len(self.fair_dims) > 1:
            pop = full_pop.mean([i for i in self.fair_dims if i in full_pop.dims])
        else:
            pop = full_pop

        # for gwr_gwr, we need to calculate regular naive_ramsey rates, get
        # discount factors, and then average them at the end
        if discounting_type == "gwr_gwr":
            array = self.global_consumption_per_capita("naive_ramsey")

            # average discount factors and expand dims to match Euler
            discount_factors = (
                self.calculate_discount_factors(array)
                .mean(dim=["ssp", "model"])
                .expand_dims({"fair_aggregation": fair_aggregation})
            )

        # naive options are calculated from the no-climate-change consumption growth rate
        elif "naive" in discounting_type:
            array = self.global_consumption_per_capita(self.discounting_type)

            # expand dims to match Euler
            discount_factors = self.calculate_discount_factors(array).expand_dims(
                {"fair_aggregation": fair_aggregation}
            )

        elif "euler" in discounting_type:

            discount_factors = []
            for agg in [i for i in fair_aggregation if i != "median"]:
                if agg == "ce":
                    factors = self.calculate_discount_factors(
                        self.ce_fair_no_pulse / pop
                    )
                elif agg == "mean":
                    factors = self.calculate_discount_factors(
                        self.global_consumption_no_pulse.mean(self.fair_dims) / pop
                    )
                elif agg == "gwr_mean":
                    factors = self.calculate_discount_factors(
                        self.global_consumption_no_pulse / full_pop
                    ).mean(self.fair_dims)
                elif agg == "median_params":

                    median_params_damages = compute_damages(
                        self.climate.fair_median_params_control,
                        betas=self.damage_function_coefficients,
                        formula=self.formula,
                    )

                    median_params_consumption = (
                        self.global_consumption - median_params_damages
                    ).expand_dims(
                        {
                            "weitzman_parameter": [
                                str(i) for i in self.weitzman_parameter
                            ]
                        }
                    )

                    if len(self.fair_dims) > 1:
                        median_params_consumption = median_params_consumption.mean(
                            [
                                i
                                for i in self.fair_dims
                                if i in median_params_consumption.dims
                            ]
                        )

                    factors = self.calculate_discount_factors(
                        median_params_consumption / pop
                    )
                elif agg == "uncollapsed":
                    factors = self.calculate_discount_factors(
                        self.global_consumption_no_pulse / full_pop
                    )

                factors = factors.assign_coords({"fair_aggregation": agg})
                discount_factors.append(factors)

            discount_factors = xr.concat(discount_factors, dim="fair_aggregation")

        return discount_factors

    @cachedproperty
    @save("discount_factors")
    def stream_discount_factors(self):
        return self.calculate_stream_discount_factors(
            discounting_type=self.discounting_type,
            fair_aggregation=self.fair_aggregation,
        )

    @cachedproperty
    @save("uncollapsed_discount_factors")
    def uncollapsed_discount_factors(self):
        pop = self.collapsed_pop.sum("region")
        pop = pop.reindex(
            year=range(pop.year.min().values, self.ext_end_year + 1),
            method="ffill",
        )
        f = self.calculate_discount_factors(
            self.global_consumption_no_pulse / pop
        ).to_dataset(name="discount_factor")
        for var in f.variables:
            f[var].encoding.clear()

        return f

    @cachedproperty
    @save("uncollapsed_marginal_damages")
    def uncollapsed_marginal_damages(self):

        md = (
            (
                (self.global_consumption_no_pulse - self.global_consumption_pulse)
                * self.climate.conversion
            )
            .rename("marginal_damages")
            .to_dataset()
        )

        for var in md.variables:
            md[var].encoding.clear()

        return md

    def ce(self, obj, dims):
        """Rechunk data appropriately and apply the certainty equivalence
        calculation. This is done in a loop to avoid memory crashes.
        Not that data MUST be chunked, otherwise Dask will take a CE over each
        chunk and sum the result.

        *** IMPORTANT NOTE ***
        This wrapper function CANNOT execute with weights as it uses a map_blocks
        function which is unable to determine how to match weight dimensions with
        its chunk. If you must weight, `c_equivalence` function must be used directly
        on the data.
        """
        for dim in dims:
            obj = obj.chunk({dim: len(obj[dim])})
            obj = obj.map_blocks(c_equivalence, kwargs=dict(dims=dim, eta=self.eta))
        return obj
