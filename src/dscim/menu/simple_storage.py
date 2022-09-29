import logging
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from dscim.descriptors import cachedproperty

logger = logging.getLogger(__name__)


class Climate:
    """
    This class wraps all climate data used in DSCIM.

    Parameters
    ---------
    gmst_path : str
        Path to GMST anomalies for damage function step.
    gmsl_path : str
        Path to GMSL anomalies for damage function step.
    gmst_fair_path : str
        Path to GMST anomalies data for FAIR step.
    gmsl_fair_path : str
        Path to GMSL anomalies data for FAIR step.
    pulse_year : int
        Year of the greenhouse gas pulse.
    damages_pulse_conversion_path: str
        Path to file containing conversion factors for each greenhouse gas
        to turn the pulse units into the appropriate units for an SCC calculation.
    ecs_mask_path : str or None, optional
        Path to a boolean NetCDF4 dataset sharing the same coordinates as self.anomalies,
        indicating which simulations should be included or excluded.
    ecs_mask_name : str or None, optional
        Name of mask to be called from within ``ecs_mask_path`` NetCDF file.
    base_period: tuple, optional
        Period for rebasing FAIR temperature anomalies. This should match the CIL projection system's base period.
    emission_scenarios: list or None, optional
        List of emission scenarios for which SCC will be calculated. Default
        is (), which gets set to ["ssp119", "ssp126", "ssp245", "ssp460", "ssp370", "ssp585"].
        Use `None` when RCP emission scenarios are not the climate projections,
        such as with RFF-SP projections.
    gases: list or None, optional
        List of greenhouse gases for which SCC will be calculated. Default is
        ["CO2_Fossil", "CH4", "N2O"].
    """

    def __init__(
        self,
        gmst_path,
        gmsl_path,
        gmst_fair_path,
        damages_pulse_conversion_path,
        pulse_year,
        gmsl_fair_path=None,
        ecs_mask_path=None,
        ecs_mask_name=None,
        base_period=(2001, 2010),
        emission_scenarios=(),
        gases=None,
    ):
        if emission_scenarios == ():
            emission_scenarios = [
                "ssp119",
                "ssp126",
                "ssp245",
                "ssp460",
                "ssp370",
                "ssp585",
            ]
        if gases is None:
            gases = ["CO2_Fossil", "CH4", "N2O"]

        self.gmst_path = gmst_path
        self.gmsl_path = gmsl_path
        self.gmst_fair_path = gmst_fair_path
        self.damages_pulse_conversion_path = damages_pulse_conversion_path
        self.gmsl_fair_path = gmsl_fair_path
        self.pulse_year = pulse_year
        self.emission_scenarios = emission_scenarios
        self.gases = gases
        self.base_period = base_period
        self.ecs_mask_path = ecs_mask_path
        self.ecs_mask_name = ecs_mask_name
        self.logger = logging.getLogger(__name__)

    @property
    def gmst(self):
        """Cached GMST anomalies"""
        gmst = pd.read_csv(self.gmst_path)

        if "temp" in gmst.columns:
            gmst = gmst.rename(columns={"temp": "anomaly"})

        return gmst

    @property
    def gmsl(self):
        """Cached GMSL anomalies"""
        gmsl = xr.open_zarr(self.gmsl_path).gmsl.to_dataframe().reset_index()

        return gmsl

    @property
    def gmst_anomalies(self):
        """This function takes FAIR GMST relative to 1765.
        It rebases it to self.base_period.
        """
        # open FAIR GMST
        temps = xr.open_dataset(
            self.gmst_fair_path,
            chunks={
                "year": 11,
            },
        )

        # calculate base period average
        base_period = temps.sel(
            year=slice(self.base_period[0], self.base_period[1])
        ).mean(dim="year")

        # subset relevant years to save compute time
        temps = temps.sel(year=slice(self.pulse_year, 2300))

        # calculate anomalies
        anomaly = temps - base_period

        return anomaly

    @property
    def gmsl_anomalies(self):
        """This function takes coastal sector's GMSL relative to 1991-2009.
        No rebasing occurs, as coastal damages are rebased to the same period.
        """
        df = xr.open_zarr(self.gmsl_fair_path)
        df = df.chunk(df.dims)

        datasets = []

        # collapse runtype dimension into two variables, and label
        # each one (control, pulse) for medians and full simulations
        for var in df.keys():
            ds = df[var].to_dataset(dim="runtype")
            ds = ds.rename({k: f"{k}_{var}" for k in ds.keys()})
            datasets.append(ds)

        anomaly = xr.combine_by_coords(datasets, combine_attrs="override")

        # drop unnecessary coordinates
        anomaly = anomaly.drop_vars(
            ["confidence", "kind", "locations", "workflow_src"],
            errors="ignore",
        )

        # rename variables
        if (
            "pulse_gmsl_median" in anomaly.keys()
            and "control_gmsl_median" in anomaly.keys()
        ):
            anomaly = anomaly.rename(
                {
                    "pulse_gmsl_median": "medianparams_pulse_gmsl",
                    "control_gmsl_median": "medianparams_control_gmsl",
                }
            )
        else:
            pass

        return anomaly

    @cachedproperty
    def anomaly_vars(self):
        """Anomaly variables to include"""
        return (
            ["temperature", "gmsl"]
            if self.gmsl_fair_path is not None
            else ["temperature"]
        )

    @cachedproperty
    def anomalies(self):
        """
        This function combines and subsets the projected GMST and GMSL anomalies by
        pulse year and emissions scenario. If applicable, it
        masks the data according to the mask passed to self.ecs_mask_path.
        """
        if self.gmsl_fair_path is not None:
            anomaly = xr.combine_by_coords([self.gmst_anomalies, self.gmsl_anomalies])
        else:
            anomaly = self.gmst_anomalies

        # subset by relevant coordinates
        anomaly = anomaly.sel(
            year=slice(self.pulse_year, 2300),
            gas=self.gases,
        )

        if self.emission_scenarios is not None:
            anomaly = anomaly.sel(rcp=self.emission_scenarios)

        if "pulse_year" in anomaly.dims:
            anomaly = anomaly.sel(pulse_year=self.pulse_year, drop=True)

        # Apply ECS mask
        if (self.ecs_mask_name is not None) and (self.ecs_mask_path is not None):
            self.logger.info(f"Masking anomalies with {self.ecs_mask_name}.")

            # load mask
            mask = xr.open_dataset(self.ecs_mask_path)[self.ecs_mask_name]

            # median variables can't be masked because they don't have a simulation dimension
            vars_no_mask = [v for v in anomaly.keys() if "median" in v]
            vars_to_mask = [v for v in anomaly.keys() if v not in vars_no_mask]

            # mask and put back together
            anomaly = anomaly[vars_no_mask].update(
                anomaly[vars_to_mask].where(mask, drop=True)
            )

        return anomaly

    @property
    def fair_control(self):
        """Anomalies without a pulse"""
        ds = self.anomalies[[f"control_{var}" for var in self.anomaly_vars]]
        return ds.rename({f"control_{var}": var for var in self.anomaly_vars})

    @property
    def fair_pulse(self):
        """Anomalies with a pulse"""
        ds = self.anomalies[[f"pulse_{var}" for var in self.anomaly_vars]]
        return ds.rename({f"pulse_{var}": var for var in self.anomaly_vars})

    @property
    def fair_median_params_control(self):
        """FAIR median parameters anomaly without a pulse"""
        ds = self.anomalies[
            [f"medianparams_control_{var}" for var in self.anomaly_vars]
        ]
        return ds.rename(
            {f"medianparams_control_{var}": var for var in self.anomaly_vars}
        )

    @property
    def fair_median_params_pulse(self):
        """FAIR median parameters anomaly with a pulse"""
        ds = self.anomalies[[f"medianparams_pulse_{var}" for var in self.anomaly_vars]]
        return ds.rename(
            {f"medianparams_pulse_{var}": var for var in self.anomaly_vars}
        )

    @property
    def conversion(self):
        """Conversion factors to turn the pulse units
        into the appropriate units for an SCC calculation"""

        conversion = (
            xr.open_dataset(self.damages_pulse_conversion_path)
            .sel(gas=self.gases)
            .to_array()
            .isel(variable=0, drop=True)
        )
        return conversion


class EconVars:
    """
    This class wraps all socioeconomic data used in DSCIM.

    Parameters
    ----------
    path_econ : str
        Path to economic data in NetCDF format.

    Notes
    ------
    Note that the input data must have population and GDP data at the IR level
    with the desired dimensions: SSP, IAM (or ``model``), IR, and year.
    """

    def __init__(self, path_econ):
        self.path = path_econ
        self.logger = logging.getLogger(__name__)

    @property
    def econ_vars(self):
        """Economic variables"""
        if self.path[-3:] == "arr":
            raw = xr.open_zarr(self.path, consolidated=True)
        else:
            raw = xr.open_dataset(self.path)
        return raw[["gdp", "pop"]]


class StackedDamages:
    """
    This class wraps all damages data used in DSCIM.

    Parameters
    ----------
    sector_path : str
        Path to input damages.
    delta : str
        Climate change damages variable.
    histclim : str
        No climate change damages variable.
    econ_vars : dscim.simple_storage.EconVars
    climate_vars : dscim.simple_storage.Climate
    subset_dict : dict
        A dictionary with coordinate values to filter data.
    eta : int
        Curvature parameter of the CRRA utility function.
    gdppc_bottom_code : int or float
        Minimum values allowed for per-capita GDP in ``self.gdppc``.
    ce_path : str, optional
        Path to directory containing certainty equivalent reduced damages and
        risk aversion data. This directory can contain `adding_up_cc.zarr` and
        `adding_up_no_cc.zarr` which have reduced damages due to climate in
        dollars (by impact region, year, etc.) for the `adding_up` recipe with
        climate change (cc) and without climate change (no cc).
        This directory should also contain `risk_aversion_{ce_type}_eta{eta}.zarr`
        as used for risk aversion calculations.
    """

    NAME = ""

    def __init__(
        self,
        sector_path,
        save_path,
        econ_vars,
        climate_vars,
        eta,
        gdppc_bottom_code,
        delta=None,
        histclim=None,
        ce_path=None,
        subset_dict=None,
    ):

        self.sector_path = sector_path
        self.save_path = save_path
        self.gdppc_bottom_code = gdppc_bottom_code
        self.subset_dict = subset_dict
        self.econ_vars = econ_vars
        self.climate = climate_vars
        self.delta = delta
        self.histclim = histclim
        self.ce_path = ce_path
        self.eta = eta

        self.logger = logging.getLogger(__name__)

    def cut(self, xr_array, end_year=2099):
        """Subset array to self.subset_dict.

        Parameters
        ----------
        xr_array :  xr.Dataset or xr.Dataarray
            An xarray object

        end_year : int
            Which year should be last in the dataset (all further data is dropped)

        Returns
        -------
        xr.Dataset or xr.Dataarray
            ``xarray`` object filtered using the dict defined in the class:
            ``self.subset_dict``
        """

        valid_keys = {
            key: self.subset_dict[key]
            for key in self.subset_dict
            if key in xr_array.coords
        }

        self.logger.debug(f"Subsetting on {valid_keys} keys.")

        xr_data = xr_array.sel(valid_keys).sel(
            year=slice(self.climate.pulse_year - 2, end_year)
        )

        return xr_data

    @property
    def cut_econ_vars(self):
        """Economic variables from SSP object"""
        if 2300 in self.econ_vars.econ_vars.year:
            # because RFF data runs to 2300, these menu runs don't need to sliced and extrapolated
            raw = self.cut(self.econ_vars.econ_vars, end_year=2300)
        else:
            # 2100 should be dropped from SSP data since CIL damages only extend to 2099
            raw = self.cut(self.econ_vars.econ_vars, end_year=2099)

        if raw is None:
            raise ValueError(
                "Economic data is not loaded. Check your config or input settings."
            )
        return raw

    @cachedproperty
    def gdp(self):
        return self.cut_econ_vars.gdp

    @cachedproperty
    def pop(self):
        return self.cut_econ_vars.pop

    @cachedproperty
    def gdppc(self):
        return np.maximum(self.gdp / self.pop, self.gdppc_bottom_code)

    @property
    def adding_up_damages(self):
        """This property calls pre-calculated adding-up IR-level 'mean' over batches."""

        mean_cc = f"{self.ce_path}/adding_up_cc.zarr"
        mean_no_cc = f"{self.ce_path}/adding_up_no_cc.zarr"

        if os.path.exists(mean_cc) and os.path.exists(mean_no_cc):
            self.logger.info(
                f"Adding up aggregated damages found at {mean_cc}, {mean_no_cc}. These are being loaded..."
            )
            damages = (
                (xr.open_zarr(mean_no_cc).no_cc - xr.open_zarr(mean_cc).cc) * self.pop
            ).sum("region")
        else:
            raise NotImplementedError(
                f"Adding up reduced damages not found: {mean_no_cc}, {mean_cc}. Please reduce damages for for `adding_up`."
            )
        return self.cut(damages)

    def risk_aversion_damages(self, ce_type):
        """This function calls pre-calculated risk-aversion IR-level 'CE' over batches.

        Parameters
        ----------
        ce_type : either `no_cc` or `cc`

        Returns
        -------
        xr.DataArray
        """
        file = f"{self.ce_path}/risk_aversion_{ce_type}_eta{self.eta}.zarr"

        if os.path.exists(file):
            self.logger.info(
                f"Risk-aversion CEs found at {file}. These are being loaded..."
            )
        else:
            raise NotImplementedError(
                "Risk-aversion CEs not found. Please run CE_calculation.ipynb for `risk_aversion`."
            )
        return self.cut(xr.open_zarr(file))
