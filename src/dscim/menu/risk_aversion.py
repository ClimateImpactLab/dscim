import pandas as pd
import xarray as xr
from dscim.menu.main_recipe import MainRecipe
from dscim.descriptors import cachedproperty


class RiskAversionRecipe(MainRecipe):
    """Risk aversion option"""

    NAME = "risk_aversion"
    __doc__ = MainRecipe.__doc__

    def ce_cc_calculation(self) -> xr.DataArray:
        """Calculate certainty equivalent over consumption with climate change

        Returns
        -------
             xr.DataArray
        """
        ce_array = self.risk_aversion_damages("cc").cc

        # for GWR options, take the CE over growth models
        if "gwr" in self.discounting_type:
            ce_array = self.ce(ce_array, dims=["ssp", "model"])

        return ce_array

    def ce_no_cc_calculation(self) -> xr.DataArray:
        """Calculate certainty equivalent consumption without climate change

        Returns
        -------
            xr.DataArray
        """
        ce_array = self.risk_aversion_damages("no_cc").no_cc

        if "gwr" in self.discounting_type:
            ce_array = self.ce(ce_array, dims=["ssp", "model"])

        return ce_array

    @cachedproperty
    def calculated_damages(self) -> xr.DataArray:
        """Calculate damages (difference between CEs)"""
        return self.ce_no_cc - self.ce_cc

    def global_damages_calculation(self) -> pd.DataFrame:
        """Aggregate damages to global level

        Returns
        --------
            pd.DataFrame
        """

        dams_collapse = (self.calculated_damages * self.collapsed_pop).sum(dim="region")
        df = dams_collapse.to_dataframe("damages").reset_index()

        if "gwr" in self.discounting_type:
            df = df.assign(
                ssp=str(list(self.gdp.ssp.values)),
                model=str(list(self.gdp.model.values)),
            )

        return df

    def country_damages_calculation(self, country) -> pd.DataFrame:
        """Aggregate damages to country level

        Returns
        --------
            pd.DataFrame
        """

        self.logger.info(f"Calculating damages for {country}")
        dams_collapse = self.calculated_damages * self.collapsed_pop

        territories = []
        for ii, row in self.countries_mapping.iterrows():
            if row['MatchedISO'] == country:
                territories.append(row['ISO'])
        self.logger.debug(f"Including territories {territories}")

        dams_collapse = dams_collapse.sel(
            region=[i for i in dams_collapse.region.values if any(j in i for j in territories)]
        )

        dams_collapse = dams_collapse.sum(dim="region").to_dataframe("damages").reset_index()

        if "gwr" in self.discounting_type:
            dams_collapse = dams_collapse.assign(
                ssp=str(list(self.gdp.ssp.values)),
                model=str(list(self.gdp.model.values)),
            )

        return dams_collapse


    def global_consumption_calculation(self, disc_type):
        """Calculate global consumption

        Returns
        -------
            xr.DataArray
        """

        if (disc_type == "constant") or ("ramsey" in disc_type):
            if self.geography == 'country':
                global_cons_no_cc = self.country_econ_vars.gdp
            else:
                global_cons_no_cc = self.gdp.sum(dim=["region"])

        elif disc_type == "constant_model_collapsed":
            global_cons_no_cc = self.gdp.sum(dim=["region"]).mean(dim=["model"])

        elif "gwr" in disc_type:
            ce_cons = self.ce(self.gdppc, dims=["ssp", "model"])
            global_cons_no_cc = (ce_cons * self.collapsed_pop).sum(dim=["region"])

        # Convert to array in case xarray becames temperamental. This is a hack
        # that need to be changed
        if isinstance(global_cons_no_cc, xr.Dataset):
            global_cons_no_cc = global_cons_no_cc.to_array()

        global_cons_no_cc.name = f"global_cons_{disc_type}"

        return global_cons_no_cc
