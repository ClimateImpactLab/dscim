import pandas as pd
import xarray as xr
import numpy as np
from dscim.utils.utils import c_equivalence
from dscim.menu.main_recipe import MainRecipe


class EquityRecipe(MainRecipe):
    """Equity option"""

    NAME = "equity"
    __doc__ = MainRecipe.__doc__

    def ce_cc_calculation(self) -> xr.DataArray:
        if "gwr" in self.discounting_type:
            dims = ["ssp", "model", "region"]
        else:
            dims = ["region"]

        ce_cc = self.risk_aversion_damages("cc").cc

        # reindex to make sure all regions are being calculated
        if len(ce_cc.region.values) < len(self.gdppc.region.values):
            ce_cc = ce_cc.reindex({"region": self.gdppc.region.values})
            # assign zero damages to places with missing damages
            ce_cc = xr.where(np.isnan(ce_cc), self.gdppc, ce_cc)

        ce_array = c_equivalence(
            ce_cc,
            dims=dims,
            weights=self.pop,
            eta=self.eta,
        )

        return ce_array.rename("cc")

    def ce_no_cc_calculation(self) -> xr.DataArray:
        if "gwr" in self.discounting_type:
            dims = ["ssp", "model", "region"]
        else:
            dims = ["region"]

        ce_no_cc = self.risk_aversion_damages("no_cc").no_cc
        # reindex to make sure all regions are being calculated
        if len(ce_no_cc.region.values) < len(self.gdppc.region.values):
            ce_no_cc = ce_no_cc.reindex({"region": self.gdppc.region.values})
            # assign zero damages to places with missing damages
            ce_no_cc = xr.where(np.isnan(ce_no_cc), self.gdppc, ce_no_cc)

        ce_no_cc_array = c_equivalence(
            ce_no_cc,
            dims=dims,
            weights=self.pop,
            eta=self.eta,
        )

        return ce_no_cc_array.rename("no_cc")

    @property
    def calculated_damages(self) -> xr.DataArray:
        return self.ce_no_cc - self.ce_cc

    def global_damages_calculation(self) -> pd.DataFrame:
        dams_collapse = self.calculated_damages * self.collapsed_pop.sum(dim="region")
        df = dams_collapse.to_dataframe("damages").reset_index()

        if "gwr" in self.discounting_type:

            df = df.assign(
                ssp=str(list(self.gdp.ssp.values)),
                model=str(list(self.gdp.model.values)),
            )

        return df

    def global_consumption_calculation(self, disc_type):

        # get global consumption certainty equivalent across regions
        ce_cons = c_equivalence(
            self.gdppc,
            dims="region",
            weights=self.pop,
            eta=self.eta,
        )

        if disc_type == "constant_model_collapsed":
            global_cons_no_cc = (ce_cons * self.pop.sum("region")).mean("model")
        elif (disc_type == "constant") | ("ramsey" in disc_type):
            global_cons_no_cc = ce_cons * self.pop.sum("region")
        elif "gwr" in disc_type:
            global_cons_no_cc = self.ce(
                ce_cons, dims=["ssp", "model"]
            ) * self.collapsed_pop.sum("region")

        # Convert to array in case xarray became temperamental
        # @TODO: remove this line
        # if isinstance(global_cons_no_cc, xr.Dataset):
        #     global_cons_no_cc = global_cons_no_cc.to_array()

        global_cons_no_cc.name = f"global_cons_{disc_type}"

        return global_cons_no_cc

    def risk_aversion_growth_rates(self):
        """Calculate risk aversion global consumption growth rates

        This function calculates the risk aversion version of global
        consumption per capita growth rates, in order to cap growth
        the equity recipe to growth in the risk aversion recipe.

        Returns
        -------
            xr.DataArray
        """
        if (self.discounting_type == "constant") or ("ramsey" in self.discounting_type):
            global_cons_no_cc = self.gdp.sum(dim=["region"])

        elif self.discounting_type == "constant_model_collapsed":
            global_cons_no_cc = self.gdp.sum(dim=["region"]).mean(dim=["model"])

        elif "gwr" in self.discounting_type:
            ce_cons = self.ce(self.gdppc, dims=["ssp", "model"])
            global_cons_no_cc = (ce_cons * self.collapsed_pop).sum(dim=["region"])

        # Calculate global consumption per capita
        globalc_pc = global_cons_no_cc / self.collapsed_pop.sum("region")

        # calculate growth rates
        growth_rates = globalc_pc.diff("year") / globalc_pc.shift(year=1)

        return growth_rates
