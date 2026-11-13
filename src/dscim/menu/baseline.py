import xarray as xr
from dscim.menu.main_recipe import MainRecipe
import os


class Baseline(MainRecipe):
    """Adding up option"""

    NAME = "adding_up"
    __doc__ = MainRecipe.__doc__

    def ce_cc(self):
        pass

    def ce_no_cc(self):
        pass

    def global_damages_calculation(self):
        """Call global damages"""
        return self.adding_up_damages.to_dataframe("damages").reset_index()

    @property
    def calculated_damages(self):
        """IR-level adding-up damages (no_cc minus cc means), for the regional path.

        Neither population-weighted nor summed over region: ``damages_dataset``
        applies ``collapsed_pop`` and the geography aggregation itself. The
        legacy path keeps using ``adding_up_damages`` (which does both).
        """
        mean_cc = f"{self.ce_path}/adding_up_cc_eta{self.eta}.zarr"
        mean_no_cc = f"{self.ce_path}/adding_up_no_cc_eta{self.eta}.zarr"

        if os.path.exists(mean_cc) and os.path.exists(mean_no_cc):
            self.logger.info(
                f"Adding up aggregated damages found at {mean_cc}, {mean_no_cc}. These are being loaded..."
            )
            damages = xr.open_zarr(mean_no_cc).no_cc - xr.open_zarr(mean_cc).cc
        else:
            raise NotImplementedError(
                f"Adding up reduced damages not found: {mean_no_cc}, {mean_cc}. Please reduce damages for for `adding_up`."
            )
        if self.individual_region:
            damages = damages.sel(region=self.individual_region, drop=True)
        return self.cut(damages)

    def ce_cc_calculation(self):
        pass

    def ce_test(self):
        pass

    def ce_no_cc_calculation(self):
        pass

    def global_consumption_calculation(self, disc_type):
        """Calculate global consumption"""

        if self.geography == "ir":
            gdp = self.gdp
        elif self.geography == "country":
            # group gdp by some grouping and collapse
            gdp = self.gdp.sum(dim=["region"])
        else:
            gdp = self.gdp.sum(dim=["region"])

        if (disc_type == "constant") or ("ramsey" in disc_type):
            global_cons_no_cc = gdp

        elif disc_type == "constant_model_collapsed":
            global_cons_no_cc = gdp.mean(dim=["model"])

        elif "gwr" in disc_type:
            global_cons_no_cc = gdp.mean(dim=["model", "ssp"])

        global_cons_no_cc.name = f"global_cons_{disc_type}"

        return global_cons_no_cc


#         def global_consumption_calculation(self, disc_type):
#         """Calculate global consumption

#         Returns
#         -------
#             xr.DataArray
#         """

#         if self.geography == 'ir':
#             gdp = self.gdp
#         elif self.geography == 'country':
#             #group gdp by some grouping and collapse
#             gdp = self.gdp.sum(dim=["region"])
#         else:
#             gdp = self.gdp.sum(dim=["region"])

#         if (disc_type == "constant") or ("ramsey" in disc_type):
#             global_cons_no_cc = gdp

#         elif disc_type == "constant_model_collapsed":
#             global_cons_no_cc = gdp.mean(dim=["model"])

#         elif "gwr" in disc_type:
#             ce_cons = self.ce(self.gdppc, dims=["ssp", "model"])
#             global_cons_no_cc = (ce_cons * self.collapsed_pop).sum(dim=["region"])

#         # Convert to array in case xarray becames temperamental. This is a hack
#         # that need to be changed
#         if isinstance(global_cons_no_cc, xr.Dataset):
#             global_cons_no_cc = global_cons_no_cc.to_array()

#         global_cons_no_cc.name = f"global_cons_{disc_type}"

#         return global_cons_no_cc
