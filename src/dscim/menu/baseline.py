import pandas as pd
import numpy as np
import xarray as xr
from dscim.menu.main_recipe import MainRecipe


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

    def calculated_damages(self):
        pass

    def ce_cc_calculation(self):
        pass

    def ce_test(self):
        pass

    def ce_no_cc_calculation(self):
        pass

    def global_consumption_calculation(self, disc_type):
        """Calculate global consumption"""

        if (disc_type == "constant") or ("ramsey" in disc_type):
            global_cons_no_cc = self.gdp.sum(dim=["region"])

        elif disc_type == "constant_model_collapsed":
            global_cons_no_cc = self.gdp.sum(dim=["region"]).mean(dim=["model"])

        elif "gwr" in disc_type:
            global_cons_no_cc = self.gdp.sum(dim=["region"]).mean(dim=["model", "ssp"])

        global_cons_no_cc.name = f"global_cons_{disc_type}"

        return global_cons_no_cc
