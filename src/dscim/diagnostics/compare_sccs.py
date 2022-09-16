import xarray as xr
import pandas as pd
import numpy as np


def compare_sccs(
    rootdict, recipe, disc, eta, rho, quantiles=(0, 0.05, 0.5, 0.95, 1), wp="0.5"
):

    this_list = []
    for name, path in rootdict.items():

        this = (
            xr.open_dataset(
                f"{path}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_sccs.nc4"
            )
            .uncollapsed_sccs.sel(
                discount_type=disc, fair_aggregation="uncollapsed", drop=True
            )
            .expand_dims({"scenario": [name]})
        )

        try:
            # for ssp calculations
            this = this.sel(
                gas="CO2_Fossil", model="IIASA GDP", rcp="ssp370", drop=True
            ).rename({"simulation": "runid"})
        except KeyError:
            # for rff calculations
            if ("simulation" in this.dims) and (len(this.simulation) == 1):
                this = this.sel(simulation=1, drop=True)
            else:
                pass

        if quantiles == "mean":
            this = this.mean("runid")
        else:
            this = this.quantile(quantiles, "runid")

        if "discrate" in this.dims:
            this = this.sel(discrate=0.02, drop=True)
        else:
            pass

        this = this.sel(weitzman_parameter=wp, drop=True)

        this_list.append(this)

    return xr.concat(this_list, "scenario")
