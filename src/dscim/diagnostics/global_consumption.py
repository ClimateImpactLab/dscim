import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from dscim.utils.utils import c_equivalence
from settings import rcp_dict


def plot_global_cons(sector, disc, recipe, root, output, wp, eta, rho):

    cc_cons = xr.open_dataset(
        f"{root}/{recipe}_{disc}_eta{eta}_rho{rho}_global_consumption_no_pulse.nc4"
    ).sel(model="IIASA GDP", ssp="SSP3", weitzman_parameter=wp)

    cc_cons = (
        c_equivalence(cc_cons, "simulation", eta=eta)
        .to_array()
        .rename("cons")
        .to_dataframe()
        .reset_index()
    )

    no_cc_cons = (
        xr.open_dataset(
            f"{root}/{recipe}_{disc}_eta{eta}_rho{rho}_global_consumption.nc4"
        )
        .sel(model="IIASA GDP", ssp="SSP3")
        .to_array()
        .rename("cons")
        .to_dataframe()
        .reset_index()
    )
    no_cc_cons["rcp"] = "No climate change"

    plot = pd.concat([cc_cons, no_cc_cons]).reset_index()
    plot["RCP"] = plot.rcp.replace(rcp_dict)
    plot["Global consumption (trillion USD)"] = plot.cons / 1e12

    fig, ax = plt.subplots(figsize=(15, 8))

    sns.lineplot(
        data=plot,
        x="year",
        y="Global consumption (trillion USD)",
        hue="RCP",
        ax=ax,
    )

    ax.set_xlabel("Year")

    plt.legend(
        bbox_to_anchor=(0.5, -0.4),
        loc="lower center",
        borderaxespad=0,
        frameon=False,
        ncol=3,
    )

    os.makedirs(output, exist_ok=True)
    plt.savefig(
        f"{output}/global_consumption_wp{wp}_{sector}_{recipe}_{disc}.png",
        dpi=300,
        bbox_inches="tight",
    )
