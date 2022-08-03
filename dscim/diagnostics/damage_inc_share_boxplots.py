import xarray as xr
import pandas as pd
import os
import sys
import yaml

USER = os.getenv("USER")
from dscim.utils.functions import get_model_weights, US_territories
from impactlab_tools.utils.weighting import weighted_quantile_xr
import numpy as np
import matplotlib.pyplot as plt


def income_boxplot(
    damages,
    delta_var,
    socioec,
    quantile,
    sector,
    rcp,
    ssp,
    model,
    year,
    USA,
    output,
):

    if quantile == "quintile":
        bin_breaks = np.linspace(0, 1, 6)
    elif quantile == "decile":
        bin_breaks = np.linspace(0, 1, 11)

    # open files and select relevant data
    socioec = xr.open_zarr(socioec).sel(ssp=ssp, model=model)
    mean_damages = xr.open_zarr(damages).sel(ssp=ssp, model=model, year=year, rcp=rcp)[
        delta_var
    ]

    if USA:
        # subset US data
        US_IRs = [
            i for i in socioec.region.values if any(j in i for j in US_territories())
        ]
        socioec = socioec.sel(region=US_IRs)
        mean_damages = mean_damages.sel(region=US_IRs)

    if "CAMEL" not in sector:
        # reduce damages for non-CAMEL sectors (CAMEL is generated differently and is thus pre-collapsed)
        weights = get_model_weights(rcp)
        mean_damages = mean_damages.mean("batch").weighted(weights).mean("gcm")

    # get damages as a share of income
    damages_inc_share = (
        mean_damages
        / socioec.gdppc.sel(year=2020).rename("damages_inc_share").to_dataset()
    )
    damages_inc_share["present_gdppc"] = socioec.gdppc.sel(year=2020)
    damages_inc_share["present_pop"] = socioec["pop"].sel(year=2020)
    damages_inc_share = damages_inc_share.dropna("region")

    # pop-weighted gdppc quantiles (determine the bin cutoffs)
    bins = weighted_quantile_xr(
        damages_inc_share.present_gdppc,
        bin_breaks,
        damages_inc_share.present_pop,
        "region",
    ).values

    groupby = damages_inc_share.groupby_bins("present_gdppc", bins)

    plot_bins = []
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    # get quantiles of damages_inc_share for each gdppc bin
    for i, j in groupby:

        bin_no = [round(i, 1) for i in bins].index(round(i.left, 1))

        values = list(
            weighted_quantile_xr(
                j.damages_inc_share, quantiles, j.present_pop, "region"
            ).values
        )

        values.append(j.damages_inc_share.weighted(j.present_pop).mean())

        plot_bins.append((bin_no + 1, values))

    # plot them
    fig, ax = plt.subplots(figsize=(15, 10), sharey=True)

    stats = []

    for bin, s in sorted(plot_bins):

        stats.append(
            {
                "label": bin,
                "mean": s[5],
                "med": s[2],
                "q1": s[1],
                "q3": s[3],
                "whislo": s[0],
                "whishi": s[4],
                "fliers": [],
            }
        )

    ax.bxp(stats, showmeans=True, meanline=False)
    ax.set_ylabel(f"{year} damages as a share of present-day GDP")
    ax.set_xlabel("Decile of present-day GDP per capita")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output, exist_ok=True)
    plt.savefig(
        f"{output}/boxplot_{sector}_domestic-{USA}_{rcp}_{ssp}_{model}_{quantile}_{year}.png",
        bbox_inches="tight",
        dpi=300,
    )
