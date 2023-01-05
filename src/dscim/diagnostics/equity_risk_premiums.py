import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from itertools import product
from dscim.utils.utils import c_equivalence

sns.set_context("poster")


def mumbai_plots(
    ce_path,
    sector,
    output,
    premium,
    ECON_ZARR,
    GMST_CSV,
    eta,
    title=True,
    aspect=1,
    selection=None,
    region_dict=None,
):
    if selection is None:
        selection = dict(year=2097)

    if region_dict is None:
        region_dict = {
            "IND.21.317.1249": "Mumbai, IND",
            "CAN.2.33.913": "Vancouver, CAN",
            "USA.14.608": "Chicago, USA",
            "EGY.11": "Cairo, EGY",
            "SDN.4.11.50.164": "Khartoum, SDN",
            "NGA.25.510": "Lagos, NGA",
            "SAU.7": "Riyadh, SAU",
            "RUS.16.430.430": "St Petersburg, RUS",
        }

    assert premium in [
        "risk_aversion",
        "equity",
    ], "Please pass a premium : ['risk_aversion', 'equity']."
    selection.update(region=[k for k, v in region_dict.items()])

    if premium == "equity":
        # select all regions, not a subset
        del selection["region"]
        # get pop for a weighted CE
        pop = xr.open_zarr(ECON_ZARR)["pop"].sel(selection)

    dt = {}

    for ce_type in ["ce_cc", "ce_no_cc"]:
        for recipe in ["adding_up", "risk_aversion"]:

            dt[f"{ce_type}_{recipe}"] = (
                xr.open_zarr(
                    f"{ce_path}/{sector}/{recipe}_{ce_type}_eta{eta}.zarr",
                    consolidated=True,
                )
                .rename({ce_type: f"{ce_type}_{recipe}"})
                .sel(selection)
            )

    gdppc = xr.open_zarr(ECON_ZARR).gdppc.sel(selection)
    GMST = (
        pd.read_csv(GMST_CSV)
        .set_index(["gcm", "rcp", "year"])
        .to_xarray()
        .sel(year=selection["year"])
    )

    merged = xr.combine_by_coords([v for k, v in dt.items()]).merge(gdppc).merge(GMST)

    if premium == "equity":
        merged["gdppc"] = merged.gdppc.weighted(pop).mean("region")
        for ce_type in ["ce_no_cc", "ce_cc"]:

            merged[f"{ce_type}_adding_up"] = (
                merged[f"{ce_type}_adding_up"].weighted(pop).mean("region")
            )
            merged[f"{ce_type}_risk_aversion"] = c_equivalence(
                merged[f"{ce_type}_risk_aversion"],
                dims=["region"],
                weights=pop,
                eta=eta,
            )
    merged = merged.rename(
        {
            "ce_cc_adding_up": "Mean consumption across climate damage draws",
            "ce_cc_risk_aversion": "Certainty equivalent consumption across climate damage draws",
            "ce_no_cc_adding_up": "Consumption without climate change",
            "ce_no_cc_risk_aversion": "Consumption without climate change, with risk premium",
        }
    )

    merged["gcm_ce_no_cc"] = c_equivalence(
        merged["Consumption without climate change, with risk premium"],
        dims=["gcm"],
        eta=eta,
    )

    merged = merged.drop(
        [
            "Consumption without climate change, with risk premium",
            "Consumption without climate change",
        ]
    )

    if premium == "equity":

        merged = merged.drop("region").expand_dims({"region": ["world"]})

    else:
        for k, v in region_dict.items():
            merged.coords["region"] = merged.region.str.replace(k, v)

    df = merged.to_dataframe().reset_index()
    df = df.melt(
        [
            "gcm",
            "rcp",
            "region",
            "model",
            "ssp",
            "year",
            "temp",
            "gdppc",
            "gcm_ce_no_cc",
        ]
    )

    wrap = 4 if premium == "risk_aversion" else 1

    for ssp, model, year in product(
        df.ssp.unique(), df.model.unique(), df.year.unique()
    ):
        print(f"Plotting {ssp} {model} {year}...")

        subset = df.loc[(df.ssp == ssp) & (df.model == model) & (df.year == year)]
        g = sns.FacetGrid(
            subset,
            col="region",
            col_wrap=wrap,
            height=10,
            aspect=aspect,
            legend_out=True,
        )
        g.map_dataframe(
            sns.scatterplot,
            x="temp",
            y="value",
            hue="variable",
            palette="husl",
            edgecolor="face",
        )
        g.map_dataframe(
            sns.lineplot, x="temp", y="gdppc", color="green", label="GDP per capita"
        )
        if premium == "equity":
            g.map_dataframe(
                sns.lineplot,
                x="temp",
                y="gcm_ce_no_cc",
                color="mediumpurple",
                label="Consumption without climate change, with risk premium",
            )

        # g.add_legend()

        for ax in g.axes:
            ax.set_ylabel("2019 PPP-adjusted USD")
            ax.set_xlabel("GMST")
            if (premium == "equity") or (not title):
                ax.set_title("")

        title_items = "_".join([ssp, model, str(year)]).replace(" ", "_")

        plt.legend(
            bbox_to_anchor=(0.5, -0.3),
            loc="lower center",
            borderaxespad=0,
            frameon=False,
        )
        # g.fig.suptitle(f"""{premium} premiums \n{sector} {title_items}""", fontsize=15)
        g.fig.subplots_adjust(top=0.9)

        os.makedirs(output, exist_ok=True)
        plt.savefig(
            f"{output}/{premium}premium_{sector}_{title_items}_{'-'.join(region_dict.keys())}.png",
            dpi=300,
            bbox_inches="tight",
        )

    print("Plotting complete.")
