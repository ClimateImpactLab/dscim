import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from dscim.utils.functions import get_model_weights
from dscim.utils.utils import c_equivalence
import geopandas as gpd
from itertools import product
import os
import seaborn as sns

sns.set_context("poster")


def make_map(
    df,
    colname,
    title,
    name_file,
    location,
    figsize=(30, 15),
    color_scale="YlOrRd",
    color_max=None,
    color_min=None,
    save_path=None,
    maxmin=True,
):

    # create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    if color_max is None:
        max_val = max(abs(df[colname].max()), abs(df[colname].min()))
        color_min, color_max = -max_val, max_val

    # add the colorbar to the figure
    sm = plt.cm.ScalarMappable(
        cmap=color_scale, norm=plt.Normalize(vmin=color_min, vmax=color_max)
    )
    fig.colorbar(sm, orientation="horizontal", fraction=0.03, pad=0.02, aspect=20)
    ax.figure.axes[1].tick_params(labelsize=14)

    # Plotting function -- from geopandas
    ax = df.plot(
        column=colname,
        cmap=color_scale,
        edgecolor="face",
        norm=mpl.colors.Normalize(vmin=color_min, vmax=color_max),
        ax=ax,
    )

    fig.text(0.5, 0.08, title, ha="center", va="center", rotation=0, fontsize=18)

    ax.set_axis_off()
    if maxmin:
        plt.annotate(
            text=f"min: {df[colname].min()}, max: {df[colname].max()}",
            xy=location,
            xycoords="axes fraction",
            fontsize=10,
        )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f"{save_path}/{name_file}", dpi=200, bbox_inches="tight")


# the main function
def maps(
    sector,
    ce_path,
    ECON_ZARR,
    eta,
    maxmin=True,
    selection=None,
    location=(0.35, -0.2),
    gcm="mean",
    prefix=None,
    variables=(
        "rp_inc_share",
        "rp_damage_share",
        "rp_cons_share",
        "rp",
        "cons",
        "cons_ce",
    ),
    maxes=(None, None, None, None, None, None),
    mins=(None, None, None, None, None, None),
    year=2097,
    plot=True,
    save_path=None,
):
    if selection is None:
        selection = dict(year=2097, ssp=["SSP2", "SSP3", "SSP4"])

    assert len(maxes) == len(
        variables
    ), "len(maxes) != len(variables). Make them equal."

    assert len(mins) == len(variables), "len(mins) != len(variables). Make them equal."

    dt = {}

    for ce_type in ["ce_cc", "ce_no_cc"]:
        for recipe in ["adding_up", "risk_aversion"]:

            dt[f"{ce_type}_{recipe}"] = (
                xr.open_zarr(
                    f"{ce_path}/{sector}/{recipe}_{ce_type}_eta{eta}.zarr",
                    consolidated=True,
                )
                .rename({ce_type: "cons"})
                .sel(selection)
            )

    gdppc = xr.open_zarr(ECON_ZARR).gdppc
    gdppc = gdppc.sel({k: v for k, v in selection.items() if k in gdppc.dims})

    damages = xr.Dataset(
        {
            "mean_cons_cc": dt["ce_cc_adding_up"].cons,
            "ce_cons_cc": dt["ce_cc_risk_aversion"].cons,
            "mean_damages": (dt["ce_no_cc_adding_up"] - dt["ce_cc_adding_up"]).cons,
            "ce_damages": (
                dt["ce_no_cc_risk_aversion"] - dt["ce_cc_risk_aversion"]
            ).cons,
            "mean_gdppc_damages": (gdppc - dt["ce_cc_adding_up"].cons),
            "ce_gdppc_damages": (gdppc - dt["ce_cc_risk_aversion"].cons),
        }
    )

    # weights
    weights = xr.concat(
        [get_model_weights("rcp45"), get_model_weights("rcp85")], "rcp"
    ).fillna(0)

    # aggregating GCMs if required
    if gcm is None:
        damages = damages
    elif gcm == "mean":
        damages = damages.weighted(weights).mean(dim="gcm")
    elif gcm == "ce":
        damages = c_equivalence(damages, dims=["gcm"])
    else:
        damages = damages.sel({"gcm": gcm})

    merged = xr.merge([damages, gdppc]).to_dataframe().reset_index()

    shp_file = gpd.read_file(
        "/shares/gcp/climate/_spatial_data/world-combo-new-nytimes/new_shapefile.shp"
    )

    data = shp_file.merge(merged, left_on=["hierid"], right_on=["region"])

    # generate variables of interest
    data["rp"] = data["ce_damages"] - data["mean_damages"]
    data["gdppc_rp"] = data["ce_gdppc_damages"] - data["mean_gdppc_damages"]
    data["rp_inc_share"] = data["rp"] / data["gdppc"] * 100
    data["rp_damage_share"] = data["rp"] / data["mean_damages"] * 100
    data["mean_damages_inc_share"] = data["mean_damages"] / data["gdppc"] * 100

    # if coastal, expand RCP dim to make the code function
    if "rcp" not in data.columns:
        data["rcp"] = ""

    if not plot:
        return data
    else:

        # cycle through all possibilities
        for col, ssp, rcp, iam in product(
            zip(variables, maxes, mins),
            data.ssp.unique(),
            data.rcp.unique(),
            data.model.unique(),
        ):

            print(f"Plotting: {sector}, {col}, {gcm}, {ssp}, {rcp}, {iam}")

            if col[0] == "rp":
                var = "Risk Premium"
                color_scale = "RdBu_r"
            elif col[0] == "gdppc_rp":
                var = "Risk Premium relative to GDPpc"
                color_scale = "Reds"
            elif col[0] == "rp_inc_share":
                var = "Risk Premium as % of GDPpc"
                color_scale = "RdBu_r"
            elif col[0] == "mean_cons_cc":
                var = "Mean of CC Consumption (over batches)"
                color_scale = "Blues"
            elif col[0] == "ce_cons_cc":
                var = "Certainty Equivalent of CC Consumption (over batches)"
                color_scale = "Blues"
            elif col[0] == "mean_damages":
                var = "Mean Damages (over batches)"
                color_scale = "RdBu_r"
            elif col[0] == "ce_damages":
                var = "Mean Damages + Risk Premium (over batches)"
                color_scale = "RdBu_r"
            elif col[0] == "mean_damages_inc_share":
                var = "Mean Damages as % of GDPpc"
                color_scale = "RdBu_r"
            else:
                print(
                    """
                Variable unrecognized. Select rp, gdppc_rp, rp_inc_share, mean_cons_cc, ce_cons_cc, mean_damages, ce_damages, mean_damages_inc_share.
                """
                )

            df = data[
                (
                    (data.rcp == rcp)
                    & (data.ssp == ssp)
                    & (data.model == iam)
                    & (data.hierid != "ATA")
                )
            ]

            # map variable
            make_map(
                df=df,
                colname=col[0],
                location=location,
                maxmin=maxmin,
                title=var,
                name_file=f"{prefix}_{sector}_{col[0]}_{year}_{gcm}_{iam}_{rcp}_{ssp}.png",
                figsize=(20, 15),
                color_max=col[1],
                color_min=col[2],
                color_scale=color_scale,
                save_path=save_path,
            )

            # variable histogram
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            sns.histplot(data=df, x=col[0], bins=500, ax=ax)

            # variable statistics
            plt.annotate(
                str(
                    round(
                        df[col[0]].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]),
                        2,
                    )
                ),
                (0.5, 0.5),
                xycoords="axes fraction",
                fontsize=10,
            )
            # fig.suptitle(
            #     f"Histogram of {var} in 2019 PPP USD \n {prefix} {sector}, IAM : {iam}, GCM : {gcm}, SSP : {ssp}, RCP: {rcp}",
            #     fontsize=10,
            # )

            plt.subplots_adjust(top=0.85)

            plt.savefig(
                f"{save_path}/{prefix}_histogram_{sector}_{col[0]}_{year}_{gcm}_{iam}_{rcp}_{ssp}.png",
                dpi=300,
                bbox_inches="tight",
            )
