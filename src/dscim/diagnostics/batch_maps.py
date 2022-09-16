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
def batch_maps(
    sector,
    DAMAGE_ZARR,
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
        selection = dict(year=2097, ssp=["SSP2", "SSP3", "SSP4"], batch="batch0")

    assert len(maxes) == len(
        variables
    ), "len(maxes) != len(variables). Make them equal."

    assert len(mins) == len(variables), "len(mins) != len(variables). Make them equal."

    damages = xr.open_zarr(
        DAMAGE_ZARR,
        consolidated=True,
    ).sel(selection)

    damages["damages"] = damages.delta_reallocation - damages.histclim_reallocation

    gdppc = xr.open_zarr(ECON_ZARR).gdppc.sel(
        {k: v for k, v in selection.items() if k not in ["rcp", "batch"]}
    )

    # weights
    weights = xr.concat(
        [get_model_weights("rcp45"), get_model_weights("rcp85")], "rcp"
    ).fillna(0)

    # aggregating GCMs if required
    if gcm == "mean":
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
    data["damages_inc_share"] = data["damages"] / data["gdppc"] * 100

    if not plot:
        return data
    else:

        # cycle through all possibilities
        for col, ssp, rcp, iam, batch in product(
            zip(variables, maxes, mins),
            data.ssp.unique(),
            data.rcp.unique(),
            data.model.unique(),
            data.batch.unique(),
        ):

            print(f"Plotting: {sector}, {col}, {gcm}, {ssp}, {rcp}, {iam}, {batch}")

            if col[0] == "damages_inc_share":
                var = "Mean Damages as % of GDPpc"
                color_scale = "RdBu_r"
            elif col[0] == "delta_reallocation":
                var = "Delta"
                color_scale = "RdBu_r"
            elif col[0] == "histclim_reallocation":
                var = "Histclim"
                color_scale = "RdBu_r"
            else:
                print(
                    """
                Variable unrecognized.
                """
                )

            df = data[
                (
                    (data.rcp == rcp)
                    & (data.ssp == ssp)
                    & (data.model == iam)
                    & (data.batch == batch)
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
                name_file=f"{prefix}_{sector}_{col[0]}_{year}_{gcm}_{iam}_{rcp}_{ssp}_{batch}.png",
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
            fig.suptitle(
                f"Histogram of {var} in 2019 PPP USD \n {prefix} {sector}, IAM : {iam}, GCM : {gcm}, SSP : {ssp}, RCP: {rcp}",
                fontsize=10,
            )

            plt.subplots_adjust(top=0.85)

            plt.savefig(
                f"{save_path}/{prefix}_histogram_{sector}_{col[0]}_{year}_{gcm}_{iam}_{rcp}_{ssp}_{batch}.png",
                dpi=300,
                bbox_inches="tight",
            )
