import os

username = os.getenv("USER")

import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
import logging

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

sns.set_style("darkgrid")
sns.set_context("talk")


def marginal_damages(
    sector,
    sector_path,
    discounting,
    recipes=("risk_aversion", "equity"),
    save_path=None,
):

    # get relevant files
    marginal_damages = {}
    for recipe in recipes:

        marginal_damages[recipe] = (
            xr.open_dataset(
                f"{sector_path}/{recipe}_{discounting}_fair_marginal_damages_ce.nc4"
            )
            .to_dataframe()
            .reset_index()
        )

        # sort for plotting consistency
        marginal_damages[recipe]["ssp-model"] = (
            marginal_damages[recipe].ssp + "-" + marginal_damages[recipe].model
        )
        marginal_damages[recipe] = marginal_damages[recipe].sort_values(["ssp-model"])

        # to avoid errors when plotting by hue and style
        if discounting == "wr":
            marginal_damages[recipe]["ssp"], marginal_damages[recipe]["model"] = "", ""

    fig, ax = plt.subplots(
        2, len(recipes), figsize=(15, 10), sharex=True, sharey=True, squeeze=False
    )

    plt.subplots_adjust(wspace=0, hspace=0, top=0.90)

    for i, recipe in enumerate(recipes):

        sns.lineplot(
            data=marginal_damages[recipe].loc[marginal_damages[recipe].rcp == "rcp45"],
            x="year",
            y="temperature",
            hue="ssp",
            style="model",
            ax=ax[0][i],
        )

        ax[0][i].set_xlabel("Year")
        ax[0][i].set_title(f"Recipe: {recipe}, rcp 4.5")

        sns.lineplot(
            data=marginal_damages[recipe].loc[marginal_damages[recipe].rcp == "rcp85"],
            x="year",
            y="temperature",
            hue="ssp",
            style="model",
            ax=ax[1][i],
        )

        ax[1][i].set_xlabel("Year")
        ax[1][i].set_title(f"Recipe: {recipe}, rcp 8.5")

    fig.suptitle(
        f"{sector}: {discounting} discounting \n Marginal damages from an additional pulse of C02"
    )

    ax[0][0].set_ylabel("Damages in 2019 USD")
    ax[1][0].set_ylabel("Damages in 2019 USD")

    if save_path is not None:
        plt.savefig(
            f"{save_path}/{sector}_{discounting}_marginal_damages.png",
            dpi=300,
            bbox_inches="tight",
        )


def global_consumption(
    sector,
    sector_path,
    discounting,
    recipes=("risk_aversion", "equity"),
    save_path=None,
    scale=10**12,
):

    gc = {}
    for recipe in recipes:

        gc[recipe] = (
            xr.open_dataset(
                f"{sector_path}/{recipe}_{discounting}_global_consumption.nc4"
            )
            .to_dataframe()
            .reset_index()
        )

        gc[recipe]["global_cons_scaled"] = (
            gc[recipe][f"global_cons_{discounting}"] / scale
        )

        if discounting == "wr":
            gc[recipe]["ssp"], gc[recipe]["model"] = "", ""

    fig, ax = plt.subplots(
        1, len(recipes), figsize=(15, 10), squeeze=False, sharey=True, sharex=True
    )

    for i, recipe in enumerate(recipes):

        sns.lineplot(
            data=gc[recipe],
            x="year",
            y="global_cons_scaled",
            hue="ssp",
            style="model",
            ax=ax[0][i],
        )

        ax[0][i].set_title(f"RECIPE: {recipe}")

    ax[0][0].set_ylabel(f'Global consumption in {"{:.0e}".format(scale)} 2019 USD')

    fig.suptitle(
        f"{sector}: {discounting} discounting \n Global consumption, no climate change"
    )

    if save_path is not None:
        plt.savefig(f"{save_path}/{sector}_{discounting}_global_consumption.pdf")


def output_scc(
    sector,
    sector_path,
    eta,
    rho,
    recipes=("adding_up", "risk_aversion", "equity", "local"),
    discounting=("constant_model_collapsed", "constant", "euler_ramsey", "euler_gwr"),
    save_path=None,
    file=None,
    subset_dict=None,
    index=(
        "discount_type",
        "discrate",
        "weitzman_parameter",
        "model",
        "ssp",
        "rcp",
        "gas",
    ),
):

    final_dfs = []

    for recipe in recipes:

        dfs = []

        for disc in discounting:

            ds = xr.open_dataset(
                f"{sector_path}/{recipe}_{disc}_eta{eta}_rho{rho}_scc.nc4"
            ).sel(subset_dict)

            if "runtype" in list(ds.coords):
                ds = ds.drop("runtype")
            else:
                pass

            if recipe == "local":
                ds = ds.drop("year")
                df = ds.to_dataframe().unstack(
                    ["first_fair_aggregation", "fair_aggregation"]
                )
            else:
                df = ds.to_dataframe().unstack("fair_aggregation")

            df = df.droplevel(0, axis=1)

            if recipe == "local":
                df.columns = [
                    df.columns[i][0] + "-" + df.columns[i][1]
                    for i in range(0, len(df.columns))
                ]

            df = df.rename(columns={col: f"{recipe}_{col}" for col in df.columns})

            if disc not in ["constant", "constant_model_collapsed"]:

                df["discrate"] = np.nan

            df = df.reset_index().set_index(index)

            dfs.append(df)

        final_dfs.append(pd.concat(dfs))

    final_df = reduce(lambda left, right: left.join(right), final_dfs)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if file is not None:
            final_df.to_csv(f"{save_path}/{file}")
        else:
            final_df.to_csv(f"{save_path}/{sector}_sccs.csv")

    return final_df


def plot_implicit_rates(
    sector,
    path,
    fair_aggregation="ce",
    ssp=None,
    model=None,
    pulse_year=2020,
    recipes=("risk_aversion", "equity"),
    discounting=("ramsey", "gwr"),
    save_path=None,
    csv=True,
):
    if ssp is None:
        ssp = ["SSP3"]

    if model is None:
        model = ["IIASA GDP"]

    for recipe in recipes:

        for disc in discounting:

            if "gwr" in disc:
                ssp = ["['SSP3', 'SSP2', 'SSP4']"]
                model = ["['IIASA GDP', 'OECD Env-Growth']"]

            naive = xr.open_dataset(
                f"{path}/{recipe}_naive_{disc}_discount_factors.nc4"
            )
            euler = xr.open_dataset(
                f"{path}/{recipe}_euler_{disc}_discount_factors.nc4"
            )

            naive.coords["weitzman_parameter"] = euler.weitzman_parameter
            naive.coords["rcp"] = "Naive Ramsey"

            if "gwr" in disc:
                naive.coords["model"] = euler.model
                naive.coords["ssp"] = euler.ssp

            subset = pd.concat(
                [euler.to_dataframe().reset_index(), naive.to_dataframe().reset_index()]
            )

            subset["rate"] = (1 / subset.__xarray_dataarray_variable__) ** (
                1 / (subset.year - pulse_year)
            ) - 1

            subset = subset.loc[
                (subset.fair_aggregation == fair_aggregation)
                & (subset.weitzman_parameter != "1")
            ]

            subset = subset.loc[(subset.ssp.isin(ssp)) & (subset.model.isin(model))]

            subset["weitzman_parameter"] = subset.weitzman_parameter.astype(
                str
            ).replace(
                {
                    "0.01": "1% of future GDP",
                    "0.1": "10% of future GDP",
                    "0.5": "50% of future GDP",
                    "858795238461.09": "1% of current GDP",
                    "1000000000000.0": "1 trillion USD",
                    "1.0": "1 USD",
                }
            )

            # palette = ['blue', 'red', 'grey']

            #             if recipe == 'equity':
            #                 subset = subset.loc[subset.rcp != 'Naive Ramsey']
            #                 palette = palette[0:2]

            if len(ssp) <= 1:
                g = sns.relplot(
                    data=subset,
                    x="year",
                    y="rate",
                    kind="line",
                    col="weitzman_parameter",
                    col_wrap=3,
                    height=10,
                    facet_kws={"sharey": True, "sharex": True, "legend_out": False},
                    hue="rcp",
                    style="model",
                )  # palette=sns.color_palette(palette))
            else:
                g = sns.relplot(
                    data=subset,
                    x="year",
                    y="rate",
                    kind="line",
                    col="weitzman_parameter",
                    row="ssp",
                    height=10,
                    facet_kws={"sharey": True, "sharex": True, "legend_out": False},
                    hue="rcp",
                    style="model",
                )  # palette=sns.color_palette(palette))

            g.fig.suptitle(
                f"""Implicit Discount Rates - Naive and Euler Discounting
            Recipe: {recipe}, Sector: {sector}, Discounting = {disc}, SSP = {ssp}"""
            )
            plt.subplots_adjust(top=0.9, wspace=0)

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(
                    f"{save_path}/{recipe}_{disc}_{ssp}_{model}_implicit_discount_rates.png",
                    dpi=300,
                    bbox_inches="tight",
                )

    #                 if csv:
    #                     cols = ['fair_aggregation', 'weitzman_parameter', 'ssp', 'model', 'rcp']
    #                     df = (subset.set_index(cols + ['year']).
    #                            rate.
    #                            sort_index(level=cols).
    #                            unstack(cols)
    #                           )

    #                     df.to_csv(f"{save_path}/{recipe}_{disc}_implicit_discount_rates.csv")

    return subset


def statistics(df):
    """Produces statistics about each SCC file.
    Undocumented :/

    """

    rcp_grouped = df.reset_index().pivot(
        index=["discount_type", "discrate", "ssp", "model", "weitzman_parameter"],
        columns=["rcp"],
    )

    for i, j in list([("rcp45", "rcp85"), ("ssp245", "ssp460"), ("ssp460", "ssp370")]):

        print(f"\n {i} > {j} \n")

        for col in rcp_grouped.columns.get_level_values("fair_aggregation").unique():
            sketch = rcp_grouped.loc[rcp_grouped[col][i] > rcp_grouped[col][j]]
            print(f"\t {col}: {round(len(sketch)/len(rcp_grouped[col])*100,2)} %")

    comparators = [
        ("risk_aversion_ce", "risk_aversion_mean"),
        ("equity_ce", "equity_mean"),
        ("equity_ce", "risk_aversion_ce"),
        ("equity_mean", "risk_aversion_mean"),
        ("risk_aversion_median", "adding_up_median"),
        ("equity_median", "risk_aversion_median"),
    ]

    df["ra_diff"] = (
        (df.risk_aversion_ce - df.risk_aversion_uncollapsed) / df.risk_aversion_ce * 100
    )
    df["eq_diff"] = (df.equity_ce - df.equity_uncollapsed) / df.equity_ce * 100

    describers = {
        "ra_diff": "(risk aversion CE - risk aversion mean(SCCs)) / risk aversion CE * 100",
        "eq_diff": "(equity CE - equity mean(SCCs)) / equity CE * 100",
    }

    group_disc = df.groupby("discount_type")

    for i, j in comparators:

        print(f"\n {i} < {j} \n")
        for name, disc in group_disc:
            share = round(len(disc.loc[disc[i] < disc[j]]) / len(disc) * 100, 2)
            print(f"\t {name}: {share} %")
            if share > 0:
                wp = disc.groupby("weitzman_parameter")
                for wp_name, wp_df in wp:
                    wp_share = round(
                        len(wp_df.loc[wp_df[i] < wp_df[j]]) / len(disc) * 100, 2
                    )
                    print(f"\t \t {wp_name}: {wp_share} %")

    for var, label in describers.items():
        print(f"\n {label} \n")
        for name, disc in group_disc:
            print(f"\t {name}: ")
            for stat in ["mean", "min", "max"]:
                print(f"\t \t {stat}: {round(disc[var].describe()[stat], 2)}")
