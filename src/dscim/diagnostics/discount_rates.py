import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys


def plot_implicit_rates(
    sector,
    path,
    eta,
    rho,
    fair_aggregation="ce",
    ssp=("SSP3",),
    model=("IIASA GDP",),
    rcp=("ssp585", "ssp245", "ssp460", "ssp370", "Naive Ramsey"),
    weitzman_parameter=("0.001", "0.1", "1.0"),
    pulse_year=2020,
    recipes=("risk_aversion", "equity"),
    discounting=("ramsey", "gwr"),
    aspect=0.4,
    save_path=None,
    csv=True,
):

    for recipe in recipes:

        for disc in discounting:

            if "gwr" in disc:
                ssp = ["['SSP2', 'SSP3', 'SSP4']"]
                model = ["['IIASA GDP', 'OECD Env-Growth']"]

            naive = xr.open_dataset(
                f"{path}/{recipe}_naive_{disc}_eta{eta}_rho{rho}_discount_factors.nc4"
            )
            euler = xr.open_dataset(
                f"{path}/{recipe}_euler_{disc}_eta{eta}_rho{rho}_discount_factors.nc4"
            )

            naive.coords["weitzman_parameter"] = euler.weitzman_parameter
            naive.coords["rcp"] = "Naive Ramsey"

            if "gwr" in disc:
                naive.coords["model"] = euler.model
                naive.coords["ssp"] = euler.ssp

            subset = pd.concat(
                [euler.to_dataframe().reset_index(), naive.to_dataframe().reset_index()]
            )

            subset["rate"] = (1 / subset.discount_factors) ** (
                1 / (subset.year - pulse_year)
            ) - 1

            subset = subset.loc[
                (subset.fair_aggregation == fair_aggregation)
                & (subset.weitzman_parameter.isin(weitzman_parameter))
                & (subset.ssp.isin(ssp))
                & (subset.model.isin(model))
                & (subset.rcp.isin(rcp))
            ].sort_values("weitzman_parameter")

            subset["weitzman_parameter"] = subset.weitzman_parameter.astype(
                str
            ).replace(
                {
                    "0.001": "0.1% of future GDP",
                    "0.01": "1% of future GDP",
                    "0.1": "10% of future GDP",
                    "0.25": "25% of future GDP",
                    "0.5": "50% of future GDP",
                    "858795238461.09": "1% of current GDP",
                    "1000000000000.0": "1 trillion USD",
                    "1.0": "100% of future GDP",
                }
            )

            subset["rcp"] = subset.rcp.replace(
                {
                    "ssp585": "8.5",
                    "ssp370": "7.0",
                    "ssp460": "6.0",
                    "ssp245": "4.5",
                }
            )

            if len(ssp) <= 1:

                g = sns.relplot(
                    data=subset.reset_index(),
                    x="year",
                    y="rate",
                    kind="line",
                    col="weitzman_parameter",
                    height=15,
                    facet_kws={"sharey": True, "sharex": True, "legend_out": False},
                    hue="rcp",
                    aspect=aspect,
                )
                g._legend.set_title("RCP")

            else:

                subset = subset.reset_index().rename(
                    columns={"ssp": "SSP", "rcp": "RCP"}
                )
                g = sns.relplot(
                    data=subset,
                    x="year",
                    y="rate",
                    kind="line",
                    col="weitzman_parameter",
                    height=15,
                    style="RCP",
                    facet_kws={"sharey": True, "sharex": True, "legend_out": False},
                    hue="SSP",
                    aspect=aspect,
                )

            plt.subplots_adjust(top=0.8, wspace=0)
            g.fig.text(
                0.0, 0.5, "Discount rate (%)", ha="center", va="center", rotation=90
            )
            g.fig.text(0.5, 0.0, "Year", ha="center", va="center", rotation=0)

            for a in g.axes.flatten():
                a.set_title(a.get_title().replace("weitzman_parameter = ", ""))
                a.set_xlabel("")
                a.set_ylabel("")

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(
                    f"{save_path}/{recipe}_{disc}_{'-'.join(ssp)}_{'-'.join(model)}_{'-'.join(rcp)}_implicit_discount_rates.png",
                    dpi=300,
                    bbox_inches="tight",
                )

    return g
