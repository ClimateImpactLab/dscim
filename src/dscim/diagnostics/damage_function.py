import os

import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from dscim.utils.utils import compute_damages
import logging

logger = logging.getLogger(__name__)

username = os.getenv("USER")


def get_legacy(sector, filepath, scale):
    """Load legacy results from available sectors: mortality and labor

    This functions gets the SSP3, 'adding up' betas from the specified sector.
    The betas are then scaled and used to predict the damage function.

    Parameters
    ----------

    Returns
    -------
    """

    # Create temp range
    temp_range = np.arange(0, 10, 0.1)
    idx = product(range(2010, 2100), temp_range)
    temps = np.stack([temp_range for _ in range(2010, 2100)]).flatten()
    temps = pd.DataFrame(temps, index=[i for i, x in idx], columns=["anomaly"])
    df = pd.read_csv(filepath)

    if sector == "mortality":
        coefs = df.loc[(df.age_adjustment == "vly") & (df.heterogeneity == "scaled")]
    elif sector == "labor":
        coefs = df

    else:
        logger.warning("Unrecognized sector. Please pass 'mortality' or 'labor'.")

    # Rename and scale betas
    coefs = coefs.set_index("year").rename(
        columns={"beta1": "beta_1", "beta2": "beta_2"}
    )
    coefs[["cons", "beta_1", "beta_2"]] = coefs[["cons", "beta_1", "beta_2"]].apply(
        lambda x: x * scale
    )

    # Join to temperature range
    coefs = temps.join(coefs).reset_index().rename(columns={"index": "year"})

    # Predict y_hat
    coefs["y_hat"] = (
        coefs.cons + coefs.anomaly * coefs.beta_1 + coefs.anomaly**2 * coefs.beta_2
    )

    return coefs[["year", "anomaly", "y_hat"]]


def damage_function(
    sector,
    sector_path,
    discounting,
    eta,
    rho,
    year=2097,
    hue_vars="ssp",
    recipes=("adding_up", "risk_aversion", "equity"),
    scale=10**12,
    x_lim=(-np.inf, np.inf),
    y_lim=(-np.inf, np.inf),
    x_var="anomaly",
    legacy=None,
    save_path=None,
    subset_dict=None,
    scatter=True,
    attributes=True,
):

    """
    This function plots the damage function for a specific year,
    discounting type, and sector.

    Parameters
    ----------

    sector : str
        Name of the sector to be plotted. Can be 'mortality' or 'labor'.
    sector_path : str
        Path to sector's saved damage function files.
    discounting : str
        Type of discounting. Can be 'constant', 'ramsey', or 'wr'.
    year: int
        Year of damage function to be plotted.
    recipes : sequence of str
        Recipe types to be plotted. Can be 'adding_up', 'risk_aversion', 'equity'
    scale : int
        Units of dollars for axis. ie., if 10**12 is passed, y axis will be in trillions.
    legacy: tuple
        If desired, plot legacy damage function on top of the plots. Can only be true for
        'mortality' and 'labor' sectors.
        legacy[0] : str, filepath to legacy betas
        legacy[1] : int, scale of legacy betas (ex. if in trillions, scale = 10**12)
    save_path : str
        Path to save
    x_lim : tuple of floats
        Bounds of x-axis.

    Note
    -----
        The function saves to ``save_path``

    Returns
    -------
        None.

    """

    points, fit, attrs = {}, {}, {}

    points, fit = {}, {}
    for recipe in recipes:

        # get relevant files
        points_file = pd.read_csv(
            f"{sector_path}/{recipe}_{discounting}_eta{eta}_rho{rho}_damage_function_points.csv"
        ).reset_index()

        fit_file = xr.open_dataset(
            f"{sector_path}/{recipe}_{discounting}_eta{eta}_rho{rho}_damage_function_fit.nc4"
        )

        # subset
        if subset_dict is not None:
            for col, val in subset_dict.items():
                points_file = points_file.loc[points_file[col].isin(val)]
            fit_file = fit_file.sel(subset_dict)

        points[recipe] = points_file.reset_index()
        fit[recipe] = fit_file.to_dataframe().reset_index()

        # save attributes to label plot
        attrs[
            recipe
        ] = f"""
            eta: {fit_file.attrs['eta']}, rho: {fit_file.attrs['rho']},
        """

        # subset years
        points[recipe] = points[recipe].loc[
            (points[recipe].year >= year - 2) & (points[recipe].year <= year + 2)
        ]
        fit[recipe] = fit[recipe].loc[fit[recipe].year == year]

        # scale damage points and betas
        points[recipe]["damages_scaled"] = points[recipe][
            "damages"
            #             f"global_damages_{discounting}"
        ].apply(lambda x: x / scale)
        fit[recipe]["y_hat_scaled"] = fit[recipe]["y_hat"].apply(lambda x: x / scale)

        fit[recipe]["hue"] = reduce(
            lambda a, b: a.str.cat(b, sep=","),
            [fit[recipe][i].astype(str) for i in hue_vars],
        )
        points[recipe]["hue"] = reduce(
            lambda a, b: a.str.cat(b, sep=","),
            [points[recipe][i].astype(str) for i in hue_vars],
        )

    # grab legacy function if passed
    if legacy is not None:
        fit["legacy"] = get_legacy(sector, filepath=legacy[0], scale=legacy[1])
        fit["legacy"] = fit["legacy"].loc[fit["legacy"].year == year]
        fit["legacy"]["y_hat_scaled"] = fit["legacy"]["y_hat"].apply(
            lambda x: x / scale
        )

    sns.set_style("darkgrid")

    fig, ax = plt.subplots(
        1, len(recipes), figsize=(15, 10), sharex=True, sharey=True, squeeze=False
    )

    plt.subplots_adjust(wspace=0, hspace=0, top=0.90)

    palette = "tab10" if discounting == "constant_model_collapsed" else "tab20"

    for i, recipe in enumerate(recipes):

        print(f"{recipe} : # of points is {len(points[recipe])}")

        if scatter:

            sns.scatterplot(
                data=points[recipe]
                .loc[
                    (points[recipe].damages_scaled >= y_lim[0])
                    & (points[recipe].damages_scaled <= y_lim[1])
                    & (points[recipe][x_var] >= x_lim[0])
                    & (points[recipe][x_var] <= x_lim[1])
                ]
                .sort_values("hue"),
                x=x_var,
                y="damages_scaled",
                hue="hue",
                palette=palette,
                s=6,
                ax=ax[0][i],
                edgecolor="face",
            )

        sns.lineplot(
            data=(
                fit[recipe]
                .loc[
                    (fit[recipe][x_var] >= x_lim[0]) & (fit[recipe][x_var] <= x_lim[1])
                ]
                .sort_values("hue")
            ),
            x=x_var,
            y="y_hat_scaled",
            hue="hue",
            palette=palette,
            ax=ax[0][i],
        )

        if legacy is not None:

            sns.lineplot(
                data=fit["legacy"],
                x=x_var,
                y="y_hat_scaled",
                color="grey",
                linestyle="dotted",
                ax=ax[0][i],
            )

        ax[0][i].set_xlabel(x_var)
        ax[0][i].set_title(f"Recipe: {recipe}")
        if attributes:
            ax[0][i].annotate(
                attrs[recipe], (0.0, -0.1), xycoords="axes fraction", fontsize=6
            )

    ltext = "\n paper damage function = grey dotted" if legacy is not None else ""
    fig.suptitle(
        f"{sector}: {discounting} discounting \n Damage functions in {year} {ltext}"
    )

    ax[0][0].set_ylabel(f'Damages in {"{:.0e}".format(scale)} 2019 USD')

    if save_path is not None:

        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{sector}_{discounting}_{year}_damage_function.pdf")

    return fig


def damage_function_w_consumption(
    sector,
    root,
    output,
    eta,
    rho,
    years=(2050, 2099),
    recipes=("equity",),
    disc="constant",
    model="IIASA GDP",
    ssp="SSP3",
):

    if "gwr" in disc:
        ssp = "['SSP2', 'SSP3', 'SSP4']"
        model = "['IIASA GDP', 'OECD Env-Growth']"

    s_list, g_list = [], []
    for recipe in recipes:

        # open fitted damage functions
        s = (
            xr.open_dataset(
                f"{root}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_fit.nc4"
            )
            .sel(model=model, ssp=ssp, year=years)
            .to_dataframe()
            .reset_index()
        )
        s["recipe"] = recipe
        s_list.append(s)

        # open consumption
        g = xr.open_dataset(
            f"{root}/{recipe}_{disc}_eta{eta}_rho{rho}_global_consumption.nc4"
        )
        if "gwr" in disc:
            g = g.expand_dims(dict(model=model, ssp=ssp))

        g = (
            g.sel(model=model, ssp=ssp, year=years)
            .to_array()
            .rename("gdp")
            .to_dataframe()
            .reset_index()
        )
        g["recipe"] = recipe
        g_list.append(g)

    subset = pd.concat(s_list)
    gdp = pd.concat(g_list)

    subset["damages_T"] = subset.y_hat / 1e12
    gdp["gdp_T"] = gdp.gdp / 1e12

    fig, axes = plt.subplots(
        min(len(recipes), len(years)),
        max(len(recipes), len(years)),
        figsize=(20, 8),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    ax = axes.flat

    for i, ry in enumerate(product(recipes, years)):

        sns.lineplot(
            data=subset.loc[(subset.year == ry[1]) & (subset.recipe == ry[0])],
            x="anomaly",
            y="damages_T",
            ax=ax[i],
        )

        ax[i].hlines(
            gdp.loc[(gdp.year == ry[1]) & (gdp.recipe == ry[0])].gdp_T.item(),
            0,
            20,
            color="green",
            linestyles="dashed",
        )

        ax[i].set_ylabel("Damages (trillion USD)")
        ax[i].set_xlabel("Δ GMST (°C)")
        ax[i].set_title(ry[1])
        ax[i].annotate(
            "100% of consumption",
            xy=(
                0,
                gdp.loc[(gdp.year == ry[1]) & (gdp.recipe == ry[0])].gdp_T.item() + 5,
            ),
            color="green",
        )

    plt.subplots_adjust(wspace=0, hspace=0)
    os.makedirs(output, exist_ok=True)
    plt.savefig(
        f"{output}/{'-'.join(recipes)}_{disc}_{'-'.join([str(i) for i in years])}_{ssp}_{model}_eta{eta}_rho{rho}_damage_function.png",
        dpi=300,
        bbox_inches="tight",
    )
