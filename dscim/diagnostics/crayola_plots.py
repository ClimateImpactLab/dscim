import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dscim import Waiter


def crayola_plots(
    var,
    config_file,
    path_saved_files=None,
    save_plot_path=None,
    bin_size=0.5,
    generate_damages=True,
    **kwargs,
):
    """Damages timeseries by temperature bins

    Plot timeline of daamges by binning global damages to temerature anomaly by
    binning. This function can either bin the damages or the ratio of damages to
    global consumption. To calculate damages and globla consumption we use the
    ```dscim.Waiter``. If ``generate_files`` option is ``False`` then
    the function will file the `damage_function_points.csv` data within the path
    matching the menu and discounting type.

    Parameters
    ----------
    var : str
        Variable to plot. Options are: ``damages``, ``ratio``
    config_file : str or dict
        Path to configuration file in local system or a ``dict`` object
    path_saved_files : str
        Path to menu output saved files
    bins_size : float
        Temperature bin size. Default is 0.5 degrees.
    save_plot_path : str
        Path to save plots. If ``None``, no plots are saved.
    generate_damages : bool
        Use ``Waiter`` to generate damages. If ``False``, data will be retrieved
        from ``path_saved_files``
    **kwargs
        Options passed to ``dscim.Waiter().menu_factory``

    Returns
    -------
        None
    """

    recipe = Waiter(config_file).menu_factory(**kwargs)

    if generate_damages:
        global_damages = recipe.damage_function_points
    else:
        damages_file = (
            f"{recipe.NAME}_{recipe.discounting_type}_damage_function_points.csv"
        )
        global_damages = pd.read_csv(os.path.join(path_saved_files, damages_file))

    # Merge global damages with global consumption
    try:
        global_damages = global_damages.merge(
            recipe.global_consumption.to_dataframe(
                name="global_consumption"
            ).reset_index(),
            on=["ssp", "year", "model"],
        )
    except KeyError:
        global_damages = global_damages.merge(
            recipe.global_consumption.to_dataframe(
                name="global_consumption"
            ).reset_index(),
            on=["year"],
        )

    # Create bins and bin temperature anomaly column
    temp_bins = np.arange(
        np.floor(global_damages.anomaly.min()),
        np.ceil(global_damages.anomaly.max()) + 1,
        bin_size,
    )
    global_damages["temp_bins"] = pd.cut(global_damages.anomaly, bins=temp_bins)

    # Get menu collapsed damages column name
    damages_col_name = global_damages.columns[
        global_damages.columns.str.startswith("global")
    ][0]

    if var == "ratio":
        global_damages["ratio_damages"] = (
            global_damages[damages_col_name] / global_damages["global_consumption"]
        )

        binned_damages = global_damages.groupby(
            ["year", "ssp", "rcp", "model", "temp_bins"], as_index=False
        )["ratio_damages"].mean()

        # Set some vars for plotting
        damages_col_name = "ratio_damages"
        y_axis_name = "Ratio Damages & Global Consumption"

    elif var == "damages":
        y_axis_name = "Damages"
        binned_damages = global_damages.groupby(
            ["year", "ssp", "rcp", "model", "temp_bins"], as_index=False
        )[damages_col_name].mean()
    else:
        raise NotImplementedError(f"{var} is not a defined option")

    with sns.plotting_context("paper", font_scale=1.3):
        sns.set_style("white")
        for ssp in binned_damages.ssp.unique():
            if recipe.discounting_type == "wr":
                g = sns.relplot(
                    x="year",
                    y=damages_col_name,
                    hue="temp_bins",
                    palette="rocket_r",
                    row="rcp",
                    kind="line",
                    legend=False,
                    data=binned_damages[
                        (binned_damages.year >= 2050) & (binned_damages.ssp == ssp)
                    ],
                )
                # Add h-line and improve plot
                (
                    g.map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
                    .set_axis_labels("Year", y_axis_name)
                    .set_titles("RCP {row_name}")
                    .tight_layout(w_pad=0)
                )
            else:
                g = sns.relplot(
                    x="year",
                    y=damages_col_name,
                    hue="temp_bins",
                    palette="rocket_r",
                    row="rcp",
                    col="model",
                    kind="line",
                    legend=False,
                    data=binned_damages[
                        (binned_damages.year >= 2050) & (binned_damages.ssp == ssp)
                    ],
                )

                # Add h-line and improve plot
                (
                    g.map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
                    .set_axis_labels("Year", y_axis_name)
                    .set_titles("{col_name} under RCP {row_name}")
                    .tight_layout(w_pad=0)
                )
            # Beef up x-axis
            (g.set(xticks=np.arange(2050, 2101, 10)))

            # Set legend
            plt.legend(
                title="GMST Bin",
                loc="upper left",
                bbox_to_anchor=(1, 2),
                labels=binned_damages.temp_bins.unique(),
            )
            # Title
            g.fig.suptitle(
                (
                    f"{y_axis_name} for {recipe.NAME} (2050-2100) \n"
                    f"under {ssp} and {recipe.discounting_type} discount \n "
                    f"{recipe.sector} with {bin_size}-degree bins"
                )
            )
            plt.subplots_adjust(top=0.85)

            # Save plot
            if save_plot_path is not None:
                if not os.path.exists(save_plot_path):
                    os.makedirs(save_plot_path, exist_ok=True)
                title = f"{var}_{recipe.discounting_type}_{recipe.NAME}_{recipe.sector}_{ssp}.png"
                g.savefig(os.path.join(save_plot_path, title))

    return (global_damages, recipe.global_consumption)
