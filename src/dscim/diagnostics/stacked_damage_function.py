import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt
from dscim.diagnostics.damage_function import damage_function


def plot_stacked(
    sector,
    recipe="risk_aversion",
    disc="constant",
    eta=2.0,
    rho=0.0,
    xlim=(-1, 8),
    years=None,
    sharey=False,
    rff=True,
):
    if years is None:
        years = [2020, 2050, 2090, 2100, 2200, 2300]

    root_rff = f"/shares/gcp/integration_replication/results/rff/{sector}/2020/"
    root_ssp = f"/shares/gcp/integration_replication/results/AR6_ssp/{sector}/2020/"
    coefs_rff = root_rff
    coefs_ssp = root_ssp
    output = f"/mnt/CIL_integration/plots/rff_diagnostics/rff_ssp_stacked_damage_functions_replication/{sector}"

    # overlap ssp and rff
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ds_rff_mean = (
        xr.open_dataset(
            f"{coefs_rff}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_coefficients.nc4"
        )
        .sel(year=years)
        .mean("runid")
    )
    ds_rff = (
        xr.open_dataset(
            f"{coefs_rff}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_coefficients.nc4"
        )
        .sel(year=years)
        .quantile(quantiles, "runid")
    )
    ds_ssp = xr.open_dataset(
        f"{coefs_ssp}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_coefficients.nc4"
    ).sel(year=years)

    temps = xr.DataArray(
        np.arange(xlim[0], xlim[1], 0.1),
        coords={"anomaly": np.arange(xlim[0], xlim[1], 0.1)},
    )

    fit_rff_mean = (
        ds_rff_mean["anomaly"] * temps
        + ds_rff_mean["np.power(anomaly, 2)"] * temps**2
    )
    fit_rff_mean = fit_rff_mean.to_dataframe("fit").reset_index()
    fit_rff_mean["model"] = "RFF mean"
    fit_rff_mean = fit_rff_mean[["fit", "year", "model", "anomaly"]]

    fit_rff = ds_rff["anomaly"] * temps + ds_rff["np.power(anomaly, 2)"] * temps**2
    fit_rff = fit_rff.to_dataframe("fit").reset_index()
    fit_rff["model"] = "quantile: " + fit_rff["quantile"].astype(str)
    fit_rff = fit_rff[["fit", "year", "model", "anomaly"]]

    fit_ssp = ds_ssp["anomaly"] * temps + ds_ssp["np.power(anomaly, 2)"] * temps**2
    fit_ssp = fit_ssp.to_dataframe("fit").reset_index()
    fit_ssp["model"] = fit_ssp.ssp + "-" + fit_ssp.model
    fit_ssp = fit_ssp[["fit", "year", "model", "anomaly"]]

    if rff:
        fit = pd.concat([fit_ssp, fit_rff, fit_rff_mean]).set_index(
            ["year", "model", "anomaly"]
        )
        pal = sns.color_palette("Paired", 6) + sns.color_palette(
            "Greys", len(quantiles) + 1
        )
    else:
        fit = fit_ssp.set_index(["year", "model", "anomaly"])
        pal = sns.color_palette("Paired", 6)

    g = sns.relplot(
        data=fit,
        x="anomaly",
        y="fit",
        hue="model",
        col="year",
        col_wrap=3,
        kind="line",
        palette=pal,
        facet_kws={"sharey": sharey, "sharex": True},
        legend="full",
    )

    g.fig.suptitle(f"{sector} {recipe} {disc} eta={eta} rho={rho}")

    plt.subplots_adjust(top=0.85)

    os.makedirs(output, exist_ok=True)
    plt.savefig(
        f"{output}/stacked_{sector}_{recipe}_{disc}_{eta}_{rho}_xlim{xlim}_rff-{rff}_years{years}.png",
        bbox_inches="tight",
        dpi=300,
    )

    # plt.close()
