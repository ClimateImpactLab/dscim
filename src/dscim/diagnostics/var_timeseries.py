import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml

USER = os.getenv("USER")
from dscim.menu.simple_storage import Climate


def get_rff_id(
    sector,
    mask,
    recipe,
    disc,
    eta,
    rho,
    kind,
    results_root,
    output,
    discrate=0.02,
    pulse_year=2020,
):

    if mask in ["epa_mask", "unmasked"]:
        pass
    elif mask == "gdppc_mask":
        print("gdppc_mask deprecated")

    sccs = (
        xr.open_dataset(
            f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_sccs.nc4"
        )
        .sel(
            weitzman_parameter="0.5",
            fair_aggregation="uncollapsed",
            gas="CO2_Fossil",
            drop=True,
        )
        .uncollapsed_sccs
    )
    if "discrate" in sccs.dims:
        sccs = sccs.sel(discrate=discrate, drop=True)
    else:
        pass

    # runid rff_sp-simulation crosswalk
    cw = xr.open_dataset("/shares/gcp/integration/rff2/rffsp_fair_sequence.nc")

    if mask == "epa_mask":
        rffsp_values = pd.read_csv(
            "/shares/gcp/integration/rff/rffsp_trials.csv"
        ).rffsp_id.unique()
        runid_idx = cw.where(cw.rff_sp.isin(rffsp_values), drop=True)
        sccs = sccs.sel(runid=runid_idx.runid)

    if kind == "zeroes":
        rff_ids = sccs.where((sccs > -1) & (sccs < 1), drop=True)
    elif kind == "p0.1":
        cutoff = sccs.quantile(0.001, "runid")
        rff_ids = sccs.where(sccs < cutoff, drop=True)
    elif kind == "p50":
        cutoff_a = sccs.quantile(0.499, "runid")
        cutoff_b = sccs.quantile(0.50, "runid")
        rff_ids = sccs.where((sccs > cutoff_a) & (sccs <= cutoff_b), drop=True)
    elif kind == "p75":
        cutoff_a = sccs.quantile(0.749, "runid")
        cutoff_b = sccs.quantile(0.75, "runid")
        rff_ids = sccs.where((sccs > cutoff_a) & (sccs <= cutoff_b), drop=True)
    elif kind == "p99.9":
        cutoff = sccs.quantile(0.999, "runid")
        rff_ids = sccs.where(sccs > cutoff, drop=True)

    csv = pd.DataFrame(list(rff_ids["runid"].values), columns=["runid"])
    if kind != "zeroes":
        assert len(csv) == 10, f"Incorrect number of runids: {len(csv)} instead of 10."
    os.makedirs(f"{output}/{sector}", exist_ok=True)
    csv.to_csv(
        f"{output}/{sector}/{recipe}_{disc}_eta{eta}_rho{rho}_{kind}_{mask}_runids.csv",
        index=False,
    )


def get_ssp_id(
    sector,
    recipe,
    disc,
    eta,
    rho,
    kind,
    output,
    results_root,
    discrate=0.02,
    pulse_year=2020,
):

    sccs = (
        xr.open_dataset(
            f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_sccs.nc4"
        )
        .sel(weitzman_parameter="0.5", fair_aggregation="uncollapsed", drop=True)
        .uncollapsed_sccs
    )

    if disc == "constant":
        sccs = sccs.sel(discrate=discrate, drop=True)

    if kind == "p1":
        cutoff = sccs.quantile(0.01, ["simulation"])
        ids = sccs.where(sccs < cutoff, drop=True)
    elif kind == "zeroes":
        ids = sccs.where((sccs > -0.01) & (sccs < 0.01), drop=True)
    elif kind == "p99":
        cutoff = sccs.quantile(0.99, ["simulation"])
        ids = sccs.where(sccs > cutoff, drop=True)

    ids = ids.to_dataframe()
    csv = ids.loc[~ids.uncollapsed_sccs.isnull()]["uncollapsed_sccs"]
    if kind != "zeroes":
        sims = csv.groupby(["ssp", "rcp", "model", "gas"]).count().unique()
        assert all([i == 23 for i in sims]), "wrong number of simulations."

    os.makedirs(f"{output}/{sector}", exist_ok=True)
    csv.to_csv(
        f"{output}/{sector}/{recipe}_{disc}_eta{eta}_rho{rho}_{kind}_simids.csv",
        index=True,
    )


def rff_timeseries(
    sector,
    mask,
    recipe,
    disc,
    eta,
    rho,
    gas,
    kind,
    results_root,
    output,
    config,
    runid_root,
    USA,
    discrate=0.02,
    pulse_year=2020,
):

    # index
    rff_ids = (
        pd.read_csv(
            f"{runid_root}/{sector}/{recipe}_{disc}_eta{eta}_rho{rho}_{kind}_{mask}_runids.csv"
        )
        .set_index("runid")
        .index
    )

    # runid rff_sp-simulation crosswalk
    cw = xr.open_dataset("/shares/gcp/integration/rff2/rffsp_fair_sequence.nc")

    # sccs
    sccs = (
        xr.open_dataset(
            f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_sccs.nc4"
        )
        .sel(
            weitzman_parameter="0.5", gas=gas, fair_aggregation="uncollapsed", drop=True
        )
        .uncollapsed_sccs.rename("SCC")
    )

    # marginal damages
    damages = (
        xr.open_zarr(
            f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_marginal_damages.zarr"
        )
        .sel(weitzman_parameter="0.5", gas=gas, drop=True)
        .marginal_damages
    )

    # discount factors
    df = (
        xr.open_zarr(
            f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_discount_factors.zarr"
        )
        .sel(weitzman_parameter="0.5", gas=gas, drop=True)
        .discount_factor.rename("discount_factors")
    )

    # discounted damages
    discounted_damages = (damages * df).rename("discounted_damages")

    # emissions
    if gas == "CO2_Fossil":
        g = "C"
    elif gas == "CH4":
        g = "CH4"
    elif gas == "N2O":
        g = "N2"
    emissions = (
        xr.open_dataset(
            "/shares/gcp/integration/rff2/climate/emissions/rff-sp_emissions_all_gases.nc"
        )
        .sel(gas=g, drop=True)
        .rename({"Year": "year", "simulation": "rff_sp"})
        .emissions.sel(rff_sp=cw.rff_sp)
    )

    # cumulative emissions
    c_emissions = emissions.cumsum("year").rename("cumulative_emissions")
    c_emissions["year"] = emissions.year

    # gmst
    with open(config) as config_file:
        params = yaml.full_load(config_file)
        params["rff_climate"].update(
            {
                "gmst_fair_path": "/shares/gcp/integration/rff2/climate/ar6_rff_fair162_control_pulse_all_gases_2020-2030-2040-2050-2060-2070-2080_emis_conc_rf_temp_lambdaeff_ohc_emissions-driven_naturalfix_v5.03_Feb072022.nc",
                "pulse_year": pulse_year,
            }
        )
    gmst = (
        Climate(**params["rff_climate"])
        .fair_pulse.temperature.sel(gas=gas, drop=True)
        .rename("gmst")
    )
    gmst_pulse = (
        Climate(**params["rff_climate"])
        .fair_pulse.temperature.sel(gas=gas, drop=True)
        .rename("gmst")
    )
    gmst_control = (
        Climate(**params["rff_climate"])
        .fair_control.temperature.sel(gas=gas, drop=True)
        .rename("gmst")
    )
    gmst_pulse_minus_control = gmst_pulse - gmst_control
    gmst_pulse_minus_control = gmst_pulse_minus_control.rename(
        "gmst_pulse_minus_control"
    )

    # gdp
    if USA:
        gdp = xr.open_dataset(
            "/shares/gcp/integration_replication/inputs/econ/rff_USA_socioeconomics.nc4"
        )
    else:
        gdp = xr.open_dataset(
            "/shares/gcp/integration_replication/inputs/econ/rff_global_socioeconomics.nc4"
        )
    gdp = gdp.drop("region").rename({"runid": "rff_sp"}).gdp.sel(rff_sp=cw.rff_sp)

    data = xr.combine_by_coords(
        [
            i.to_dataset()
            for i in [
                sccs,
                damages,
                df,
                discounted_damages,
                emissions,
                c_emissions,
                gmst,
                gmst_pulse_minus_control,
                gdp,
            ]
        ]
    )

    if "discrate" in data.dims:
        data = data.sel(discrate=discrate, drop=True)
    else:
        pass

    data = data.sel(runid=rff_ids).to_dataframe().reset_index()

    fig, ax = plt.subplots(8, 1, figsize=(10, 15), sharex=True)

    for i, yvar in enumerate(
        [
            "cumulative_emissions",
            "emissions",
            "gmst",
            "gmst_pulse_minus_control",
            "marginal_damages",
            "discount_factors",
            "discounted_damages",
            "gdp",
        ]
    ):

        sns.lineplot(
            data=data,
            x="year",
            y=yvar,
            hue="SCC",
            legend="full",
            ax=ax[i],
        )

        ax[i].set_title(yvar)
        if i > 0:
            ax[i].get_legend().remove()
        else:
            h, lth = ax[i].get_legend_handles_labels()
            lth = [str(round(float(i), 2)) for i in lth]
            ax[i].legend(
                h,
                lth,
                loc="upper center",
                bbox_to_anchor=(0.5, 2.1),
                ncol=3,
                title="SCC value",
            )

    plt.subplots_adjust(top=0.85)
    fig.suptitle(
        f"mask={mask}, SCC subset={kind} \n {sector} {recipe} {disc}, eta={eta} rho={rho}"
    )
    os.makedirs(f"{output}/{sector}", exist_ok=True)
    plt.savefig(
        f"{output}/{sector}/vars_timeseries_{gas}_{sector}_{recipe}_{disc}_{eta}_{rho}_{kind}_{mask}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def ssp_timeseries(
    sector,
    kind,
    ssp,
    rcp,
    model,
    recipe,
    disc,
    eta,
    rho,
    gas,
    results_root,
    simid_root,
    config,
    output,
    pulse_year=2020,
    results_mask="unmasked",
    discrate=0.02,
):

    # index
    ids = pd.read_csv(
        f"{simid_root}/{sector}/{recipe}_{disc}_eta{eta}_rho{rho}_{kind}_simids.csv"
    )
    ids = (
        ids.loc[(ids.ssp == ssp) & (ids.model == model) & (ids.rcp == rcp)]
        .set_index(["simulation"])
        .index
    )

    # sccs
    sccs = (
        xr.open_dataset(
            f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_sccs.nc4"
        )
        .sel(
            weitzman_parameter="0.5", gas=gas, fair_aggregation="uncollapsed", drop=True
        )
        .uncollapsed_sccs.rename("SCC")
    )

    # marginal damages
    damages = (
        xr.open_zarr(
            f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_marginal_damages.zarr"
        )
        .sel(weitzman_parameter="0.5", gas=gas, drop=True)
        .marginal_damages
    )

    # discount factors
    df = xr.open_zarr(
        f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_discount_factors.zarr"
    )

    if "coastal" not in sector:
        # coastal is missing the gas dimension for EPA results
        df = df.sel(gas=gas, drop=True)

    df = df.sel(weitzman_parameter="0.5", drop=True).discount_factor.rename(
        "discount_factors"
    )

    # discounted damages
    discounted_damages = (damages * df).rename("discounted_damages")

    # emissions
    emissions = xr.open_dataset(
        "/shares/gcp/integration/float32/dscim_input_data/climate/AR6/ar6_fair162_control_pulse_2020-2030-2040-2050-2060-2070-2080_emis_conc_rf_temp_lambdaeff_emissions-driven_naturalfix_v4.0_Jan212022.nc"
    ).emissions.sel(gas=gas, drop=True)

    # cumulative emissions
    c_emissions = emissions.cumsum("year").rename("cumulative_emissions")
    c_emissions["year"] = emissions.year

    # gmst
    with open(config) as config_file:
        params = yaml.full_load(config_file)
        params["AR6_ssp_climate"].update({"pulse_year": pulse_year})

    # gdp
    gdp = xr.open_dataset(
        f"{results_root}/{recipe}_{disc}_eta{eta}_rho{rho}_global_consumption.nc4"
    ).__xarray_dataarray_variable__.rename("gdp")

    anom_var = "gmsl" if "coastal" in sector else "temperature"

    anomaly = Climate(**params["AR6_ssp_climate"]).fair_pulse[anom_var]

    if "coastal" not in sector:
        anomaly = anomaly.sel(gas=gas, drop=True)

    data = xr.combine_by_coords(
        [
            i.to_dataset()
            for i in [
                sccs,
                damages,
                df,
                discounted_damages,
                emissions,
                c_emissions,
                anomaly,
                gdp,
            ]
        ]
    )
    data = data.sel(
        simulation=ids, year=slice(2000, 2300), ssp=ssp, model=model, rcp=rcp
    )

    if disc == "constant":
        data = data.sel(discrate=discrate, drop=True)

    data = data.to_dataframe().reset_index()

    fig, ax = plt.subplots(7, 1, figsize=(10, 15), sharex=True)

    for i, yvar in enumerate(
        [
            "cumulative_emissions",
            "emissions",
            anom_var,
            "marginal_damages",
            "discount_factors",
            "discounted_damages",
            "gdp",
        ]
    ):

        sns.lineplot(
            data=data,
            x="year",
            y=yvar,
            hue="SCC",
            ax=ax[i],
            legend="full",
        )

        ax[i].set_title(yvar)
        if i > 0:
            ax[i].get_legend().remove()
        else:
            h, lth = ax[i].get_legend_handles_labels()
            lth = [str(round(float(i), 2)) for i in lth]
            ax[i].legend(
                h,
                lth,
                loc="upper center",
                bbox_to_anchor=(0.5, 2.1),
                ncol=6,
                title="SCC value",
            )

    plt.subplots_adjust(top=0.82)
    fig.suptitle(
        f"SCC subset={kind} \n {sector} {recipe} {disc}, eta={eta} rho={rho} \n {ssp} {model} {rcp}"
    )
    os.makedirs(f"{output}/{sector}", exist_ok=True)
    plt.savefig(
        f"{output}/{sector}/vars_timeseries_{sector}_{recipe}_{disc}_{eta}_{rho}_{kind}_{ssp}_{rcp}_{model}.png",
        bbox_inches="tight",
        dpi=300,
    )
