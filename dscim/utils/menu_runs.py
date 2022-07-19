from dscim.menu.simple_storage import Climate, EconVars
import dscim.menu.baseline
import dscim.menu.risk_aversion
import dscim.menu.equity

import os, gc, time, yaml

USER = os.getenv("USER")
import numpy as np
import pandas as pd
import xarray as xr
import dask

dask.config.set(**{"array.slicing.split_large_chunks": False})
from dscim import ProWaiter
from itertools import product

MENU_OPTIONS = {
    "adding_up": dscim.menu.baseline.Baseline,
    "risk_aversion": dscim.menu.risk_aversion.RiskAversionRecipe,
    "equity": dscim.menu.equity.EquityRecipe,
}


def run_ssps(
    sectors,
    pulse_years,
    menu_discs,
    eta_rhos,
    config,
    USA,
    AR,
    masks=[None],
    fair_dims_list=[["simulation"]],
    global_cons=False,
    factors=False,
    marginal_damages=False,
    order="damage_function",
):

    with open(config, "r") as stream:
        conf = yaml.safe_load(stream)

    for sector, pulse_year, menu_disc, eta_rho, mask, fair_dims in product(
        sectors, pulse_years, menu_discs, eta_rhos.items(), masks, fair_dims_list
    ):

        menu_option, discount_type = menu_disc
        save_path = f"{conf['paths'][f'AR{AR}_ssp_results']}/{sector}/{pulse_year}/"

        if mask is not None:
            save_path = save_path + "/" + mask

        if fair_dims != ["simulation"]:
            save_path = (
                save_path
                + "/"
                + f"fair_collapsed_{'_'.join([i for i in fair_dims if i!='simulation'])}"
            )

        if USA == True:
            econ = EconVars(path_econ=conf["econdata"]["USA_ssp"])
        else:
            econ = EconVars(path_econ=conf["econdata"]["global_ssp"])

        add_kwargs = {
            "econ_vars": econ,
            "climate_vars": Climate(
                **conf[f"AR{AR}_ssp_climate"],
                pulse_year=pulse_year,
                ecs_mask_name=mask,
            ),
            "formula": conf["sectors"][sector if USA == False else sector[:-4]][
                "formula"
            ],
            "discounting_type": discount_type,
            "sector": sector,
            "ce_path": f"{conf['paths']['reduced_damages_library']}/{sector}/",
            "save_path": save_path,
            "eta": eta_rho[0],
            "rho": eta_rho[1],
            "fair_dims": fair_dims,
        }

        kwargs = conf["global_parameters"].copy()
        for k, v in add_kwargs.items():
            assert (
                k not in kwargs.keys()
            ), f"{k} already set in config. Please check `global_parameters`."
            kwargs.update({k: v})

        if "CAMEL" in sector:
            kwargs.update(
                {
                    "damage_function_path": f"{conf['paths']['ssp_damage_function_library']}/{sector}/2020/",
                    "save_files": [
                        "damage_function_points",
                        "marginal_damages",
                        "discount_factors",
                        "uncollapsed_sccs",
                        "scc",
                    ],
                }
            )

        menu_item = MENU_OPTIONS[menu_option](**kwargs)
        menu_item.order_plate(order)

        if global_cons == True:
            menu_item.global_consumption_no_pulse.to_netcdf(
                f"{save_path}/{menu_option}_{discount_type}_eta{menu_item.eta}_rho{menu_item.rho}_global_consumption_no_pulse.nc4"
            )
            menu_item.global_consumption.to_netcdf(
                f"{save_path}/{menu_option}_{discount_type}_eta{menu_item.eta}_rho{menu_item.rho}_global_consumption.nc4"
            )

        if marginal_damages == True:
            md = (
                menu_item.global_consumption_no_pulse
                - menu_item.global_consumption_pulse
            ) * menu_item.climate.conversion
            md = md.rename("marginal_damages").to_dataset()
            for var in md.variables:
                md[var].encoding.clear()
            md.chunk(
                {
                    "discount_type": 1,
                    "weitzman_parameter": 1,
                    "ssp": 1,
                    "model": 1,
                    "gas": 1,
                    "year": 10,
                }
            ).to_zarr(
                f"{save_path}/{menu_option}_{discount_type}_eta{menu_item.eta}_rho{menu_item.rho}_uncollapsed_marginal_damages.zarr",
                consolidated=True,
                mode="w",
            )

        if factors == True:

            # holding population constant
            # from 2100 to 2300 with 2099 values
            pop = menu_item.collapsed_pop.sum("region")
            pop = pop.reindex(
                year=range(pop.year.min().values, menu_item.ext_end_year + 1),
                method="ffill",
            )

            df = menu_item.calculate_discount_factors(
                menu_item.global_consumption_no_pulse / pop
            ).to_dataset(name="discount_factor")
            for var in df.variables:
                df[var].encoding.clear()
            df.chunk(
                {
                    "discount_type": 1,
                    "weitzman_parameter": 1,
                    "ssp": 1,
                    "model": 1,
                    # "gas":1,
                    "year": 10,
                }
            ).to_zarr(
                f"{save_path}/{menu_option}_{discount_type}_eta{menu_item.eta}_rho{menu_item.rho}_uncollapsed_discount_factors.zarr",
                consolidated=True,
                mode="w",
            )


def run_rff(
    sectors,
    pulse_years,
    menu_discs,
    eta_rhos,
    config,
    USA,
    global_cons=True,
    factors=True,
    marginal_damages=True,
    order="scc",
):

    with open(config, "r") as stream:
        conf = yaml.safe_load(stream)

    for sector, pulse_year, menu_disc, eta_rho in product(
        sectors, pulse_years, menu_discs, eta_rhos.items()
    ):

        menu_option, discount_type = menu_disc
        save_path = f"{conf['paths']['rff_results']}/{sector}/{pulse_year}/"

        if USA == True:
            econ = EconVars(
                path_econ=f"{conf['rffdata']['socioec_output']}/rff_USA_socioeconomics.nc4"
            )
        else:
            econ = EconVars(
                path_econ=f"{conf['rffdata']['socioec_output']}/rff_global_socioeconomics.nc4"
            )

        add_kwargs = {
            "econ_vars": econ,
            "climate_vars": Climate(**conf["rff_climate"], pulse_year=pulse_year),
            "formula": conf["sectors"][sector if USA == False else sector[:-4]][
                "formula"
            ],
            "discounting_type": discount_type,
            "sector": sector,
            "ce_path": None,
            "save_path": save_path,
            "eta": eta_rho[0],
            "rho": eta_rho[1],
            "damage_function_path": f"{conf['paths']['rff_damage_function_library']}/{sector}/2020",
            "save_files": ["uncollapsed_sccs"],
            "ecs_mask_path": None,
            "ecs_mask_name": None,
        }

        kwargs = conf["global_parameters"].copy()
        for k, v in add_kwargs.items():
            assert (
                k not in kwargs.keys()
            ), f"{k} already set in config. Please check `global_parameters`."
            kwargs.update({k: v})

        menu_item = MENU_OPTIONS[menu_option](**kwargs)
        menu_item.order_plate(order)

        if global_cons == True:
            menu_item.global_consumption_no_pulse.to_netcdf(
                f"{save_path}/{menu_option}_{discount_type}_eta{menu_item.eta}_rho{menu_item.rho}_global_consumption_no_pulse.nc4"
            )
            menu_item.global_consumption.to_netcdf(
                f"{save_path}/{menu_option}_{discount_type}_eta{menu_item.eta}_rho{menu_item.rho}_global_consumption.nc4"
            )

        if marginal_damages == True:
            md = (
                (
                    (
                        menu_item.global_consumption_no_pulse
                        - menu_item.global_consumption_pulse
                    )
                    * menu_item.climate.conversion
                )
                .rename("marginal_damages")
                .to_dataset()
            )

            for var in md.variables:
                md[var].encoding.clear()

            md.chunk(
                {
                    "discount_type": 1,
                    "weitzman_parameter": 14,
                    "runid": 10000,
                    "gas": 1,
                    "year": 10,
                }
            ).to_zarr(
                f"{save_path}/{menu_option}_{discount_type}_eta{menu_item.eta}_rho{menu_item.rho}_uncollapsed_marginal_damages.zarr",
                consolidated=True,
                mode="w",
            )
        if factors == True:

            f = menu_item.calculate_discount_factors(
                menu_item.global_consumption_no_pulse / menu_item.pop
            ).to_dataset(name="discount_factor")

            for var in f.variables:
                f[var].encoding.clear()

            f.chunk(
                {
                    "discount_type": 1,
                    "weitzman_parameter": 14,
                    "runid": 10000,
                    "gas": 1,
                    "region": 1,
                    "year": 10,
                }
            ).to_zarr(
                f"{save_path}/{menu_option}_{discount_type}_eta{menu_item.eta}_rho{menu_item.rho}_uncollapsed_discount_factors.zarr",
                consolidated=True,
                mode="w",
            )
