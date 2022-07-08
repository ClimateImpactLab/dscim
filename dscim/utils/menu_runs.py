import os, gc, time

USER = os.getenv("USER")
import numpy as np
import pandas as pd
import xarray as xr
import dask

dask.config.set(**{"array.slicing.split_large_chunks": False})
from dscim import ProWaiter
from itertools import product


def run_AR6_epa_ssps(
    sectors,
    pulse_years,
    menu_discs,
    eta_rhos,
    config,
    results_root,
    reduced_damages_library,
    ssp_damage_function_library,
    global_cons=True,
    factors=True,
    marginal_damages=True,
    order="scc",
):

    w = ProWaiter(path_to_config=config)

    for sector, pulse_year, menu_disc, eta_rho in product(
        sectors, pulse_years, menu_discs, eta_rhos.items()
    ):

        menu_option = menu_disc[0]
        discount_type = menu_disc[1]

        save_path = f"{results_root}/{sector}/{pulse_year}/"

        kwargs = {
            "discounting_type": discount_type,
            "sector": sector,
            "ce_path": f"{reduced_damages_library}/{sector}/",
            "save_path": save_path,
            "weitzman_parameter": weitzman_values,
            "pulse_year": pulse_year,
            "eta": eta_rho[0],
            "rho": eta_rho[1],
        }

        if "CAMEL" in sector:
            kwargs.update(
                {
                    "damage_function_path": f"{ssp_damage_function_library}/{sector}/",
                }
            )

        menu_item = w.menu_factory(menu_key=menu_option, sector=sector, kwargs=kwargs)

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

        menu_item.order_plate(order)


def run_epa_rff(
    sectors,
    pulse_years,
    menu_discs,
    eta_rhos,
    config,
    results_root,
    rff_damage_function_library,
    global_cons=True,
    factors=True,
    marginal_damages=True,
    order="scc",
):

    w = ProWaiter(path_to_config=config)

    for sector, pulse_year, menu_disc, eta_rho in product(
        sectors, pulse_years, menu_discs, eta_rhos.items()
    ):

        menu_option = menu_disc[0]
        discount_type = menu_disc[1]

        save_path = f"{results_root}/{sector}/{pulse_year}/"
        os.makedirs(save_path, exist_ok=True)

        kwargs = {
            "discounting_type": discount_type,
            "sector": sector,
            "damage_function_path": f"{rff_damage_function_library}/{sector}",
            "save_path": save_path,
            "save_files": ["uncollapsed_sccs"],
            "weitzman_parameter": weitzman_values,
            "pulse_year": pulse_year,
            "ecs_mask_path": None,
            "ecs_mask_name": None,
            "eta": eta_rho[0],
            "rho": eta_rho[1],
        }

        menu_item = w.menu_factory(menu_key=menu_option, sector=sector, kwargs=kwargs)

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
            print("done saving discount factor")

        menu_item.order_plate(order)
