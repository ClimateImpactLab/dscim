from dscim.menu.simple_storage import Climate, EconVars
import dscim.menu.baseline
import dscim.menu.risk_aversion
import dscim.menu.equity

import os
import gc
import time
import yaml

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
    masks=None,
    fair_dims_list=None,
    order="damage_function",
):
    if masks is None:
        masks = [None]
    if fair_dims_list is None:
        fair_dims_list = [["simulation"]]

    with open(config, "r") as stream:
        conf = yaml.safe_load(stream)

    for sector, pulse_year, menu_disc, eta_rho, mask, fair_dims in product(
        sectors, pulse_years, menu_discs, eta_rhos.items(), masks, fair_dims_list
    ):

        menu_option, discount_type = menu_disc
        save_path = f"{conf['paths'][f'AR{AR}_ssp_results']}/{sector}/{pulse_year}/"

        if mask is not None:
            save_path = save_path + "/" + mask
        else:
            save_path = save_path + "/" + "unmasked"

        if fair_dims != ["simulation"]:
            save_path = (
                save_path
                + "/"
                + f"fair_collapsed_{'_'.join([i for i in fair_dims if i!='simulation'])}"
            )

        if USA:
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
            "formula": conf["sectors"][sector if not USA else sector[:-4]]["formula"],
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
                    "damage_function_path": f"{conf['paths']['ssp_damage_function_library']}/{sector}/2020/unmasked",
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


def run_rff(
    sectors,
    pulse_years,
    menu_discs,
    eta_rhos,
    config,
    USA,
    order="scc",
):

    with open(config, "r") as stream:
        conf = yaml.safe_load(stream)

    for sector, pulse_year, menu_disc, eta_rho in product(
        sectors, pulse_years, menu_discs, eta_rhos.items()
    ):

        menu_option, discount_type = menu_disc
        save_path = f"{conf['paths']['rff_results']}/{sector}/{pulse_year}/unmasked"

        if USA:
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
            "formula": conf["sectors"][sector if not USA else sector[:-4]]["formula"],
            "discounting_type": discount_type,
            "sector": sector,
            "ce_path": None,
            "save_path": save_path,
            "eta": eta_rho[0],
            "rho": eta_rho[1],
            "damage_function_path": f"{conf['paths']['rff_damage_function_library']}/{sector}/2020/unmasked",
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
