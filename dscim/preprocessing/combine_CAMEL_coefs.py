""" Combine the coastal and AMEL separately-estimated
coefficients into a single file for CAMEL.
"""

import xarray as xr
import pandas as pd
import numpy as np
import os, sys, argparse

USER = os.getenv("USER")

# read passed arguments,
# otherwise use defaults

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--recipes",
    required=False,
    type=str,
    nargs="*",
    default=["adding_up", "risk_aversion"],
    dest="recipes",
    metavar="<recipe list>",
    help="Recipe list",
)

parser.add_argument(
    "-e",
    "--etas",
    required=True,
    type=str,
    nargs="*",
    default=["2.0"],
    dest="etas",
    metavar="<eta list>",
    help="Eta list",
)

parser.add_argument(
    "-d",
    "--discs",
    required=False,
    type=str,
    nargs="*",
    default=["constant", "euler_ramsey", "euler_gwr"],
    dest="discs",
    metavar="<discounting_type list>",
    help="Discounting type list",
)
parser.add_argument(
    "-l",
    "--library",
    required=False,
    type=str,
    default=f"/home/{USER}/repos/integration/input_data/damage_function_library_AR6_bc39",
    dest="library",
    metavar="<library str>",
    help="library str",
)
parser.add_argument(
    "-c",
    "--coastal",
    required=True,
    type=str,
    default="coastal",
    dest="coastal",
    metavar="<coastal str>",
    help="coastal str",
)
parser.add_argument(
    "-A",
    "--AMEL",
    required=True,
    type=str,
    default="AMEL_clipped",
    dest="AMEL",
    metavar="<AMEL str>",
    help="AMEL str",
)
parser.add_argument(
    "-C",
    "--CAMEL",
    required=True,
    type=str,
    default="CAMEL_clipped",
    dest="CAMEL",
    metavar="<CAMEL str>",
    help="CAMEL str",
)
parser.add_argument("--fit", dest="fit", action="store_true")
parser.add_argument("--no-fit", dest="fit", action="store_false")

args = parser.parse_args()

# loop through options to create merged CAMEL
# and save into damage function library

eta_rhos = {
    "2.0": "0.0",
    "1.016010255": "9.149608e-05",
    "1.244459066": "0.00197263997",
    "1.421158116": "0.00461878399",
    "1.567899395": "0.00770271076",
}

for recipe in args.recipes:
    for disc in args.discs:
        for eta in args.etas:

            rho = eta_rhos[eta]
            print(f"Creating {recipe} {disc} for {args.CAMEL}...")

            coefs, fit = {}, {}

            for sector in [args.coastal, args.AMEL]:
                coefs[
                    sector
                ] = f"{args.library}/{sector}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_coefficients.nc4"

            coefs["combined"] = xr.combine_by_coords(
                [
                    xr.open_dataset(coefs[args.AMEL]),
                    xr.open_dataset(coefs[args.coastal]),
                ],
                combine_attrs="drop",
            )

            coefs["combined"].attrs = {
                "sector": args.CAMEL,
                "Run type": recipe,
                "Discount type": disc,
                "Coastal path": coefs[args.coastal],
                "AMEL path": coefs[args.AMEL],
            }

            coefs["combined"].to_netcdf(
                f"{args.library}/{args.CAMEL}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_coefficients.nc4"
            )

            if args.fit == True:

                for sector in [args.coastal, args.AMEL]:
                    fit[
                        sector
                    ] = f"{args.library}/{sector}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_fit.nc4"

                fit["combined"] = xr.open_dataset(fit[args.coastal]) + xr.open_dataset(
                    fit[args.AMEL]
                )

                fit["combined"].attrs = {
                    "sector": args.CAMEL,
                    "Run type": recipe,
                    "Discount type": disc,
                    "Coastal path": fit[args.coastal],
                    "AMEL path": fit[args.AMEL],
                }

                fit["combined"].to_netcdf(
                    f"{args.library}/{args.CAMEL}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_fit.nc4"
                )
