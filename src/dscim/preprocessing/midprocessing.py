import os
import sys
import shutil

USER = os.getenv("USER")
import xarray as xr


def update_damage_function_library(
    input,
    output,
    recipe,
    disc,
    eta,
    rho,
    fit=True,
):

    os.makedirs(output, exist_ok=True)
    shutil.copy2(
        f"{input}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_coefficients.nc4",
        output,
    )

    if fit:
        shutil.copy2(
            f"{input}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_fit.nc4", output
        )


def combine_CAMEL_coefs(
    recipe,
    disc,
    eta,
    rho,
    CAMEL,
    coastal,
    AMEL,
    input_dir,
    mask="unmasked",
    pulse_year=2020,
    fit=True,
):

    print(f"Creating {recipe} {disc} for {CAMEL}...")
    os.makedirs(f"{input_dir}/{CAMEL}/{pulse_year}/{mask}", exist_ok=True)

    coefs, fit = {}, {}

    for sector in [coastal, AMEL]:
        coefs[
            sector
        ] = f"{input_dir}/{sector}/{pulse_year}/{mask}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_coefficients.nc4"

    coefs["combined"] = xr.combine_by_coords(
        [
            xr.open_dataset(coefs[AMEL]),
            xr.open_dataset(coefs[coastal]),
        ],
        combine_attrs="drop",
    )

    coefs["combined"].attrs = {
        "sector": CAMEL,
        "Run type": recipe,
        "Discount type": disc,
        "Coastal path": coefs[coastal],
        "AMEL path": coefs[AMEL],
    }

    coefs["combined"].to_netcdf(
        f"{input_dir}/{CAMEL}/{pulse_year}/{mask}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_coefficients.nc4"
    )

    if fit:

        for sector in [coastal, AMEL]:
            fit[
                sector
            ] = f"{input_dir}/{sector}/{pulse_year}/{mask}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_fit.nc4"

        fit["combined"] = xr.open_dataset(fit[coastal]) + xr.open_dataset(fit[AMEL])

        fit["combined"].attrs = {
            "sector": CAMEL,
            "Run type": recipe,
            "Discount type": disc,
            "Coastal path": fit[coastal],
            "AMEL path": fit[AMEL],
        }

        fit["combined"].to_netcdf(
            f"{input_dir}/{CAMEL}/{pulse_year}/{mask}/{recipe}_{disc}_eta{eta}_rho{rho}_damage_function_fit.nc4"
        )
