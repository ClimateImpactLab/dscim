import xarray as xr
import pandas as pd
import numpy as np
from datetime import date
from p_tqdm import p_map


def convert_old_to_newformat_AR(
    oldpaths,
    gas="CO2_Fossil",
    var="temperature",
):
    """
    Convert AR6 files into integration-processable format.
    """

    med = xr.open_dataset(oldpaths["median"])
    sims = xr.open_dataset(oldpaths["sims"])

    mednew = (
        med[var]
        .astype(np.float32)
        .to_dataset(dim="runtype")
        .rename(
            {
                "control": f"medianparams_control_{var}",
                "pulse": f"medianparams_pulse_{var}",
            }
        )
        .expand_dims(
            dim={
                "gas": [
                    gas,
                ],
            }
        )
    )

    simsnew = sims.temperature.astype(np.float32).to_dataset(dim="runtype")
    simsnew = simsnew.rename({"control": f"control_{var}", "pulse": f"pulse_{var}"})
    simsnew = simsnew.expand_dims(
        dim={
            "gas": [
                gas,
            ],
        }
    )

    new = xr.merge([mednew, simsnew])

    return new


def stack_gases(gas_dict, date=str(date.today()), var="temperature"):
    """Convert RFF climate data into integration-processable files."""

    gases = []
    conversions = {}

    for gas in gas_dict:

        # open climate data
        this = xr.open_dataset(
            "/shares/gcp/integration/rff2/climate/"
            + f"ar6_rff_fair162_control_pulse_{gas}_2020-2030-2040-2050-2060-2070-2080_emis_conc_rf_temp_lambdaeff_ohc_emissions-driven_naturalfix_v5.03_{gas_dict[gas]}.nc"
        )
        conversions[gas] = this.attrs["damages_pulse_conversion"]

        # reformat climate data
        this = this.temperature.astype(np.float32).to_dataset(dim="runtype")
        this = this.rename({"control": f"control_{var}", "pulse": f"pulse_{var}"})
        this = this.expand_dims({"gas": [gas]})

        gases.append(this)

    # save out climate data
    xr.combine_by_coords(gases).to_netcdf(
        "/shares/gcp/integration/rff2/climate/"
        + f"ar6_rff_fair162_control_pulse_all_gases_2020-2030-2040-2050-2060-2070-2080_emis_conc_rf_temp_lambdaeff_ohc_emissions-driven_naturalfix_v5.03_{date}.nc"
    )

    # save out conversions
    conversion = xr.DataArray(
        [i for i in conversions.values()], {"gas": [i for i in conversions.keys()]}
    )

    conversion.to_netcdf(
        f"/shares/gcp/integration/rff2/climate/conversion_v5.03_{date}.nc4"
    )
