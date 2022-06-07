import xarray as xr
import numpy as np
import pandas as pd
from p_tqdm import p_map
from itertools import product

quantiles = [
    (0, 1),
    (0.05, 0.95),
    (0.01, 0.99),
]

years = range(2020, 2081, 10)


def mask_year(args):

    year, q = args

    ssp_climate = "/shares/gcp/integration/float32/dscim_input_data/climate/AR6/ar6_fair162_sim_and_medianparams_control_pulse_2020-2030-2040-2050-2060-2070-2080_emis_conc_rf_temp_lambdaeff_emissions-driven_naturalfix_v4.0_Jan212022.nc"
    rff_folder = "/shares/gcp/integration/rff/climate/CO2_Fossil/"
    rff_file = "ar6_rff_iter0-19_fair162_CO2_Fossil_control_pulse_{}_temp_v5.02_newformat_Jan72022.nc"
    out_folder = "/shares/gcp/integration/rff/climate/CO2_Fossil/sspclipped/"

    rff = xr.open_dataset(rff_folder + rff_file.format(year))[
        [
            "control_temperature",
            "pulse_temperature",
        ]
    ]

    ssp_min = (
        xr.open_dataset(ssp_climate)
        .sel(pulse_year=year)
        .quantile(q[0], ["simulation"])
        .min("rcp")
    )
    ssp_max = (
        xr.open_dataset(ssp_climate)
        .sel(pulse_year=year)
        .quantile(q[1], ["simulation"])
        .max("rcp")
    )

    for var in rff.keys():

        # index of sim-rff_sps which are out of bounds
        index = (
            rff[var]
            .where(((rff[var] > ssp_max[var]) | (rff[var] < ssp_min[var])), drop=True)
            .to_dataframe()
            .reset_index()
        )
        index = index.loc[~index[var].isnull()]  # [['rff_sp', 'simulation']]

        # replace values of out-of-bounds sim-rff_sps
        rff[var] = xr.where(rff[var] > ssp_max[var], ssp_max[var], rff[var])
        rff[var] = xr.where(rff[var] < ssp_min[var], ssp_min[var], rff[var])

    rff.to_netcdf(
        out_folder
        + rff_file.format(year).replace(".nc", f"_q{q[0]}_q{q[1]}_sspmasked.nc")
    )

    index.to_csv(
        out_folder
        + rff_file.format(year).replace(".nc", f"_q{q[0]}_q{q[1]}_sspmasked_index.csv"),
        index=False,
    )


p_map(mask_year, list(product(years, quantiles)), num_cpus=20)
