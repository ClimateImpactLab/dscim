import xarray as xr
import pandas as pd
import numpy as np
import os, sys
from numpy.testing import assert_allclose


def prep_rff_socioeconomics(
    inflation_path="/mnt/Global_ACP/MORTALITY/release_2020/data/3_valuation/inputs/adjustments/fed_income_inflation.csv",
    rff_path="/shares/gcp/integration/rff/socioeconomics/rff-sp_socioeconomics_all_runs_feather_files.nc",
    runid_path="/shares/gcp/integration/rff2/rffsp_fair_sequence.nc",
    out_path="/shares/gcp/integration/rff/socioeconomics/",
    USA=False,
):

    # Load Fed GDP deflator
    fed_gdpdef = pd.read_csv(inflation_path).set_index("year")["gdpdef"].to_dict()

    # transform 2011 USD to 2019 USD
    inflation_adj = fed_gdpdef[2019] / fed_gdpdef[2011]

    # read in RFF data
    socioec = xr.open_dataset(rff_path)

    if USA == False:
        print("Summing to globe.")
        socioec = socioec.sum("Country")
    else:
        print("USA output.")
        socioec = socioec.sel(Country="USA", drop=True)

    # interpolate with log -> linear interpolation -> exponentiate
    socioec = np.exp(
        np.log(socioec).interp({"Year": range(2020, 2301, 1)}, method="linear")
    ).rename({"runid": "rff_sp", "Year": "year", "Pop": "pop", "GDP": "gdp"})

    socioec["pop"] = socioec["pop"] * 1000
    socioec["gdp"] = socioec["gdp"] * 1e6 * inflation_adj

    # read in RFF runids and update coordinates with them
    run_id = xr.open_dataset(runid_path)
    socioec = socioec.sel(rff_sp=run_id.rff_sp, drop=True)

    if USA == False:
        socioec.expand_dims({"region": ["world"]}).to_netcdf(
            f"{out_path}/rff_global_socioeconomics.nc4"
        )
    else:
        socioec.expand_dims({"region": ["USA"]}).to_netcdf(
            f"{out_path}/rff_USA_socioeconomics.nc4"
        )
