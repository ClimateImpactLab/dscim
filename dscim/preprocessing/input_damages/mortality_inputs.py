import pandas as pd
import xarray as xr
import numpy as np
import glob, os, sys
from pathlib import Path
from p_tqdm import p_map
import seaborn as sns
import matplotlib.pyplot as plt
from dscim.menu.simple_storage import EconVars

## NOTE: currently, mortality damages are per capita but individual components (deaths, costs) are NOT per capita. This will change soon. Confirm with Stefan/Emile whether it's been changed. Date: 15/09/2021

gcm = [
    "ACCESS1-0",
    "CCSM4",
    "GFDL-CM3",
    "IPSL-CM5A-LR",
    "MIROC-ESM-CHEM",
    "bcc-csm1-1",
    "CESM1-BGC",
    "GFDL-ESM2G",
    "IPSL-CM5A-MR",
    "MPI-ESM-LR",
    "BNU-ESM",
    "CNRM-CM5",
    "GFDL-ESM2M",
    "MIROC5",
    "MPI-ESM-MR",
    "CanESM2",
    "CSIRO-Mk3-6-0",
    "inmcm4",
    "MIROC-ESM",
    "MRI-CGCM3",
    "NorESM1-M",
    "surrogate_GFDL-CM3_89",
    "surrogate_GFDL-ESM2G_11",
    "surrogate_CanESM2_99",
    "surrogate_GFDL-ESM2G_01",
    "surrogate_MRI-CGCM3_11",
    "surrogate_CanESM2_89",
    "surrogate_GFDL-CM3_94",
    "surrogate_MRI-CGCM3_01",
    "surrogate_CanESM2_94",
    "surrogate_GFDL-CM3_99",
    "surrogate_MRI-CGCM3_06",
    "surrogate_GFDL-ESM2G_06",
]

gcm.reverse()

##########################
# IR-level VSL
##########################
delta_paths = list(
    Path(f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/").glob(
        "mortality_damages_IR_batch[0-9].nc4"
    )
) + list(
    Path(f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/").glob(
        "mortality_damages_IR_batch[0-9][0-9].nc4"
    )
)
histclim_paths = list(
    Path(
        f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/nosubtract/"
    ).glob("mortality_damages_IR_batch*_histclim.nc4")
)

delta_in = "monetized_damages_vsl_epa_scaled"
delta_out = "monetized_damages_vsl_epa_scaled"
histclim_in = "monetized_deaths_vsl_epa_scaled"
histclim_out = "monetized_deaths_vsl_epa_scaled"
outpath = "/shares/gcp/integration/float32/input_data_histclim/mortality_data/impacts-darwin-montecarlo-damages-vsl-ir-valuation-histclim-delta.zarr"

##########################
# mortality paper, integration paper
##########################
# delta_paths = list(
#     Path(f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/").glob(
#         "mortality_damages_IR_batch[0-9].nc4"
#     )
# ) + list(
#     Path(f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/").glob(
#         "mortality_damages_IR_batch[0-9][0-9].nc4"
#     )
# )
# histclim_paths = list(
#         Path(
#             f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/nosubtract/"
#         ).glob("mortality_damages_IR_batch*_histclim.nc4")
#     )

# delta_in = 'monetized_damages_vly_epa_scaled'
# delta_out = 'delta_monetized_damages_vly_epa_scaled'
# histclim_in = 'monetized_deaths_vly_epa_scaled'
# histclim_out = 'histclim_monetized_deaths_vly_epa_scaled'
# outpath = "/shares/gcp/integration/float32/input_data_histclim/mortality_data/impacts-darwin-montecarlo-damages-vly-ir-valuation-histclim-delta.zarr"

##########################
# EPA deliverables
##########################

# delta_paths = Path(f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/").glob(
#         "mortality_damages_IR_batch*_iso_income.nc4"
#     )
# histclim_paths = Path(f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/nosubtract/").glob(
#         "mortality_damages_IR_batch*_histclim_iso_income.nc4"
#     )

# delta_in = 'monetized_damages_vsl_epa_scaled'
# delta_out = 'delta_hybrid_damages_vsl_epa_scaled'
# histclim_in = 'monetized_deaths_vsl_epa_scaled'
# histclim_out = 'histclim_deaths_vsl_epa_scaled'
# outpath = f"/shares/gcp/integration/float32/input_data_histclim/mortality_data/impacts-darwin-montecarlo-damages-hybrid-histclim-delta.zarr"

##########################
# global average VSL
##########################

# delta_paths = list(
#     Path(
#         "/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/vsl_popavg/"
#     ).glob("mortality_damages_IR_batch[0-9].nc4")
# ) + list(
#     Path(
#         "/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/vsl_popavg/"
#     ).glob("mortality_damages_IR_batch[0-9][0-9].nc4")
# )

# histclim_paths = list(
#     Path(
#         "/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/vsl_popavg/"
#     ).glob("mortality_damages_IR_batch*_histclim.nc4")
# )

# delta_in = "monetized_damages_vsl_epa_popavg"
# delta_out = "monetized_damages_vsl_epa_popavg"
# histclim_in = "monetized_deaths_vsl_epa_popavg"
# histclim_out = "monetized_deaths_vsl_epa_popavg"
# outpath = f"/shares/gcp/integration/float32/input_data_histclim/mortality_data/impacts-darwin-montecarlo-damages-vsl_popavg-histclim-delta.zarr"

##########################
# running the function
##########################

print("# delta files: ", len(delta_paths))
print("# histclim files: ", len(histclim_paths))


def resave_mortality_histclim(i):
    def prep(ds, i=i):
        return ds.sel(gcm=i).drop("gcm")

    ec = EconVars(
        path_econ="/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr"
    )

    delta = xr.open_mfdataset(delta_paths, preprocess=prep, parallel=True)

    histclim = xr.open_mfdataset(histclim_paths, preprocess=prep, parallel=True)

    damages = xr.Dataset(
        {
            delta_out: delta[delta_in],
            # making histclim per capita because deaths are currently *NOT* per capita, whereas damages are
            histclim_out: histclim[histclim_in] / ec.econ_vars.pop.load(),
        }
    ).expand_dims({"gcm": [str(i)]})

    damages = damages.chunk(
        {"batch": 15, "ssp": 1, "model": 1, "rcp": 1, "gcm": 1, "year": 10}
    )
    damages.coords.update({"batch": [f"batch{i}" for i in damages.batch.values]})

    # convert to EPA VSL
    damages = damages * 0.90681089

    if i == "surrogate_GFDL-ESM2G_06":
        damages.to_zarr(
            outpath,
            consolidated=True,
            mode="w",
        )
    else:
        damages.to_zarr(
            outpath,
            consolidated=True,
            append_dim="gcm",
        )

    delta.close()
    histclim.close()
    damages.close()


for i, g in enumerate(gcm):
    print(i)
    resave_mortality_histclim(g)
