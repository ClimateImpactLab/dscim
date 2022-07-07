import xarray as xr
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from dscim.utils.rff import rff_damage_functions

USER = os.getenv("USER")
eta_rhos = {
    "2.0": "0.0",
    "1.016010255": "9.149608e-05",
    "1.244459066": "0.00197263997",
    "1.421158116": "0.00461878399",
    "1.567899395": "0.00770271076",
}

sectors = [
    # "CAMEL_m4_c0.21.4",
    # "AMEL_m4",
    "CAMEL_m1_c0.20",
    # "mortality_v4",
    # "energy",
    # "labor",
    # "agriculture",
]

USA = True

if USA == True:
    sectors = [f"{i}_USA" for i in sectors]
    ssp_gdp = "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39_USA.zarr"
    rff_gdp = f"/shares/gcp/integration/rff/socioeconomics/rff_USA_socioeconomics.nc4"
else:
    ssp_gdp = "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr"
    rff_gdp = "/shares/gcp/integration/rff/socioeconomics/rff_global_socioeconomics.nc4"

rff_damage_functions(
    sectors=sectors,
    eta_rhos=eta_rhos,
    USA=USA,
    ssp_gdp=ssp_gdp,
    rff_gdp=rff_gdp,
    recipes_discs=[("adding_up", "constant"), ("risk_aversion", "euler_ramsey")],
    in_library="/mnt/CIL_integration/damage_function_library/damage_function_library_ssp",
    out_library="/mnt/CIL_integration/damage_function_library/damage_function_library_rff_test",
    runid_path="/shares/gcp/integration/rff2/rffsp_fair_sequence.nc",
    weights_path="/shares/gcp/integration/rff/damage_function_weights/damage_function_weights3.nc4",
)
