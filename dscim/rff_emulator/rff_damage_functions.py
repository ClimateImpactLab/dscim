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
