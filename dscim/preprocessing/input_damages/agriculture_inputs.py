import os
import numpy as np
from functools import partial
import xarray as xr
from p_tqdm import p_map
from pathlib import Path
from dask.distributed import Client
from dscim.utils.calculate_damages import compute_ag_damages
from dscim.menu.simple_storage import EconVars
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, progress

print("testing message: version jun 25")
# DO NOT USE A DASK CLIENT FOR THIS SCRIPT. IT'LL MESS UP P_MAP.

def agriculture_inputs(input_root = "/shares/gcp/outputs/agriculture/impacts-mealy/gdp_weights_delta", 
                       output_root = "/shares/gcp/integration/float32/input_data_histclim/ag_data/gdp_weights_delta/test", 
                       path_econ = "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr", 
                       versions = [""]):
    # versions = [""]
    # ["all_topcodes", 'v0.0_topcodes', 'v0.0_elasticities', 'v0.0_globe_market', 'v0.0_iso_market', 'v0.0_no_CO2']

    paths = {
        f"{input_root}/{k}": f"{output_root}"
        for k in versions
    }

    # if __name__ == "__main__":

    ec = EconVars(
        path_econ=path_econ
    )

    scalar = float(1 - 0.55)
    print(scalar)

    for input_path, save_path in paths.items():

        compute_ag_damages(
            input_path=input_path,
            pop=ec.econ_vars.pop,
            topcode="agshare_10",
            integration=True,
            save_path=save_path,
            vars=["wc_no_reallocation"],
            scalar=scalar,
        )

    print("COMPLETED.")
