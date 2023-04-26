import xarray as xr
import pandas as pd
import numpy as np
import os
import sys
import yaml
from p_tqdm import p_map
from itertools import product

USER = os.getenv("USER")

# parameters
dir = "/shares/gcp/integration/rff/climate/masks/CO2_Fossil"

stack_list = [["gdppc", "emissions"], ["gdppc", "emissions", "climate"]]

for stack in stack_list:

    # open all the relevant masks
    masks = p_map(xr.open_dataset, [f"{dir}/{s}_based_masks.nc4" for s in stack])

    # find the union of the first two
    union_mask = xr.ufuncs.logical_and(*masks[:2])
    if len(masks) > 2:
        # for each further mask, cumulatively find union with preceding masks
        for i in masks[2:]:
            union_mask = xr.ufuncs.logical_and(union_mask, i)

    union_mask.to_netcdf(f"{dir}/{'_'.join(stack)}_based_masks.nc4")
