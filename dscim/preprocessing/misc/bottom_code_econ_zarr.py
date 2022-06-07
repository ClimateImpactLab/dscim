import xarray as xr
import numpy as np

dir_nc4 = "/shares/gcp/integration/float32/dscim_input_data/econvars/vsl"
dir_zarr = "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs"
bottom_code = 39.39265060424805

nc4s = []
for i in range(1, 6):
    nc4 = xr.open_dataset(f"{dir_nc4}/SSP{i}.nc4")
    nc4 = nc4.assign_coords(ssp=f"SSP{i}")
    nc4 = nc4.expand_dims(dim="ssp")
    nc4s.append(nc4)

all_ssps = xr.combine_by_coords(nc4s)[["gdp", "pop"]]

all_ssps["gdppc"] = np.maximum(all_ssps.gdp / all_ssps.pop, bottom_code)

all_ssps.to_zarr(
    "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr",
    consolidated=True,
    mode="w",
)
