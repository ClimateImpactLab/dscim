import xarray as xr

d = xr.open_zarr(
    "/shares/gcp/integration/float32/input_data_histclim/coastal_data/coastal_damages_v0.19.zarr"
)

d.sel(adapt_type="optimal").drop("adapt_type").to_zarr(
    "/shares/gcp/integration/float32/input_data_histclim/coastal_data/coastal_damages_v0.19-optimal.zarr",
    consolidated=True,
    mode="w",
)
