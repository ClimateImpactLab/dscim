import xarray as xr

v = 0.21

d = xr.open_zarr(
    f"/shares/gcp/integration/float32/input_data_histclim/coastal_data/coastal_damages_v{v}.zarr"
)

d.sel(adapt_type="optimal", vsl_valuation="global", drop=True).to_zarr(
    f"/shares/gcp/integration/float32/input_data_histclim/coastal_data/coastal_damages_v{v}-global-optimal.zarr",
    consolidated=True,
    mode="w",
)

d.sel(adapt_type="optimal", vsl_valuation="row", drop=True).to_zarr(
    f"/shares/gcp/integration/float32/input_data_histclim/coastal_data/coastal_damages_v{v}-row-optimal.zarr",
    consolidated=True,
    mode="w",
)
