import xarray as xr

print("testing message: version jun 25")

def coastal_inputs(
    vsl_valuation,
    adapt_type,
    path = "/shares/gcp/integration/float32/input_data_histclim/coastal_data/",
    v=0.21,
):
    
    d = xr.open_zarr(f"{input_path}/coastal_damages_v{v}.zarr")

    d.sel(adapt_type=adapt_type, vsl_valuation=vsl_valuation, drop=True).to_zarr(
        f"{output_path}/coastal_damages_v{v}-{adapt_type}-{vsl_valuation}.zarr",
        consolidated=True,
        mode="w",
    )