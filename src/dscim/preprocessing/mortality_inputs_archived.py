import xarray as xr
from dscim.utils.functions import gcms
from pathlib import Path
from dscim.preprocessing.input_damages import prep_mortality_damages

#########################
# MORTALITY V4
#########################

prep_mortality_damages(
    gcms=gcms(),
    paths=dict(
        delta_deaths=[
            Path(
                f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/vsl_popavg/mortality_damages_IR_batch{i}.nc4"
            )
            for i in range(15)
        ],
        delta_costs=[
            Path(
                f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/mortality_damages_IR_batch{i}.nc4"
            )
            for i in range(15)
        ],
        histclim_deaths=[
            Path(
                f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/vsl_popavg/mortality_damages_IR_batch{i}_histclim.nc4"
            )
            for i in range(15)
        ],
    ),
    vars=dict(
        delta_deaths="monetized_deaths_vsl_epa_popavg",
        delta_costs="monetized_costs_vsl_epa_scaled",
        histclim_deaths="monetized_deaths_vsl_epa_popavg",
    ),
    outpath="/shares/gcp/integration/float32/sectoral_ir_damages/mortality_data/impacts-darwin-montecarlo-damages-v4.zarr",
)

#########################
# MORTALITY V5
#########################

# prep_mortality_damages(
#     gcms=gcms(),
#     paths=dict(
#     delta_deaths = [
#         Path(f"/shares/gcp/estimation/mortality/release_2020/data/3_valuation/impact_region/row/mortality_damages_IR_batch{i}.nc4")
#         for i in range(15)
#     ],
#     delta_costs = [
#         Path(f"/shares/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/mortality_damages_IR_batch{i}.nc4")
#         for i in range(15)
#     ],
#     histclim_deaths = [
#         Path(f"/shares/gcp/estimation/mortality/release_2020/data/3_valuation/impact_region/row/mortality_damages_IR_batch{i}.nc4")
#         for i in range(15)
#     ],
# ),
#     vars=dict(
#     delta_deaths = 'monetized_deaths_vsl_epa_scaled',
#     delta_costs = 'monetized_costs_vsl_epa_scaled',
#     histclim_deaths = 'monetized_histclim_deaths_vsl_epa_scaled',
# ),
#     outpath="/shares/gcp/integration/float32/sectoral_ir_damages/mortality_data/impacts-darwin-montecarlo-damages-v5.zarr",
# )
