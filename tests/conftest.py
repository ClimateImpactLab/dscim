import os
import pytest

from dscim.menu.simple_storage import Climate, EconVars
from dscim.menu.baseline import Baseline
from dscim.menu.risk_aversion import RiskAversionRecipe

# @MIKE: CHANGE ME
data_dir = "."


@pytest.fixture(scope="module")
def climate():
    datadir = os.path.join(os.path.dirname(__file__), "data", "climate")
    climate = Climate(
        gmsl_path=os.path.join(data_dir, "gmsl.csv"),
        gmst_path=os.path.join(datadir, "gmst.csv"),
        gmst_fair_path=os.path.join(datadir, "fair_temps_sims.nc4"),
        gmsl_fair_path=os.path.join(datadir, "fair_slr_sims.zarr"),
        damages_pulse_conversion_path=os.path.join(
            datadir,
            "scenario_rcp45-rcp85-ssp245-ssp460-ssp370_CO2_Fossil-CH4-N2O_conversion_pulseyears_2020-2100_5yrincrements_v3.0_newformat.nc",
        ),
        ecs_mask_path=None,
        ecs_mask_name=None,
        pulse_year=2020,
        base_period=[2001, 2010],
        emission_scenarios=["ssp370"],
        gases=["CO2_Fossil", "CH4", "N2O"],
    )

    return climate


@pytest.fixture(scope="module")
def econ():
    datadir = os.path.join(os.path.dirname(__file__), "data/zarrs/all_ssps.zarr")
    econvars = EconVars(path_econ=datadir)

    return econvars


all_discount_types = [
    "constant",
    "constant_model_collapsed",
    "naive_ramsey",
    "naive_gwr",
    "euler_ramsey",
    "euler_gwr",
    "gwr_gwr",
]


@pytest.fixture(params=all_discount_types, scope="module")
def discount_types(request):
    return request.param


all_menu_classes = [Baseline, RiskAversionRecipe]


@pytest.fixture(params=all_menu_classes, scope="module")
def menu_class(request):
    return request.param


@pytest.fixture(scope="module")
def menu_instance(menu_class, discount_types, econ, climate):
    datadir = os.path.join(os.path.dirname(__file__), "data")
    yield menu_class(
        sector_path=[{"dummy_sector": os.path.join(datadir, "damages")}],
        save_path=None,
        discrete_discounting=True,
        econ_vars=econ,
        climate_vars=climate,
        fit_type="ols",
        variable=[{"dummy_sector": "damages"}],
        sector="dummy_sector",
        discounting_type=discount_types,
        ext_method="global_c_ratio",
        save_files=[
            "damage_function_points",
            "global_consumption",
            "damage_function_coefficients",
            "damage_function_fit",
            "marginal_damages",
            "discount_factors",
            "uncollapsed_sccs",
            "scc",
        ],
        ce_path=os.path.join(datadir, "CEs"),
        subset_dict={
            "ssp": ["SSP2", "SSP3", "SSP4"],
            "region": [
                "IND.21.317.1249",
                "CAN.2.33.913",
                "USA.14.608",
                "EGY.11",
                "SDN.4.11.50.164",
                "NGA.25.510",
                "SAU.7",
                "RUS.16.430.430",
                "SOM.2.5",
            ],
        },
        formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
        extrap_formula=None,
        fair_aggregation=["median_params", "ce", "mean"],
        weitzman_parameter=[0.1],
    )
