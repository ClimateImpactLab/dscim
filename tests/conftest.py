import os
import pytest

from dscim.menu.simple_storage import Climate, EconVars
from dscim.menu.baseline import Baseline
from dscim.menu.risk_aversion import RiskAversionRecipe
from dscim.menu.equity import EquityRecipe
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import xarray as xr

# @MIKE: CHANGE ME
data_dir = "."


@pytest.fixture(scope="module")
def climate():
    datadir = os.path.join(os.path.dirname(__file__), "data", "climate")
    climate = Climate(
        gmsl_path=os.path.join(data_dir, "gmsl.csv"),
        gmst_path=os.path.join(datadir, "gmst.csv"),
        gmst_fair_path=os.path.join(datadir, "fair_temps_sims.nc4"),
        gmsl_fair_path=os.path.join(datadir, "fair_slr_sims.nc4"),
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


all_menu_classes = [Baseline, RiskAversionRecipe, EquityRecipe]


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


@pytest.fixture
def weights_unclean_fixture(tmp_path):
    """
    Create and save out a dummy uncleaned weights file
    """
    d = Path(tmp_path) / "clean_root"
    d.mkdir()

    filename = str(d / "emulate-fivebean-1234.csv")

    alpha = list(
        product(
            np.arange(2010, 2021, 5),
            ["alpha"],
            ["high", "low"],
            ["SSP2", "SSP3", "SSP4"],
            [1],
        )
    )
    error = list(product(np.arange(2010, 2021, 5), ["USA", "ARG"], ["error"], [1]))
    alpha_file = pd.DataFrame(alpha, columns=["year", "param", "model", "ssp", "value"])
    alpha_file["name"] = alpha_file.apply(
        lambda x: "/".join([":".join([str(x.year), x.model]), x.ssp]), axis=1
    )
    alpha_file = alpha_file.drop(columns=["model", "ssp"])

    error_file = pd.DataFrame(error, columns=["year", "country", "param", "value"])
    error_file["name"] = error_file.apply(
        lambda x: ":".join([x.country, str(x.year)]), axis=1
    )
    error_file = error_file.drop(columns=["country"])
    out = pd.concat([alpha_file, error_file]).reset_index().drop(columns="index")

    out.to_csv(filename, index=False)
    with open(filename, "r+") as f:
        lines = f.readlines()
        lines.insert(0, "#\n" * 9)
        f.seek(0)
        f.writelines(lines)


@pytest.fixture
def save_ssprff_econ(tmp_path):
    """
    Create and save dummy ssp/rff socioeconomics files
    """
    d = tmp_path / "econ"
    d.mkdir(exist_ok=True)

    ssp_econ = xr.Dataset(
        {
            "gdp": (["ssp", "region", "model", "year"], np.ones((1, 2, 2, 4))),
            "gdppc": (["ssp", "region", "model", "year"], np.ones((1, 2, 2, 4))),
            "pop": (["ssp", "region", "model", "year"], np.ones((1, 2, 2, 4))),
        },
        coords={
            "ssp": (["ssp"], ["SSP3"]),
            "region": (["region"], ["ZWE.test_region", "USA.test_region"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "year": (["year"], [2021, 2022, 2023, 2099]),
        },
    )

    rff_econ = xr.Dataset(
        {
            "pop": (["region", "year", "runid"], np.ones((1, 5, 5))),
            "gdp": (["region", "year", "runid"], np.ones((1, 5, 5))),
        },
        coords={
            "region": (["region"], ["world"]),
            "year": (["year"], [2021, 2022, 2023, 2099, 2100]),
            "runid": (["runid"], np.arange(1, 6)),
        },
    )

    ssp_econ.to_zarr(d / "integration-econ-bc39.zarr")
    rff_econ.to_netcdf(d / "rff_global_socioeconomics.nc4")


# ---------------------------------------------------------------------------
# Geography fixtures (tests/test_geography.py).
# ---------------------------------------------------------------------------

# Region subset shared by the geography tests. INDIVIDUAL_REGION is one of
# them, so a single-region run has a counterpart inside the all-region run.
GEOGRAPHY_REGIONS = [
    "IND.21.317.1249",
    "CAN.2.33.913",
    "USA.14.608",
    "EGY.11",
    "SDN.4.11.50.164",
    "NGA.25.510",
    "SAU.7",
    "RUS.16.430.430",
    "SOM.2.5",
]
INDIVIDUAL_REGION = "USA.14.608"


@pytest.fixture(scope="module")
def menu_instance_with_geography(menu_class, discount_types, econ, climate):
    """Menu instance built exactly like ``menu_instance`` with the geography
    keywords spelled out at their defaults."""
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
        ],
        ce_path=os.path.join(datadir, "CEs"),
        subset_dict={"ssp": ["SSP2", "SSP3", "SSP4"], "region": GEOGRAPHY_REGIONS},
        formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
        extrap_formula=None,
        fair_aggregation=["median_params", "ce", "mean"],
        weitzman_parameter=[0.1],
        geography="globe",
        country_mapping_path=None,
        individual_region=None,
    )


@pytest.fixture(scope="module")
def geography_menu_factory(econ, climate):
    """Build a menu with the standard test configuration and any overrides.

    ``all_ssps=True`` drops the SSP subset so SSP1 and SSP5 are present, which
    is what exercises the illegal-combination masking.
    """
    datadir = os.path.join(os.path.dirname(__file__), "data")

    def _build(
        menu_class=Baseline,
        discounting_type="euler_ramsey",
        all_ssps=False,
        ce_path=None,
        **overrides,
    ):
        subset = {"region": GEOGRAPHY_REGIONS}
        if not all_ssps:
            subset["ssp"] = ["SSP2", "SSP3", "SSP4"]
        kwargs = dict(
            sector_path=[{"dummy_sector": os.path.join(datadir, "damages")}],
            save_path=None,
            discrete_discounting=True,
            econ_vars=econ,
            climate_vars=climate,
            fit_type="ols",
            variable=[{"dummy_sector": "damages"}],
            sector="dummy_sector",
            discounting_type=discounting_type,
            ext_method="global_c_ratio",
            save_files=[],
            ce_path=ce_path or os.path.join(datadir, "CEs"),
            subset_dict=subset,
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            extrap_formula=None,
            fair_aggregation=["median_params", "ce", "mean"],
            weitzman_parameter=[0.1],
        )
        kwargs.update(overrides)
        return menu_class(**kwargs)

    return _build


@pytest.fixture(scope="module")
def country_mapping_csv(tmp_path_factory):
    """ISO -> MatchedISO csv for GEOGRAPHY_REGIONS; SOM has no MatchedISO so it
    lands in the "nopop" bucket."""
    path = tmp_path_factory.mktemp("country_mapping") / "matched_CIL_ISO.csv"
    isos = ["IND", "CAN", "USA", "EGY", "SDN", "NGA", "SAU", "RUS", "SOM"]
    pd.DataFrame(
        {"ISO": isos, "MatchedISO": isos[:-1] + [np.nan]},
    ).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def legacy_ce_paths(tmp_path_factory):
    """Three ce_path directories built from tests/data/CEs:
    ``legacy`` has only the unsuffixed adding_up stores (dscim <= 0.7.0 naming),
    ``both`` has the unsuffixed and the eta-suffixed stores, ``empty`` has
    neither."""
    import shutil

    src = os.path.join(os.path.dirname(__file__), "data", "CEs")
    eta = 1.421158116
    root = tmp_path_factory.mktemp("adding_up_stores")
    legacy, both, empty = root / "legacy", root / "both", root / "empty"
    for d in (legacy, both, empty):
        d.mkdir()
    for red in ("cc", "no_cc"):
        suffixed = f"adding_up_{red}_eta{eta}.zarr"
        shutil.copytree(os.path.join(src, suffixed), legacy / f"adding_up_{red}.zarr")
        shutil.copytree(os.path.join(src, suffixed), both / f"adding_up_{red}.zarr")
        shutil.copytree(os.path.join(src, suffixed), both / suffixed)
    return {
        "suffixed": src,
        "legacy": str(legacy),
        "both": str(both),
        "empty": str(empty),
    }
