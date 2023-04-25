from itertools import product
import numpy as np
import pandas as pd
from dscim.utils.rff import (
    solve_optimization,
    process_ssp_sample,
    process_rff_sample,
    clean_simulation,
    clean_error,
    weight_df,
    rff_damage_functions,
    prep_rff_socioeconomics,
)
import pytest
import os
from pathlib import Path
import xarray as xr


@pytest.fixture
def ssp_df_in_fixture(tmp_path):
    """
    Create fake SSP input data for tests and save to csv
    """
    HEADER = """test
    header
    -
    -
    -
    -
    -
    -
    -
    -
    --
    """

    ssp_df_in = pd.DataFrame(
        data={
            "year": [2005] * 4
            + [2015] * 3
            + [2020] * 4
            + [2005] * 4
            + [2015] * 4
            + [2020] * 4,
            "model": ["low"] * 11 + ["high"] * 12,
            "scenario": ["SSP2"] * 2
            + ["SSP3"] * 2
            + ["SSP2"]
            + ["SSP3"] * 2
            + ["SSP2", "SSP2", "SSP3", "SSP3"] * 4,
            "iso": ["ZWE", "USA"] * 2 + ["USA"] + ["ZWE", "USA"] * 9,
            "value": [1] * 23,
        }
    )
    with open(os.path.join(tmp_path, "ssp_in.csv"), "w") as ssp_in:
        ssp_in.write(HEADER.strip() + "\n")
        ssp_df_in.to_csv(ssp_in, index=False)


def test_solve_optimization(
    tmp_path,
    ssp_df_in_fixture,
):
    """
    Test that solve optimization correctly applies the emulation scheme to produce a set of SSP-growth model weights and country-level errors
    """
    rff_df_in = pd.DataFrame(
        data={
            "iso": ["USA"],
            "year": [2020],
            "value": [1],
            "loginc": [np.log(1)],
            "isoyear": ["USA:2020"],
            "Pop": [1],
            "GDP": [1],
        }
    )
    ssp_df = process_ssp_sample(os.path.join(tmp_path, "ssp_in.csv"))
    ds_out_expected = pd.DataFrame(
        data={
            "year": [2020] * 6,
            "param": ["error"] * 2 + ["alpha"] * 4,
            "name": [
                "ZWE:2020",
                "USA:2020",
                "2020:low/SSP2",
                "2020:low/SSP3",
                "2020:high/SSP2",
                "2020:high/SSP3",
            ],
            "value": [0.0] * 2 + [1.0] + [0.0] * 3,
        }
    )
    ds_out_actual = solve_optimization(ssp_df, rff_df_in)
    pd.testing.assert_frame_equal(
        ds_out_actual.reset_index(drop=True), ds_out_expected.reset_index(drop=True)
    )


def test_solve_optimization_exception(
    tmp_path,
    ssp_df_in_fixture,
    capsys,
):
    """
    Test that solve optimization returns an exception when countires with missing value for any SSP-growth model occured in the loop
    """
    rff_df_in = pd.DataFrame(
        data={
            "iso": ["USA", "ZWE"],
            "year": [2015] * 2,
            "value": [1] * 2,
            "loginc": [np.log(1)] * 2,
            "isoyear": ["USA:2015", "ZWE:2015"],
            "Pop": [1] * 2,
            "GDP": [1] * 2,
            "weight": [1] * 2,
        }
    )
    ssp_df = process_ssp_sample(os.path.join(tmp_path, "ssp_in.csv"))
    solve_optimization(ssp_df, rff_df_in)
    captured = capsys.readouterr()
    assert "Exception! Keep going.." in captured.out


def test_process_ssp_sample(
    tmp_path,
    ssp_df_in_fixture,
):
    """
    Test that process ssp sample correctly cleans GDP per capita out of the SSP input file
    """
    ds_out_expected = pd.DataFrame(
        data={
            "year": [2015] * 3 + [2020] * 4 + [2015] * 4 + [2020] * 4,
            "model": ["low"] * 7 + ["high"] * 8,
            "scenario": ["SSP2"] + ["SSP3"] * 2 + ["SSP2", "SSP2", "SSP3", "SSP3"] * 3,
            "iso": ["USA"] + ["ZWE", "USA"] * 7,
            "value": [1] * 15,
            "loginc": [np.log(1)] * 15,
            "isoyear": ["USA:2015"]
            + ["ZWE:2015", "USA:2015"]
            + ["ZWE:2020", "USA:2020"] * 2
            + ["ZWE:2015", "USA:2015"] * 2
            + ["ZWE:2020", "USA:2020"] * 2,
            "yearscen": ["2015:low/SSP2"]
            + ["2015:low/SSP3"] * 2
            + ["2020:low/SSP2"] * 2
            + ["2020:low/SSP3"] * 2
            + ["2015:high/SSP2"] * 2
            + ["2015:high/SSP3"] * 2
            + ["2020:high/SSP2"] * 2
            + ["2020:high/SSP3"] * 2,
        }
    )
    ds_out_actual = process_ssp_sample(os.path.join(tmp_path, "ssp_in.csv"))
    pd.testing.assert_frame_equal(
        ds_out_actual.reset_index(drop=True), ds_out_expected.reset_index(drop=True)
    )


def test_process_rff_sample(
    tmp_path,
    ssp_df_in_fixture,
):
    """
    Test that process rff sample correctly cleans RFF input data and passes the cleaned data to the emulation scheme to produce a set of SSP-growth model weights and country-level errors
    """
    rff_raw_in = pd.DataFrame(
        data={
            "Country": ["ZWE", "USA"],
            "Year": [2020] * 2,
            "Pop": [1000, 1000],
            "GDP": [98.71 / 88.58, 98.71 / 88.58],
        }
    )
    rff_raw_in.to_feather(os.path.join(tmp_path, "run_1.feather"))
    ds_out_expected = pd.DataFrame(
        data={
            "year": [2015] * 6 + [2020] * 6,
            "param": ["error"] * 2 + ["alpha"] * 4 + ["error"] * 2 + ["alpha"] * 4,
            "name": ["USA:2015"]
            + ["ZWE:2015"]
            + ["2015:low/SSP2"]
            + ["2015:low/SSP3"]
            + ["2015:high/SSP2"]
            + ["2015:high/SSP3"]
            + ["ZWE:2020"]
            + ["USA:2020"]
            + ["2020:low/SSP2"]
            + ["2020:low/SSP3"]
            + ["2020:high/SSP2"]
            + ["2020:high/SSP3"],
            "value": [0.0] * 2 + [1.0] + [0.0] * 5 + [1.0] + [0.0] * 3,
        }
    )
    test_HEADER = """test
    header
    """
    process_rff_sample(
        i=1,
        rffpath=tmp_path,
        ssp_df=process_ssp_sample(os.path.join(tmp_path, "ssp_in.csv")),
        outdir=tmp_path,
        HEADER=test_HEADER,
    )
    ds_out_actual = pd.read_csv(os.path.join(tmp_path, "emulate-1.csv"), skiprows=2)
    pd.testing.assert_frame_equal(
        ds_out_actual.reset_index(drop=True), ds_out_expected.reset_index(drop=True)
    )


def test_clean_weights_alpha(tmp_path, weights_unclean_fixture):
    """
    Test that clean weights correctly cleans alpha and error out of the weights file
    """
    alpha = list(
        product(
            np.arange(2010, 2021, 5),
            ["alpha"],
            ["OECD Env-Growth", "IIASA GDP"],
            ["SSP2", "SSP3", "SSP4"],
            [1],
        )
    )
    alpha = pd.DataFrame(alpha, columns=["year", "var", "model", "ssp", "value"])
    alpha["rff_sp"] = 1234
    out_expected = (
        alpha.set_index(["model", "ssp", "rff_sp", "year"])
        .drop(columns="var")
        .to_xarray()["value"]
    )
    out_actual = clean_simulation(1234, str(Path(tmp_path) / "clean_root"))

    xr.testing.assert_equal(out_actual, out_expected)


def test_clean_weights_error(tmp_path, weights_unclean_fixture):
    """
    Test that clean weights correctly cleans alpha and error out of the weights file
    """
    error = list(product(np.arange(2010, 2021, 5), ["USA", "ARG"], ["error"], [1]))
    error = pd.DataFrame(error, columns=["year", "iso", "var", "value"])
    error["rff_sp"] = 1234
    out_expected = (
        error.set_index(["iso", "year", "rff_sp"])
        .drop(columns="var")
        .to_xarray()["value"]
    )
    out_actual = clean_error(1234, str(Path(tmp_path) / "clean_root"))

    xr.testing.assert_equal(out_actual, out_expected)


def test_weight_df(tmp_path):
    """
    Test that weight_df correctly weights SSP damage functions to produce RFF damage functions
    """
    d = tmp_path / "weighting"
    d.mkdir()

    sector = "sector_dummy"
    pulse_year = "2020"
    recipe = "euler_ramsey"
    disc = "risk_aversion"
    eta_rho = [1, 2]
    file = "damage_functions"
    mask = "unmasked"

    df_in_dir = d / sector / pulse_year / mask
    df_out_dir = d / "out" / sector / pulse_year / mask

    df_in_file = (
        df_in_dir / f"{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_{file}.nc4"
    )
    outfile_path = (
        df_out_dir / f"{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_{file}.nc4"
    )
    fractional_path = (
        df_out_dir
        / f"{recipe}_{disc}_eta{eta_rho[0]}_rho{eta_rho[1]}_fractional_{file}.nc4"
    )

    Path(df_in_dir).mkdir(parents=True, exist_ok=True)
    Path(df_out_dir).mkdir(parents=True, exist_ok=True)
    df_in = xr.Dataset(
        {
            "anomaly": (
                ["discount_type", "ssp", "model", "year"],
                np.ones((1, 2, 2, 3)),
            ),
        },
        coords={
            "discount_type": (["discount_type"], ["euler_ramsey"]),
            "ssp": (["ssp"], ["SSP3", "SSP4"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "year": (["year"], [2021, 2022, 2099]),
        },
    )
    df_in.to_netcdf(df_in_file)

    # Relative gdp of 2100-2300 as a portion of 2099 gdp to extrapolate damage functions
    factors = xr.Dataset(
        {
            "anomaly": (
                ["discount_type", "runid", "year"],
                np.ones((1, 2, 3)),
            ),
        },
        coords={
            "discount_type": (["discount_type"], ["euler_ramsey"]),
            "runid": (["runid"], [5000, 9999]),
            "year": (["year"], [2100, 2101, 2102]),
        },
    )

    out_expected = xr.Dataset(
        {
            "anomaly": (
                ["discount_type", "year", "runid"],
                np.ones((1, 6, 2)),
            ),
        },
        coords={
            "discount_type": (["discount_type"], ["euler_ramsey"]),
            "year": (["year"], [2021, 2022, 2099, 2100, 2101, 2102]),
            "runid": (["runid"], [5000, 9999]),
        },
    )

    # Dataset of damage functions as a fraction of gdp
    out_expected_fractional = xr.Dataset(
        {
            "anomaly": (["discount_type", "year"], np.ones((1, 3))),
        },
        coords={
            "discount_type": (["discount_type"], ["euler_ramsey"]),
            "year": (["year"], [2021, 2022, 2099]),
        },
    )

    weight_df(
        sector,
        eta_rho,
        recipe,
        disc,
        file,
        d,
        d / "out",
        1,
        2,
        0.5,
        factors,
        pulse_year,
        True,
        mask="unmasked",
    )

    xr.testing.assert_equal(out_expected, xr.open_dataset(outfile_path))
    xr.testing.assert_equal(out_expected_fractional, xr.open_dataset(fractional_path))


def test_rff_damage_functions(tmp_path, save_ssprff_econ):
    """
    Test that rff_damage_functions correctly runs weight_df to produce RFF damage functions
    """
    d = tmp_path / "weighting"
    d.mkdir(exist_ok=True)

    rff = d / "rff"
    rff.mkdir(exist_ok=True)

    ssp = d / "ssp"
    ssp.mkdir(exist_ok=True)

    sector = "dummy_sector"
    eta_rhos = [[1.0, 2.0]]
    USA = False

    weights = xr.Dataset(
        {
            "value": (["model", "ssp", "rff_sp", "year"], np.ones((2, 1, 5, 4))),
        },
        coords={
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "ssp": (["ssp"], ["SSP3"]),
            "rff_sp": (["rff_sp"], np.arange(1, 6)),
            "year": (["year"], [2021, 2022, 2023, 2099]),
        },
    )

    weights.to_netcdf(d / "damage_function_weights.nc4")

    runids = xr.Dataset(
        {
            "simulation": (["runid"], np.arange(4, -1, -1)),
            "rff_sp": (["runid"], np.arange(1, 6)),
        },
        coords={
            "runid": (["runid"], np.arange(1, 6)),
        },
    )

    runids.to_netcdf(d / "dummy_runids.nc")

    df_in = xr.Dataset(
        {
            "anomaly": (
                ["discount_type", "ssp", "model", "year"],
                np.ones((1, 2, 2, 3)),
            ),
        },
        coords={
            "discount_type": (["discount_type"], ["euler_ramsey"]),
            "ssp": (["ssp"], ["SSP3", "SSP4"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
            "year": (["year"], [2021, 2022, 2099]),
        },
    )

    (ssp / sector / "2020" / "unmasked").mkdir(parents=True, exist_ok=True)
    df_in.to_netcdf(
        ssp
        / sector
        / "2020"
        / "unmasked"
        / "risk_aversion_euler_ramsey_eta1.0_rho2.0_damage_function_coefficients.nc4"
    )

    rff_damage_functions(
        sectors=[sector],
        eta_rhos=eta_rhos,
        USA=USA,
        ssp_gdp=tmp_path / "econ" / "integration-econ-bc39.zarr",
        rff_gdp=tmp_path / "econ" / "rff_global_socioeconomics.nc4",
        recipes_discs=[
            ("risk_aversion", "euler_ramsey"),
        ],
        in_library=ssp,
        out_library=rff,
        runid_path=d / "dummy_runids.nc",
        weights_path=d,
        pulse_year=2020,
        mask="unmasked",
    )

    out_actual = xr.open_dataset(
        rff
        / sector
        / "2020"
        / "unmasked"
        / "risk_aversion_euler_ramsey_eta1.0_rho2.0_damage_function_coefficients.nc4"
    )
    out_expected = xr.Dataset(
        {
            "anomaly": (
                ["discount_type", "year", "runid"],
                np.ones((1, 4, 5)),
            ),
        },
        coords={
            "discount_type": (["discount_type"], ["euler_ramsey"]),
            "runid": (["runid"], np.arange(1, 6)),
            "rff_sp": (["runid"], np.arange(1, 6)),
            "year": (["year"], [2021, 2022, 2099, 2100]),
        },
    )

    xr.testing.assert_equal(out_actual, out_expected)


@pytest.mark.parametrize("USA", [True, False])
def test_prep_rff_socioeconomics(tmp_path, USA):
    """
    Test that rff socioeconomics generation is correctly interpolating and deflating
    """
    # May want to use "real" numbers for this test, since there is some more intensive calculation
    deflator_file = pd.DataFrame({"year": [2011, 2019], "gdpdef": [1, 2]})
    deflator_file.to_csv(tmp_path / "fed_income_inflation.csv")

    rff_econ = xr.Dataset(
        {
            "Pop": (["runid", "Country", "Year"], np.ones((5, 2, 5))),
            "GDP": (["runid", "Country", "Year"], np.ones((5, 2, 5))),
        },
        coords={
            "Country": (["Country"], ["USA", "ZWE"]),
            "Year": (["Year"], [2021, 2022, 2023, 2099, 2100]),
            "runid": (["runid"], np.arange(1, 6)),
        },
    )

    rff_econ.to_netcdf(tmp_path / "rff-sp_socioeconomics_all_runs_feather_files.nc")

    runids = xr.Dataset(
        {
            "simulation": (["runid"], np.arange(4, -1, -1)),
            "rff_sp": (["runid"], np.arange(1, 6)),
        },
        coords={
            "runid": (["runid"], np.arange(1, 6)),
        },
    )

    runids.to_netcdf(tmp_path / "dummy_runids.nc")

    prep_rff_socioeconomics(
        tmp_path / "fed_income_inflation.csv",
        tmp_path / "rff-sp_socioeconomics_all_runs_feather_files.nc",
        tmp_path / "dummy_runids.nc",
        tmp_path,
        USA,
    )
    file_USA = "USA" if USA else "global"
    coord_USA = "USA" if USA else "world"
    gdp_USA = 1 if USA else 2

    out_expected = xr.Dataset(
        {
            "pop": (["region", "runid", "year"], np.ones((1, 5, 5)) * gdp_USA * 1e3),
            "gdp": (["region", "runid", "year"], np.ones((1, 5, 5)) * gdp_USA * 2e6),
        },
        coords={
            "region": (["region"], [coord_USA]),
            "runid": (["runid"], np.arange(1, 6)),
            "rff_sp": (["runid"], np.arange(1, 6)),
            "year": (["year"], [2021, 2022, 2023, 2099, 2100]),
        },
    )
    out_actual = xr.open_dataset(tmp_path / f"rff_{file_USA}_socioeconomics.nc4").sel(
        year=[2021, 2022, 2023, 2099, 2100]
    )

    xr.testing.assert_equal(out_expected, out_actual)
