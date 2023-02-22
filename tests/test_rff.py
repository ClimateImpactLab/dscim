from itertools import product
import numpy as np
import pandas as pd
from dscim.utils.rff import (
    clean_simulation,
    clean_error,
    weight_df,
    rff_damage_functions,
)
import pytest
from pathlib import Path
import xarray as xr


@pytest.mark.parametrize("param", ["alpha", "error"])
def test_clean_weights(tmp_path, weights_unclean, param):
    if param == "alpha":
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
    else:
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
    d = tmp_path / "weighting"
    d.mkdir()

    sector = "sector_dummy"
    pulse_year = "2020"
    recipe = "euler_ramsey"
    disc = "risk_aversion"
    eta_rho = [1, 2]
    file = "damage_functions"

    df_in_dir = d / sector / pulse_year
    df_out_dir = d / "out" / sector / pulse_year

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

    factors = xr.Dataset(
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
            "year": (["year"], [2100, 2101, 2102]),
        },
    )
    df_in.to_netcdf(df_in_file)

    out_expected = xr.Dataset(
        {
            "anomaly": (
                ["discount_type", "year", "ssp", "model"],
                np.ones((1, 6, 2, 2)),
            ),
        },
        coords={
            "discount_type": (["discount_type"], ["euler_ramsey"]),
            "year": (["year"], [2021, 2022, 2099, 2100, 2101, 2102]),
            "ssp": (["ssp"], ["SSP3", "SSP4"]),
            "model": (["model"], ["IIASA GDP", "OECD Env-Growth"]),
        },
    )

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
    )

    xr.testing.assert_allclose(out_expected, xr.open_dataset(outfile_path))
    xr.testing.assert_allclose(
        out_expected_fractional, xr.open_dataset(fractional_path)
    )


def test_rff_damage_functions(tmp_path, save_ssprff_econ):
    d = tmp_path / "weighting"
    d.mkdir(exist_ok=True)

    rff = d / "rff"
    rff.mkdir(exist_ok=True)

    ssp = d / "ssp"
    ssp.mkdir(exist_ok=True)

    sector = "dummy_sector"
    eta_rhos = {1.0: 2.0}
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

    (ssp / sector / "2020").mkdir(parents=True, exist_ok=True)
    df_in.to_netcdf(
        ssp
        / sector
        / "2020"
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
    )

    out_actual = xr.open_dataset(
        rff
        / sector
        / "2020"
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

    xr.testing.assert_allclose(out_actual, out_expected)
