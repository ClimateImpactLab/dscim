from itertools import product
import numpy as np
import pandas as pd
from dscim.utils.rff import clean_simulation, clean_error
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
