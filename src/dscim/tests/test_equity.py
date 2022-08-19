# import os
# import pytest
# import pandas as pd
# import xarray as xr

# from pathlib import Path
# from pandas.testing import assert_frame_equal
# from xarray.testing import assert_equal, assert_allclose

# from dscim.tests import open_example_dataset, open_zipped_results
# from dscim.menu.equity import EquityRecipe


# @pytest.fixture(
#     params=[
#         "constant",
#         # "constant_model_collapsed", # not used
#         "naive_ramsey",
#         "euler_ramsey",
#         "naive_gwr",
#         "gwr_gwr",
#         "euler_gwr",
#     ]
# )
# def discount_types(request):
#     return request.param


# @pytest.fixture
# def equity(discount_types, econ, climate):
#     datadir = os.path.join(os.path.dirname(__file__), "data")

#     recipe = EquityRecipe(
#         sector_path=[{"dummy_sector": os.path.join(datadir, "damages")}],
#         save_path=None,
#         econ_vars=econ,
#         climate_vars=climate,
#         fit_type="ols",
#         variable=[{"dummy_sector": "damages"}],
#         sector="dummy_sector",
#         discounting_type=discount_types,
#         ext_method="global_c_ratio",
#         ce_path= os.path.join(datadir, "CEs"),

#         subset_dict={
#             "ssp": ["SSP2", "SSP3", "SSP4"],
#             "region": [
#                 "IND.21.317.1249",
#                 "CAN.2.33.913",
#                 "USA.14.608",
#                 "EGY.11",
#                 "SDN.4.11.50.164",
#                 "NGA.25.510",
#                 "SAU.7",
#                 "RUS.16.430.430",
#                 "SOM.2.5",
#             ],
#         },
#         fair_aggregation=["ce", "median_params", "mean"],
#         extrap_formula=None,
#         formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
#     )

#     yield recipe

# @pytest.mark.xfail
# def test_equity_points(equity, discount_types):
#     path = f"equity_{discount_types}_eta{equity.eta}_rho{equity.rho}_damage_function_points.csv"
#     expected = open_zipped_results(path)
#     actual = equity.damage_function_points
#     assert_frame_equal(expected, actual, rtol=1e-4, atol=1e-4)

# @pytest.mark.xfail
# def test_equity_coefficients(equity, discount_types):
#     path = f"equity_{discount_types}_eta{equity.eta}_rho{equity.rho}_damage_function_coefficients.nc4"
#     expected = open_zipped_results(path)
#     actual = equity.damage_function_coefficients
#     assert_allclose(
#         expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
#         actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
#     )

# @pytest.mark.xfail
# def test_equity_fit(equity, discount_types):
#     path = f"equity_{discount_types}_eta{equity.eta}_rho{equity.rho}_damage_function_fit.nc4"
#     expected = open_zipped_results(path)
#     actual = equity.damage_function_fit
#     assert_allclose(
#         expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
#         actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
#     )

# @pytest.mark.xfail
# def test_equity_global_consumption(equity, discount_types):
#     path = f"equity_{discount_types}_eta{equity.eta}_rho{equity.rho}_global_consumption.nc4"
#     expected = open_zipped_results(path)
#     actual = equity.global_consumption.squeeze()
#     # Small format hack from I/O
#     if isinstance(expected, xr.Dataset):
#         expected = expected.to_array().squeeze().drop("variable")

#     assert_allclose(
#         expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
#         actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
#     )


# @pytest.mark.xfail
# def test_equity_scc(equity, discount_types):
#     path = f"equity_{discount_types}_eta{equity.eta}_rho{equity.rho}_scc.nc4"
#     expected = open_zipped_results(path)
#     actual = equity.calculate_scc.squeeze()

#     # Small format hack from I/O
#     if isinstance(expected, xr.Dataset):
#         expected = expected.to_array().squeeze().drop("variable")

#     assert_allclose(
#         expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
#         actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
#         rtol=1.5e-4,
#         atol=1e-4,
#     )
