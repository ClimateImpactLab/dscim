"""Tests for geography functionality and backward compatibility."""

import pandas
import xarray as xr
import numpy as np
import pytest

from dscim.menu.risk_aversion import RiskAversionRecipe


class TestGlobeGeographyEquivalence:
    """Tests that xarray path produces same results as pandas path for globe."""

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_damages_dataset_equals_global_damages_calculation(self, menu_instance):
        df_pandas = menu_instance.global_damages_calculation()
        ds_xarray = menu_instance.damages_dataset(geography="globe")

        df_xarray = ds_xarray.to_dataframe().reset_index()

        assert "damages" in df_pandas.columns
        assert "damages" in df_xarray.columns

        damages_pandas = df_pandas["damages"].sort_values().reset_index(drop=True)
        damages_xarray = df_xarray["damages"].sort_values().reset_index(drop=True)

        np.testing.assert_allclose(
            damages_pandas.values,
            damages_xarray.values,
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize(
        "discount_types", ["euler_ramsey", "euler_gwr", "constant"], indirect=True
    )
    def test_damages_dataset_returns_dataset(self, menu_instance):
        result = menu_instance.damages_dataset(geography="globe")
        assert isinstance(result, xr.Dataset)
        assert "damages" in result.data_vars


class TestGeographyAggregation:
    """Tests for _aggregate_by_geography method."""

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_aggregate_globe_sums_all_regions(self, menu_instance):
        damages = menu_instance.calculated_damages * menu_instance.collapsed_pop

        expected = damages.sum(dim="region")
        actual = menu_instance._aggregate_by_geography(damages, "globe")

        assert actual.region.values == ["globe"]

        actual_values = actual.squeeze(dim="region", drop=True)
        xr.testing.assert_allclose(expected, actual_values)

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_aggregate_ir_preserves_regions(self, menu_instance):
        damages = menu_instance.calculated_damages * menu_instance.collapsed_pop

        result = menu_instance._aggregate_by_geography(damages, "ir")

        xr.testing.assert_allclose(result, damages)

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_invalid_geography_raises_error(self, menu_instance):
        damages = menu_instance.calculated_damages * menu_instance.collapsed_pop

        with pytest.raises(ValueError, match="Unknown geography"):
            menu_instance._aggregate_by_geography(damages, "invalid_geography")

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_country_without_mapping_raises_error(self, menu_instance):
        damages = menu_instance.calculated_damages * menu_instance.collapsed_pop

        menu_instance.country_mapping = None

        with pytest.raises(ValueError, match="country_mapping"):
            menu_instance._aggregate_by_geography(damages, "country")


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing API."""

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_global_damages_calculation_returns_dataframe(self, menu_instance):
        result = menu_instance.global_damages_calculation()
        assert isinstance(result, pandas.DataFrame)
        assert "region" not in result.columns

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_damage_function_points_returns_dataframe(self, menu_instance):
        result = menu_instance.damage_function_points
        assert isinstance(result, pandas.DataFrame)

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_default_geography_is_globe(self, menu_instance):
        assert menu_instance.geography == "globe"


class TestDualPathEquivalence:
    """Tests for pandas vs xarray path equivalence."""

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_pandas_path_used_for_globe(self, menu_instance):
        assert menu_instance.geography == "globe"

        result = menu_instance.damage_function_points
        assert isinstance(result, pandas.DataFrame)

        expected = menu_instance._damage_function_points_pandas()
        pandas.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_xarray_path_matches_pandas_path_for_globe(self, menu_instance):
        # Compare full pipeline: damages, climate merge, illegal filtering
        pandas_result = menu_instance._damage_function_points_pandas()

        original_geography = menu_instance.geography
        menu_instance.geography = "globe"
        xarray_result = menu_instance._damage_function_points_xarray()
        menu_instance.geography = original_geography

        assert isinstance(pandas_result, pandas.DataFrame)
        assert isinstance(xarray_result, pandas.DataFrame)

        assert "damages" in pandas_result.columns
        assert "damages" in xarray_result.columns

        sort_cols = [
            c
            for c in ["year", "ssp", "model", "gcm", "rcp"]
            if c in pandas_result.columns
        ]
        pandas_sorted = pandas_result.sort_values(sort_cols).reset_index(drop=True)
        xarray_sorted = xarray_result.sort_values(sort_cols).reset_index(drop=True)

        np.testing.assert_allclose(
            pandas_sorted["damages"].values,
            xarray_sorted["damages"].values,
            rtol=1e-10,
            atol=1e-10,
        )

        if "anomaly" in pandas_sorted.columns and "anomaly" in xarray_sorted.columns:
            pandas_nan = pandas_sorted["anomaly"].isna()
            xarray_nan = xarray_sorted["anomaly"].isna()
            assert (pandas_nan == xarray_nan).all()

            pandas_valid = pandas_sorted.loc[~pandas_nan, "anomaly"].values
            xarray_valid = xarray_sorted.loc[~xarray_nan, "anomaly"].values
            np.testing.assert_allclose(
                pandas_valid,
                xarray_valid,
                rtol=1e-10,
                atol=1e-10,
            )


class TestCountryAggregation:
    """Tests for country-level aggregation."""

    @pytest.mark.skip(reason="Requires country_mapping fixture")
    def test_country_aggregation(self):
        pass


class TestIndividualRegion:
    """Tests for individual region calculations."""

    @pytest.mark.skip(reason="For future individual_region support")
    def test_individual_region_filter(self):
        pass
