"""Geography regression tests: normalisation, the dual pandas/xarray path,
illegal-combination masking on the xarray path, country mapping, legacy
adding_up stores and individual_region.

The warning tests check that each deprecation or legacy warning fires exactly
where intended and nowhere else.
"""

import warnings

import numpy as np
import pandas
import pytest
import xarray as xr

from dscim.menu.baseline import Baseline
from dscim.menu.equity import EquityRecipe
from dscim.menu.risk_aversion import RiskAversionRecipe
from dscim.menu.simple_storage import (
    GEOGRAPHIES,
    StackedDamages,
    adding_up_store,
    normalize_geography,
)

from .conftest import GEOGRAPHY_REGIONS, INDIVIDUAL_REGION

ETA = 1.421158116


def _dscim_warnings(records, *needles):
    """Warnings among ``records`` whose message contains every needle."""
    return [w for w in records if all(str(n) in str(w.message) for n in needles)]


def _bare_stacked_damages(econ, climate, **kw):
    return StackedDamages(
        sector_path=None,
        save_path=None,
        econ_vars=econ,
        climate_vars=climate,
        eta=ETA,
        gdppc_bottom_code=0.0,
        **kw,
    )


# ---------------------------------------------------------------------------
# geography normalisation: one spelling
# ---------------------------------------------------------------------------
class TestGeographyNormalisation:
    def test_normalize_geography_values(self):
        assert normalize_geography(None) == "globe"
        for g in GEOGRAPHIES:
            assert normalize_geography(g) == g
        with pytest.raises(ValueError, match="Unknown geography"):
            normalize_geography("planet")

    def test_normalize_geography_global_alias_warns(self):
        with pytest.warns(DeprecationWarning, match="geography='global'"):
            assert normalize_geography("global") == "globe"

    def test_default_geography_is_globe(self, geography_menu_factory):
        assert geography_menu_factory().geography == "globe"

    def test_default_construction_emits_no_dscim_warning(self, geography_menu_factory):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            geography_menu_factory()
        assert not _dscim_warnings(records, "geography")
        assert not _dscim_warnings(records, "country_ISOs")
        assert not _dscim_warnings(records, "legacy store")

    @pytest.mark.parametrize("menu_class", [Baseline, RiskAversionRecipe, EquityRecipe])
    def test_global_alias_warns_once_and_normalises(
        self, geography_menu_factory, menu_class
    ):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            menu = geography_menu_factory(menu_class=menu_class, geography="global")
        found = _dscim_warnings(records, "geography='global'", "geography='globe'")
        assert len(found) == 1
        assert issubclass(found[0].category, DeprecationWarning)
        # stacklevel: the frame that called the constructor (the factory)
        assert found[0].filename == geography_menu_factory.__code__.co_filename
        assert menu.geography == "globe"

    def test_unknown_geography_raises(self, geography_menu_factory):
        with pytest.raises(ValueError, match="Unknown geography"):
            geography_menu_factory(geography="planet")

    def test_equity_rejects_regional_geographies(self, geography_menu_factory):
        for g in ("ir", "country"):
            with pytest.raises(NotImplementedError, match="EquityRecipe"):
                geography_menu_factory(menu_class=EquityRecipe, geography=g)

    def test_stacked_damages_alone_normalises(self, econ, climate):
        assert _bare_stacked_damages(econ, climate).geography == "globe"
        with pytest.warns(DeprecationWarning, match="geography='global'"):
            sd = _bare_stacked_damages(econ, climate, geography="global")
        assert sd.geography == "globe"


# ---------------------------------------------------------------------------
# globe is the pandas path of dscim <= 0.7.0; the xarray path agrees with it
# ---------------------------------------------------------------------------
class TestGlobeGeographyEquivalence:
    """The xarray path at "globe" produces the same numbers as the pandas path."""

    @pytest.mark.parametrize(
        "menu_class", [Baseline, RiskAversionRecipe, EquityRecipe], indirect=True
    )
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_damages_dataset_equals_global_damages_calculation(self, menu_instance):
        df_pandas = menu_instance.global_damages_calculation()
        ds_xarray = menu_instance.damages_dataset(geography="globe")

        assert isinstance(ds_xarray, xr.Dataset)
        assert list(ds_xarray.region.values) == ["globe"]

        damages_pandas = np.sort(df_pandas["damages"].values)
        damages_xarray = np.sort(ds_xarray["damages"].values.ravel())
        np.testing.assert_allclose(
            damages_pandas, damages_xarray, rtol=1e-10, atol=1e-10, equal_nan=True
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
    """_aggregate_by_geography, the single implementation all recipes share."""

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_aggregate_globe_sums_all_regions(self, menu_instance):
        damages = menu_instance.calculated_damages * menu_instance.collapsed_pop

        expected = damages.sum(dim="region")
        actual = menu_instance._aggregate_by_geography(damages, "globe")

        assert list(actual.region.values) == ["globe"]
        xr.testing.assert_allclose(expected, actual.squeeze(dim="region", drop=True))

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_aggregate_ir_preserves_regions(self, menu_instance):
        damages = menu_instance.calculated_damages * menu_instance.collapsed_pop
        xr.testing.assert_allclose(
            menu_instance._aggregate_by_geography(damages, "ir"), damages
        )

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
        assert menu_instance.country_mapping is None
        with pytest.raises(ValueError, match="country_mapping_path"):
            menu_instance._aggregate_by_geography(damages, "country")


class TestBackwardCompatibility:
    """With the default geography the public API is that of dscim <= 0.7.0."""

    @pytest.mark.parametrize(
        "menu_class", [Baseline, RiskAversionRecipe, EquityRecipe], indirect=True
    )
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_global_damages_calculation_returns_dataframe(self, menu_instance):
        result = menu_instance.global_damages_calculation()
        assert isinstance(result, pandas.DataFrame)
        assert "region" not in result.columns

    @pytest.mark.parametrize(
        "menu_class", [Baseline, RiskAversionRecipe, EquityRecipe], indirect=True
    )
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_damage_function_points_returns_dataframe(self, menu_instance):
        result = menu_instance.damage_function_points
        assert isinstance(result, pandas.DataFrame)
        assert "region" not in result.columns

    @pytest.mark.parametrize(
        "menu_class", [Baseline, RiskAversionRecipe, EquityRecipe], indirect=True
    )
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_default_geography_is_globe(self, menu_instance):
        assert menu_instance.geography == "globe"

    @pytest.mark.parametrize(
        "menu_class", [Baseline, RiskAversionRecipe, EquityRecipe], indirect=True
    )
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_no_region_dimension_on_legacy_outputs(self, menu_instance):
        assert "region" not in menu_instance.global_consumption.dims
        assert "region" not in menu_instance.uncollapsed_discount_factors.dims

    @pytest.mark.parametrize(
        "menu_class", [Baseline, RiskAversionRecipe, EquityRecipe], indirect=True
    )
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_explicit_globe_matches_default(
        self, menu_instance, menu_instance_with_geography
    ):
        pandas.testing.assert_frame_equal(
            menu_instance_with_geography.damage_function_points,
            menu_instance.damage_function_points,
            check_exact=True,
        )


class TestDualPathEquivalence:
    """pandas vs xarray path."""

    @pytest.mark.parametrize("menu_class", [RiskAversionRecipe], indirect=True)
    @pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
    def test_pandas_path_used_for_globe(self, menu_instance):
        assert menu_instance.geography == "globe"
        result = menu_instance.damage_function_points
        assert isinstance(result, pandas.DataFrame)
        pandas.testing.assert_frame_equal(
            result, menu_instance._damage_function_points_pandas()
        )

    @pytest.mark.parametrize("menu_class", [Baseline, RiskAversionRecipe])
    def test_xarray_path_matches_pandas_path_for_globe(
        self, geography_menu_factory, menu_class
    ):
        """Full pipeline with all five SSPs: damages, climate merge and the
        illegal-combination masks agree row for row."""
        menu = geography_menu_factory(menu_class=menu_class, all_ssps=True)
        pandas_result = menu._damage_function_points_pandas()
        xarray_result = (
            menu._damage_function_points_xarray()
            .to_dataframe()
            .reset_index()
            .drop(columns="region")
        )
        assert isinstance(pandas_result, pandas.DataFrame)

        keys = ["ssp", "model", "year", "gcm", "rcp"]
        merged = pandas_result.merge(
            xarray_result, on=keys, suffixes=("_pd", "_xr"), how="outer", indicator=True
        )
        assert (merged["_merge"] == "both").all()
        assert len(merged) == len(pandas_result)
        np.testing.assert_allclose(
            merged.damages_pd, merged.damages_xr, rtol=1e-10, equal_nan=True
        )
        assert (merged.anomaly_pd.isna() == merged.anomaly_xr.isna()).all()
        np.testing.assert_allclose(
            merged.anomaly_pd, merged.anomaly_xr, rtol=1e-10, equal_nan=True
        )
        # the masks did real work: SSP1/rcp85 and SSP5/rcp45 rows are NaN
        assert pandas_result.anomaly.isna().sum() > 0


# ---------------------------------------------------------------------------
# regional path and illegal-combination masking
# ---------------------------------------------------------------------------
class TestRegionalPath:
    @pytest.mark.parametrize("menu_class", [Baseline, RiskAversionRecipe])
    def test_ir_returns_dataset_with_region_dimension(
        self, geography_menu_factory, menu_class
    ):
        menu = geography_menu_factory(menu_class=menu_class, geography="ir")
        pts = menu.damage_function_points
        assert isinstance(pts, xr.Dataset)
        assert pts.sizes["region"] == len(GEOGRAPHY_REGIONS)
        assert {"damages", "anomaly"} <= set(pts.data_vars)
        assert "region" in menu.global_consumption.dims

    @pytest.mark.parametrize("menu_class", [Baseline, RiskAversionRecipe])
    def test_ir_damages_sum_to_globe_damages(self, geography_menu_factory, menu_class):
        pts = geography_menu_factory(
            menu_class=menu_class, geography="ir"
        ).damage_function_points
        globe = geography_menu_factory(
            menu_class=menu_class
        ).global_damages_calculation()
        np.testing.assert_allclose(
            np.sort(pts["damages"].sum("region").values.ravel()),
            np.sort(globe["damages"].values),
            rtol=1e-9,
        )

    @pytest.mark.parametrize("menu_class", [Baseline, RiskAversionRecipe])
    def test_ir_masks_illegal_ssp_rcp_combinations(
        self, geography_menu_factory, menu_class
    ):
        """SSP1/rcp85 and SSP5/rcp45 have no finite anomaly; damages are
        never masked."""
        pts = geography_menu_factory(
            menu_class=menu_class, geography="ir", all_ssps=True
        ).damage_function_points
        illegal = ((pts.ssp == "SSP1") & (pts.rcp == "rcp85")) | (
            (pts.ssp == "SSP5") & (pts.rcp == "rcp45")
        )
        anomaly = pts["anomaly"].broadcast_like(pts["damages"])
        illegal = illegal.broadcast_like(pts["damages"])
        assert int(illegal.sum()) > 0
        assert int(np.isfinite(anomaly.where(illegal)).sum()) == 0
        legal_anomaly = anomaly.where(~illegal)
        assert int(np.isfinite(legal_anomaly).sum()) > 0
        # the mask touches anomaly only
        assert (
            pts["damages"].notnull().sum()
            == pts["damages"].where(~illegal).notnull().sum()
            + pts["damages"].where(illegal).notnull().sum()
        )

    def test_ir_without_illegal_ssps_is_untouched(self, geography_menu_factory):
        """With SSP2-4 only (the regional-scc configuration) the mask does not
        run: anomaly keeps the source's dims and NaN pattern."""
        menu = geography_menu_factory(geography="ir")
        pts = menu.damage_function_points
        assert "ssp" not in pts["anomaly"].dims
        source = menu.climate.gmst.set_index(["gcm", "rcp", "year"]).to_xarray()
        (var,) = list(source.data_vars)
        xr.testing.assert_identical(
            pts["anomaly"].isnull(),
            source[var].reindex_like(pts["anomaly"]).isnull().rename("anomaly"),
        )


# ---------------------------------------------------------------------------
# country mapping
# ---------------------------------------------------------------------------
class TestCountryAggregation:
    @pytest.mark.parametrize("menu_class", [Baseline, RiskAversionRecipe])
    def test_country_geography(
        self, geography_menu_factory, country_mapping_csv, menu_class
    ):
        menu = geography_menu_factory(
            menu_class=menu_class,
            geography="country",
            country_mapping_path=country_mapping_csv,
        )
        pts = menu.damage_function_points
        assert isinstance(pts, xr.Dataset)
        assert set(pts.region.values) == {
            "IND",
            "CAN",
            "USA",
            "EGY",
            "SDN",
            "NGA",
            "SAU",
            "RUS",
            "nopop",
        }
        globe = geography_menu_factory(
            menu_class=menu_class
        ).global_damages_calculation()
        np.testing.assert_allclose(
            np.sort(pts["damages"].sum("region").values.ravel()),
            np.sort(globe["damages"].values),
            rtol=1e-9,
        )
        assert menu.global_consumption.sizes["region"] == 9

    def test_country_without_mapping_raises(self, geography_menu_factory):
        menu = geography_menu_factory(geography="country")
        with pytest.raises(ValueError, match="country_mapping_path"):
            menu.damage_function_points

    def test_country_mapping_path_is_silent(
        self, geography_menu_factory, country_mapping_csv
    ):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            menu = geography_menu_factory(country_mapping_path=country_mapping_csv)
        assert not _dscim_warnings(records, "country_ISOs")
        assert isinstance(menu.country_mapping, pandas.DataFrame)
        assert len(menu.country_mapping) == 9

    def test_no_mapping_means_none_and_no_legacy_attributes(
        self, geography_menu_factory
    ):
        menu = geography_menu_factory()
        assert menu.country_mapping is None
        assert not hasattr(menu, "countries_mapping")
        assert not hasattr(menu, "countries")

    def test_country_isos_warns_once_and_populates_country_mapping(
        self, geography_menu_factory, country_mapping_csv
    ):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            menu = geography_menu_factory(country_ISOs=country_mapping_csv)
        found = _dscim_warnings(records, "country_ISOs", "country_mapping_path")
        assert len(found) == 1
        assert issubclass(found[0].category, DeprecationWarning)
        assert found[0].filename == geography_menu_factory.__code__.co_filename
        expected = geography_menu_factory(country_mapping_path=country_mapping_csv)
        pandas.testing.assert_frame_equal(
            menu.country_mapping, expected.country_mapping
        )
        assert not hasattr(menu, "countries_mapping")

    def test_country_isos_still_works_end_to_end(
        self, geography_menu_factory, country_mapping_csv
    ):
        """What regional-scc passes today keeps producing the same result."""
        with pytest.warns(DeprecationWarning, match="country_ISOs"):
            legacy = geography_menu_factory(
                geography="country", country_ISOs=country_mapping_csv
            )
        new = geography_menu_factory(
            geography="country", country_mapping_path=country_mapping_csv
        )
        xr.testing.assert_identical(
            legacy.damage_function_points, new.damage_function_points
        )

    def test_stacked_damages_alone(self, econ, climate, country_mapping_csv):
        sd = _bare_stacked_damages(
            econ, climate, country_mapping_path=country_mapping_csv
        )
        assert isinstance(sd.country_mapping, pandas.DataFrame)
        with pytest.warns(DeprecationWarning, match="country_ISOs"):
            sd2 = _bare_stacked_damages(econ, climate, country_ISOs=country_mapping_csv)
        pandas.testing.assert_frame_equal(sd2.country_mapping, sd.country_mapping)


# ---------------------------------------------------------------------------
# legacy adding_up stores (read side; the write side is in
# test_preprocessing.py::test_reduce_damages)
# ---------------------------------------------------------------------------
class TestLegacyAddingUpStores:
    def test_suffixed_store_is_preferred_and_silent(self, legacy_ce_paths):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            path = adding_up_store(legacy_ce_paths["both"], "cc", ETA)
        assert path.endswith(f"adding_up_cc_eta{ETA}.zarr")
        assert not _dscim_warnings(records, "legacy store")

    def test_legacy_store_is_used_with_a_warning_naming_both(self, legacy_ce_paths):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            path = adding_up_store(legacy_ce_paths["legacy"], "cc", ETA)
        assert path.endswith("/adding_up_cc.zarr")
        found = _dscim_warnings(
            records, f"adding_up_cc_eta{ETA}.zarr", "adding_up_cc.zarr"
        )
        assert len(found) == 1
        assert issubclass(found[0].category, UserWarning)

    def test_missing_both_returns_suffixed_path_silently(self, legacy_ce_paths):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            path = adding_up_store(legacy_ce_paths["empty"], "cc", ETA)
        assert path.endswith(f"adding_up_cc_eta{ETA}.zarr")
        assert not _dscim_warnings(records, "legacy store")

    def test_globe_path_falls_back_to_legacy_stores(
        self, geography_menu_factory, legacy_ce_paths
    ):
        reference = geography_menu_factory().damage_function_points
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            pts = geography_menu_factory(
                ce_path=legacy_ce_paths["legacy"]
            ).damage_function_points
        assert len(_dscim_warnings(records, "legacy store")) == 2  # cc and no_cc
        pandas.testing.assert_frame_equal(pts, reference, check_exact=True)

    def test_regional_path_falls_back_to_legacy_stores(
        self, geography_menu_factory, legacy_ce_paths
    ):
        reference = geography_menu_factory(geography="ir").calculated_damages
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            actual = geography_menu_factory(
                geography="ir", ce_path=legacy_ce_paths["legacy"]
            ).calculated_damages
        assert len(_dscim_warnings(records, "legacy store")) == 2
        xr.testing.assert_allclose(actual, reference)

    def test_suffixed_stores_emit_no_legacy_warning(self, geography_menu_factory):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            geography_menu_factory().damage_function_points
            geography_menu_factory(geography="ir").calculated_damages
        assert not _dscim_warnings(records, "legacy store")

    def test_missing_stores_raise_naming_the_suffixed_path(
        self, geography_menu_factory, legacy_ce_paths
    ):
        with pytest.raises(NotImplementedError, match=f"adding_up_cc_eta{ETA}.zarr"):
            geography_menu_factory(
                ce_path=legacy_ce_paths["empty"]
            ).damage_function_points


# ---------------------------------------------------------------------------
# individual_region
# ---------------------------------------------------------------------------
class TestIndividualRegion:
    def test_damages_dataset_has_no_region_dimension(self, geography_menu_factory):
        one = geography_menu_factory(
            geography="ir", individual_region=INDIVIDUAL_REGION
        )
        ds = one.damages_dataset(geography="ir")
        assert "region" not in ds.dims

    def test_damages_dataset_matches_that_slice_of_the_all_region_run(
        self, geography_menu_factory
    ):
        one = geography_menu_factory(
            geography="ir", individual_region=INDIVIDUAL_REGION
        )
        every = geography_menu_factory(geography="ir")
        actual = one.damages_dataset(geography="ir").damages
        expected = every.damages_dataset(geography="ir").damages.sel(
            region=INDIVIDUAL_REGION, drop=True
        )
        xr.testing.assert_allclose(actual, expected)

    def test_damages_dataset_differs_from_other_regions(self, geography_menu_factory):
        one = geography_menu_factory(
            geography="ir", individual_region=INDIVIDUAL_REGION
        )
        every = geography_menu_factory(geography="ir")
        single = one.damages_dataset(geography="ir").damages
        multi = every.damages_dataset(geography="ir").damages
        assert multi.sizes["region"] > 1
        for other in [r for r in multi.region.values if r != INDIVIDUAL_REGION]:
            assert not np.allclose(
                single.values, multi.sel(region=other).values, equal_nan=True
            )

    def test_global_consumption_follows_the_region(self, geography_menu_factory):
        one = geography_menu_factory(
            geography="ir", individual_region=INDIVIDUAL_REGION
        )
        every = geography_menu_factory(geography="ir")
        single = one.global_consumption
        assert "region" not in single.dims
        expected = every.global_consumption.sel(region=INDIVIDUAL_REGION, drop=True)
        xr.testing.assert_allclose(single, expected)
        globe = geography_menu_factory().global_consumption
        assert float(single.sum()) < float(globe.sum())

    def test_calculated_damages_is_a_property_not_a_method(
        self, geography_menu_factory
    ):
        menu = geography_menu_factory(
            geography="ir", individual_region=INDIVIDUAL_REGION
        )
        assert isinstance(menu.calculated_damages, xr.DataArray)
        assert "region" not in menu.calculated_damages.dims

    def test_calculated_damages_is_not_population_weighted(
        self, geography_menu_factory
    ):
        """damages_dataset multiplies by collapsed_pop; weighting here too would
        double count population."""
        menu = geography_menu_factory(geography="ir")
        cc = xr.open_zarr(f"{menu.ce_path}/adding_up_cc_eta{menu.eta}.zarr").cc
        no_cc = xr.open_zarr(f"{menu.ce_path}/adding_up_no_cc_eta{menu.eta}.zarr").no_cc
        expected = menu.cut(no_cc - cc)
        xr.testing.assert_allclose(menu.calculated_damages, expected)
        assert not np.isclose(
            float(menu.calculated_damages.sum()),
            float((expected * menu.collapsed_pop).sum()),
        )
