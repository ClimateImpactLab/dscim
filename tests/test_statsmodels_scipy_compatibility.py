#!/usr/bin/env python3
"""
Compatibility test: statsmodels vs scipy

This script verifies that statsmodels >= 0.14.5 is compatible with scipy >= 1.16.0,
resolving the import error caused by the removal of `_lazywhere` in scipy internals.

Reference:
- statsmodels 0.14.5 release notes: https://github.com/statsmodels/statsmodels/releases/tag/v0.14.5

Usage:
    python test_statsmodels_scipy_compatibility.py
    pytest test_statsmodels_scipy_compatibility.py
"""

import sys
import importlib


def check_versions():
    """Print installed versions of relevant packages."""
    print("Installed package versions:")
    for pkg in ["scipy", "statsmodels", "numpy"]:
        try:
            mod = importlib.import_module(pkg)
            print(f"  {pkg:<12} {mod.__version__}")
        except ImportError:
            print(f"  {pkg:<12} not installed")
    print()


def test_statsmodels_import():
    """Ensure statsmodels can be imported and has expected version."""
    import statsmodels
    assert statsmodels.__version__ >= "0.14.5"


def test_statsmodels_api_import():
    """Ensure statsmodels.api can be imported without triggering ImportError."""
    import statsmodels.api as sm
    assert hasattr(sm, "OLS")
    assert len(dir(sm)) > 10  # Sanity check


def test_problematic_modules():
    """Test modules previously affected by _lazywhere import failure."""
    problematic_modules = [
        "statsmodels.genmod._tweedie_compound_poisson",
        "statsmodels.distributions.discrete",
    ]

    for module_name in problematic_modules:
        module = importlib.import_module(module_name)
        assert module is not None


def run_cli_mode():
    """Run all tests manually in CLI mode with status output."""
    print("=" * 60)
    print("Running statsmodels/scipy compatibility checks")
    print("=" * 60)
    check_versions()

    all_passed = True

    try:
        test_statsmodels_import()
        print("statsmodels import: PASSED")
    except Exception as e:
        print(f"statsmodels import: FAILED ({e})")
        all_passed = False

    try:
        test_statsmodels_api_import()
        print("statsmodels.api import: PASSED")
    except Exception as e:
        print(f"statsmodels.api import: FAILED ({e})")
        all_passed = False

    try:
        test_problematic_modules()
        print("problematic module imports: PASSED")
    except Exception as e:
        print(f"problematic module imports: FAILED ({e})")
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("All compatibility checks passed.")
        sys.exit(0)
    else:
        print("One or more compatibility checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_cli_mode()
