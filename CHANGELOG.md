# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased


## [0.7.0] - 2025-08-15

### Added
- [Documentation pages](climateimpactlab.github.io/dscim/) added using mkdocs ([PR #254](https://github.com/ClimateImpactLab/dscim/pull/254), [@JMGilbert](https://github.com/JMGilbert))
- Added discounting option `constant_gwr`, which applies discounting across SSPs ([PR #405](https://github.com/ClimateImpactLab/dscim/pull/405), [@JMGilbert](https://github.com/JMGilbert)).

### Changed

- The function signature for `calculate_labor_batch_damages()` in `src/dscim/preprocessing/input_damages.py` was updated to include additional args with default values that allow the labor SCC application to run without modifying `dscim` code in the future. This is backwards compatible. ([PR #415](https://github.com/ClimateImpactLab/dscim/pull/415), [@JMGilbert](https://github.com/JMGilbert)).
- Python version for running automated tests in CI upgraded from Python 3.10 to 3.12 ([PR #270](https://github.com/ClimateImpactLab/dscim/pull/270), [@brews](https://github.com/brews)).

### Fixed

- Fixed how quantile regression SCCs (`quantreg`) are calculated by allowing for the full cloud of damage points in the damage function fit stage (previously the `batch` dimension was incorrectly reduced before damage function fit even if `quantreg=True`) ([PR #405](https://github.com/ClimateImpactLab/dscim/pull/405), [@JMGilbert](https://github.com/JMGilbert)). 
- Minor code cleanup. Switch old %-string formatting to use f-strings ([PR #351](https://github.com/ClimateImpactLab/dscim/pull/351), [@brews](https://github.com/brews)).
- Pin `numcodecs` package to 0.15.1 to fix automated tests in CI. This works with `zarr < 3`. ([PR #406](https://github.com/ClimateImpactLab/dscim/pull/406), [@JMGilbert](https://github.com/JMGilbert)).
- Pin `statsmodels` to 0.14.5 to fix automated tests in CI. ([PR #429](https://github.com/ClimateImpactLab/dscim/pull/429), [@C1587S](https://github.com/C1587S)). 

### Removed

- Removed [`preprocessing/climate`](https://github.com/ClimateImpactLab/dscim/tree/25dfb39637d5716662a3ec636028d5066ddb10bb/src/dscim/preprocessing/climate) and [`preprocessing/misc`](https://github.com/ClimateImpactLab/dscim/tree/25dfb39637d5716662a3ec636028d5066ddb10bb/src/dscim/preprocessing/misc) subpackages. ([PR #249](https://github.com/ClimateImpactLab/dscim/pull/249), [@JMGilbert](https://github.com/JMGilbert))
    - The modules in these subpackages referenced hard coded filepaths and were used for old versions of climate files and inputs that are now properly formatted by default
- Removed [`utils/generate_yaml`](https://github.com/ClimateImpactLab/dscim/blob/25dfb39637d5716662a3ec636028d5066ddb10bb/src/dscim/utils/generate_yaml.py) and [`utils/plotting_utils`](https://github.com/ClimateImpactLab/dscim/blob/25dfb39637d5716662a3ec636028d5066ddb10bb/src/dscim/utils/plotting_utils.py) modules. ([PR #249](https://github.com/ClimateImpactLab/dscim/pull/249), [@JMGilbert](https://github.com/JMGilbert))
    - `generate_yaml` seems to have been designed for an old version of `dscim-epa` and that functionality has now been transferred to the `scripts/directory_setup.py` script in the `dscim-epa` and `dscim-facts-epa` repositories
    - `plotting utils` was only in use for a single diagnostic and was transferred to the script that generated that diagnostic
- Removed `midprocessing` [`update_damage_function_library`](https://github.com/ClimateImpactLab/dscim/blob/25dfb39637d5716662a3ec636028d5066ddb10bb/src/dscim/preprocessing/midprocessing.py#L8-L26) and `utils` [`constant_equivalent_discount_rate`](https://github.com/ClimateImpactLab/dscim/blob/main/src/dscim/utils/functions.py#L25-L61), [`calculate_constant_equivalent_discount_rate`](https://github.com/ClimateImpactLab/dscim/blob/main/src/dscim/utils/functions.py#L64-L142), and [`get_model_weights`](https://github.com/ClimateImpactLab/dscim/blob/main/src/dscim/utils/functions.py#L145-L216) functions. ([PR #249](https://github.com/ClimateImpactLab/dscim/pull/249), [@JMGilbert](https://github.com/JMGilbert))
    - `update_damage_function_library` was previously used to move files prior to the functionality that directly saved files into the appropriate location
    - `constant_equivalent_discount_rate` and `calculate_constant_equivalent_discount_rate` are used for integration paper tables, and have been transferred to the appropriate scripts
    - `get_model_weights` is used for a few diagnostics and has been transferred to the appropriate scripts


## [0.6.0] - 2024-04-24

### Added

- Add an option for producing SCC ranges that account for only statistical uncertainty. ([PR #143](https://github.com/ClimateImpactLab/dscim/pull/143), [@davidrzhdu](https://github.com/davidrzhdu))

### Fixed

- Fix concatenate_energy_damages netcdf saving functionality which was not clearing data encoding causing some coordinates to be truncated. ([PR #229](https://github.com/ClimateImpactLab/dscim/pull/229), [@JMGilbert](https://github.com/JMGilbert))
- Fix tests broken by sorting update in pandas v2.2.1 ([PR #216](https://github.com/ClimateImpactLab/dscim/pull/216), [@JMGilbert](https://github.com/JMGilbert))

## [0.5.0] - 2023-11-17

### Added

- Add naive list of package dependencies to pyproject.toml.([PR #123](https://github.com/ClimateImpactLab/dscim/pull/123), [@brews](https://github.com/brews))
- CI, coverage, DOI badges on README. ([PR #134](https://github.com/ClimateImpactLab/dscim/pull/134), [@brews](https://github.com/brews))

### Changed

- Dropped optional/unused dependencies `click`, `dask-jobqueue`, `geopandas`, `gurobipy`, `ipywidgets`, `seaborn`. ([PR #99](https://github.com/ClimateImpactLab/dscim/pull/99), [@brews](https://github.com/brews))
- Switch build system from `setuptools` to `hatchling`. ([PR #128](https://github.com/ClimateImpactLab/dscim/pull/128), [@brews](https://github.com/brews))
- Clean up unit test for `dscim.utils.utils.c_equivalence`. ([PR #135](https://github.com/ClimateImpactLab/dscim/pull/135), [@brews](https://github.com/brews))
- Reformat gmst/gmsl pulse files by removing unnecessary dimensions and indices. ([PR #169](https://github.com/ClimateImpactLab/dscim/pull/169), [@JMGilbert](https://github.com/JMGilbert))

### Fixed

- Fix DeprecationWarning on import. ([PR #128](https://github.com/ClimateImpactLab/dscim/pull/128), [@brews](https://github.com/brews))
- Fix write-to-copy warning in `process_rff_sample()`. ([PR #116](https://github.com/ClimateImpactLab/dscim/pull/116), [@brews](https://github.com/brews))
- Fix exception from indexing with dask-backed boolean array and input climate Dataset attrs collision with xarray >= v2023.3.0. ([PR #129](https://github.com/ClimateImpactLab/dscim/pull/129), [@brews](https://github.com/brews))
- Fix bad release header links in CHANGELOG.md. ([PR #105](https://github.com/ClimateImpactLab/dscim/pull/105), [@brews](https://github.com/brews))
- Fixed broken code quality checks in CI. Now using `ruff` instead of `flake8`. ([PR #107](https://github.com/ClimateImpactLab/dscim/pull/107), [@brews](https://github.com/brews))
- Minor code style cleanup. ([PR #133](https://github.com/ClimateImpactLab/dscim/pull/133), [@brews](https://github.com/brews))

## [0.4.0] - 2023-07-06

### Added

- Functions to concatenate input damages across batches. ([PR #83](https://github.com/ClimateImpactLab/dscim/pull/83), [@davidrzhdu](https://github.com/davidrzhdu))
- New unit tests for [dscim/utils/input_damages.py](https://github.com/ClimateImpactLab/dscim/blob/main/src/dscim/preprocessing/input_damages.py). ([PR #68](https://github.com/ClimateImpactLab/dscim/pull/68), [@davidrzhdu](https://github.com/davidrzhdu))
- New unit tests for [dscim/utils/rff.py](https://github.com/ClimateImpactLab/dscim/blob/main/src/dscim/utils/rff.py). ([PR #73](https://github.com/ClimateImpactLab/dscim/pull/73), [@JMGilbert](https://github.com/JMGilbert))
- New unit tests for [dscim/dscim/preprocessing.py](https://github.com/ClimateImpactLab/dscim/blob/main/src/dscim/preprocessing/preprocessing.py). ([PR #67](https://github.com/ClimateImpactLab/dscim/pull/67), [@JMGilbert](https://github.com/JMGilbert))
- Functions used for producing RFF weights. ([PR #66](https://github.com/ClimateImpactLab/dscim/pull/66), [@davidrzhdu](https://github.com/davidrzhdu))

### Changed

- Re-enable equity menu option tests. ([PR #84](https://github.com/ClimateImpactLab/dscim/pull/84), [@JMGilbert](https://github.com/JMGilbert))
- Changed `coastal_inputs` function to work with new version of coastal outputs. ([PR #75](https://github.com/ClimateImpactLab/dscim/pull/75), [@davidrzhdu](https://github.com/davidrzhdu))
- Changed `prep_mortality_damages` function to work with new format mortality outputs. ([PR #74](https://github.com/ClimateImpactLab/dscim/pull/74) and [PR #68](https://github.com/ClimateImpactLab/dscim/pull/68), [@JMGilbert](https://github.com/JMGilbert))
- Included US territories in damages and economic variable subsetting. ([PR #78](https://github.com/ClimateImpactLab/dscim/pull/78), [@JMGilbert](https://github.com/JMGilbert))
- Changed format of `eta_rhos` to allow for multiple values of `rho` for the same `eta`. ([PR #65](https://github.com/ClimateImpactLab/dscim/pull/65), [@JMGilbert](https://github.com/JMGilbert))
- Removed incomplete "time_trend" extrapolation option from `dscim.utils.utils.model_outputs()`, along with unused function arguments. This is a breaking change. ([PR #53](https://github.com/ClimateImpactLab/dscim/pull/53), [@brews](https://github.com/brews))

### Removed

- Removed `clip_damage` function in `dscim/preprocessing/preprocessing.py`. ([PR #67](https://github.com/ClimateImpactLab/dscim/pull/67), [@JMGilbert](https://github.com/JMGilbert))
- Removed climate reformatting functions and files -- to be added back with climate file generation. ([PR #67](https://github.com/ClimateImpactLab/dscim/pull/67), [@JMGilbert](https://github.com/JMGilbert))
- Remove diagnostics module. ([PR #60](https://github.com/ClimateImpactLab/dscim/pull/60), [@JMGilbert](https://github.com/JMGilbert))
- Remove old/unnecessary files. ([PR #57](https://github.com/ClimateImpactLab/dscim/pull/57), [@JMGilbert](https://github.com/JMGilbert))
- Remove unused “save_path” and “ec_cls” from `read_energy_files_parallel()`. ([PR #56](https://github.com/ClimateImpactLab/dscim/pull/56), [@davidrzhdu](https://github.com/davidrzhdu))

### Fixed

- Make all input damages output files with correct chunksizes. ([PR #83](https://github.com/ClimateImpactLab/dscim/pull/83), [@JMGilbert](https://github.com/JMGilbert))
- Add `.load()` to every loading of population data from EconVars. ([PR #82](https://github.com/ClimateImpactLab/dscim/pull/82), [@davidrzhdu](https://github.com/davidrzhdu))
- Make `compute_ag_damages` function correctly save outputs in float32. ([PR #72](https://github.com/ClimateImpactLab/dscim/pull/72) and [PR #82](https://github.com/ClimateImpactLab/dscim/pull/82), [@davidrzhdu](https://github.com/davidrzhdu))
- Make rff damage functions read in and save out in the proper filepath structure. ([PR #79](https://github.com/ClimateImpactLab/dscim/pull/79), [@JMGilbert](https://github.com/JMGilbert))
- Enter the proper functional form of isoelastic utility when `eta = 1`. ([PR #65](https://github.com/ClimateImpactLab/dscim/pull/65), [@JMGilbert](https://github.com/JMGilbert))
- Pin numpy version to stop tests failing. ([PR #60](https://github.com/ClimateImpactLab/dscim/pull/60), [@JMGilbert](https://github.com/JMGilbert))


## [0.3.0] - 2022-09-29

### Added

- New unit tests. ([PR #50](https://github.com/ClimateImpactLab/dscim/pull/50), [PR #52](https://github.com/ClimateImpactLab/dscim/pull/52), [@brews](https://github.com/brews))

### Changed

- Removed unused “pulseyrs” and “global_cons” from `convert_old_to_newformat_AR()` and `run_rff()`. Note this is a breaking change. ([PR #51](https://github.com/ClimateImpactLab/dscim/pull/51), [@davidrzhdu](https://github.com/davidrzhdu), [@kemccusker](https://github.com/kemccusker))
- Updated README with additional technical details. ([PR #49](https://github.com/ClimateImpactLab/dscim/pull/49), [@brews](https://github.com/brews))

### Fixed

- Fix xarray `.drop()` deprecation. ([PR #54](https://github.com/ClimateImpactLab/dscim/pull/54), [@brews](https://github.com/brews))
- Fix pathlib.Path/str `TypeError` in `preprocessing.clip_damages()`. ([PR #55](https://github.com/ClimateImpactLab/dscim/pull/55), [@brews](https://github.com/brews))
- Minor fixes to docstrs. ([PR #50](https://github.com/ClimateImpactLab/dscim/pull/50), [PR #52](https://github.com/ClimateImpactLab/dscim/pull/52), [@brews](https://github.com/brews))


## [0.2.1] - 2022-09-22

### Fixed

- Fix issue [#45](https://github.com/ClimateImpactLab/dscim/issues/45) by allowing for `emission_scenario` to be `None`. ([PR #46](https://github.com/ClimateImpactLab/dscim/pull/46), [PR #47](https://github.com/ClimateImpactLab/dscim/pull/47), [@JMGilbert](https://github.com/JMGilbert))

## [0.2.0] - 2022-09-16

### Changed

- Remove mutable argument defaults to avoid gotchas. ([PR #44](https://github.com/ClimateImpactLab/dscim/pull/44), [@brews](https://github.com/brews))
- Quiet unused(?), common, logging messages to terminal. ([PR #14](https://github.com/ClimateImpactLab/dscim/pull/14), [@brews](https://github.com/brews))

### Fixed

- Add missing `self` arg to `global_consumption_calculation` abstract method. ([PR #43](https://github.com/ClimateImpactLab/dscim/pull/43), [@brews](https://github.com/brews))

## [0.1.0] - 2022-08-30

- Initial release.

[unreleased]: https://github.com/climateimpactlab/dscim/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/climateimpactlab/dscim/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/climateimpactlab/dscim/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/climateimpactlab/dscim/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/climateimpactlab/dscim/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/climateimpactlab/dscim/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/climateimpactlab/dscim/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/climateimpactlab/dscim/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/climateimpactlab/dscim/releases/tag/v0.1.0
