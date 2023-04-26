# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Unreleased
### Added
- New unit tests for [dscim/utils/input_damages.py](https://github.com/ClimateImpactLab/dscim/blob/main/src/dscim/preprocessing/input_damages.py). ([PR #68](https://github.com/ClimateImpactLab/dscim/pull/68), [@davidrzhdu](https://github.com/davidrzhdu))
- Functions used for producing RFF weights. ([PR #66](https://github.com/ClimateImpactLab/dscim/pull/66), [@davidrzhdu](https://github.com/davidrzhdu))
### Changed
- Changed `coastal_inputs` function to work with new version of coastal outputs. ([PR #75](https://github.com/ClimateImpactLab/dscim/pull/75), [@davidrzhdu](https://github.com/davidrzhdu))
- Changed `prep_mortality_damages` function to work with new format mortality outputs. ([PR #74](https://github.com/ClimateImpactLab/dscim/pull/74) and [PR #68](https://github.com/ClimateImpactLab/dscim/pull/68), [@JMGilbert](https://github.com/JMGilbert))
- Included US territories in damages and economic variable subsetting. ([PR #78](https://github.com/ClimateImpactLab/dscim/pull/78), [@JMGilbert](https://github.com/JMGilbert))
- Changed format of `eta_rhos` to allow for multiple values of `rho` for the same `eta`. ([PR #65](https://github.com/ClimateImpactLab/dscim/pull/65), [@JMGilbert](https://github.com/JMGilbert))
- Remove diagnostics module. ([PR #60](https://github.com/ClimateImpactLab/dscim/pull/60), [@JMGilbert](https://github.com/JMGilbert))
- Remove old/unnecessary files. ([PR #57](https://github.com/ClimateImpactLab/dscim/pull/57), [@JMGilbert](https://github.com/JMGilbert))
### Fixed
- Add `.load()` to every loading of population data from EconVars. ([PR #82](https://github.com/ClimateImpactLab/dscim/pull/82), [@davidrzhdu](https://github.com/davidrzhdu))
- Make `compute_ag_damages` function correctly save outputs in float32. ([PR #72](https://github.com/ClimateImpactLab/dscim/pull/72) and [PR #82](https://github.com/ClimateImpactLab/dscim/pull/82), [@davidrzhdu](https://github.com/davidrzhdu))
- Make rff damage functions read in and save out in the proper filepath structure. ([PR #79](https://github.com/ClimateImpactLab/dscim/pull/79), [@JMGilbert](https://github.com/JMGilbert))
- Enter the proper functional form of isoelastic utility when `eta = 1`. ([PR #65](https://github.com/ClimateImpactLab/dscim/pull/65), [@JMGilbert](https://github.com/JMGilbert))
- Pin numpy version to stop tests failing. ([PR #60](https://github.com/ClimateImpactLab/dscim/pull/60), [@JMGilbert](https://github.com/JMGilbert))
- Remove unused “save_path” and “ec_cls” from `read_energy_files_parallel()`. ([PR #56](https://github.com/ClimateImpactLab/dscim/pull/56), [@davidrzhdu](https://github.com/davidrzhdu))


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
