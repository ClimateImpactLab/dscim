# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


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
