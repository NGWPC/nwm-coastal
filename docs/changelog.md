# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-06

### Added

- Initial release of NWM Coastal
- SCHISM coastal model workflow support
- YAML configuration with variable interpolation
- Configuration inheritance with `_base` field
- CLI commands: `init`, `validate`, `submit`, `run`, `stages`
- Python API for programmatic workflow control
- Automatic data download from NWM and STOFS sources
- Support for TPXO and STOFS boundary conditions
- Support for four coastal domains: Hawaii, PRVI, Atlantic/Gulf, Pacific
- Interactive and non-interactive job submission modes
- Partial workflow execution with `--start-from` and `--stop-after`
- Smart default paths with interpolation templates
- Comprehensive configuration validation
- MkDocs documentation with Material theme

### Supported Data Sources

- NWM Retrospective 3.0 (1979-02-01 to 2023-01-31)
- NWM Analysis (2018-09-17 to present)
- STOFS water levels (2020-12-30 to present)
- GLOFS (Great Lakes OFS, 2005-09-30 to present)
- TPXO tidal model (local installation required)

[0.1.0]: https://github.com/NGWPC/nwm-coastal/releases/tag/v0.1.0
[unreleased]: https://github.com/NGWPC/nwm-coastal/compare/v0.1.0...HEAD
