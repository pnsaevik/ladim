# Changelog

All notable changes to the project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Issues]
### Change
- Velocity in forcing module should return "grid speed" velocity. Rescaling
  should happen within the forcing module, not tracking module.
- New grid and forcing module should have a clearer separation. Grid should
  take care of coordinate system changes, while forcing should return static
  fields.
### Add
- Multiplicity to the releaser module


## [2.0.1] - 2024-05-27
### Fixed
- No longer throws error if there are no released particles at simulation start
- Non-second units in continuous release are respected

## [2.0.0] - 2024-03-20
### Changed
- Legacy modules are removed. This may lead to nuance changes in ladim output.
### Fixed
- Tracker module no longer gives error if particles are deactivated


## [1.3.5] - 2024-01-30 
### Added
- Text releaser module can add default values other than zero
### Fixed
- Allow mixture of unix and windows path slash in config file
- Particles close to edge no longer causes errors
### Changed
- Output module is now called at the end of each timestep
- New output module
- New release module
- New tracker module


## [1.3.4] - 2024-01-25
### Fixed
- Package now works with pandas 2.2.0


## [1.3.2] - 2022-10-20
### Added
- Automatically publish to GitHub Releases and PyPI


## [1.3.1] - 2022-10-19
### Changed
- Moved to GitHub Actions CI
### Fixed
- ROMS module no longer produces masked arrays


## [1.3] - 2022-06-23
### Added
- CI integration server
### Changed
- More flexible modules


## [1.2] - 2022-04-06


Initial fork from Bjørn Ådlandsvik's ladim version
Forked commit: 73567f0e04e33d56556887b9fd14c67b69bc600d
