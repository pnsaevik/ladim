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

## [2.5.1] - 2026-09-28
### Fixed
- Updated deprecated CI scripts


## [2.5.0] - 2026-09-28
### Added
- Tracker: Midpoint integration method (advection: RK2)
### Changed
- Tracker: The integrators (EF, RK2, RK4) are multithreaded numba kernels.
  About 0.35 s per time step faster for 1.5 million particles.
- Tracker: The random walk uses uniform random numbers with variance 1
  (instead of normal deviates) from a counter-based generator: a hash of
  the particle identifier and a key drawn from np.random each time step.
  Runs are reproducible with numerics.seed, and independent of the number
  of threads and the particle order. The random stream differs from
  earlier versions, so results with diffusion are not bit-identical.
- ibms.light.surface_light is a multithreaded numba kernel with the same
  results. The numpy version is kept as surface_light_numpy.
- The release schedule is sorted by release_start (stable sort)
- The program exits with an error if the release times are not sorted, or
  if releases at the same time have different release intervals
### Fixed
- Release groups without repeats (discrete releases and warm start
  particles) were re-scanned at every later time step. With 1.5 million
  particles in one discrete release this cost 0.34 s per time step.
- Warm start particles were released 'mult' times when the release file
  has a 'mult' column

## [2.4.0] - 2026-09-28
### Added
- Chunk loader (ladim.gridforce.chunkloader): Reads and decodes HDF5 chunks
  of the forcing files directly, in parallel, only for the chunks needed by
  the particles, and prefetches the chunks of the next forcing frame in the
  background. Falls back to netCDF4 for unsupported file layouts.
- The number of threads is set with the environment variable
  LADIM_NUM_THREADS (or the gridforce setting num_threads). Default: all
  CPUs available to the process.
- ROMS forcing: Forcing.field() samples variables defined on w levels (such
  as w and AKs) and at u and v points correctly, based on the dimensions of
  the variable.
- ROMS forcing: The interpolation of each field is set with the gridforce
  setting "interpolation" (e.g. {w: tz}) or the "linear" argument of
  Forcing.field(), as the letters of the dimensions (t, z, y, x) to
  interpolate linearly. Default: "tz" for w, no interpolation otherwise.
### Changed
- The ROMS grid and forcing module is rewritten for speed and memory use.
  ladim.gridforce.ROMS is now a thin wrapper around ladim.gridforce.roms
  (coordinate conversion, grid and forcing) and the chunk loader. With 1.5
  million particles on NorKyst 800 m forcing, a simulation runs about 11
  times faster than before, with a peak memory use of 8 GB instead of 15 GB.
- New dependencies: h5py, deflate and isal
- Grid.z_r and Grid.z_w are computed when first accessed
- Linear interpolation of velocity at positions outside the grid uses the
  value at the boundary (constant extension) instead of extrapolation
### Removed
- ROMS forcing: The forcing fields are no longer available as full arrays
  (Forcing.U, Forcing.V, Forcing.temp, forcing["temp"], etc.). Use
  Forcing.field() and Forcing.velocity() instead. The gridforce setting
  ibm_forcing is no longer needed.
- ROMS forcing: The attributes steps, stepdiff, file_idx, frame_idx and
  has_been_initialized, and the method find_files()
- ladim.gridforce.ROMS no longer exports sample2D, bilin_inv, s_stretch,
  sdepth, z2s, sample3D and sample3DUV. The ROMS-specific functions are
  available in ladim.gridforce.roms.coords, the others in ladim.sample.
### Fixed
- ROMS forcing: A jump backwards in time could read forcing from the wrong
  file

## [2.3.7] - 2026-09-28
### Fixed
- ROMS forcing: Scalar fields (e.g. temp, salt), and velocity with
  method="nearest", now return the lowest s-level for positions below it
  (constant extrapolation). Previously, the second lowest level was used.

## [2.3.6] - 2026-09-28
### Fixed
- ROMS forcing: In the first forcing interval of a simulation, scalar fields
  (e.g. temp, salt) are now taken from the latest forcing time at or before
  the model time, as in the rest of the simulation. Previously, the next
  forcing time was used if the simulation started at a forcing time, and a
  value interpolated to one time step before the start otherwise.

## [2.3.5] - 2026-09-28
### Fixed
- ROMS forcing: If the simulation skipped ahead to a later first release
  time, scalar fields (e.g. temp, salt) were extrapolated to time step -1
  until the next forcing time. If the skip ended within the first forcing
  interval, neither velocity nor scalar fields were updated again.

## [2.3.4] - 2026-09-23
### Fixed
- Error in loading intermediate u/v, introduced by commit a191e21

## [2.3.3] - 2026-09-11
### Fixed
- Performance is no longer hindered by having a long explicit list of
  particles to be released.

## [2.3.2] - 2026-08-26
### Changed
- Output now uses "seconds since 1970-01-01 00:00:00" instead of "seconds since 1970-01-01",
  for improved backwards compatibility

## [2.3.1] - 2026-08-03
### Fixed
- Output split now works also on continuous releases with infrequent writes


## [2.3.0] - 2026-02-11
### Added
- Warm start capabilities
### Changed
- Skip forward if no particles
- Use sphinx autoapi to generate documentation


## [2.2.0] - 2025-10-01
### Added
- Output can now be split between several files


## [2.1.8] - 2025-09-22
### Changed
- Use arctan2 instead of atan2 to improve numpy compatibility


## [2.1.7] - 2026-02-04
### Fixed
- Compatibility with pandas 3


## [2.1.7] - 2026-02-04
### Fixed
- Compatibility with pandas 3


## [2.1.6] - 2025-06-10
### Added
- Experimental njit functions


## [2.1.5] - 2025-04-14
### Added
- Allow custom parameters in custom gridforce modules


## [2.0.9] - 2025-03-01
### Fixed
- Subgrid configuration was ignored when loading velocities, this is now fixed


## [2.0.6] - 2025-02-28
### Fixed
- Ladim output is now flushed every time step


## [2.0.5] - 2025-01-27
### Fixed
- Subgrid configuration was ignored by Ladim, this is now fixed


## [2.0.4] - 2024-10-08
### Fixed
- References to legacy salmon lice model in ladim.yaml are
  converted to ladim_plugins version of the module
- Accepts empty module config in yaml file


## [2.0.3] - 2024-09-19
### Fixed
- Can import local IBM and gridforce modules
- zROMS module now works with legacy config file
- Simulation no longer breaks if particles reach domain boundary
### Changed
- Logger now outputs current time


## [2.0.2] - 2024-09-17
### Fixed
- Accepts setattr-style assignments in ibm module


## [2.0.1] - 2024-05-27
### Fixed
- No longer throws error if there are no released particles at simulation start
- Non-second units in continuous release are respected

## [2.0.0] - 2024-03-20
### Changed
- Legacy modules are removed. This may lead to nuance changes in ladim output.
### Fixed
- Tracker module no longer gives error if particles are deactivated
- Multiplicity to the releaser module


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
