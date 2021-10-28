# PlanetModel-SPH
Hobby project to implement SPH in Unity targeting real-time physically-reasonable simulation of Planets and Stars on a gaming laptop using Unity.

# Setup notes
Needs unity 2021.1.23f1

# Current Roadmap

## Roadmap key
| Symbol | Meaning |
|--------|---------|
| 🔴 | Very little understanding |
| 🟠 | Learning stage: Learning how this works and if it is a good idea|
| 🟡 | Design stage: Scoping the feature and figuring out how it will work |
| 🟢 | Implementation stage: Writing code & Testing|
| ✅ | Implementation done |

## Jupiter v1
Goals: Spherical mass simulation that reaches hydrostatic equilibrium, using physically applicable units
- ✅ Particle Creation from initial conditions
- ✅ Rendering of particles as spheres
- ✅ SPH calculation of Kernels
- ✅ SPH calculation of Density
- ✅ Naive Gravity O(n^2) 
- 🟡 Pressure equation of state
- 🟡 Acceleration and time integration of position
- 🟡 Units & Scale
  - Allow choice of units/scale for internal calculations
  - Alter Authoring and display to display with units
  - Use a library for this? or just build something lightweight
- 🟠 Text UI for aggregate physical quantities
  - Totals: Energy, Momentum, Mass
  - Average/max/min: Temp, Pressure, Density, Grav Field

## Future
- 🟠 Variable smothing kernel lengths
  - May be required to hit physical realism in collisions
- 🟠 Gravity in O(NlogN) using Physics system Tree
  - May require surgury to Unity.Physics Broadphase
  - Speedups on large number of particles
- 🟠 Gravity Kernel which conserves energy
  - See Price & Monaghan 2007
  - Affects particle kernel, because kernels are related by derivatives.
- 🟠 Unit Tests for mathematics
  - Should be able to spot-test most math routines using Unity's testing system
- 🔴 Heat equation of state.
  - Temperature affects pressure and pressure affects temperature
- 🔴 Render particle groups as blackbodies
  - Temperature at surface should determine BB spectrum, color, luminosity
  - How to Map abstract particles -> pixels / what's enough particles that it's a group?
- 🔴 Graph UI to plot physical quantities
  - Spatial graph of field quantities
  - Time graph of aggregate quantities
  - Consider: https://github.com/pandr/unity-debug-overlay
- 🔴 Remove Unity.Physics Dependency
  - Broadphase construction
  - Interacting Pairs list from Broadphase
  - Position updates
- 🔴 Simulation tests
  - Put some particles in a space, optionally evolve time, test that physical properties converge to expected values.
- 🔴 Numerical error checking
  - Surveil calculation for floating point issues (Sampling, FP exceptions?)
- 🔴 Move Physics Computations to Compute Shader
  - Primary reason would be performance


