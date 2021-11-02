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
Goals: Spherical mass simulation that reaches hydrostatic equilibrium.
- ✅ Particle Creation from initial conditions
- ✅ Rendering of particles as spheres
- ✅ SPH calculation of Kernels
- ✅ SPH calculation of Density
- ✅ Naive Gravity O(n^2) 
- 🟡 Pressure equation of state
- 🟡 Acceleration and time integration of position

## Jupiter v2
Goal: Analyze performance outside editor, understand feasibility scenarios
- 🟠 Mouse-based Camera control in built executable
- 🟠 Windows build which runs standalone
- 🟠 Text UI for aggregate physical quantities
  - Totals: Energy, Momentum, Mass
  - Average/max/min: Temp, Pressure, Density, Grav Field

## Jupiter v3
Goals: Sufficient physics to render a planet rather than particles,
- ✅ Skybox and directional lighting
- 🟡 Units & Scale
  - Allow choice of units/scale for internal calculations at compile time
  - Alter Authoring and physical data display to display with units
  - First version: 
    - Use CGS for I/O
    - Define compile-time scales in space, time, mass for internal calc
- 🔴 Heat equation of state.
  - Temperature affects pressure and pressure affects temperature
- 🔴 Rendering of groups of particles based on blackbody spectrum

## Demo-polish
Goal: Polished experience which clearly showcases current capabilities, makes program correctness obvious to a technical audience
- 🔴 Ability to specify initial conditions at runtime
- 🟠 UI for launching and choosing initial state (depends on runtime initial conditions)
  - A Gaseous planet
  - A collision of gaseous planets
  - A over-rotating gaseous planet
- 🟠 Unit Tests for mathematics
  - Should be able to spot-test most math routines using Unity's testing system
- 🟠 2D labeled planar grid for visual scale

## Future
Cool things that aren't a priority yet, but have come up.
- 🟠 Variable smothing kernel lengths
  - May be required to hit physical realism in collisions
- 🟠 Gravity in O(NlogN) using Physics system Tree
  - May require surgury to Unity.Physics Broadphase
  - Speedups on large number of particles
- 🟠 Gravity Kernel which conserves energy
  - See Price & Monaghan 2007
  - Affects particle kernel, because kernels are related by derivatives.

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


