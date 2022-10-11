# LTSimulationEngine

## Build instrictions (Windows)

### Clone

```
git clone --recurse-submodules git@svleddar-gitlab.leddartech.local:simulation/LTSimulationEngine.git
```

or using https:

```
git clone --recurse-submodules https://svleddar-gitlab.leddartech.local/simulation/LTSimulationEngine.git
```

### Install dependencies:

- **Cmake** (https://cmake.org/)
- **Ninja** (https://ninja-build.org/)
  The Ninja buiild system is used for speed. This behaviour can be changed by editing `build_solution.bat`
- **CUDA 11.4** (https://developer.nvidia.com/cuda-downloads)


### Build commands

```
./build_solution.bat
./build_runtime.bat
```

Tile will create a Build folder and build the project.
