# Visualization interface with raylib
____________________________________________________________
## Cloning
Clone the parent repository with `git clone --recursive ...` or `git submodule update --init --recursive` if you already cloned without submodule.\
If there are submodule errors you can clone the raylib dependency yourself:
```bash
cd visualization/
git clone https://github.com/raysan5/raylib.git
```

## Building
This project requires CMake, a build system like make/Ninja and a C++ compiler (duh :))). Should work on Unix and Windows
```bash
mkdir visualization/build
cd visualization/build
cmake ..      
make
```

Building with Visual Studio
```bash
mkdir visualization/build
cd visualization/build
cmake .. -G "Visual Studio 16 2019" -A x64     # Or another VS version
GridWorldEnv.sln
```

## Usage
After building the project:
```bash
cd visualization/build/bin                       # If not already in build/bin
./GridWorldEnv <size> <num_block_types> <s|l> <file_name>          

# <env_size>     : size of the environment (n x n x n)
# <num_block_types>: number of block types
# <s|l>          : whether to save to file or load from file
# <file_name>    : file to load from / save to 
#
# NOTE: Loading and saving file location is relative to the executable
```

## WIP
* ~~Add input form for grid resizing (or just take the dimension size by command line for simplicity). See `raygui.h`.~~ 
* ~~Save the structure into storage.~~
* Done?
