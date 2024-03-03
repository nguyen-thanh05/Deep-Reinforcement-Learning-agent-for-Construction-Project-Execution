# Visualization interface with raylib
____________________________________________________________
## Building
This project requires CMake, a build system like make/Ninja and a C++ compiler (duh :))). Should work on Unix and Windows(?)
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
./GridWorldEnv <size> <s|l> <file_name>          

# <env_size>     : size of the environment (n x n x n)
# <s|l>          : whether to save to file or load from file
# <file_name>    : file to load from / save to 
#
# NOTE: Loading from file assumes the file is in the visualization/targets
# directory, no need to prefix it with relative path.
# Writing to file also assumes the name contains no prefix,
# will place the file into visualization/targets
```

## WIP
* ~~Add input form for grid resizing (or just take the dimension size by command line for simplicity). See `raygui.h`.~~ 
* ~~Save the structure into storage.~~
* Done?
