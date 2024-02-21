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

## WIP
* ~~Add input form for grid resizing (or just take the dimension size by command line for simplicity). See `raygui.h`.~~ 
* ~~Save the structure into storage.~~
* Done?
