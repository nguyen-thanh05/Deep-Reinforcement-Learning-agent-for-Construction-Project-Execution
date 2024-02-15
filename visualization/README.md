# Visualization interface with raylib
____________________________________________________________
## Building
This project requires CMake, make and a C++ compiler (duh :))). Should work on Unix and Windows(?)
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
* Add input form for grid resizing. See `raygui.h`.
* Save the structure into storage.