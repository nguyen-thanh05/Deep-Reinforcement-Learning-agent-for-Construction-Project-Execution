cmake_minimum_required(VERSION 3.10)
project(GridWorldEnv)

# Set the path to raylib directory
set(RAYLIB_DIR "${CMAKE_SOURCE_DIR}/raylib")

# Add raylib as a subdirectory
add_subdirectory(${RAYLIB_DIR})

# Set your project source files
file(GLOB SOURCES "src/*.cpp" "src/*.h")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} raylib)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)

target_include_directories(${PROJECT_NAME} PRIVATE ${RAYLIB_DIR}/src)
