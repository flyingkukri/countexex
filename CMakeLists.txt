cmake_minimum_required(VERSION 3.2)

# set the project name and version
project(countexex VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# include the ExternalProject module
include(ExternalProject)

# set the install location
set(mlpack_INSTALL_DIR ${CMAKE_BINARY_DIR}/mlpack_install)

# add mlpack as an external project
ExternalProject_Add(mlpack_external
  PREFIX            ${CMAKE_BINARY_DIR}/mlpack
  GIT_REPOSITORY    https://github.com/mlpack/mlpack.git
  GIT_TAG           master
  CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${mlpack_INSTALL_DIR}
  INSTALL_DIR       ${mlpack_INSTALL_DIR}
  BUILD_BYPRODUCTS  ${mlpack_INSTALL_DIR}/lib/libmlpack.so
)

set(MLPACK_INCLUDE_DIR ${mlpack_INSTALL_DIR}/include)

add_subdirectory(storm)

if (NOT TARGET storm)
  message(FATAL_ERROR "storm not found")
endif()

set(storm_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include/")
message("storm include dir: ${storm_INCLUDE_DIR}")

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(tests)


enable_testing()

