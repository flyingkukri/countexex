cmake_minimum_required(VERSION 3.2)

# set the project name and version
project(countexex VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# search for Storm library
find_package(storm REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)


add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(tests)

enable_testing()

# Note that other libraries/includes from Storm might need to be added depending on the used functionality.
# For example, to enable counterexamples, include directory ${storm-counterexamples_INCLUDE_DIR} and link library storm-counterexamples need to be added.

