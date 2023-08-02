set(SOURCE_FILES main.cpp)

# set executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Search for Boost program_options lib
find_package(Boost QUIET REQUIRED COMPONENTS program_options)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost Not found")
endif()

# Set include directories and dependencies
target_include_directories(${PROJECT_NAME} PUBLIC ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR} ${storm-permissive_INCLUDE_DIR} ${libmain_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE storm storm-parsers storm-permissive libmain Boost::program_options)