set(SOURCE_FILES main.cpp)

# set executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Set include directories and dependencies
target_include_directories(${PROJECT_NAME} PUBLIC ${SOURCE_FILES} ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR} ${storm-permissive_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE storm storm-parsers storm-permissive libmain)