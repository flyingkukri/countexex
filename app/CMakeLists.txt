set(SOURCE_FILES main.cpp)

# set executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Set include directories and dependencies
target_include_directories(${PROJECT_NAME} PUBLIC ${SOURCE_FILES} ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR} ${storm-permissive_INCLUDE_DIR} ${libmain_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/src/train_data_generation ${PROJECT_SOURCE_DIR}/src/model_builder ${PROJECT_SOURCE_DIR}/src/importance_calculation ${PROJECT_SOURCE_DIR}/src/dtree_visualization)
target_link_libraries(${PROJECT_NAME} PRIVATE storm storm-parsers storm-permissive libmain )