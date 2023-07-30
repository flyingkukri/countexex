set(SOURCE_FILES train_data_generation/genTrainData.cpp model_builder/buildModel.cpp importance_calculation/impCalc.cpp dtree_visualization/dtreeToDot.cpp)
set(HEADER_FILES train_data_generation/genTrainData.h model_builder/buildModel.h importance_calculation/impCalc.h dtree_visualization/dtreeToDot.h)

add_library(libmain ${SOURCE_FILES} ${HEADER_FILES})

# Set include directories and dependencies
# we have to include 
target_include_directories(libmain PUBLIC ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR} ${storm-permissive_INCLUDE_DIR} ${MLPACK_INCLUDE_DIR}
                                   ${PROJECT_SOURCE_DIR}/src/train_data_generation ${PROJECT_SOURCE_DIR}/src/model_builder ${PROJECT_SOURCE_DIR}/src/importance_calculation ${PROJECT_SOURCE_DIR}/src/dtree_visualization INTERFACE ./)
target_link_libraries(libmain PRIVATE storm storm-parsers storm-permissive)