add_subdirectory(lib/Catch2)

add_executable(testscountex
        genTrainDataTest.cpp
        ${PROJECT_SOURCE_DIR}/src/train_data_generation/genTrainData.cpp
        )
    
target_include_directories(testscountex PRIVATE ${storm_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/src/train_data_generation)
target_link_libraries(testscountex PRIVATE storm Catch2::Catch2WithMain libmain)

include(CTest)
include(Catch)
catch_discover_tests(testscountex)