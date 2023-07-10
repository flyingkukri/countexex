#
#
#include_directories(${PROJECT_SOURCE_DIR}/src)
#
## TODO make the non-main into a library
#add_executable(libtest genTrainDataTest.cpp ${PROJECT_SOURCE_DIR}/src/genTrainData.cpp)
#
#target_include_directories(libtest PUBLIC ${storm_INCLUDE_DIR})
#target_link_libraries(libtest PRIVATE storm)
#
#
#enable_testing()
#add_test(NAME libtest
#        COMMAND $<TARGET_FILE:libtest>)

add_subdirectory(lib/Catch2)

add_executable(testscountex
        genTrainDataTest.cpp
        ${PROJECT_SOURCE_DIR}/src/genTrainData.cpp
        )
    
target_include_directories(testscountex PRIVATE ${storm_INCLUDE_DIR})
target_link_libraries(testscountex PRIVATE storm Catch2::Catch2WithMain libmain)

include(CTest)
include(Catch)
catch_discover_tests(testscountex)