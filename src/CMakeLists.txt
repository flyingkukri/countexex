set(SOURCE_FILES genTrainData.cpp buildModel.cpp impCalc.cpp )#utils.hpp)
set(HEADER_FILES genTrainData.h buildModel.h impCalc.h utils.h)

add_library(libmain ${SOURCE_FILES} ${HEADER_FILES})

# Set include directories and dependencies
# we have to include 
target_include_directories(libmain PRIVATE ${SOURCE_FILES} ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR} ${storm-permissive_INCLUDE_DIR}
                                   INTERFACE ./)
target_link_libraries(libmain PRIVATE storm storm-parsers storm-permissive)