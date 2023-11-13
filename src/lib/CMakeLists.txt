# Import thrust
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${isoext_SOURCE_DIR}/include/lib/*.cuh")
file(GLOB SOURCE_LIST CONFIGURE_DEPENDS "${isoext_SOURCE_DIR}/src/lib/*.cu")

add_library(lib ${HEADER_LIST} ${SOURCE_LIST})
set_target_properties(lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(lib PUBLIC "${isoext_SOURCE_DIR}/include/lib/")
target_link_libraries(lib Thrust)