file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${diff_voxel_SOURCE_DIR}/include/diff_voxel/*.hpp")
file(GLOB SOURCE_LIST CONFIGURE_DEPENDS "${diff_voxel_SOURCE_DIR}/src/*.cpp")

message(STATUS "fuck ${HEADER_LIST}")
message(STATUS "fuck ${SOURCE_LIST}")

add_library(diff_voxel ${HEADER_LIST} ${SOURCE_LIST})
set_target_properties(diff_voxel PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(diff_voxel PUBLIC "${diff_voxel_SOURCE_DIR}/include/diff_voxel")