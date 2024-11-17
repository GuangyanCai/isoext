cmake_minimum_required(VERSION 3.15...3.26)

project(isoext LANGUAGES CXX CUDA)

if(NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

include(cmake/get_cpm.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY nvidia/cccl
  GIT_TAG v2.6.1 # Fixing the version to avoid breaking changes
)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${isoext_SOURCE_DIR}/include/*.cuh")
file(GLOB SOURCE_LIST CONFIGURE_DEPENDS "${isoext_SOURCE_DIR}/src/*.cu")

# Enable CUDA lambda expressions
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# We are now ready to compile the actual extension module
nanobind_add_module(

  # Name of the extension
  isoext_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  ${HEADER_LIST}
  ${SOURCE_LIST}
)

target_link_libraries(isoext_ext PRIVATE CCCL::CCCL)
target_include_directories(isoext_ext PUBLIC "${isoext_SOURCE_DIR}/include/")

# Install directive for scikit-build-core
install(TARGETS isoext_ext LIBRARY DESTINATION isoext)
