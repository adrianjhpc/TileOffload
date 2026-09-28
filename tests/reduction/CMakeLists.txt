cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(TileOffloadReductionValidation LANGUAGES NONE)
endif()

include(CTest)

get_filename_component(_tileoff_DEFAULT_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(tileoff_ROOT "${_tileoff_DEFAULT_ROOT}" CACHE PATH
  "Path to the TileOffload repository root")
unset(_tileoff_DEFAULT_ROOT)

set(_tileoff_REDUCTION_TEST_DRIVER_DEFAULT "${tileoff_ROOT}/bin/TileOffload-flang")
if(DEFINED tileoff_DRIVER AND NOT "${tileoff_DRIVER}" STREQUAL "")
  set(_tileoff_REDUCTION_TEST_DRIVER_DEFAULT "${tileoff_DRIVER}")
elseif(DEFINED ENV{tileoff_DRIVER} AND NOT "$ENV{tileoff_DRIVER}" STREQUAL "")
  set(_tileoff_REDUCTION_TEST_DRIVER_DEFAULT "$ENV{tileoff_DRIVER}")
endif()

set(tileoff_REDUCTION_TEST_DRIVER
  "${_tileoff_REDUCTION_TEST_DRIVER_DEFAULT}" CACHE FILEPATH
  "Path to the TileOffload-flang compiler driver")
unset(_tileoff_REDUCTION_TEST_DRIVER_DEFAULT)

set(LLVM_BUILD "$ENV{LLVM_BUILD}" CACHE PATH
  "llvm-project build directory used by TileOffload-flang")
set(TRITON_OPT "$ENV{TRITON_OPT}" CACHE FILEPATH
  "Path to triton-opt used by TileOffload-flang")
set(MLIR_TRANSLATE "$ENV{MLIR_TRANSLATE}" CACHE FILEPATH
  "Path to mlir-translate used by TileOffload-flang")
set(LLC "$ENV{LLC}" CACHE FILEPATH
  "Path to llc used by TileOffload-flang")
set(CUDA_LIB_DIR "$ENV{CUDA_LIB_DIR}" CACHE PATH
  "Optional CUDA driver library directory")

set(tileoff_SM "80" CACHE STRING
  "CUDA SM architecture number, e.g. 80 or 90")
set(tileoff_NUM_WARPS "1" CACHE STRING
  "Number of Triton warps for the reduction validation kernels")
set(tileoff_THREADS_PER_WARP "32" CACHE STRING
  "Threads per Triton warp")
set(tileoff_NUM_STAGES "3" CACHE STRING
  "Number of Triton pipeline stages")
set(tileoff_REDUCTION_TEST_COMPILE_FLAGS "-O3" CACHE STRING
  "Additional flags passed to TileOffload-flang")

option(tileoff_REDUCTION_TEST_BUILD_BY_DEFAULT
  "Build the reduction validation executable as part of the default build" ON)
option(tileoff_REDUCTION_TEST_KEEP_INTERMEDIATES
  "Keep TileOffload-flang intermediate files for the reduction test" OFF)
option(tileoff_REDUCTION_TEST_VERBOSE
  "Print commands run internally by TileOffload-flang" OFF)

if(NOT EXISTS "${tileoff_REDUCTION_TEST_DRIVER}")
  message(FATAL_ERROR
    "TileOffload-flang was not found at ${tileoff_REDUCTION_TEST_DRIVER}. "
    "Set tileoff_REDUCTION_TEST_DRIVER or tileoff_DRIVER.")
endif()

if(NOT IS_DIRECTORY "${LLVM_BUILD}")
  message(FATAL_ERROR
    "LLVM_BUILD is not a directory: ${LLVM_BUILD}. "
    "Configure with -DLLVM_BUILD=/path/to/llvm-project/build.")
endif()

foreach(tool_var IN ITEMS TRITON_OPT MLIR_TRANSLATE LLC)
  if(NOT EXISTS "${${tool_var}}")
    message(FATAL_ERROR
      "${tool_var} was not found at ${${tool_var}}. "
      "Set -D${tool_var}=/path/to/tool.")
  endif()
endforeach()

if(NOT "${CUDA_LIB_DIR}" STREQUAL "" AND NOT IS_DIRECTORY "${CUDA_LIB_DIR}")
  message(FATAL_ERROR "CUDA_LIB_DIR is not a directory: ${CUDA_LIB_DIR}")
endif()

if("${CMAKE_CURRENT_BINARY_DIR}" MATCHES " ")
  message(FATAL_ERROR
    "The TileOffload reduction test build directory must not contain whitespace.")
endif()

separate_arguments(_tileoff_REDUCTION_TEST_COMPILE_FLAGS
  UNIX_COMMAND "${tileoff_REDUCTION_TEST_COMPILE_FLAGS}")

set(_tileoff_REDUCTION_TEST_CONTROL_FLAGS)
if(tileoff_REDUCTION_TEST_KEEP_INTERMEDIATES)
  list(APPEND _tileoff_REDUCTION_TEST_CONTROL_FLAGS --TileOffload-keep)
endif()
if(tileoff_REDUCTION_TEST_VERBOSE)
  list(APPEND _tileoff_REDUCTION_TEST_CONTROL_FLAGS --TileOffload-verbose)
endif()

set(_tileoff_REDUCTION_TEST_SOURCE
  "${CMAKE_CURRENT_LIST_DIR}/reduction-validation.f90")
set(tileoff_REDUCTION_TEST_EXECUTABLE
  "${CMAKE_CURRENT_BINARY_DIR}/TileOffload-reduction-validation")
set(_tileoff_REDUCTION_TEST_INTERMEDIATES
  "${CMAKE_CURRENT_BINARY_DIR}/intermediates")
set(_tileoff_REDUCTION_TEST_MODULES
  "${CMAKE_CURRENT_BINARY_DIR}/modules")
set(_tileoff_REDUCTION_TEST_CONFIG
  "${CMAKE_CURRENT_BINARY_DIR}/TileOffload-reduction-validation.config")

string(CONCAT _tileoff_REDUCTION_TEST_CONFIG_CONTENT
  "driver=${tileoff_REDUCTION_TEST_DRIVER}\n"
  "sm=${tileoff_SM}\n"
  "num_warps=${tileoff_NUM_WARPS}\n"
  "threads_per_warp=${tileoff_THREADS_PER_WARP}\n"
  "num_stages=${tileoff_NUM_STAGES}\n"
  "compile_flags=${tileoff_REDUCTION_TEST_COMPILE_FLAGS}\n")
file(GENERATE OUTPUT "${_tileoff_REDUCTION_TEST_CONFIG}"
  CONTENT "${_tileoff_REDUCTION_TEST_CONFIG_CONTENT}")
unset(_tileoff_REDUCTION_TEST_CONFIG_CONTENT)

set(_tileoff_REDUCTION_TEST_ENVIRONMENT
  "LLVM_BUILD=${LLVM_BUILD}"
  "TRITON_OPT=${TRITON_OPT}"
  "MLIR_TRANSLATE=${MLIR_TRANSLATE}"
  "LLC=${LLC}"
  "CUDA_LIB_DIR=${CUDA_LIB_DIR}")

add_custom_command(
  OUTPUT "${tileoff_REDUCTION_TEST_EXECUTABLE}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory
    "${_tileoff_REDUCTION_TEST_INTERMEDIATES}"
    "${_tileoff_REDUCTION_TEST_MODULES}"
  COMMAND "${CMAKE_COMMAND}" -E env
    ${_tileoff_REDUCTION_TEST_ENVIRONMENT}
    "${tileoff_REDUCTION_TEST_DRIVER}"
      ${_tileoff_REDUCTION_TEST_CONTROL_FLAGS}
      --TileOffload-workdir "${_tileoff_REDUCTION_TEST_INTERMEDIATES}"
      --TileOffload-sm "${tileoff_SM}"
      --TileOffload-num-warps "${tileoff_NUM_WARPS}"
      --TileOffload-threads-per-warp "${tileoff_THREADS_PER_WARP}"
      --TileOffload-num-stages "${tileoff_NUM_STAGES}"
      ${_tileoff_REDUCTION_TEST_COMPILE_FLAGS}
      -I "${_tileoff_REDUCTION_TEST_MODULES}"
      -J "${_tileoff_REDUCTION_TEST_MODULES}"
      "${_tileoff_REDUCTION_TEST_SOURCE}"
      -o "${tileoff_REDUCTION_TEST_EXECUTABLE}"
  DEPENDS
    "${tileoff_REDUCTION_TEST_DRIVER}"
    "${_tileoff_REDUCTION_TEST_SOURCE}"
    "${_tileoff_REDUCTION_TEST_CONFIG}"
  COMMENT "Building TileOffload reduction validation executable"
  VERBATIM
)

if(tileoff_REDUCTION_TEST_BUILD_BY_DEFAULT)
  add_custom_target(TileOffload-reduction-validation-build ALL
    DEPENDS "${tileoff_REDUCTION_TEST_EXECUTABLE}")
else()
  add_custom_target(TileOffload-reduction-validation-build
    DEPENDS "${tileoff_REDUCTION_TEST_EXECUTABLE}")
endif()

if(BUILD_TESTING)
  add_test(NAME TileOffload-reduction-validation
    COMMAND "${CMAKE_CURRENT_LIST_DIR}/run-reduction-validation.sh"
      "${tileoff_REDUCTION_TEST_EXECUTABLE}")
endif()
