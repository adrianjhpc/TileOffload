cmake_minimum_required(VERSION 3.20)

project(TileOffloadBenchmarks LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

set(TILEOFF_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH
  "Path to external TileOffload repository root")

set(_TILEOFF_DRIVER_DEFAULT "${TILEOFF_ROOT}/bin/tileoffload-flang-driver")
if(DEFINED ENV{TILEOFF_DRIVER} AND NOT "$ENV{TILEOFF_DRIVER}" STREQUAL "")
  set(_TILEOFF_DRIVER_DEFAULT "$ENV{TILEOFF_DRIVER}")
endif()

set(TILEOFF_DRIVER "${_TILEOFF_DRIVER_DEFAULT}" CACHE FILEPATH
  "Path to the traditional tileoffload-flang-driver")
unset(_TILEOFF_DRIVER_DEFAULT)

set(LLVM_BUILD "$ENV{LLVM_BUILD}" CACHE PATH
  "llvm-project build directory used by tileoffload-flang-driver")

set(TRITON_OPT "$ENV{TRITON_OPT}" CACHE FILEPATH
  "Path to triton-opt used by tileoffload-flang-driver")

set(MLIR_TRANSLATE "$ENV{MLIR_TRANSLATE}" CACHE FILEPATH
  "Path to mlir-translate used by tileoffload-flang-driver")

set(LLC "$ENV{LLC}" CACHE FILEPATH
  "Path to llc used by tileoffload-flang-driver")

set(CUDA_LIB_DIR "$ENV{CUDA_LIB_DIR}" CACHE PATH
  "Optional CUDA driver library directory")

set(TILEOFF_SM "90" CACHE STRING
  "CUDA SM architecture number, e.g. 80, 90")

set(CUDA_ARCH "sm_${TILEOFF_SM}" CACHE STRING
  "CUDA architecture passed to nvcc, e.g. sm_80")

set(OPENACC_GPU_ARCH "cc${TILEOFF_SM}" CACHE STRING
  "OpenACC GPU architecture passed to nvfortran, e.g. cc80")

option(TILEOFF_ENABLE_TILEOFF_BENCHMARKS
  "Build TileOffload benchmarks" ON)

option(TILEOFF_ENABLE_OPENMP_BASELINES
  "Build OpenMP CPU baselines" ON)

option(TILEOFF_ENABLE_OPENMP_GPU_BASELINES
  "Build OpenMP GPU offloading baselines" ON)

option(TILEOFF_ENABLE_OPENACC_BASELINES
  "Build OpenACC baselines with nvfortran" OFF)

option(TILEOFF_ENABLE_CUDA_BASELINES
  "Build CUDA baselines with nvcc" ON)

option(TILEOFF_BENCH_BUILD_BY_DEFAULT
  "Build all enabled benchmark targets by default" ON)

option(TILEOFF_KEEP_INTERMEDIATES
  "Keep tileoffload-flang-driver intermediate directories" OFF)

option(TILEOFF_VERBOSE
  "Print commands executed internally by tileoffload-flang-driver" OFF)

set(NVFORTRAN "" CACHE FILEPATH
  "Path to nvfortran for OpenACC baselines")

set(NVCC "" CACHE FILEPATH
  "Path to nvcc for CUDA baselines")

set(OPENMP_GPU_OFFLOAD_FLAGS "-fopenmp-targets=nvptx64-nvidia-cuda" CACHE STRING
  "Compiler flags for OpenMP target offloading (e.g. -fopenmp-targets=nvptx64-nvidia-cuda or -foffload=nvptx-none)")

set(TILEOFF_NUM_WARPS "1" CACHE STRING
  "Number of Triton warps for tileoffload-generated kernels")

set(TILEOFF_THREADS_PER_WARP "32" CACHE STRING
  "Triton/CUDA threads per warp")

set(TILEOFF_MATMUL_NUM_WARPS "4" CACHE STRING
  "Number of Triton warps for TileOffload matmul benchmarks")

set(TILEOFF_MATMUL_TF32_NUM_WARPS "4" CACHE STRING
  "Number of Triton warps for TileOffload tf32 matmul benchmarks")

set(TILEOFF_MATMUL_F64_NUM_WARPS "4" CACHE STRING
  "Number of Triton warps for TileOffload f64 matmul benchmarks")

set(TILEOFF_NUM_STAGES "3" CACHE STRING
  "Triton pipeline stages")

set(TILEOFF_F64_MATMUL_STRATEGY "reduce" CACHE STRING
  "TileOffload f64 matmul strategy: reduce, fma, or dot")

set(TILEOFF_COMPILE_FLAGS "-O3" CACHE STRING
  "Additional flags passed through tileoffload-flang-driver while compiling benchmarks")

set(TILEOFF_LINK_FLAGS "" CACHE STRING
  "Additional flags passed through tileoffload-flang-driver while linking benchmarks")

set(OPENACC_NVFORTRAN_FLAGS "-O3 -tp=zen3 -acc" CACHE STRING
  "nvfortran flags for OpenACC baselines")

set(OPENACC_GPU_FLAGS "-gpu=${OPENACC_GPU_ARCH}" CACHE STRING
  "nvfortran GPU flags for OpenACC baselines")


# -----------------------------------------------------------------------------
# Tool discovery
# -----------------------------------------------------------------------------

if(TILEOFF_ENABLE_CUDA_BASELINES AND NOT NVCC)
  find_program(NVCC nvcc)
endif()

if(TILEOFF_ENABLE_OPENACC_BASELINES AND NOT NVFORTRAN)
  find_program(NVFORTRAN nvfortran)
endif()

if(TILEOFF_ENABLE_OPENMP_BASELINES OR TILEOFF_ENABLE_OPENMP_GPU_BASELINES)
  enable_language(Fortran OPTIONAL)
endif()

# -----------------------------------------------------------------------------
# Common paths
# -----------------------------------------------------------------------------

set(BENCH_COMMON "${CMAKE_CURRENT_LIST_DIR}/common/bench_utils.f90")

if(NOT EXISTS "${BENCH_COMMON}")
  message(FATAL_ERROR "Missing ${BENCH_COMMON}")
endif()

if(TILEOFF_ENABLE_TILEOFF_BENCHMARKS AND NOT EXISTS "${TILEOFF_DRIVER}")
  message(FATAL_ERROR
    "tileoffload-flang-driver not found: ${TILEOFF_DRIVER}. "
    "Configure with -DTILEOFF_DRIVER=/path/to/tileoffload-flang-driver")
endif()

if(TILEOFF_ENABLE_TILEOFF_BENCHMARKS AND NOT IS_DIRECTORY "${LLVM_BUILD}")
  message(FATAL_ERROR
    "LLVM_BUILD is not a build directory: ${LLVM_BUILD}. "
    "Configure with -DLLVM_BUILD=/path/to/llvm-project/build")
endif()

foreach(tool_var IN ITEMS TRITON_OPT MLIR_TRANSLATE LLC)
  if(TILEOFF_ENABLE_TILEOFF_BENCHMARKS AND NOT EXISTS "${${tool_var}}")
    message(FATAL_ERROR
      "${tool_var} not found: ${${tool_var}}. "
      "Configure with -D${tool_var}=/path/to/tool")
  endif()
endforeach()

if(TILEOFF_ENABLE_TILEOFF_BENCHMARKS
    AND NOT "${CUDA_LIB_DIR}" STREQUAL ""
    AND NOT IS_DIRECTORY "${CUDA_LIB_DIR}")
  message(FATAL_ERROR "CUDA_LIB_DIR is not a directory: ${CUDA_LIB_DIR}")
endif()

set(ALL_BENCHMARK_TARGETS)

# -----------------------------------------------------------------------------
# TileOffload route
# -----------------------------------------------------------------------------

function(add_tileoff_benchmark name kernel main)
  set(workdir "${CMAKE_CURRENT_BINARY_DIR}/tileoffload/${name}")
  set(objdir "${workdir}/obj")
  set(moddir "${workdir}/mod")
  set(intermediate_parent "${workdir}/intermediates")
  set(exe "${workdir}/${name}")

  set(kernel_source "${CMAKE_CURRENT_LIST_DIR}/tileoffload/${kernel}")
  set(main_source "${CMAKE_CURRENT_LIST_DIR}/tileoffload/${main}")
  set(kernel_object "${objdir}/${name}.kernel.o")
  set(common_object "${objdir}/${name}.common.o")
  set(main_object "${objdir}/${name}.main.o")

  set(effective_tileoff_num_warps "${TILEOFF_NUM_WARPS}")

  if("${name}" STREQUAL "tileoffload-matmul-2d")
    set(effective_tileoff_num_warps "${TILEOFF_MATMUL_NUM_WARPS}")
  endif()

  if("${name}" STREQUAL "tileoffload-matmul-2d-tf32" OR
     "${name}" STREQUAL "tileoffload-matmul-2d-tf32-packed" OR
     "${name}" STREQUAL "tileoffload-matmul-2d-tf32-repack")
    set(effective_tileoff_num_warps "${TILEOFF_MATMUL_TF32_NUM_WARPS}")
  endif()

  if("${name}" STREQUAL "tileoffload-matmul-2d-f64")
    set(effective_tileoff_num_warps "${TILEOFF_MATMUL_F64_NUM_WARPS}")
  endif()

  separate_arguments(tileoff_compile_flags UNIX_COMMAND "${TILEOFF_COMPILE_FLAGS}")
  separate_arguments(tileoff_link_flags UNIX_COMMAND "${TILEOFF_LINK_FLAGS}")

  set(tileoff_control_flags)
  if(TILEOFF_KEEP_INTERMEDIATES)
    list(APPEND tileoff_control_flags --tileoff-keep)
  endif()
  if(TILEOFF_VERBOSE)
    list(APPEND tileoff_control_flags --tileoff-verbose)
  endif()

  set(tileoff_environment
    "LLVM_BUILD=${LLVM_BUILD}"
    "TRITON_OPT=${TRITON_OPT}"
    "MLIR_TRANSLATE=${MLIR_TRANSLATE}"
    "LLC=${LLC}"
    "CUDA_LIB_DIR=${CUDA_LIB_DIR}")

  # Compile the TileOffload translation unit into a self-contained relocatable
  # object.  Its host code, PTX, JSON metadata, and registration constructor
  # are all carried by kernel_object, so the final link needs no sidecar files.
  add_custom_command(
    OUTPUT "${kernel_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "${objdir}" "${moddir}" "${intermediate_parent}"
    COMMAND "${CMAKE_COMMAND}" -E env
      ${tileoff_environment}
      "${TILEOFF_DRIVER}"
        ${tileoff_control_flags}
        --tileoff-workdir "${intermediate_parent}"
        --tileoff-sm "${TILEOFF_SM}"
        --tileoff-num-warps "${effective_tileoff_num_warps}"
        --tileoff-threads-per-warp "${TILEOFF_THREADS_PER_WARP}"
        --tileoff-num-stages "${TILEOFF_NUM_STAGES}"
        --tileoff-f64-matmul-strategy "${TILEOFF_F64_MATMUL_STRATEGY}"
        ${tileoff_compile_flags}
        -I "${moddir}"
        -J "${moddir}"
        -c "${kernel_source}"
        -o "${kernel_object}"
    DEPENDS
      "${TILEOFF_DRIVER}"
      "${kernel_source}"
    COMMENT "Compiling TileOffload kernel object for ${name}"
    VERBATIM
  )

  # bench_utils is compiled separately for each benchmark so its module and
  # object live entirely inside that benchmark's build directory.
  add_custom_command(
    OUTPUT "${common_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${objdir}" "${moddir}"
    COMMAND "${CMAKE_COMMAND}" -E env
      ${tileoff_environment}
      "${TILEOFF_DRIVER}"
        ${tileoff_control_flags}
        ${tileoff_compile_flags}
        -I "${moddir}"
        -J "${moddir}"
        -c "${BENCH_COMMON}"
        -o "${common_object}"
    DEPENDS
      "${TILEOFF_DRIVER}"
      "${BENCH_COMMON}"
    COMMENT "Compiling TileOffload benchmark utilities for ${name}"
    VERBATIM
  )

  # The main program can USE modules emitted by both preceding compilations.
  # Depending on the objects also serializes creation of those .mod files.
  add_custom_command(
    OUTPUT "${main_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${objdir}" "${moddir}"
    COMMAND "${CMAKE_COMMAND}" -E env
      ${tileoff_environment}
      "${TILEOFF_DRIVER}"
        ${tileoff_control_flags}
        ${tileoff_compile_flags}
        -I "${moddir}"
        -J "${moddir}"
        -c "${main_source}"
        -o "${main_object}"
    DEPENDS
      "${TILEOFF_DRIVER}"
      "${main_source}"
      "${kernel_object}"
      "${common_object}"
    COMMENT "Compiling TileOffload benchmark main object for ${name}"
    VERBATIM
  )

  # Link the three ordinary objects.  --tileoff-runtime explicitly adds the
  # TileOffload runtime, CUDA driver library, and required runtime search path; this
  # also works on systems where no llvm-nm/nm is available for auto-detection.
  add_custom_command(
    OUTPUT "${exe}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${workdir}"
    COMMAND "${CMAKE_COMMAND}" -E env
      ${tileoff_environment}
      "${TILEOFF_DRIVER}"
        ${tileoff_control_flags}
        --tileoff-runtime
        "${main_object}"
        "${kernel_object}"
        "${common_object}"
        ${tileoff_link_flags}
        -o "${exe}"
    DEPENDS
      "${TILEOFF_DRIVER}"
      "${main_object}"
      "${kernel_object}"
      "${common_object}"
    COMMENT "Linking TileOffload benchmark ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${exe}")
  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(TILEOFF_ENABLE_TILEOFF_BENCHMARKS)
  add_tileoff_benchmark(tileoffload-vector-add
    vector_add_kernel.f90
    vector_add_main.f90)

  add_tileoff_benchmark(tileoffload-vector-add-f64
    vector_add_f64_kernel.f90
    vector_add_f64_main.f90)

  add_tileoff_benchmark(tileoffload-saxpy
    saxpy_kernel.f90
    saxpy_main.f90)

  add_tileoff_benchmark(tileoffload-daxpy
    daxpy_kernel.f90
    daxpy_main.f90)

  add_tileoff_benchmark(tileoffload-axpby
    axpby_kernel.f90
    axpby_main.f90)

  add_tileoff_benchmark(tileoffload-axpby-f64
    axpby_f64_kernel.f90
    axpby_f64_main.f90)

  add_tileoff_benchmark(tileoffload-matrix-add-2d
    matrix_add_2d_kernel.f90
    matrix_add_2d_main.f90)

  add_tileoff_benchmark(tileoffload-matrix-add-2d-f64
    matrix_add_2d_f64_kernel.f90
    matrix_add_2d_f64_main.f90)

  add_tileoff_benchmark(tileoffload-matmul-2d
    matmul_2d_kernel.f90
    matmul_2d_main.f90)

  add_tileoff_benchmark(tileoffload-matmul-2d-tf32
    matmul_2d_tf32_kernel.f90
    matmul_2d_tf32_main.f90)

  # Same data/reference as TF32; explicit AP(k,i) storage, pack-once or per-call.
  add_tileoff_benchmark(tileoffload-matmul-2d-tf32-packed
    matmul_2d_tf32_packed_kernel.f90
    matmul_2d_tf32_packed_main.f90)

  add_tileoff_benchmark(tileoffload-matmul-2d-tf32-repack
    matmul_2d_tf32_packed_kernel.f90
    matmul_2d_tf32_repack_main.f90)

  add_tileoff_benchmark(tileoffload-matmul-2d-f64
    matmul_2d_f64_kernel.f90
    matmul_2d_f64_main.f90)

  add_tileoff_benchmark(tileoffload-reduction-sum
    reduction_sum_kernel.f90
    reduction_sum_main.f90)

  add_tileoff_benchmark(tileoffload-reduction-dot
    reduction_dot_kernel.f90
    reduction_dot_main.f90)

  add_tileoff_benchmark(tileoffload-reduction-dot-f64
    reduction_dot_f64_kernel.f90
    reduction_dot_f64_main.f90)

endif()

# -----------------------------------------------------------------------------
# OpenMP route
# -----------------------------------------------------------------------------

function(add_openmp_benchmark name source)
  if(NOT CMAKE_Fortran_COMPILER)
    message(WARNING "No Fortran compiler available; skipping ${name}")
    return()
  endif()

  set(source_path "${CMAKE_CURRENT_LIST_DIR}/openmp/${source}")

  if(NOT EXISTS "${source_path}")
    message(WARNING "OpenMP benchmark source missing; skipping ${name}: ${source_path}")
    return()
  endif()

  set(openmp_outdir "${CMAKE_CURRENT_BINARY_DIR}/openmp")
  set(openmp_moddir "${CMAKE_CURRENT_BINARY_DIR}/openmp/mod/${name}")

  add_executable("${name}"
    "${BENCH_COMMON}"
    "${source_path}")

  target_compile_options("${name}" PRIVATE -O3 -fopenmp)
  target_link_options("${name}" PRIVATE -fopenmp)

  target_include_directories("${name}" PRIVATE "${openmp_moddir}")

  set_target_properties("${name}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${openmp_outdir}"
    Fortran_MODULE_DIRECTORY "${openmp_moddir}")

  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(TILEOFF_ENABLE_OPENMP_BASELINES)
  if(CMAKE_Fortran_COMPILER)
    add_openmp_benchmark(openmp-vector-add vector_add_openmp.f90)
    add_openmp_benchmark(openmp-vector-add-f64 vector_add_f64_openmp.f90)
    add_openmp_benchmark(openmp-saxpy saxpy_openmp.f90)
    add_openmp_benchmark(openmp-daxpy daxpy_openmp.f90)
    add_openmp_benchmark(openmp-axpby axpby_openmp.f90)
    add_openmp_benchmark(openmp-matrix-add-2d matrix_add_2d_openmp.f90)
    add_openmp_benchmark(openmp-matrix-add-2d-f64 matrix_add_2d_f64_openmp.f90)
    add_openmp_benchmark(openmp-matmul-2d matmul_2d_openmp.f90)
    add_openmp_benchmark(openmp-matmul-2d-f64 matmul_2d_f64_openmp.f90)
  else()
    message(WARNING "OpenMP baselines requested but no Fortran compiler was found")
  endif()
endif()

# -----------------------------------------------------------------------------
# OpenMP GPU route
# -----------------------------------------------------------------------------

function(add_openmp_gpu_benchmark name source)
  if(NOT CMAKE_Fortran_COMPILER)
    message(WARNING "No Fortran compiler available; skipping ${name}")
    return()
  endif()

  # Assuming you save the new GPU files in a directory called 'openmp_gpu'
  set(source_path "${CMAKE_CURRENT_LIST_DIR}/openmp_gpu/${source}")

  if(NOT EXISTS "${source_path}")
    message(WARNING "OpenMP GPU benchmark source missing; skipping ${name}: ${source_path}")
    return()
  endif()

  set(openmp_gpu_outdir "${CMAKE_CURRENT_BINARY_DIR}/openmp_gpu")
  set(openmp_gpu_moddir "${CMAKE_CURRENT_BINARY_DIR}/openmp_gpu/mod/${name}")

  add_executable("${name}"
    "${BENCH_COMMON}"
    "${source_path}")

  # Separate OMP flags into a list if multiple are provided by the user
  separate_arguments(OMP_OFFLOAD_LIST UNIX_COMMAND "${OPENMP_GPU_OFFLOAD_FLAGS}")

  target_compile_options("${name}" PRIVATE -O3 -fopenmp ${OMP_OFFLOAD_LIST})
  target_link_options("${name}" PRIVATE -fopenmp ${OMP_OFFLOAD_LIST})

  target_include_directories("${name}" PRIVATE "${openmp_gpu_moddir}")

  set_target_properties("${name}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${openmp_gpu_outdir}"
    Fortran_MODULE_DIRECTORY "${openmp_gpu_moddir}")

  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(TILEOFF_ENABLE_OPENMP_GPU_BASELINES)
  if(CMAKE_Fortran_COMPILER)
    add_openmp_gpu_benchmark(openmp-gpu-vector-add vector_add_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-vector-add-f64 vector_add_f64_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-saxpy saxpy_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-daxpy daxpy_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-axpby axpby_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-matrix-add-2d matrix_add_2d_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-matrix-add-2d-f64 matrix_add_2d_f64_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-matmul-2d matmul_2d_openmp.f90)
    add_openmp_gpu_benchmark(openmp-gpu-matmul-2d-f64 matmul_2d_f64_openmp.f90)

  else()
    message(WARNING "OpenMP GPU baselines requested but no Fortran compiler was found")
  endif()
endif()

# -----------------------------------------------------------------------------
# OpenACC route
# -----------------------------------------------------------------------------

function(add_openacc_benchmark name source)
  if(NOT NVFORTRAN)
    message(WARNING "nvfortran not found; skipping ${name}")
    return()
  endif()

  set(source_path "${CMAKE_CURRENT_LIST_DIR}/openacc/${source}")

  if(NOT EXISTS "${source_path}")
    message(WARNING "OpenACC benchmark source missing; skipping ${name}: ${source_path}")
    return()
  endif()

  set(outdir "${CMAKE_CURRENT_BINARY_DIR}/openacc")
  # CMake enters WORKING_DIRECTORY before running any COMMAND, so it must
  # already exist when the build tool launches the custom command.  Give each
  # benchmark a private directory as nvfortran may also emit module and other
  # intermediate files in its current directory during parallel builds.
  set(workdir "${outdir}/${name}.work")
  file(MAKE_DIRECTORY "${outdir}" "${workdir}")
  set(exe "${outdir}/${name}")

  separate_arguments(OPENACC_NVFORTRAN_FLAGS_LIST UNIX_COMMAND "${OPENACC_NVFORTRAN_FLAGS}")
  separate_arguments(OPENACC_GPU_FLAGS_LIST UNIX_COMMAND "${OPENACC_GPU_FLAGS}")

  add_custom_command(
    OUTPUT "${exe}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${outdir}" "${workdir}"
    COMMAND "${NVFORTRAN}"
      ${OPENACC_NVFORTRAN_FLAGS_LIST}
      ${OPENACC_GPU_FLAGS_LIST}
      "${BENCH_COMMON}"
      "${source_path}"
      -o "${exe}"
    DEPENDS
      "${BENCH_COMMON}"
      "${source_path}"
    WORKING_DIRECTORY "${workdir}"
    COMMENT "Building OpenACC benchmark ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${exe}")

  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(TILEOFF_ENABLE_OPENACC_BASELINES)
  if(NVFORTRAN)
    add_openacc_benchmark(openacc-vector-add vector_add_openacc.f90)
    add_openacc_benchmark(openacc-vector-add-f64 vector_add_f64_openacc.f90)
    add_openacc_benchmark(openacc-saxpy saxpy_openacc.f90)
    add_openacc_benchmark(openacc-daxpy daxpy_openacc.f90)
    add_openacc_benchmark(openacc-axpby axpby_openacc.f90)
    add_openacc_benchmark(openacc-reduction reduction_openacc.f90)
    add_openacc_benchmark(openacc-matrix-add-2d matrix_add_2d_openacc.f90)
    add_openacc_benchmark(openacc-matrix-add-2d-f64 matrix_add_2d_f64_openacc.f90)
    add_openacc_benchmark(openacc-matmul-2d matmul_2d_openacc.f90)
    add_openacc_benchmark(openacc-matmul-2d-f64 matmul_2d_f64_openacc.f90)
  else()
    message(WARNING
      "OpenACC baselines requested but nvfortran was not found. "
      "Configure with -DNVFORTRAN=/path/to/nvfortran")
  endif()
endif()

# -----------------------------------------------------------------------------
# CUDA route
# -----------------------------------------------------------------------------

function(add_cuda_benchmark name source)
  if(NOT NVCC)
    message(WARNING "nvcc not found; skipping ${name}")
    return()
  endif()

  set(outdir "${CMAKE_CURRENT_BINARY_DIR}/cuda")
  set(exe "${outdir}/${name}")

  add_custom_command(
    OUTPUT "${exe}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${outdir}"
    COMMAND "${NVCC}"
      -O3
      "-arch=${CUDA_ARCH}"
      "${CMAKE_CURRENT_LIST_DIR}/cuda/${source}"
      -o "${exe}"
    DEPENDS "${CMAKE_CURRENT_LIST_DIR}/cuda/${source}"
    COMMENT "Building CUDA benchmark ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${exe}")
  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(TILEOFF_ENABLE_CUDA_BASELINES)
  if(NVCC)
    add_cuda_benchmark(cuda-vector-add vector_add_cuda.cu)
    add_cuda_benchmark(cuda-vector-add-f64 vector_add_f64_cuda.cu)
    add_cuda_benchmark(cuda-saxpy saxpy_cuda.cu)
    add_cuda_benchmark(cuda-daxpy daxpy_cuda.cu)
    add_cuda_benchmark(cuda-axpby axpby_cuda.cu)
    add_cuda_benchmark(cuda-axpby-f64 axpby_f64_cuda.cu)
    add_cuda_benchmark(cuda-matrix-add-2d matrix_add_2d_cuda.cu)
    add_cuda_benchmark(cuda-matrix-add-2d-f64 matrix_add_2d_f64_cuda.cu)
    add_cuda_benchmark(cuda-matmul-2d matmul_2d_cuda.cu)
    add_cuda_benchmark(cuda-matmul-2d-f64 matmul_2d_f64_cuda.cu)
    add_cuda_benchmark(cuda-reduction-sum reduction_sum_cuda.cu)
    add_cuda_benchmark(cuda-reduction-dot reduction_dot_cuda.cu)
    add_cuda_benchmark(cuda-reduction-dot-f64 reduction_dot_f64_cuda.cu)
  else()
    message(WARNING
      "CUDA baselines requested but nvcc was not found. "
      "Configure with -DNVCC=/path/to/nvcc")
  endif()
endif()

function(add_cuda_cublas_benchmark name source)
  if(NOT NVCC)
    message(WARNING "nvcc not found; skipping ${name}")
    return()
  endif()

  set(outdir "${CMAKE_CURRENT_BINARY_DIR}/cuda")
  set(exe "${outdir}/${name}")

  add_custom_command(
    OUTPUT "${exe}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${outdir}"
    COMMAND "${NVCC}"
      -O3
      "-arch=${CUDA_ARCH}"
      "${CMAKE_CURRENT_LIST_DIR}/cuda/${source}"
      -lcublas
      -o "${exe}"
    DEPENDS "${CMAKE_CURRENT_LIST_DIR}/cuda/${source}"
    COMMENT "Building CUDA cuBLAS benchmark ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${exe}")
  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(TILEOFF_ENABLE_CUDA_BASELINES)
  if(NVCC)
    add_cuda_cublas_benchmark(cuda-matmul-2d-cublas matmul_2d_cublas.cu)
    add_cuda_cublas_benchmark(cuda-matmul-2d-cublas-f64 matmul_2d_f64_cublas.cu)
  else()
    message(WARNING
      "CUDA baselines requested but nvcc was not found. "
      "Configure with -DNVCC=/path/to/nvcc")
  endif()
endif()


# -----------------------------------------------------------------------------
# Aggregate target
# -----------------------------------------------------------------------------

if(TILEOFF_BENCH_BUILD_BY_DEFAULT)
  add_custom_target(tileoffload-all-benchmarks ALL
    DEPENDS ${ALL_BENCHMARK_TARGETS})
else()
  add_custom_target(tileoffload-all-benchmarks
    DEPENDS ${ALL_BENCHMARK_TARGETS})
endif()

message(STATUS "TileOffload benchmark targets: ${ALL_BENCHMARK_TARGETS}")
