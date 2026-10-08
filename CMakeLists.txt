cmake_minimum_required(VERSION 3.15)

project(e8qlib VERSION 1.0.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(E8Q_BUILD_SAMPLES "Build C++ sample programs" ON)
option(E8Q_BUILD_TESTS "Build C++ tests" ON)
option(E8Q_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(E8Q_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(
    E8Q_ENABLE_NATIVE_OPTIMIZATION
    "Optimize generic code for the build machine"
    ON
)

# Compile exactly one SIMD backend. generic = scalar TUs only; avx2 / avx512
# add that backend's kernel TUs. Runtime dispatch never calls a kernel that
# was not compiled. Default avx2 (paper AVX2 ledger).
set(E8Q_ISA "avx2" CACHE STRING "Compile-time ISA: generic, avx2, or avx512")
set_property(CACHE E8Q_ISA PROPERTY STRINGS generic avx2 avx512)
string(TOLOWER "${E8Q_ISA}" E8Q_ISA_LOWER)

find_package(OpenMP REQUIRED)

add_library(e8q_compile_options INTERFACE)
target_compile_features(e8q_compile_options INTERFACE cxx_std_17)
target_compile_options(e8q_compile_options INTERFACE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-O3>
    $<$<AND:$<BOOL:${E8Q_ENABLE_NATIVE_OPTIMIZATION}>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-march=native>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-fno-math-errno>
)
target_link_libraries(e8q_compile_options INTERFACE OpenMP::OpenMP_CXX)
if(E8Q_ENABLE_SANITIZERS)
    target_compile_options(e8q_compile_options INTERFACE
        $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-fsanitize=address,undefined;-fno-omit-frame-pointer>
    )
    target_link_options(e8q_compile_options INTERFACE
        $<$<CXX_COMPILER_ID:Clang,GNU>:-fsanitize=address,undefined>
    )
endif()

function(e8q_enable_warnings target)
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-Wall;-Wextra;-Wconversion;-Wno-sign-conversion>"
    )
endfunction()

set(E8Q_GENERIC_SOURCES
    src/utils/cpu_features.cpp
    src/utils/rotator.cpp
    src/quantization/e8q_codec.cpp
    src/quantization/me8q_plan.cpp
    src/quantization/me8q_codec.cpp
    src/index/ivf_route.cpp
    src/index/e8q_scan.cpp
    src/index/e8q_build.cpp
    src/index/e8q_io.cpp
    src/index/e8q_search.cpp
    src/index/e8q_oracle.cpp
    src/index/me8q_common.cpp
    src/index/me8q_io.cpp
    src/index/me8q_measure.cpp
    src/index/me8q_build.cpp
    src/index/me8q_search.cpp
    src/index/me8q_fast.cpp
    src/simd/dispatch.cpp
    src/simd/kernels_generic.cpp
    src/simd/kernels_dispatch.cpp
    src/simd/scan_kernels_generic.cpp
    src/simd/scan_kernels_dispatch.cpp
    src/simd/fastscan_generic.cpp
    src/simd/fastscan_dispatch.cpp
    src/simd/me8q_score_generic.cpp
    src/simd/me8q_score_dispatch.cpp
    src/simd/e8q_search_generic.cpp
    src/simd/e8q_search_dispatch.cpp
    src/simd/me8q_prune_generic.cpp
    src/simd/me8q_prune_dispatch.cpp
)

set(E8Q_AVX2_KERNELS
    src/simd/kernels_avx2.cpp
    src/simd/scan_kernels_avx2.cpp
    src/simd/fastscan_avx2.cpp
    src/simd/me8q_score_avx2.cpp
    src/simd/e8q_search_avx2.cpp
    src/simd/me8q_prune_avx2.cpp
)

set(E8Q_AVX512_KERNELS
    src/simd/kernels_avx512.cpp
    src/simd/scan_kernels_avx512.cpp
    src/simd/fastscan_avx512.cpp
    src/simd/me8q_score_avx512.cpp
    src/simd/e8q_search_avx512.cpp
    src/simd/me8q_prune_avx512.cpp
)

set(E8Q_CORE_SOURCES ${E8Q_GENERIC_SOURCES})
set(E8Q_HAVE_AVX2 OFF)
set(E8Q_HAVE_AVX512 OFF)

if(E8Q_ISA_LOWER STREQUAL "generic")
    message(STATUS "e8qlib ISA: generic (scalar kernels only)")
    if(E8Q_ENABLE_NATIVE_OPTIMIZATION)
        message(WARNING "E8Q_ISA=generic with E8Q_ENABLE_NATIVE_OPTIMIZATION=ON may emit host ISA in generic TUs")
    endif()
elseif(E8Q_ISA_LOWER STREQUAL "avx2")
    message(STATUS "e8qlib ISA: avx2")
    list(APPEND E8Q_CORE_SOURCES ${E8Q_AVX2_KERNELS})
    set(E8Q_HAVE_AVX2 ON)
elseif(E8Q_ISA_LOWER STREQUAL "avx512")
    message(STATUS "e8qlib ISA: avx512")
    list(APPEND E8Q_CORE_SOURCES ${E8Q_AVX512_KERNELS})
    set(E8Q_HAVE_AVX512 ON)
else()
    message(FATAL_ERROR "E8Q_ISA must be generic, avx2, or avx512 (got '${E8Q_ISA}')")
endif()

add_library(e8q_core STATIC ${E8Q_CORE_SOURCES})

target_include_directories(e8q_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(e8q_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(e8q_core PUBLIC e8q_compile_options)
set_target_properties(e8q_core PROPERTIES
    EXPORT_NAME core
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME e8q
)
e8q_enable_warnings(e8q_core)

if(E8Q_HAVE_AVX2)
    target_compile_definitions(e8q_core PUBLIC E8Q_HAVE_AVX2)
    set_source_files_properties(${E8Q_AVX2_KERNELS} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma;-mpopcnt"
    )
endif()
if(E8Q_HAVE_AVX512)
    target_compile_definitions(e8q_core PUBLIC E8Q_HAVE_AVX512)
    set_source_files_properties(${E8Q_AVX512_KERNELS} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mavx512vbmi;-mfma"
    )
endif()

add_library(e8qlib::core ALIAS e8q_core)

install(TARGETS e8q_core e8q_compile_options
    EXPORT e8qlibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/e8qlib DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
    cmake/e8qlibConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/e8qlibConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/e8qlib
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/e8qlibConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/e8qlibConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/e8qlibConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/e8qlib
)
install(EXPORT e8qlibTargets
    FILE e8qlibTargets.cmake
    NAMESPACE e8qlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/e8qlib
)

if(E8Q_BUILD_SAMPLES)
    add_subdirectory(sample/cpp)
endif()

if(E8Q_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(E8Q_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()
