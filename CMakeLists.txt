cmake_minimum_required(VERSION 3.22)
project(mc_definitivo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

# ---------------------------------------------------------------------------
# Opciones
# ---------------------------------------------------------------------------
set(MC_ENABLE_CUDA "AUTO" CACHE STRING "Backend CUDA: AUTO (si hay nvcc), ON (obligatorio) u OFF")
set_property(CACHE MC_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
set(MC_ARCH "" CACHE STRING "Opt-in: 'native' compila el motor CPU con -march=native")
option(MC_BUILD_TESTS    "Compilar la suite de tests (doctest)" ON)
option(MC_BUILD_EXAMPLES "Compilar los ejemplos (backend CPU y, si hay nvcc, GPU)" ON)
option(MC_BUILD_BENCH    "Compilar bench_cpu (benchmarks del motor CPU)" ON)
option(MC_BUILD_GUI      "Compilar mc_gui (interfaz web local)" ON)

# ---------------------------------------------------------------------------
# CUDA opcional: el build solo-CPU debe funcionar sin nvcc, sin Eigen ni lapack.
# ---------------------------------------------------------------------------
set(MC_HAS_CUDA OFF)
if(NOT MC_ENABLE_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "80")   # A-100 sm_80
        endif()
        set(MC_HAS_CUDA ON)
    elseif(MC_ENABLE_CUDA STREQUAL "ON")
        message(FATAL_ERROR "MC_ENABLE_CUDA=ON pero no se encontró nvcc")
    endif()
endif()
message(STATUS "Backend CUDA: ${MC_HAS_CUDA}")

find_package(Threads REQUIRED)

# MinGW: enlazado estático de libstdc++/libgcc/winpthread. Evita 0xc0000139 cuando en el
# PATH conviven varias distribuciones de MinGW/MSYS2 (DLLs de runtime incompatibles) y
# permite lanzar los .exe fuera de la shell de MSYS.
if(MINGW)
    add_link_options(-static)
endif()

# Avisos comunes (sin -Werror: los CUDA/MSVC añaden ruido propio)
add_library(mc_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mc_warnings INTERFACE -Wall -Wextra)
elseif(MSVC)
    target_compile_options(mc_warnings INTERFACE /W3)
endif()

# ---------------------------------------------------------------------------
# Núcleo: tipos, fórmulas analíticas, BB/PCA (CPU), estadísticos, barrido
# ---------------------------------------------------------------------------
add_library(mc_core STATIC utils.cpp sweep.cpp reference_prices.cpp)
target_include_directories(mc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mc_core PUBLIC Threads::Threads PRIVATE mc_warnings)

# ---------------------------------------------------------------------------
# Motor CPU multihilo (sin dependencias externas).
# -ffp-contract=off y sin fast-math: el contrato de determinismo (mismo binario => mismos
# bits con cualquier nº de hilos) no admite FMA contraidos de forma distinta por función.
# -march=native es opt-in (MC_ARCH=native): binarios portables por defecto.
# ---------------------------------------------------------------------------
add_library(mc_cpu STATIC
    cpu/thread_pool.cpp
    cpu/normal.cpp
    cpu/noise.cpp
    cpu/sobol.cpp
    cpu/joe_kuo_data.cpp
    cpu/params.cpp
    cpu/kernels.cpp
    cpu/path_sim.cpp
    cpu/mc_cpu.cpp
    cpu/qmc_noise.cpp
    cpu/qmc_cpu.cpp
    cpu/mlmc_cpu.cpp
    cpu/vr_cpu.cpp
    cpu/path_sampler.cpp
)
target_include_directories(mc_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mc_cpu PUBLIC mc_core PRIVATE mc_warnings)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mc_cpu PRIVATE -ffp-contract=off)
    if(MC_ARCH STREQUAL "native")
        target_compile_options(mc_cpu PUBLIC -march=native)
    endif()
elseif(MSVC)
    target_compile_options(mc_cpu PRIVATE /fp:precise)
endif()

# ---------------------------------------------------------------------------
# Backend CUDA (solo si hay nvcc)
# ---------------------------------------------------------------------------
if(MC_HAS_CUDA)
    add_library(mc_cuda STATIC methods_cuda.cu)
    target_include_directories(mc_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(mc_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>)
    target_link_libraries(mc_cuda PUBLIC cublas curand mc_core)
endif()

# ---------------------------------------------------------------------------
# Fachada: un único punto de entrada para CPU y GPU (engine/api.hpp).
# Sin CUDA se enlaza engine/cuda_stub.cpp (mismas firmas que methods_cuda.cuh, lanzan excepción):
# el adaptador y los ejemplos se compilan igual en cualquier máquina.
# ---------------------------------------------------------------------------
add_library(mc_engine STATIC
    engine/facade.cpp
    engine/plan.cpp
    engine/cuda_adapter.cpp
)
target_include_directories(mc_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mc_engine PUBLIC mc_cpu mc_core PRIVATE mc_warnings)
if(MC_HAS_CUDA)
    target_compile_definitions(mc_engine PUBLIC MC_HAS_CUDA=1)
    target_link_libraries(mc_engine PUBLIC mc_cuda)
else()
    target_sources(mc_engine PRIVATE engine/cuda_stub.cpp)
endif()

# ---------------------------------------------------------------------------
# Ejemplos (--backend=cpu|cuda --threads=N)
# ---------------------------------------------------------------------------
if(MC_BUILD_EXAMPLES)
    set(EXAMPLES
        01_european
        02_asian
        03_lookback
        04_barrier
        05_heston
        06_dupire
        07_basket_uncorr
        08_basket_corr
        09_asian_cv
        10_dupire_cv
        11_otm_is
    )
    foreach(EX ${EXAMPLES})
        string(REGEX MATCH "^[0-9]+" IDX "${EX}")
        set(TARGET "ejemplo${IDX}")
        add_executable(${TARGET} examples/ejemplo${EX}.cpp)
        target_link_libraries(${TARGET} PRIVATE mc_engine)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(MC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(MC_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# ---------------------------------------------------------------------------
# GUI: servidor HTTP local + interfaz web embebida
# ---------------------------------------------------------------------------
if(MC_BUILD_GUI)
    add_subdirectory(gui)
endif()
