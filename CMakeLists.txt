cmake_minimum_required(VERSION 3.22)
project(mc_definitivo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------------------- #
# CUDA es opcional: sin nvcc se compila la parte CPU (programa "estructurados"
# con motor CPU y tests). Con nvcc se compila además todo el motor GPU.
# ------------------------------------------------------------------------- #
option(MC_USAR_CUDA "Compilar los motores GPU (requiere nvcc)" ON)
set(MC_CON_CUDA OFF)
if(MC_USAR_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "80")   # A-100 sm_80
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(MC_CON_CUDA ON)
    else()
        message(WARNING "No se ha encontrado nvcc: se compila solo la parte CPU (estructurados + tests).")
    endif()
endif()

find_package(Threads REQUIRED)

# ------------------------------------------------------------------------- #
# Precios de referencia analíticos/semianalíticos (C++ puro)
# ------------------------------------------------------------------------- #
add_library(mc_referencias STATIC referencias.cpp)
target_include_directories(mc_referencias PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------------------------------- #
# Productos estructurados: definición (JSON), plantillas, asistente y motores
# ------------------------------------------------------------------------- #
set(EST_FUENTES
    estructurados/json.cpp
    estructurados/expresion.cpp
    estructurados/producto.cpp
    estructurados/perfil.cpp
    estructurados/vanilla.cpp
    estructurados/cotizaciones.cpp
    estructurados/replicacion.cpp
    estructurados/optimizacion.cpp
    estructurados/mercado.cpp
    estructurados/calibracion.cpp
    estructurados/historico.cpp
    estructurados/ficha.cpp
    estructurados/plantillas.cpp
    estructurados/asistente.cpp
    estructurados/valoracion.cpp
    estructurados/motor_cpu.cpp
    estructurados/sobol_host.cpp
)
if(MC_CON_CUDA)
    list(APPEND EST_FUENTES estructurados/motor_gpu.cu)
else()
    list(APPEND EST_FUENTES estructurados/motor_gpu_stub.cpp)
endif()
add_library(estructurados_core STATIC ${EST_FUENTES})
target_include_directories(estructurados_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(estructurados_core PUBLIC mc_referencias Threads::Threads)
if(MC_CON_CUDA)
    # cuRAND en host aporta los vectores de Sobol (QMC también en CPU)
    target_compile_definitions(estructurados_core PUBLIC MC_CON_CURAND=1)
    target_link_libraries(estructurados_core PUBLIC CUDA::curand CUDA::cudart)
    target_compile_options(estructurados_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>)
endif()

add_executable(estructurados estructurados/main.cpp)
target_link_libraries(estructurados PRIVATE estructurados_core)

enable_testing()
add_executable(test_estructurados tests/test_estructurados.cpp)
target_link_libraries(test_estructurados PRIVATE estructurados_core)
target_compile_definitions(test_estructurados PRIVATE MC_DIR_FUENTES="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME estructurados_cpu COMMAND test_estructurados rapido)

# ------------------------------------------------------------------------- #
# Motor GPU original (MC / QMC / MLMC / MLQMC) y ejemplos
# ------------------------------------------------------------------------- #
if(MC_CON_CUDA)
    set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Directorio de cabeceras de Eigen 3")

    add_library(mc_core STATIC utils.cpp)
    target_include_directories(mc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(mc_core PUBLIC lapack blas)

    add_library(mc_cuda STATIC methods_cuda.cu)
    target_include_directories(mc_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
    target_compile_options(mc_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>)
    target_link_libraries(mc_cuda PUBLIC cublas curand mc_core mc_referencias)

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
        12_merton
        13_bates
        14_sabr
        15_nig
    )

    foreach(EX ${EXAMPLES})
        string(REGEX MATCH "^[0-9]+" IDX "${EX}")
        set(TARGET "ejemplo${IDX}")
        add_executable(${TARGET} examples/ejemplo${EX}.cpp)
        target_link_libraries(${TARGET} PRIVATE mc_cuda)
    endforeach()
endif()
