cmake_minimum_required(VERSION 3.22)
project(mc_definitivo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES "80")   # A-100 sm_80

add_library(mc_core STATIC utils.cpp)
target_include_directories(mc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} /usr/include/eigen3)
target_link_libraries(mc_core PUBLIC lapack blas)

add_library(mc_cuda STATIC methods_cuda.cu)
target_include_directories(mc_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} /usr/include/eigen3)
target_compile_options(mc_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>)
target_link_libraries(mc_cuda PUBLIC cublas curand mc_core)

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
    target_link_libraries(${TARGET} PRIVATE mc_cuda)
endforeach()
