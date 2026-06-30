cmake_minimum_required(VERSION 3.20)
project(mc_ponytail LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)          # A-100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

find_package(Eigen3 REQUIRED NO_MODULE)

# Biblioteca principal: utils + CUDA
add_library(mc_core STATIC utils.cpp methods_cuda.cu)
target_include_directories(mc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mc_core PUBLIC Eigen3::Eigen cublas curand)
set_target_properties(mc_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON)

# Función auxiliar para registrar un ejecutable de ejemplo
function(add_ejemplo name)
    add_executable(${name} examples/${name}.cpp)
    target_link_libraries(${name} PRIVATE mc_core)
endfunction()

add_ejemplo(ejemplo01_european)
add_ejemplo(ejemplo02_asian)
add_ejemplo(ejemplo03_lookback)
add_ejemplo(ejemplo04_barrier)
add_ejemplo(ejemplo05_heston)
add_ejemplo(ejemplo06_dupire)
add_ejemplo(ejemplo07_basket_uncorr)
add_ejemplo(ejemplo08_basket_corr)
add_ejemplo(ejemplo09_asian_cv)
add_ejemplo(ejemplo10_dupire_cv)
add_ejemplo(ejemplo11_otm_is)
