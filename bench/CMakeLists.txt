# Benchmarks del motor CPU (no se ejecutan en CTest; ver docs/perf/).
find_package(Git QUIET)
set(MC_GIT_HASH "desconocido")
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    OUTPUT_VARIABLE MC_GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_dirty)
        set(MC_GIT_HASH "${MC_GIT_HASH}+dirty")
    endif()
endif()

add_executable(bench_cpu bench_cpu.cpp bench_workloads.cpp)
target_link_libraries(bench_cpu PRIVATE mc_cpu mc_core mc_warnings)
target_compile_definitions(bench_cpu PRIVATE
    MC_GIT_HASH="${MC_GIT_HASH}"
    MC_BUILD_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${MC_ARCH}")

add_executable(bench_micro bench_micro.cpp)
target_link_libraries(bench_micro PRIVATE mc_cpu mc_core mc_warnings)
