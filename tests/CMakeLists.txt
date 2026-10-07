# Suite de tests (doctest vendorizado en third_party/doctest). Una sola ejecutable con
# varias "test suites" de doctest, registradas en CTest con etiquetas:
#   fast          pruebas unitarias rápidas (segundos)
#   stat          pruebas estadísticas (semillas fijas, umbrales calibrados)
#   determinism   resultados bit a bit idénticos con cualquier nº de hilos
add_executable(mc_tests
    test_main.cpp
    test_sweep_format.cpp
    test_utils.cpp
    test_rng_normal.cpp
    test_thread_pool.cpp
    test_sobol.cpp
    test_mc_cpu.cpp
    test_qmc_cpu.cpp
    test_mlmc_cpu.cpp
    test_vr_cpu.cpp
)
target_include_directories(mc_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party/doctest)
target_compile_definitions(mc_tests PRIVATE MC_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(mc_tests PRIVATE mc_cpu mc_core mc_warnings)

foreach(SUITE fast stat determinism)
    add_test(NAME mc_${SUITE} COMMAND mc_tests --test-suite=${SUITE})
    set_tests_properties(mc_${SUITE} PROPERTIES LABELS ${SUITE})
endforeach()
