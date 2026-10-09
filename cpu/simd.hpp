#pragma once
// Selección del juego de instrucciones en tiempo de ejecución.
//
// El binario se compila para x86-64 genérico (SSE2). Las pocas rutinas que dominan el tiempo
// (generador de normales, potencia de Dupire, inversa de la normal) tienen además una versión AVX2
// que se elige al arrancar si el procesador la admite. Las dos versiones de cada rutina hacen las
// mismas operaciones en el mismo orden y dan EXACTAMENTE los mismos bits (los tests lo comprueban),
// así que el resultado de una simulación no depende del procesador.

namespace mc::cpu {

enum class SimdLevel { Scalar = 0, Avx2 = 1 };

SimdLevel simd_level_available();        // lo máximo que admite esta máquina y esta compilación
SimdLevel simd_level();                  // el nivel en uso (por defecto, el máximo disponible)
void set_simd_level(SimdLevel level);    // para tests y mediciones; se recorta a simd_level_available()

} // namespace mc::cpu

// ---- uso interno (ficheros .cpp con intrínsecos) --------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
#define MC_X86_64 1
#endif

// Función compilada para AVX2 aunque el resto del fichero no lo esté.
#if defined(__GNUC__) || defined(__clang__)
#define MC_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define MC_TARGET_AVX2
#endif
