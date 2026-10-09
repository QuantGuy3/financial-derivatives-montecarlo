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
#if defined(MC_X86_64) && (defined(__GNUC__) || defined(__clang__))
#define MC_TARGET_AVX2 __attribute__((target("avx2")))
#define MC_ALWAYS_INLINE inline __attribute__((always_inline))
// Con GCC/Clang un bucle escrito una vez puede compilarse dos veces: el cuerpo va en una función
// MC_ALWAYS_INLINE y se llama desde un envoltorio normal y desde otro MC_TARGET_AVX2, donde el
// compilador lo vectoriza a 256 bits. Solo vale para bucles cuyos acumuladores son independientes
// (el orden de las sumas de cada elemento no cambia), así que los bits son los mismos.
#define MC_HAVE_AVX2_CLONES 1
#else
#define MC_TARGET_AVX2
#if defined(_MSC_VER)
#define MC_ALWAYS_INLINE __forceinline
#else
#define MC_ALWAYS_INLINE inline
#endif
#endif
