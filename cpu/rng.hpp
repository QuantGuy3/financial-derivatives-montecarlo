#pragma once
// Generadores pseudoaleatorios del motor CPU.
//
// Diseño (ver plan): la semilla es POR CAMINO, no por chunk ni por hilo. El estado de
// xoshiro256++ de cada camino se deriva por hash de (semilla, etiqueta de flujo, nivel,
// índice de camino). Así el valor simulado de un camino depende solo de su índice:
//   * el resultado no depende del nº de hilos ni de cómo se reparte el trabajo;
//   * propiedad de prefijo: los primeros n caminos son los mismos sea cual sea N, por lo
//     que ampliar N (MLMC "extra", afinar precisión desde la GUI) es consistente.

#include "lanes.hpp"

#include <bit>
#include <cstdint>

namespace mc::cpu {

// ---- SplitMix64 (Vigna) ---------------------------------------------------------------
inline constexpr uint64_t splitmix64_next(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Finalizador de 64 bits (mezcla sin incremento): sirve para hashear tuplas de enteros.
inline constexpr uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Etiquetas de flujo: separan los números aleatorios de fases distintas de un algoritmo
// (p. ej. piloto frente a corrida principal) aunque compartan semilla.
enum class Stream : uint64_t {
    Pilot    = 1,
    Main     = 2,
    Extra    = 3,
    Control  = 4,
    Paths    = 5,   // trayectorias de la GUI
    Test     = 100,
};

// ---- xoshiro256++ (Blackman & Vigna) -----------------------------------------------------
struct Xoshiro256pp {
    uint64_t s[4];

    static inline uint64_t rotl(uint64_t x, int k) { return std::rotl(x, k); }

    inline uint64_t next() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // U[0,1) con 53 bits
    inline double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    // U(0,1) estricto: nunca 0 ni 1 (seguro para log y para la inversa de la normal)
    inline double uniform_open() { return ((double)(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

    // Conversión de una palabra de 64 bits a uniforme (las mismas fórmulas que uniform/uniform_open).
    static inline double to_uniform(uint64_t bits) { return (double)(bits >> 11) * (1.0 / 9007199254740992.0); }
    static inline double to_uniform_open(uint64_t bits) {
        return ((double)(bits >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }

    // Parte del hash que no depende del camino: se calcula una vez por flujo, no una vez por camino.
    static inline constexpr uint64_t path_hash_base(uint64_t seed, Stream stream, uint64_t level) {
        uint64_t h = mix64(seed + 0x2545F4914F6CDD1DULL);
        h = mix64(h ^ ((uint64_t)stream * 0xD6E8FEB86659FD93ULL));
        return mix64(h ^ (level * 0xA0761D6478BD642FULL + 0x1234567ULL));
    }
    static inline constexpr uint64_t path_hash(uint64_t base, uint64_t path) {
        return mix64(base ^ (path * 0xE7037ED1A0B428DBULL + 0x9E3779B97F4A7C15ULL));
    }

    // Estado derivado de la tupla (semilla, flujo, nivel, camino).
    static inline Xoshiro256pp for_path(uint64_t seed, Stream stream, uint64_t level, uint64_t path) {
        Xoshiro256pp g;
        uint64_t sm = path_hash(path_hash_base(seed, stream, level), path);
        g.s[0] = splitmix64_next(sm);
        g.s[1] = splitmix64_next(sm);
        g.s[2] = splitmix64_next(sm);
        g.s[3] = splitmix64_next(sm);
        return g;
    }
};

// ---- kLanes generadores en paso sincronizado --------------------------------------------------
//
// Un generador por camino del bloque, con el estado en disposición SoA. La recurrencia de xoshiro
// es una cadena de dependencias (cada palabra necesita el estado anterior); con kLanes cadenas
// independientes intercaladas el procesador las solapa, y el compilador vectoriza next_all. El
// carril l produce EXACTAMENTE la misma secuencia que Xoshiro256pp::for_path(..., first + l).
struct LaneRng {
    uint64_t s0[kLanes], s1[kLanes], s2[kLanes], s3[kLanes];
    uint64_t key[kLanes];   // hash del camino: semilla de los sorteos auxiliares (ver normal.cpp)
    uint64_t pos = 0;       // normales ya generadas por carril (índice de la siguiente)

    // Carril l <- generador del camino first + l (base = Xoshiro256pp::path_hash_base).
    inline void seed(uint64_t base, uint64_t first) {
        pos = 0;
        for (int l = 0; l < kLanes; l++) {
            uint64_t sm = Xoshiro256pp::path_hash(base, first + (uint64_t)l);
            key[l] = sm;
            s0[l] = splitmix64_next(sm);
            s1[l] = splitmix64_next(sm);
            s2[l] = splitmix64_next(sm);
            s3[l] = splitmix64_next(sm);
        }
    }

    // Una palabra de 64 bits de cada carril.
    inline void next_all(uint64_t* out) {
        for (int l = 0; l < kLanes; l++) {
            out[l] = std::rotl(s0[l] + s3[l], 23) + s0[l];
            const uint64_t t = s1[l] << 17;
            s2[l] ^= s0[l];
            s3[l] ^= s1[l];
            s1[l] ^= s2[l];
            s0[l] ^= s3[l];
            s2[l] ^= t;
            s3[l] = std::rotl(s3[l], 45);
        }
    }

    inline Xoshiro256pp get(int l) const { return Xoshiro256pp{{s0[l], s1[l], s2[l], s3[l]}}; }
    inline void set(int l, const Xoshiro256pp& g) { s0[l] = g.s[0]; s1[l] = g.s[1]; s2[l] = g.s[2]; s3[l] = g.s[3]; }
};

} // namespace mc::cpu
