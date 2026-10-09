#pragma once
// Normales estándar: inversa de la CDF (AS 241, precisión ~1e-16) y generadores.

#include "rng.hpp"
#include "simd.hpp"

#include <cmath>

namespace mc::cpu {

// Inversa de la CDF normal: Phi^{-1}(p), 0 < p < 1 (Wichura 1988, Algoritmo AS 241,
// PPND16; coeficientes públicos del algoritmo publicado). Precisión relativa ~1e-16.
// Fuera de (0,1) devuelve ±infinito.
double norm_inv_cdf(double p);

// CDF de la normal estándar.
inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Algoritmo con el que se generan las normales de un camino pseudoaleatorio.
enum class NormalMethod {
    BoxMuller,    // 2 uniformes -> 2 normales (log, sqrt, sincos)
    InverseCdf,   // 1 uniforme -> 1 normal (AS 241)
    Ziggurat,     // Marsaglia-Tsang con 256 capas: ~1 palabra aleatoria por normal, sin funciones transcendentes
};

// Rellena z[0..n) con normales N(0,1) i.i.d. consumiendo el generador g, una detrás de otra.
// Es el generador secuencial clásico; el motor NO lo usa (usa fill_normals_lanes).
void fill_normals(NormalMethod method, Xoshiro256pp& g, double* z, int n);

// Generador del motor, por bloque: Z[d*ld + l] = scale * (normal nº g.pos + d del carril l), d en
// [0,D), l en [0,n), n <= kLanes. Los carriles >= n no se tocan. Avanza g.pos en D.
//   * BoxMuller / InverseCdf: los mismos números que fill_normals camino a camino.
//   * Ziggurat: cada normal consume exactamente una palabra del generador del camino; la cuña y la
//     cola usan un generador auxiliar sembrado con (camino, d). No coincide con el Ziggurat secuencial
//     de fill_normals (otra tabla y otro uso de los bits), pero el resultado no depende del nivel SIMD,
//     del nº de carriles ni del troceado: la normal d del camino p es función de (semilla, flujo,
//     nivel, p, d).
void fill_normals_lanes(NormalMethod method, LaneRng& g, int n, double* Z, int ld, int D, double scale);

} // namespace mc::cpu
