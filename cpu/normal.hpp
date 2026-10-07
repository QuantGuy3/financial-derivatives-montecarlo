#pragma once
// Normales estándar: inversa de la CDF (AS 241, precisión ~1e-16) y generadores.

#include "rng.hpp"

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
};

// Rellena z[0..n) con normales N(0,1) i.i.d. consumiendo el generador g.
void fill_normals(NormalMethod method, Xoshiro256pp& g, double* z, int n);

} // namespace mc::cpu
