#pragma once
#include <vector>
#include <cmath>
#include <variant>

// La Cholesky cabe en _constant_ memory si d*d*sizeof(float) <= 64KB → d = 128
// Usamos 64 como margen de seguridad con espacio para otros campos del struct.
inline constexpr int CONST_DIM_MAX  = 64;
inline constexpr int D_MAX_SOBOL    = 20000; // Límite de dimensión de secuencias Sobol

// -------------------------------------------//
// Structs con los parámetros de los ejemplos  //
// -------------------------------------------//

struct GBMParams {
    // dS = μ·S·dt + σ·S·dW₁     ← precio
    double S0    = 100.0;
    double mu    = 0.05;
    double sigma = 0.20;
    double T     = 1.0;
};

struct HestonParams {
    // dS = μ·S·dt + √v·S·dW₁        ← precio
    // dv = κ(θ - v)·dt + ξ·√v·dW₂   ← varianza
    double S0    = 100.0;
    double mu    = 0.05;
    double kappa = 2.0;
    double theta = 0.04;
    double xi    = 0.5;
    double rho   = -0.9;
    double v0    = 0.04;
    double T     = 1.0;

    // Descomposición de Cholesky 2×2 para los incrementos correlacionados:
    // dW₁ = Z₁,  dW₂ = ρ·Z₁ + √(1-ρ²)·Z₂
    double L[2][2] = {};

    void compute_cholesky() {
        L[0][0] = 1.0;   L[0][1] = 0.0;
        L[1][0] = rho;   L[1][1] = std::sqrt(1.0 - rho * rho);
    }
};

struct DupireLocalParams {
    // dS = μ·S·dt + σ_loc(S,t)·S·dW               ← precio
    // σ_loc(S,t) = σ₀ · exp(-α·t) · (S/S₀)^(β-1)  ← volatilidad local
    double S0     = 100.0;
    double mu     = 0.05;
    double sigma0 = 0.20;
    double alpha  = 0.5;
    double beta_d = 0.7;
    double T      = 1.0;
};

struct MultiDupireParams {
    // dSᵢ = μ·Sᵢ·dt + σ_loc(Sᵢ,t)·Sᵢ·dWᵢ    para i = 1,...,n
    // σ_loc(Sᵢ,t) = σ₀ · exp(-α·t) · (Sᵢ/S₀ᵢ)^(β-1)
    int    n           = 100;
    double mu          = 0.05;
    double sigma0      = 0.20;
    double alpha       = 0.5;
    double beta_d      = 0.7;
    double T           = 1.0;
    bool   uncorrelated = false; // true → ρ = I, omite la multiplicación Cholesky
    std::vector<double> S0;      // longitud n
    std::vector<double> L;       // Cholesky n×n, fila principal
};

using ModelVariant = std::variant<
    GBMParams, HestonParams, DupireLocalParams, MultiDupireParams>;

// ---------------------------------//
// Auxiliares para acceder a datos  //
// ---------------------------------//

// Dimensión del movimiento browniano (número de procesos de Wiener independientes)
inline int model_noise_dim(const ModelVariant& mv) {
    return std::visit([](const auto& m) -> int {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, GBMParams>)          return 1;
        if constexpr (std::is_same_v<T, HestonParams>)        return 2;
        if constexpr (std::is_same_v<T, DupireLocalParams>)   return 1;
        if constexpr (std::is_same_v<T, MultiDupireParams>)   return m.n;
        return 1;
    }, mv);
}

inline double model_T(const ModelVariant& mv) {
    return std::visit([](const auto& m) { return m.T; }, mv);
}

inline double model_S0(const ModelVariant& mv) {
    return std::visit([](const auto& m) -> double {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, MultiDupireParams>)
            return m.S0.empty() ? 100.0 : m.S0[0];
        else
            return m.S0;
    }, mv);
}
