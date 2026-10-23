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

// ------------------------------------------------------------------------- //
// Modelos añadidos (se simulan en logaritmo: el paso es exacto en Merton y   //
// NIG; Bates usa truncamiento completo en la varianza). `mu` es la deriva    //
// riesgo neutro r - q, igual que en los modelos anteriores.                 //
// ------------------------------------------------------------------------- //

struct MertonParams {
    // dS/S = (μ - λ·k̄)·dt + σ·dW + (e^J - 1)·dN,  J ~ N(μ_J, σ_J²),  N ~ Poisson(λ)
    // k̄ = E[e^J - 1] = exp(μ_J + σ_J²/2) - 1 (compensador de los saltos)
    double S0     = 100.0;
    double mu     = 0.05;
    double sigma  = 0.15;
    double lambda = 0.5;
    double muJ    = -0.10;
    double sigmaJ = 0.15;
    double T      = 1.0;
};

struct BatesParams {
    // Heston con saltos de Merton:
    // dS/S = (μ - λ·k̄)·dt + √v·dW₁ + (e^J - 1)·dN
    // dv   = κ(θ - v)·dt + ξ·√v·dW₂,   d<W₁,W₂> = ρ·dt
    double S0     = 100.0;
    double mu     = 0.05;
    double kappa  = 1.5;
    double theta  = 0.04;
    double xi     = 0.5;
    double rho    = -0.7;
    double v0     = 0.04;
    double lambda = 0.5;
    double muJ    = -0.10;
    double sigmaJ = 0.15;
    double T      = 1.0;
};

struct SABRParams {
    // dF = α·F^β·dW₁ (sobre el forward), dα = ν·α·dW₂, d<W₁,W₂> = ρ·dt.
    // `alpha` es la volatilidad en el forward inicial: σ_ef = α·(F/F₀)^(β-1),
    // es decir, α_Hagan = alpha·F₀^(1-β).
    double S0    = 100.0;
    double mu    = 0.05;
    double alpha = 0.20;
    double beta  = 0.7;
    double nu    = 0.4;
    double rho   = -0.5;
    double T     = 1.0;
};

struct NIGParams {
    // Normal Inverse Gaussian (Cont-Tankov): X_t = θ·G_t + σ·W(G_t) con G un
    // subordinador IG de media t y varianza κ·t; S_t = S₀·exp((μ + ω)t + X_t),
    // ω = -(1 - √(1 - σ²κ - 2θκ))/κ (corrección de martingala).
    double S0    = 100.0;
    double mu    = 0.05;
    double sigma = 0.20;
    double theta = -0.10;
    double kappa = 0.20;
    double T     = 1.0;
};

using ModelVariant = std::variant<
    GBMParams, HestonParams, DupireLocalParams, MultiDupireParams,
    MertonParams, BatesParams, SABRParams, NIGParams>;

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
        // Normales por paso: principal + extras (nº de saltos y tamaño, varianza,
        // vol SABR, o las dos del muestreo IG de la NIG)
        if constexpr (std::is_same_v<T, MertonParams>)        return 3;
        if constexpr (std::is_same_v<T, BatesParams>)         return 4;
        if constexpr (std::is_same_v<T, SABRParams>)          return 2;
        if constexpr (std::is_same_v<T, NIGParams>)           return 3;
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
