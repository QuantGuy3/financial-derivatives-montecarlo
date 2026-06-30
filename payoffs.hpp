#pragma once
#include <variant>
#include <cmath>

// Broadie, Glasserman, Kou (1997) "A Continuity Correction for Discrete Barrier Options"
// Mathematical Finance 7(4):325-349, ec. (1.2). Convierte el sesgo O(sqrt(h)) a O(h).
inline constexpr double BGK_BETA   = 0.5826;  // CPU
inline constexpr float  BGK_BETA_F = 0.5826f; // GPU

// ------------------//
// Structs de payoff //
// ------------------//

struct European  { double K, r, T; };  // max(S_T - K, 0)
struct Asian     { double K;        };  // max(Ā - K, 0),   Ā = (1/N) Σ S_{t_k}
struct GeomAsian { double K;        };  // max(G - K, 0),   G = exp((1/N) Σ log S_{t_k})
struct Lookback  { double sigma;    };  // S_T - min_{0≤t≤T} S_t
struct Barrier   { double K, B, sigma, r, T; }; // max(S_T-K,0)·1{max S_t < B}
struct Basket    { double K, r, T; int n_assets; }; // max(Ā_activos - K, 0)

using PayoffVariant = std::variant<
    European, Asian, GeomAsian, Lookback, Barrier, Basket>;

// ----------------------- //
// Auxiliares de consulta  //
// ----------------------- //

// Indica si el payoff requiere guardar la trayectoria completa (no solo el valor terminal)
inline bool payoff_needs_full_path(const PayoffVariant& pv) {
    return std::visit([](const auto& p) -> bool {
        using T = std::decay_t<decltype(p)>;
        return std::is_same_v<T, Asian>     ||
               std::is_same_v<T, GeomAsian> ||
               std::is_same_v<T, Lookback>  ||
               std::is_same_v<T, Barrier>;
    }, pv);
}

// Factor de descuento e^{-rT}; vale 1 para payoffs sin tasa de interés explícita
inline double payoff_discount(const PayoffVariant& pv) {
    return std::visit([](const auto& p) -> double {
        using T = std::decay_t<decltype(p)>;
        if constexpr (
            std::is_same_v<T, European> ||
            std::is_same_v<T, Barrier>  ||
            std::is_same_v<T, Basket>
        ) return std::exp(-p.r * p.T);
        return 1.0;
    }, pv);
}
