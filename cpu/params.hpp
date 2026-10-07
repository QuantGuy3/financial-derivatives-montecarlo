#pragma once
// Parámetros en doble precisión, "aplanados", de los modelos y payoffs para el motor CPU.
// Equivalente a KernelParams de methods_cuda.cu (float en __constant__).

#include "../models.hpp"
#include "../payoffs.hpp"

#include <type_traits>
#include <vector>

namespace mc::cpu {

// Mismos valores enteros que ModelKind/PayoffKind de methods_cuda.cu (que viven solo en el
// .cu y no se tocan). Coinciden con el índice de las alternativas de los std::variant.
enum class ModelKind : int { GBM = 0, Heston = 1, Dupire = 2, MultiDupire = 3 };
enum class PayoffKind : int { European = 0, Asian = 1, GeomAsian = 2, Lookback = 3, Barrier = 4, Basket = 5 };
inline constexpr int kNumModels  = 4;
inline constexpr int kNumPayoffs = 6;

static_assert(std::is_same_v<std::variant_alternative_t<0, ModelVariant>, GBMParams>);
static_assert(std::is_same_v<std::variant_alternative_t<1, ModelVariant>, HestonParams>);
static_assert(std::is_same_v<std::variant_alternative_t<2, ModelVariant>, DupireLocalParams>);
static_assert(std::is_same_v<std::variant_alternative_t<3, ModelVariant>, MultiDupireParams>);
static_assert(std::is_same_v<std::variant_alternative_t<0, PayoffVariant>, European>);
static_assert(std::is_same_v<std::variant_alternative_t<1, PayoffVariant>, Asian>);
static_assert(std::is_same_v<std::variant_alternative_t<2, PayoffVariant>, GeomAsian>);
static_assert(std::is_same_v<std::variant_alternative_t<3, PayoffVariant>, Lookback>);
static_assert(std::is_same_v<std::variant_alternative_t<4, PayoffVariant>, Barrier>);
static_assert(std::is_same_v<std::variant_alternative_t<5, PayoffVariant>, Basket>);

struct CpuModel {
    ModelKind kind = ModelKind::GBM;
    int    noise_dim = 1;      // procesos de Wiener independientes por paso
    double T = 1.0, S0 = 100.0, mu = 0.05;
    double sigma = 0.2;                                       // GBM
    double kappa = 0, theta = 0, xi = 0, rho = 0, v0 = 0;     // Heston
    double l21 = 0, l22 = 1;                                  // Cholesky 2x2: dW2 = l21*a1 + l22*a2
    double sigma0 = 0, alpha = 0, beta_d = 1;                 // Dupire local
    // Cesta (MultiDupire)
    int    n_assets = 1;
    bool   uncorrelated = true;
    std::vector<double> S0v;   // S0 por activo
    std::vector<double> L;     // Cholesky n x n fila-principal (vacío si no correlada)
};

struct CpuPayoff {
    PayoffKind kind = PayoffKind::European;
    double K = 0, B = 0, r = 0, discount = 1.0;
    bool   bgk = false;        // corrección de Broadie-Glasserman-Kou (Lookback, Barrier)
    double sigma_bgk = 0;
    int    n_assets = 1;
};

// Qué se evalúa sobre cada camino. Plain = el payoff del modelo; CV* = payoff principal con
// variable de control (Y_cv = Y_main - beta·(Y_ctrl - E_ctrl)), solo las dos parejas que
// implementa la GPU; IsGbmCall = importance sampling de la call europea bajo GBM
// (desplazamiento z_step en cada incremento, con cociente de verosimilitud).
struct EvalSpec {
    enum class Kind { Plain, CvAsianGeom, CvDupireGbm, IsGbmCall };
    Kind   kind = Kind::Plain;
    double beta = 0.0;      // CV
    double E_ctrl = 0.0;    // CV: valor esperado analítico del control
    double z_star = 0.0;    // IS: desplazamiento total; por paso es z_star/sqrt(n_pasos)
};

CpuModel  make_cpu_model(const ModelVariant& mv);
CpuPayoff make_cpu_payoff(const PayoffVariant& pv);

} // namespace mc::cpu
