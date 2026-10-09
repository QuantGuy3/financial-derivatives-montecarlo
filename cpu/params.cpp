#include "params.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace mc::cpu {

CpuModel make_cpu_model(const ModelVariant& mv) {
    CpuModel m;
    std::visit([&](const auto& p) {
        using T = std::decay_t<decltype(p)>;
        m.T = p.T;
        m.mu = p.mu;
        if constexpr (std::is_same_v<T, GBMParams>) {
            m.kind = ModelKind::GBM; m.noise_dim = 1;
            m.S0 = p.S0; m.sigma = p.sigma;
        } else if constexpr (std::is_same_v<T, HestonParams>) {
            m.kind = ModelKind::Heston; m.noise_dim = 2;
            m.S0 = p.S0; m.kappa = p.kappa; m.theta = p.theta; m.xi = p.xi;
            m.rho = p.rho; m.v0 = p.v0;
            // Se parte de rho (no de p.L, que exige llamar a compute_cholesky()).
            m.l21 = p.rho; m.l22 = std::sqrt(1.0 - p.rho * p.rho);
        } else if constexpr (std::is_same_v<T, DupireLocalParams>) {
            m.kind = ModelKind::Dupire; m.noise_dim = 1;
            m.S0 = p.S0; m.sigma0 = p.sigma0; m.alpha = p.alpha; m.beta_d = p.beta_d;
            m.sigma = p.sigma0;
        } else {
            static_assert(std::is_same_v<T, MultiDupireParams>);
            m.kind = ModelKind::MultiDupire;
            m.n_assets = p.n;
            m.noise_dim = p.n;
            if ((int)p.S0.size() < p.n)
                throw std::runtime_error("MultiDupireParams.S0 tiene " + std::to_string(p.S0.size())
                                         + " elementos, se esperaban " + std::to_string(p.n));
            m.S0v.assign(p.S0.begin(), p.S0.begin() + p.n);
            m.S0 = m.S0v[0];
            m.sigma0 = p.sigma0; m.alpha = p.alpha; m.beta_d = p.beta_d;
            m.uncorrelated = p.uncorrelated || p.L.empty();
            if (!m.uncorrelated) {
                if ((long long)p.L.size() < (long long)p.n * p.n)
                    throw std::runtime_error("MultiDupireParams.L tiene " + std::to_string(p.L.size())
                                             + " elementos, se esperaban " + std::to_string((long long)p.n * p.n));
                m.L.assign(p.L.begin(), p.L.begin() + (long long)p.n * p.n);
            }
        }
    }, mv);
    return m;
}

CpuPayoff make_cpu_payoff(const PayoffVariant& pv) {
    CpuPayoff c;
    std::visit([&](const auto& p) {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, European>) {
            c.kind = PayoffKind::European; c.K = p.K; c.r = p.r; c.discount = std::exp(-p.r * p.T);
        } else if constexpr (std::is_same_v<T, Asian>) {
            c.kind = PayoffKind::Asian; c.K = p.K; c.discount = 1.0;
        } else if constexpr (std::is_same_v<T, GeomAsian>) {
            c.kind = PayoffKind::GeomAsian; c.K = p.K; c.discount = 1.0;
        } else if constexpr (std::is_same_v<T, Lookback>) {
            c.kind = PayoffKind::Lookback; c.discount = 1.0;
            c.bgk = true; c.sigma_bgk = p.sigma;
        } else if constexpr (std::is_same_v<T, Barrier>) {
            c.kind = PayoffKind::Barrier; c.K = p.K; c.B = p.B; c.r = p.r;
            c.discount = std::exp(-p.r * p.T);
            c.bgk = true; c.sigma_bgk = p.sigma;
        } else {
            static_assert(std::is_same_v<T, Basket>);
            c.kind = PayoffKind::Basket; c.K = p.K; c.r = p.r; c.discount = std::exp(-p.r * p.T);
            c.n_assets = p.n_assets;
        }
    }, pv);
    return c;
}

} // namespace mc::cpu
