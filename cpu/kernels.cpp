#include "kernels.hpp"

#include <stdexcept>

namespace mc::cpu {

namespace {

template <ModelKind MK>
struct SingleRow {
    static constexpr SingleFn fns[kNumPayoffs] = {
        &kernel_single<MK, PayoffKind::European>,
        &kernel_single<MK, PayoffKind::Asian>,
        &kernel_single<MK, PayoffKind::GeomAsian>,
        &kernel_single<MK, PayoffKind::Lookback>,
        &kernel_single<MK, PayoffKind::Barrier>,
        &kernel_single<MK, PayoffKind::Basket>,
    };
};

} // namespace

SingleFn select_single_kernel(ModelKind mk, PayoffKind pk) {
    const int p = (int)pk;
    switch (mk) {
    case ModelKind::GBM:    return SingleRow<ModelKind::GBM>::fns[p];
    case ModelKind::Heston: return SingleRow<ModelKind::Heston>::fns[p];
    case ModelKind::Dupire: return SingleRow<ModelKind::Dupire>::fns[p];
    default: break;
    }
    throw std::invalid_argument("select_single_kernel: modelo no soportado (use kernel_basket para cestas)");
}

void kernel_basket(const KCtx& c, const double* dW, int ld, double* Y, double* S) {
    constexpr int W = kLanes;
    const CpuModel& m = *c.m;
    const CpuPayoff& pp = *c.p;
    const int n = m.n_assets;
    const double h = c.h;

    for (int a = 0; a < n; a++)
        for (int l = 0; l < W; l++) S[a * W + l] = m.S0v[a];

    for (int k = 0; k < c.n_steps; k++) {
        const double* dw = dW + (size_t)k * n * ld;
        const double e_t = std::exp(-m.alpha * (k * h));
        for (int a = 0; a < n; a++) {
            const double* dwa = dw + (size_t)a * ld;
            const double S0a = m.S0v[a];
            for (int l = 0; l < W; l++)
                S[a * W + l] = euler_dupire(m, S[a * W + l], dwa[l], h, e_t, S0a);
        }
    }
    for (int l = 0; l < W; l++) {
        double sum = 0.0;
        for (int a = 0; a < n; a++) sum += S[a * W + l];
        const double mean = sum / n;
        Y[l] = terminal_payoff<PayoffKind::Basket>(pp, mean, 0.0, c.n_steps);
    }
}

} // namespace mc::cpu
