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
        const double e_t = c.exp_at[(size_t)k];
        for (int a = 0; a < n; a++) {
            euler_dupire_block(m, S + (size_t)a * W, dw + (size_t)a * ld, h, e_t, m.S0v[a]);
        }
    }
    for (int l = 0; l < W; l++) {
        double sum = 0.0;
        for (int a = 0; a < n; a++) sum += S[a * W + l];
        const double mean = sum / n;
        Y[l] = terminal_payoff<PayoffKind::Basket>(pp, mean, 0.0, c.n_steps);
    }
}

namespace {

template <ModelKind MK>
struct CoupledRow {
    static constexpr CoupledFn fns[kNumPayoffs] = {
        &kernel_coupled<MK, PayoffKind::European>,
        &kernel_coupled<MK, PayoffKind::Asian>,
        &kernel_coupled<MK, PayoffKind::GeomAsian>,
        &kernel_coupled<MK, PayoffKind::Lookback>,
        &kernel_coupled<MK, PayoffKind::Barrier>,
        &kernel_coupled<MK, PayoffKind::Basket>,
    };
};

} // namespace

CoupledFn select_coupled_kernel(ModelKind mk, PayoffKind pk) {
    const int p = (int)pk;
    switch (mk) {
    case ModelKind::GBM:    return CoupledRow<ModelKind::GBM>::fns[p];
    case ModelKind::Heston: return CoupledRow<ModelKind::Heston>::fns[p];
    case ModelKind::Dupire: return CoupledRow<ModelKind::Dupire>::fns[p];
    default: break;
    }
    throw std::invalid_argument("select_coupled_kernel: MLMC no admite cestas multi-activo");
}

// ---- evaluadores con reducción de varianza ----------------------------------------------------------

namespace {

// GBM: Asian aritmética (principal) + Asian geométrica (control) sobre el mismo camino.
void eval_cv_asian(const KCtx& c, const double* dW, int ld, double* Y0, double* Y1) {
    constexpr int W = kLanes;
    const CpuModel& m = *c.m;
    double S[W], arith[W], logsum[W];
    for (int l = 0; l < W; l++) { S[l] = m.S0; arith[l] = 0.0; logsum[l] = 0.0; }
    for (int k = 0; k < c.n_steps; k++) {
        const double* dw = dW + (size_t)k * ld;
        for (int l = 0; l < W; l++) {
            S[l] = euler_gbm(m, S[l], dw[l], c.h);
            arith[l] += S[l];
            logsum[l] += std::log(S[l]);
        }
    }
    const double K = c.p->K;
    for (int l = 0; l < W; l++) {
        Y0[l] = std::max(arith[l] / c.n_steps - K, 0.0);
        Y1[l] = std::max(std::exp(logsum[l] / c.n_steps) - K, 0.0);
    }
}

// Dupire europea (principal) + GBM con sigma0 europea (control), mismos incrementos.
void eval_cv_dupire(const KCtx& c, const double* dW, int ld, double* Y0, double* Y1) {
    constexpr int W = kLanes;
    const CpuModel& m = *c.m;
    double Sd[W], Sg[W];
    for (int l = 0; l < W; l++) Sd[l] = Sg[l] = m.S0;
    for (int k = 0; k < c.n_steps; k++) {
        const double* dw = dW + (size_t)k * ld;
        euler_dupire_block(m, Sd, dw, c.h, c.exp_at[(size_t)k], m.S0);
        for (int l = 0; l < W; l++) Sg[l] = euler_gbm(m, Sg[l], dw[l], c.h);
    }
    const double K = c.p->K, disc = c.p->discount;
    for (int l = 0; l < W; l++) {
        Y0[l] = std::max(Sd[l] - K, 0.0) * disc;
        Y1[l] = std::max(Sg[l] - K, 0.0) * disc;
    }
}

// Importance sampling GBM + call: dW es el incremento SIN desplazar; z_k = dW_k/sqrt(h) (válido
// para cualquier construcción del ruido: Raw, BB o PCA).
void eval_is_call(const KCtx& c, const double* dW, int ld, double* Y0, double* /*Y1*/) {
    constexpr int W = kLanes;
    const CpuModel& m = *c.m;
    double S[W], zsum[W];
    for (int l = 0; l < W; l++) { S[l] = m.S0; zsum[l] = 0.0; }
    for (int k = 0; k < c.n_steps; k++) {
        const double* dw = dW + (size_t)k * ld;
        for (int l = 0; l < W; l++) {
            zsum[l] += dw[l] / c.sqrt_h;
            S[l] = euler_gbm(m, S[l], dw[l] + c.z_step * c.sqrt_h, c.h);
        }
    }
    const double K = c.p->K, disc = c.p->discount;
    for (int l = 0; l < W; l++) {
        const double lr = std::exp(-c.z_step * zsum[l] - 0.5 * c.z_step * c.z_step * c.n_steps);
        Y0[l] = std::max(S[l] - K, 0.0) * disc * lr;
    }
}

// MLMC + CV (GBM, Asian aritmética + control geométrica) aplicado DENTRO de cada nivel.
void coupled_cv_asian(const CKCtx& c, const double* dW, int ld, double* Yf, double* Yc) {
    constexpr int W = kLanes;
    const CpuModel& m = *c.m;
    double Sf[W], Sc[W], af[W], lf[W], ac[W], lc[W], acc[W];
    for (int l = 0; l < W; l++) { Sf[l] = Sc[l] = m.S0; af[l] = lf[l] = ac[l] = lc[l] = acc[l] = 0.0; }
    for (int k = 0; k < c.n_fine; k++) {
        const double* dw = dW + (size_t)k * ld;
        for (int l = 0; l < W; l++) {
            acc[l] += dw[l];
            Sf[l] = euler_gbm(m, Sf[l], dw[l], c.h_f);
            af[l] += Sf[l];
            lf[l] += std::log(Sf[l]);
        }
        if ((k + 1) % c.M == 0) {
            for (int l = 0; l < W; l++) {
                Sc[l] = Sc[l] + m.mu * Sc[l] * c.h_c + m.sigma * Sc[l] * acc[l];
                acc[l] = 0.0;
                if (c.n_coarse > 0) { ac[l] += Sc[l]; lc[l] += std::log(Sc[l]); }
            }
        }
    }
    const double K = c.p->K;
    for (int l = 0; l < W; l++) {
        const double ya = std::max(af[l] / c.n_fine - K, 0.0);
        const double yg = std::max(std::exp(lf[l] / c.n_fine) - K, 0.0);
        Yf[l] = ya - c.beta * (yg - c.E_ctrl);
        if (c.n_coarse > 0) {
            const double yac = std::max(ac[l] / c.n_coarse - K, 0.0);
            const double ygc = std::max(std::exp(lc[l] / c.n_coarse) - K, 0.0);
            Yc[l] = yac - c.beta * (ygc - c.E_ctrl);
        } else {
            Yc[l] = 0.0;
        }
    }
}

// MLMC + IS (GBM + call): desplaza los incrementos finos; el grueso agrega los ya desplazados; un
// único cociente de verosimilitud (del camino fino) pondera ambos.
void coupled_is_call(const CKCtx& c, const double* dW, int ld, double* Yf, double* Yc) {
    constexpr int W = kLanes;
    const CpuModel& m = *c.m;
    const double sqrt_hf = std::sqrt(c.h_f);
    double Sf[W], Sc[W], acc[W], zsum[W];
    for (int l = 0; l < W; l++) { Sf[l] = Sc[l] = m.S0; acc[l] = zsum[l] = 0.0; }
    for (int k = 0; k < c.n_fine; k++) {
        const double* dw = dW + (size_t)k * ld;
        for (int l = 0; l < W; l++) {
            zsum[l] += dw[l] / sqrt_hf;
            const double d = dw[l] + c.z_level * sqrt_hf;
            acc[l] += d;
            Sf[l] = euler_gbm(m, Sf[l], d, c.h_f);
        }
        if ((k + 1) % c.M == 0) {
            for (int l = 0; l < W; l++) {
                Sc[l] = Sc[l] + m.mu * Sc[l] * c.h_c + m.sigma * Sc[l] * acc[l];
                acc[l] = 0.0;
            }
        }
    }
    const double K = c.p->K, disc = c.p->discount;
    for (int l = 0; l < W; l++) {
        const double lr = std::exp(-c.z_level * zsum[l] - 0.5 * c.z_level * c.z_level * c.n_fine);
        Yf[l] = std::max(Sf[l] - K, 0.0) * disc * lr;
        Yc[l] = (c.n_coarse > 0) ? std::max(Sc[l] - K, 0.0) * disc * lr : 0.0;
    }
}

} // namespace

EvalFn select_eval_kernel(EvalSpec::Kind kind) {
    switch (kind) {
    case EvalSpec::Kind::CvAsianGeom: return &eval_cv_asian;
    case EvalSpec::Kind::CvDupireGbm: return &eval_cv_dupire;
    case EvalSpec::Kind::IsGbmCall:   return &eval_is_call;
    default: return nullptr;
    }
}

CoupledFn select_coupled_eval_kernel(EvalSpec::Kind kind) {
    switch (kind) {
    case EvalSpec::Kind::CvAsianGeom: return &coupled_cv_asian;
    case EvalSpec::Kind::IsGbmCall:   return &coupled_is_call;
    default: break;
    }
    throw std::invalid_argument("MLMC: esta variante de reducción de varianza solo existe para GBM+Asian (CV) y GBM+call (IS)");
}

} // namespace mc::cpu
