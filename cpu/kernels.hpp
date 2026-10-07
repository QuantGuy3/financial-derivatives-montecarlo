#pragma once
// Núcleos de simulación del motor CPU (equivalentes a kernel_mc / kernel_multi_dupire de la GPU).
//
// Cada llamada simula un BLOQUE de kLanes caminos en paso sincronizado ("lockstep"):
// el recurrente de Euler de un camino es una cadena de dependencias, pero kLanes cadenas
// independientes avanzan intercaladas (paralelismo a nivel de instrucción) y el compilador
// puede vectorizar los bucles internos sobre los carriles.
//
// Entrada: incrementos brownianos dW YA escalados (sqrt(h) incluido), en el layout de la GPU
//   dW[(k*noise_dim + c)*ld + lane],   k = paso, c = componente del ruido, lane = camino.
// Heston recibe dos componentes no correlacionadas; el núcleo aplica la Cholesky 2x2.
// Salida: Y[lane] = payoff (descontado donde corresponda) de cada camino del bloque.

#include "params.hpp"

#include <algorithm>
#include <cmath>

namespace mc::cpu {

#ifndef MC_LANES
#define MC_LANES 8
#endif
inline constexpr int kLanes = MC_LANES;

struct KCtx {
    const CpuModel*  m = nullptr;
    const CpuPayoff* p = nullptr;
    int    n_steps = 1;
    double h = 1.0;
    double sqrt_h = 1.0;
    double em = 1.0;       // exp(-kappa*h) (Heston)
    // Evaluadores con reducción de varianza (ver EvalSpec)
    double beta = 0.0, E_ctrl = 0.0;
    double z_step = 0.0;   // IS: desplazamiento por paso = z_star / sqrt(n_steps)
};

inline KCtx make_kctx(const CpuModel& m, const CpuPayoff& p, int n_steps, const EvalSpec& ev = {}) {
    KCtx c;
    c.m = &m; c.p = &p; c.n_steps = n_steps;
    c.h = m.T / n_steps;
    c.sqrt_h = std::sqrt(c.h);
    c.em = std::exp(-m.kappa * c.h);
    c.beta = ev.beta; c.E_ctrl = ev.E_ctrl;
    c.z_step = ev.z_star / std::sqrt((double)n_steps);
    return c;
}

// ---- payoffs --------------------------------------------------------------------------------

template <PayoffKind PK>
inline constexpr bool is_path_dep = (PK == PayoffKind::Asian || PK == PayoffKind::GeomAsian ||
                                     PK == PayoffKind::Lookback || PK == PayoffKind::Barrier);

template <PayoffKind PK>
inline double running_init(double S0) {
    if constexpr (PK == PayoffKind::Lookback) return S0;
    else return 0.0;
}

template <PayoffKind PK>
inline void running_update(double& run, double S) {
    if constexpr (PK == PayoffKind::Asian) run += S;
    else if constexpr (PK == PayoffKind::GeomAsian) run += std::log(S);
    else if constexpr (PK == PayoffKind::Lookback) run = std::min(run, S);
    else if constexpr (PK == PayoffKind::Barrier) run = std::max(run, S);
}

// Corrección BGK (monitorización discreta -> continua) aplicada al acumulador al final.
template <PayoffKind PK>
inline void apply_bgk(const CpuPayoff& p, double& run, double h_step) {
    if (!p.bgk) return;
    const double corr = std::exp(-BGK_BETA * p.sigma_bgk * std::sqrt(h_step));
    if constexpr (PK == PayoffKind::Lookback) run *= corr;
    else if constexpr (PK == PayoffKind::Barrier) run /= corr;
}

template <PayoffKind PK>
inline double terminal_payoff(const CpuPayoff& p, double S_T, double run, int n_steps) {
    if constexpr (PK == PayoffKind::European)
        return std::max(S_T - p.K, 0.0) * p.discount;
    else if constexpr (PK == PayoffKind::Asian)
        return std::max(run / n_steps - p.K, 0.0);
    else if constexpr (PK == PayoffKind::GeomAsian)
        return std::max(std::exp(run / n_steps) - p.K, 0.0);
    else if constexpr (PK == PayoffKind::Lookback)
        return S_T - run;
    else if constexpr (PK == PayoffKind::Barrier)
        return (run >= p.B) ? 0.0 : std::max(S_T - p.K, 0.0) * p.discount;
    else   // Basket: el llamante pasa la media de los activos como S_T
        return std::max(S_T - p.K, 0.0) * p.discount;
}

// ---- pasos de Euler por modelo (un carril) ---------------------------------------------------

inline double euler_gbm(const CpuModel& m, double S, double dw, double h) {
    return S + m.mu * S * h + m.sigma * S * dw;
}

// sigma_loc(S,t) = sigma0 * exp(-alpha t) * (S/S0)^(beta-1);  e_t = exp(-alpha t) se pasa
// calculado (compartido por todos los carriles del paso). S es absorbente en 0.
inline double euler_dupire(const CpuModel& m, double S, double dw, double h, double e_t, double S0_ref) {
    if (!(S > 0.0)) return 0.0;
    const double sigma_loc = m.sigma0 * e_t * std::pow(S / S0_ref, m.beta_d - 1.0);
    const double Sn = S + m.mu * S * h + sigma_loc * S * dw;
    return Sn > 0.0 ? Sn : 0.0;
}

// Heston: S usa la varianza ANTES de actualizar; V se actualiza con el esquema exacto en media
// (Milstein exacto en v, como d_euler_heston). dw1, dw2 son los incrementos YA correlacionados.
inline void euler_heston(const CpuModel& m, double& S, double& V, double dw1, double dw2, double h, double em) {
    const double Vp = std::max(V, 0.0);
    const double sq = std::sqrt(Vp);
    S = S + m.mu * S * h + sq * S * dw1;
    V = m.theta + em * (V - m.theta) + m.xi * sq * dw2;
}

// ---- núcleo de un bloque ---------------------------------------------------------------------

template <ModelKind MK, PayoffKind PK>
void kernel_single(const KCtx& c, const double* dW, int ld, double* Y) {
    constexpr int W = kLanes;
    constexpr int dim = (MK == ModelKind::Heston) ? 2 : 1;
    const CpuModel& m = *c.m;
    const CpuPayoff& pp = *c.p;
    const double h = c.h;

    double S[W], V[W], run[W];
    for (int l = 0; l < W; l++) {
        S[l] = m.S0;
        V[l] = m.v0;
        run[l] = running_init<PK>(m.S0);
    }

    for (int k = 0; k < c.n_steps; k++) {
        const double* dw = dW + (size_t)k * dim * ld;
        if constexpr (MK == ModelKind::GBM) {
            for (int l = 0; l < W; l++) S[l] = euler_gbm(m, S[l], dw[l], h);
        } else if constexpr (MK == ModelKind::Dupire) {
            const double e_t = std::exp(-m.alpha * (k * h));
            for (int l = 0; l < W; l++) S[l] = euler_dupire(m, S[l], dw[l], h, e_t, m.S0);
        } else {
            static_assert(MK == ModelKind::Heston);
            for (int l = 0; l < W; l++) {
                const double a1 = dw[l];
                const double a2 = dw[ld + l];
                euler_heston(m, S[l], V[l], a1, m.l21 * a1 + m.l22 * a2, h, c.em);
            }
        }
        if constexpr (is_path_dep<PK>)
            for (int l = 0; l < W; l++) running_update<PK>(run[l], S[l]);
    }
    if constexpr (is_path_dep<PK>)
        for (int l = 0; l < W; l++) apply_bgk<PK>(pp, run[l], h);
    for (int l = 0; l < W; l++) Y[l] = terminal_payoff<PK>(pp, S[l], run[l], c.n_steps);
}

// Cesta Dupire multi-activo (payoff Basket). dW ya CORRELACIONADO; scratch >= n_assets*kLanes.
void kernel_basket(const KCtx& c, const double* dW, int ld, double* Y, double* scratch);

using SingleFn = void (*)(const KCtx&, const double*, int, double*);
// Núcleo [modelo x payoff] para GBM/Heston/Dupire (no MultiDupire).
SingleFn select_single_kernel(ModelKind mk, PayoffKind pk);

// ---- núcleo MLMC: caminos fino y grueso acoplados ----------------------------------------------------

struct CKCtx {
    const CpuModel*  m = nullptr;
    const CpuPayoff* p = nullptr;
    int    n_fine = 1, n_coarse = 0, M = 2;
    double h_f = 1.0, h_c = 1.0;
    double em_f = 1.0, em_c = 1.0;   // exp(-kappa h) fino/grueso (Heston)
    double beta = 0.0, E_ctrl = 0.0; // CV
    double z_level = 0.0;            // IS: z_star / sqrt(n_fine)
};

inline CKCtx make_ckctx(const CpuModel& m, const CpuPayoff& p, int level, int M,
                        const EvalSpec& ev = {}) {
    CKCtx c;
    c.m = &m; c.p = &p; c.M = M;
    c.n_fine = 1;
    for (int i = 0; i < level; i++) c.n_fine *= M;
    c.n_coarse = (level == 0) ? 0 : c.n_fine / M;
    c.h_f = m.T / c.n_fine;
    c.h_c = m.T / std::max(c.n_coarse, 1);
    c.em_f = std::exp(-m.kappa * c.h_f);
    c.em_c = std::exp(-m.kappa * c.h_c);
    c.beta = ev.beta; c.E_ctrl = ev.E_ctrl;
    c.z_level = ev.z_star / std::sqrt((double)c.n_fine);
    return c;
}

// El mismo ruido fino alimenta ambos caminos: el grueso avanza cada M pasos finos con la SUMA de
// los M incrementos finos (acoplamiento de Giles). Yc = 0 en el nivel 0 (sin grueso).
// dW es el incremento fino ya escalado, layout [(k*dim+c)*ld + lane] como kernel_single.
template <ModelKind MK, PayoffKind PK>
void kernel_coupled(const CKCtx& c, const double* dW, int ld, double* Yf, double* Yc) {
    constexpr int W = kLanes;
    constexpr int dim = (MK == ModelKind::Heston) ? 2 : 1;
    const CpuModel& m = *c.m;
    const CpuPayoff& pp = *c.p;

    double Sf[W], Vf[W], Sc[W], Vc[W], run_f[W], run_c[W], acc1[W], acc2[W];
    for (int l = 0; l < W; l++) {
        Sf[l] = Sc[l] = m.S0;
        Vf[l] = Vc[l] = m.v0;
        run_f[l] = run_c[l] = running_init<PK>(m.S0);
        acc1[l] = acc2[l] = 0.0;
    }
    int coarse_k = 0;

    for (int k = 0; k < c.n_fine; k++) {
        const double* dw = dW + (size_t)k * dim * ld;
        if constexpr (MK == ModelKind::GBM) {
            for (int l = 0; l < W; l++) {
                acc1[l] += dw[l];
                Sf[l] = euler_gbm(m, Sf[l], dw[l], c.h_f);
            }
        } else if constexpr (MK == ModelKind::Dupire) {
            const double e_t = std::exp(-m.alpha * (k * c.h_f));
            for (int l = 0; l < W; l++) {
                acc1[l] += dw[l];
                Sf[l] = euler_dupire(m, Sf[l], dw[l], c.h_f, e_t, m.S0);
            }
        } else {
            static_assert(MK == ModelKind::Heston);
            for (int l = 0; l < W; l++) {
                const double a1 = dw[l], a2 = dw[ld + l];
                const double dw1 = a1, dw2 = m.l21 * a1 + m.l22 * a2;
                acc1[l] += dw1; acc2[l] += dw2;
                euler_heston(m, Sf[l], Vf[l], dw1, dw2, c.h_f, c.em_f);
            }
        }
        if constexpr (is_path_dep<PK>)
            for (int l = 0; l < W; l++) running_update<PK>(run_f[l], Sf[l]);

        if ((k + 1) % c.M == 0) {
            if constexpr (MK == ModelKind::GBM) {
                for (int l = 0; l < W; l++) {
                    Sc[l] = Sc[l] + m.mu * Sc[l] * c.h_c + m.sigma * Sc[l] * acc1[l];
                    acc1[l] = 0.0;
                }
            } else if constexpr (MK == ModelKind::Dupire) {
                const double e_t = std::exp(-m.alpha * (coarse_k * c.h_c));
                for (int l = 0; l < W; l++) {
                    Sc[l] = euler_dupire(m, Sc[l], acc1[l], c.h_c, e_t, m.S0);
                    acc1[l] = 0.0;
                }
            } else {
                for (int l = 0; l < W; l++) {
                    euler_heston(m, Sc[l], Vc[l], acc1[l], acc2[l], c.h_c, c.em_c);
                    acc1[l] = acc2[l] = 0.0;
                }
            }
            if constexpr (is_path_dep<PK>)
                for (int l = 0; l < W; l++) running_update<PK>(run_c[l], Sc[l]);
            ++coarse_k;
        }
    }
    if constexpr (is_path_dep<PK>)
        for (int l = 0; l < W; l++) {
            apply_bgk<PK>(pp, run_f[l], c.h_f);
            apply_bgk<PK>(pp, run_c[l], c.h_c);
        }
    for (int l = 0; l < W; l++) {
        Yf[l] = terminal_payoff<PK>(pp, Sf[l], run_f[l], c.n_fine);
        Yc[l] = (c.n_coarse > 0) ? terminal_payoff<PK>(pp, Sc[l], run_c[l], c.n_coarse) : 0.0;
    }
}

using CoupledFn = void (*)(const CKCtx&, const double*, int, double*, double*);
CoupledFn select_coupled_kernel(ModelKind mk, PayoffKind pk);

// ---- evaluadores con variable de control / importance sampling (un nivel) -----------------------
// Y0, Y1: valores por carril. CV: (Y_principal, Y_control); IS: Y0 = payoff·LR (Y1 no se usa).
using EvalFn = void (*)(const KCtx&, const double*, int, double*, double*);
// Devuelve nullptr para EvalSpec::Kind::Plain.
EvalFn select_eval_kernel(EvalSpec::Kind kind);

// Versiones acopladas fino/grueso para MLMC (Yf, Yc ya con CV/IS aplicados; Yc = 0 en el nivel 0).
CoupledFn select_coupled_eval_kernel(EvalSpec::Kind kind);

} // namespace mc::cpu
