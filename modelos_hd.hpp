#pragma once
// ---------------------------------------------------------------------------- //
// Dinámicas de los subyacentes, compartidas por CPU y GPU (__host__ __device__) //
//                                                                              //
// Todos los modelos se simulan en logaritmo del precio, x = ln S, bajo la      //
// medida riesgo neutro: E[S_t] = S_0 · exp(∫(r - q)).                          //
//                                                                              //
//  black_scholes  dS/S = (r-q)dt + σ dW                         (exacto)        //
//  heston         dS/S = (r-q)dt + √v dW₁,  dv = κ(θ-v)dt + ξ√v dW₂             //
//                 (Euler con truncamiento completo, Lord et al. 2010)           //
//  merton         Black-Scholes + saltos lognormales compuestos de Poisson     //
//                 (intensidad λ, log-salto ~ N(μ_J, δ²)); exacto               //
//  bates          Heston + saltos de Merton                                    //
//  vol_local      σ_loc(S,t) = σ₀·e^{-αt}·(S/S₀)^{β-1}  (paramétrica tipo       //
//                 Dupire/CEV, la misma familia que DupireLocalParams)          //
//  sabr           dF = α F^β dW₁, dα = ν α dW₂ (sobre el forward; se simula el  //
//                 spot con σ_ef = α·(F/F₀)^{β-1})                              //
//  nig            Normal Inverse Gaussian (Lévy de saltos puros): browniano    //
//                 con deriva θ y vol σ subordinado por un proceso IG de media  //
//                 t y varianza κt (parametrización de Cont-Tankov); exacto     //
//                                                                              //
// Ruido por paso: cada activo consume 1 normal "principal" (la que se         //
// correlaciona entre activos vía Cholesky) y n_extra normales propias:        //
//   Heston: z_v | Merton: z_N (nº de saltos), z_J (tamaño) | Bates: z_v, z_N,  //
//   z_J | SABR: z_α | NIG: z_IG, z_U (muestreo IG de Michael-Schucany-Haas).    //
// Las normales que se usan como uniformes (z_N, z_U) se transforman con Φ(z),  //
// de modo que todo el ruido sigue siendo normal y compatible con QMC.         //
// ---------------------------------------------------------------------------- //

#include "hd_comun.hpp"

enum class TipoModelo : int {
    BlackScholes = 0,
    Heston       = 1,
    Merton       = 2,
    Bates        = 3,
    VolLocal     = 4,
    SABR         = 5,
    NIG          = 6,
};
inline constexpr int N_TIPOS_MODELO = 7;

// Constante de Broadie-Glasserman-Kou para la corrección de continuidad
inline constexpr double BGK_BETA_HD = 0.5826;

HD_INLINE int ruidos_extra(TipoModelo t) {
    switch (t) {
        case TipoModelo::Heston: return 1;
        case TipoModelo::Merton: return 2;
        case TipoModelo::Bates:  return 3;
        case TipoModelo::SABR:   return 1;
        case TipoModelo::NIG:    return 2;
        default:                 return 0;
    }
}

// Modelos cuyo paso es exacto en distribución (sin error de discretización si
// solo se observan fechas discretas): pueden simularse solo en las fechas clave.
HD_INLINE bool modelo_exacto(TipoModelo t) {
    return t == TipoModelo::BlackScholes || t == TipoModelo::Merton || t == TipoModelo::NIG;
}

// Parámetros de un subyacente, ya preparados para la simulación (POD, se sube
// tal cual a la GPU). Un único struct para todos los modelos: cada modelo usa
// solo sus campos.
template<typename Real>
struct ParamsActivo {
    int  tipo;        // TipoModelo
    int  n_extra;     // normales extra por paso (además de la principal)
    Real x0;          // ln(spot) al inicio de la simulación
    Real x_ancla;     // ln(S₀) de anclaje de la vol local / SABR (no se mueve al calcular deltas)
    Real s_ref_inv;   // 1 / nivel inicial (referencia de los rendimientos)
    Real q;           // rentabilidad por dividendo (continua)
    Real sigma;       // BS, Merton: σ | VolLocal: σ₀ | SABR: α₀ | NIG: σ
    Real v0, kappa, theta, xi;          // Heston / Bates
    Real rho, rho_c;                    // correlación propia y √(1-ρ²) (Heston, Bates, SABR)
    Real lambda, mu_j, sigma_j, comp_j; // saltos: λ, μ_J, δ, compensador λ·(e^{μ_J+δ²/2}-1)
    Real alpha_lv;                      // VolLocal: decaimiento temporal α
    Real beta_m1;                       // VolLocal / SABR: β - 1
    Real nu;                            // SABR: vol de la vol
    Real theta_nig, kappa_nig, omega_nig; // NIG: θ, κ y corrección de martingala ω
    Real vol_max;                       // tope de la vol instantánea (vol local / SABR)
};

// ---------------------------------------------------------- //
// Bloques elementales (también los usan los kernels vanilla) //
// ---------------------------------------------------------- //

// Inversa de la Poisson de media m en la uniforme u (en doble precisión: el bucle
// acumula probabilidades y en float podría no alcanzar u cerca de 1).
HD_INLINE int poisson_inv(double m, double u) {
    if (m <= 0.0) return 0;
    if (m > 60.0) { // aproximación normal (solo con intensidades/plazos enormes)
        double n = floor(m + sqrt(m) * hd::inv_norm_cdf(u) + 0.5);
        return n < 0.0 ? 0 : (int)n;
    }
    double p = exp(-m), F = p;
    int n = 0;
    const int nmax = (int)(m + 12.0 * sqrt(m) + 20.0);
    while (u > F && n < nmax) {
        ++n;
        p *= m / n;
        F += p;
    }
    return n;
}

// Suma de los log-saltos de un paso: N ~ Poisson(λ·dt) a partir de z_n y, dado N,
// la suma de N saltos N(μ_J, δ²) es N(N·μ_J, N·δ²) a partir de z_j.
template<typename Real>
HD_INLINE Real salto_compuesto(Real lambda_dt, Real mu_j, Real sigma_j, Real z_n, Real z_j) {
    if (lambda_dt <= Real(0)) return Real(0);
    double u = hd::norm_cdf((double)z_n);
    int n = poisson_inv((double)lambda_dt, u);
    if (n == 0) return Real(0);
    return (Real)n * mu_j + hd::sqrt_((Real)n) * sigma_j * z_j;
}

// Incremento del subordinador IG de la NIG en un paso dt: IG de media dt y
// parámetro de forma dt²/κ (varianza κ·dt), por el método de Michael, Schucany y
// Haas (1976) con una normal z1 y una uniforme Φ(z2). Se usa la forma estable
// x = μ / (1 + a + √(a² + 2a)), a = μ·z1²/(2λ), sin cancelaciones.
template<typename Real>
HD_INLINE Real ig_incremento(Real dt, Real kappa, Real z1, Real z2) {
    Real y = z1 * z1;
    Real a = y * kappa / (Real(2) * dt);
    Real xg = dt / (Real(1) + a + hd::sqrt_(a * a + Real(2) * a));
    Real u = hd::norm_cdf(z2);
    return (u * (dt + xg) <= dt) ? xg : dt * dt / xg;
}

// Vol local paramétrica: σ₀·exp(-α t + (β-1)(x - x_ancla)), con tope vol_max
template<typename Real>
HD_INLINE Real vol_local_param(const ParamsActivo<Real>& a, Real x, Real t) {
    Real s = a.sigma * hd::exp_(-a.alpha_lv * t + a.beta_m1 * (x - a.x_ancla));
    return hd::min_(s, a.vol_max);
}

// Vol efectiva de SABR: α·(F/F₀)^{β-1}; con tipos deterministas F/F₀ = S·e^{-∫(r-q)}/S₀,
// así que basta el log-spot corregido por el arrastre acumulado.
template<typename Real>
HD_INLINE Real vol_sabr(const ParamsActivo<Real>& a, Real alpha_t, Real x, Real arrastre_t) {
    Real s = alpha_t * hd::exp_(a.beta_m1 * (x - a.x_ancla - arrastre_t));
    return hd::min_(s, a.vol_max);
}

// Estado inicial de la variable auxiliar v (varianza Heston/Bates, α en SABR)
template<typename Real>
HD_INLINE Real estado_aux_inicial(const ParamsActivo<Real>& a) {
    switch ((TipoModelo)a.tipo) {
        case TipoModelo::Heston:
        case TipoModelo::Bates: return a.v0;
        case TipoModelo::SABR:  return a.sigma;
        default:                return Real(0);
    }
}

// Volatilidad instantánea (para la corrección de continuidad BGK de barreras)
template<typename Real>
HD_INLINE Real vol_instantanea(const ParamsActivo<Real>& a, Real x, Real v, Real t, Real rcum_t) {
    switch ((TipoModelo)a.tipo) {
        case TipoModelo::Heston:
        case TipoModelo::Bates:    return hd::sqrt_(hd::max_(v, Real(0)));
        case TipoModelo::VolLocal: return vol_local_param(a, x, t);
        case TipoModelo::SABR:     return vol_sabr(a, v, x, rcum_t - a.q * t);
        default:                   return a.sigma; // BS, Merton (difusión), NIG (aprox.)
    }
}

// ------------------------------------------------------------------------- //
// Un paso del log-precio x (y de la variable auxiliar v) desde t hasta t+dt. //
//   rdt    = ∫_t^{t+dt} r(s) ds         (curva de tipos determinista)        //
//   rcum_t = ∫_0^t r(s) ds                                                    //
//   w      = normal principal (ya correlacionada con el resto de activos)    //
//   e0..e2 = normales extra propias del activo                                //
// ------------------------------------------------------------------------- //
template<typename Real>
HD_INLINE void paso_modelo(const ParamsActivo<Real>& a, Real& x, Real& v,
                           Real t, Real dt, Real sqdt, Real rdt, Real rcum_t,
                           Real w, Real e0, Real e1, Real e2) {
    const Real arrastre = rdt - a.q * dt; // ∫(r - q) en el paso
    switch ((TipoModelo)a.tipo) {
    case TipoModelo::BlackScholes: {
        x += arrastre - Real(0.5) * a.sigma * a.sigma * dt + a.sigma * sqdt * w;
    } break;
    case TipoModelo::Heston: {
        Real vp = hd::max_(v, Real(0));
        Real sv = hd::sqrt_(vp);
        x += arrastre - Real(0.5) * vp * dt + sv * sqdt * w;
        v += a.kappa * (a.theta - vp) * dt + a.xi * sv * sqdt * (a.rho * w + a.rho_c * e0);
    } break;
    case TipoModelo::Merton: {
        Real J = salto_compuesto(a.lambda * dt, a.mu_j, a.sigma_j, e0, e1);
        x += arrastre - (Real(0.5) * a.sigma * a.sigma + a.comp_j) * dt + a.sigma * sqdt * w + J;
    } break;
    case TipoModelo::Bates: {
        Real vp = hd::max_(v, Real(0));
        Real sv = hd::sqrt_(vp);
        Real J = salto_compuesto(a.lambda * dt, a.mu_j, a.sigma_j, e1, e2);
        x += arrastre - (Real(0.5) * vp + a.comp_j) * dt + sv * sqdt * w + J;
        v += a.kappa * (a.theta - vp) * dt + a.xi * sv * sqdt * (a.rho * w + a.rho_c * e0);
    } break;
    case TipoModelo::VolLocal: {
        Real s = vol_local_param(a, x, t);
        x += arrastre - Real(0.5) * s * s * dt + s * sqdt * w;
    } break;
    case TipoModelo::SABR: {
        Real s = vol_sabr(a, v, x, rcum_t - a.q * t);
        x += arrastre - Real(0.5) * s * s * dt + s * sqdt * w;
        // α es un GBM sin deriva: paso exacto
        v *= hd::exp_(a.nu * sqdt * (a.rho * w + a.rho_c * e0) - Real(0.5) * a.nu * a.nu * dt);
    } break;
    case TipoModelo::NIG: {
        Real G = ig_incremento(dt, a.kappa_nig, e0, e1);
        x += arrastre + a.omega_nig * dt + a.theta_nig * G + a.sigma * hd::sqrt_(G) * w;
    } break;
    }
}
