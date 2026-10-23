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
//  rough_bergomi  V_t = ξ0·exp(ηY_t - ½η²t^{2H}), Y_t = √(2H)∫(t-s)^{H-½}dW_s  //
//                 (Bayer-Friz-Gatheral): paso cercano exacto (esquema híbrido  //
//                 de Bennedsen-Lunde-Pakkanen) y el resto del núcleo con una   //
//                 suma de exponenciales (factores markovianos)                 //
//  rough_heston   V_t = V0 + ∫K(t-s)κ(θ-V_s)ds + ∫K(t-s)ξ√V_s dW_s,            //
//                 K(t) = t^{H-½}/Γ(H+½) (El Euch-Rosenbaum): se simula la      //
//                 varianza forward media de cada paso con el esquema QE de     //
//                 Andersen (positiva, dos momentos exactos), el núcleo con     //
//                 factores markovianos y corrección de martingala exacta       //
//  fmls           log-estable de momentos finitos (Carr-Wu): incrementos       //
//                 α-estables con β = -1 por Chambers-Mallows-Stuck; exacto     //
//                                                                              //
// Ruido por paso: cada activo consume 1 normal "principal" (la que se         //
// correlaciona entre activos vía Cholesky) y n_extra normales propias:        //
//   Heston: z_v | Merton: z_N (nº de saltos), z_J (tamaño) | Bates: z_v, z_N,  //
//   z_J | SABR: z_α | NIG: z_IG, z_U (muestreo IG de Michael-Schucany-Haas) |   //
//   rough Bergomi: z_v, z_N (parte propia del paso cercano) | rough Heston: z_v //
//   | FMLS: z_W (la exponencial de CMS; el ángulo sale de la principal).        //
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
    RoughBergomi = 7,
    RoughHeston  = 8,
    FMLS         = 9,
};
inline constexpr int N_TIPOS_MODELO = 10;

// Modelos rugosos: factores markovianos del núcleo fraccionario (estado por
// trayectoria) y subyacentes rugosos por producto
inline constexpr int MAX_FAC_HD = 33;
inline constexpr int MAX_RUGOSOS_HD = 4;

// Constante de Broadie-Glasserman-Kou para la corrección de continuidad
inline constexpr double BGK_BETA_HD = 0.5826;

HD_INLINE int ruidos_extra(TipoModelo t) {
    switch (t) {
        case TipoModelo::Heston: return 1;
        case TipoModelo::Merton: return 2;
        case TipoModelo::Bates:  return 3;
        case TipoModelo::SABR:   return 1;
        case TipoModelo::NIG:    return 2;
        case TipoModelo::RoughBergomi: return 2;
        case TipoModelo::RoughHeston:  return 1;
        case TipoModelo::FMLS:   return 1;
        default:                 return 0;
    }
}

HD_INLINE bool modelo_rugoso(TipoModelo t) {
    return t == TipoModelo::RoughBergomi || t == TipoModelo::RoughHeston;
}

// Modelos cuyo paso es exacto en distribución (sin error de discretización si
// solo se observan fechas discretas): pueden simularse solo en las fechas clave.
HD_INLINE bool modelo_exacto(TipoModelo t) {
    return t == TipoModelo::BlackScholes || t == TipoModelo::Merton || t == TipoModelo::NIG ||
           t == TipoModelo::FMLS;
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
    // Modelos rugosos (rough Heston usa además v0, theta, kappa, xi, rho y rho_c)
    int  n_fac;                         // factores markovianos del núcleo (0 si no es rugoso)
    int  tab;                           // inicio de su tabla por pasos en tab_frac (-1 si no tiene)
    int  tab_paso;                      // longitud del registro de cada paso: 3·n_fac + 3
    int  ranura;                        // índice de su estado de factores (-1 si no es rugoso)
    Real eta, xi0;                      // rough Bergomi: η y ξ0
    Real v_ini;                         // varianza inicial del estado auxiliar
    // FMLS: α, 1/α, constantes de Chambers-Mallows-Stuck con β = -1, escala y deriva
    Real alpha_st, inv_alpha_st, cms_b, cms_s, escala_st, mu_st;
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
        case TipoModelo::RoughBergomi:
        case TipoModelo::RoughHeston: return a.v_ini;
        default:                return Real(0);
    }
}

// Volatilidad instantánea (para la corrección de continuidad BGK de barreras)
template<typename Real>
HD_INLINE Real vol_instantanea(const ParamsActivo<Real>& a, Real x, Real v, Real t, Real rcum_t) {
    switch ((TipoModelo)a.tipo) {
        case TipoModelo::Heston:
        case TipoModelo::Bates:
        case TipoModelo::RoughBergomi:
        case TipoModelo::RoughHeston: return hd::sqrt_(hd::max_(v, Real(0)));
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
//   u      = factores markovianos del activo (modelos rugosos) o nullptr      //
//   tab    = registro de este paso en la tabla del activo (rugosos) o nullptr //
// ------------------------------------------------------------------------- //
template<typename Real>
HD_INLINE void paso_modelo(const ParamsActivo<Real>& a, Real& x, Real& v,
                           Real t, Real dt, Real sqdt, Real rdt, Real rcum_t,
                           Real w, Real e0, Real e1, Real e2,
                           Real* u = nullptr, const Real* tab = nullptr) {
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
    case TipoModelo::RoughBergomi: {
        // Registro: a_j·e^{-x_jΔ} | e^{-x_jΔ} | (1-e^{-x_jΔ})/(x_jΔ) | β, γ, ln ξ0 - ½η²t_{k+1}^{2H}
        const int nf = a.n_fac;
        const Real* ae = tab;
        const Real* e = tab + nf;
        const Real* wj = tab + 2 * nf;
        const Real beta = tab[3 * nf], gam = tab[3 * nf + 1], ct = tab[3 * nf + 2];
        const Real vp = hd::max_(v, Real(0)), sv = hd::sqrt_(vp);
        x += arrastre - Real(0.5) * vp * dt + sv * sqdt * w;   // varianza al inicio del paso
        const Real dW = sqdt * (a.rho * w + a.rho_c * e0);
        Real Y = beta * dW + gam * e1;                         // tramo cercano exacto
        for (int j = 0; j < nf; ++j) {                         // resto del núcleo
            Y += ae[j] * u[j];
            u[j] = e[j] * u[j] + wj[j] * dW;
        }
        v = hd::exp_(a.eta * Y + ct);
    } break;
    case TipoModelo::RoughHeston: {
        // v = varianza forward media del paso (≥ 0). Registro:
        //   e^{-x_jΔ} | (1-e^{-x_jΔ})/(x_jΔ) | c_j·w_j(Δ') | kd', A1', A0
        // (Δ' es el paso siguiente: la QE sortea directamente su varianza media)
        const int nf = a.n_fac;
        const Real* e = tab;
        const Real* wj = tab + nf;
        const Real* cw = tab + 2 * nf;
        const Real kd = tab[3 * nf], A1 = tab[3 * nf + 1], A0 = tab[3 * nf + 2];
        const Real vb = hd::max_(v, Real(0)), sv = hd::sqrt_(vb);
        const Real der = a.kappa * (a.theta - vb);
        // Media y desviación condicionadas de la varianza media del paso siguiente
        Real m = a.v0 + der * kd + A1 * der * dt;
        for (int j = 0; j < nf; ++j) m += cw[j] * e[j] * u[j];
        m = hd::max_(m, Real(1e-12));
        const Real s = A1 * a.xi * sv * sqdt;
        const Real G = a.rho * w + a.rho_c * e0;              // normal de la varianza
        // Exponente de la corrección de martingala: E[exp(tt·Z)]
        const Real tt = a.rho * sv * sqdt * (Real(1) - Real(0.5) * a.rho * a.xi * A0 * dt);
        const Real psi = (s * s) / (m * m);
        Real xn, Z, lnM;
        if (!(psi > Real(1e-8))) {
            xn = m; Z = Real(0); lnM = Real(0);
        } else if (psi <= Real(1.5)) {                        // QE: rama cuadrática
            const Real ip = Real(2) / psi;
            const Real b2 = ip - Real(1) + hd::sqrt_(ip) * hd::sqrt_(ip - Real(1));
            const Real aq = m / (Real(1) + b2), b = hd::sqrt_(b2);
            xn = aq * (b + G) * (b + G);
            Z = (xn - m) / s;
            const Real tau = tt * aq / s;
            lnM = -tt * m / s + tau * b2 / (Real(1) - Real(2) * tau) - Real(0.5) * hd::log_(Real(1) - Real(2) * tau);
        } else {                                              // QE: rama exponencial
            const Real p = (psi - Real(1)) / (psi + Real(1)), be = (Real(1) - p) / m;
            const Real uc = hd::norm_cdf(-G);                 // 1 - U
            xn = (Real(1) - uc <= p) ? Real(0) : hd::log_((Real(1) - p) / uc) / be;
            Z = (xn - m) / s;
            lnM = -tt * m / s + hd::log_(p + (Real(1) - p) * be / (be - tt / s));
        }
        // Varianza integrada del paso, con su respuesta al ruido del propio paso
        const Real I = hd::max_(dt * (vb + a.xi * A0 * sqdt * sv * Z), Real(0));
        const Real eperp = a.rho_c * w - a.rho * e0;          // independiente de G
        x += arrastre - Real(0.5) * I + a.rho * sv * sqdt * Z + a.rho_c * hd::sqrt_(I) * eperp +
             Real(0.5) * a.rho * a.rho * vb * dt - lnM;
        const Real dM = der * dt + a.xi * sv * sqdt * Z;
        for (int j = 0; j < nf; ++j) u[j] = e[j] * u[j] + wj[j] * dM;
        v = xn;
    } break;
    case TipoModelo::FMLS: {
        // Chambers-Mallows-Stuck con β = -1: ángulo de la normal principal (cópula
        // gaussiana entre subyacentes; X es creciente en el ángulo) y W ~ Exp(1)
        const Real lim = Real(1.5707960);                     // π/2 - 3e-7
        const Real V = hd::clamp_(Real(3.14159265358979) * (hd::norm_cdf(w) - Real(0.5)), -lim, lim);
        const Real W = -hd::log_(hd::max_(hd::norm_cdf(-e0), Real(1e-30)));
        const Real al = a.alpha_st, ab = al * (V + a.cms_b);
        const Real c2 = hd::max_(hd::cos_(V - ab), Real(0));
        const Real X = a.cms_s * hd::sin_(ab) / hd::pow_(hd::cos_(V), a.inv_alpha_st) *
                       hd::pow_(c2 / hd::max_(W, Real(1e-30)), (Real(1) - al) * a.inv_alpha_st);
        x += arrastre + a.mu_st * dt + a.escala_st * hd::pow_(dt, a.inv_alpha_st) * X;
    } break;
    }
}
