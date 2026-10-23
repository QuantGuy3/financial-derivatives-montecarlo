#pragma once
// ---------------------------------------------------------------------------- //
// Plan de simulación: del Producto validado a los arreglos planos (en host)    //
// que lee simular_trayectoria(). El motor CPU usa la vista tal cual; el motor  //
// GPU copia los mismos arreglos a la memoria del dispositivo.                  //
//                                                                              //
// Rejilla temporal: cada intervalo entre fechas de observación consecutivas se //
// divide en pasos iguales de longitud <= 1/pasos_por_anio, de modo que todas   //
// las fechas de observación son exactamente nodos de la rejilla. Con           //
// pasos_por_anio = 0 solo se simulan las fechas de observación (válido cuando  //
// todos los modelos son exactos y no hay vigilancia diaria/continua).          //
// ---------------------------------------------------------------------------- //

#include "producto.hpp"
#include "../fraccional/fraccional.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace est {

template<typename Real>
struct PlanSim {
    std::vector<ParamsActivo<Real>> act;
    std::vector<Real> chol, pesos;
    std::vector<Real> dt, sqdt, rdt, t_ini, rcum_ini, df_fin;
    std::vector<int>  obs_paso;
    std::vector<Real> t_obs, df_obs;
    std::vector<Real> cup_imp, cup_bar;
    std::vector<int>  ac_activa;
    std::vector<Real> ac_bar, ac_imp;
    std::vector<BloqueSim<Real>> bloques;
    std::vector<InstrFormula<Real>> formulas;
    std::vector<Real> tab_frac;
    SimProducto<Real> vista{};
    double pasos_por_anio = 0.0;

    PlanSim() = default;
    PlanSim(const PlanSim&) = delete;
    PlanSim& operator=(const PlanSim&) = delete;

    long long dimension() const { return (long long)vista.n_pasos * vista.dim_paso; }

    // Apunta la vista a los vectores de este objeto
    void enlazar() {
        vista.act = act.data();
        vista.chol = chol.empty() ? nullptr : chol.data();
        vista.pesos = pesos.data();
        vista.dt = dt.data();
        vista.sqdt = sqdt.data();
        vista.rdt = rdt.data();
        vista.t_ini = t_ini.data();
        vista.rcum_ini = rcum_ini.data();
        vista.df_fin = df_fin.data();
        vista.obs_paso = obs_paso.data();
        vista.t_obs = t_obs.data();
        vista.df_obs = df_obs.data();
        vista.cup_imp = cup_imp.empty() ? nullptr : cup_imp.data();
        vista.cup_bar = cup_bar.empty() ? nullptr : cup_bar.data();
        vista.ac_activa = ac_activa.empty() ? nullptr : ac_activa.data();
        vista.ac_bar = ac_bar.empty() ? nullptr : ac_bar.data();
        vista.ac_imp = ac_imp.empty() ? nullptr : ac_imp.data();
        vista.bloques = bloques.empty() ? nullptr : bloques.data();
        vista.formulas = formulas.empty() ? nullptr : formulas.data();
        vista.tab_frac = tab_frac.empty() ? nullptr : tab_frac.data();
    }
};

// Parámetros de simulación de un subyacente
template<typename Real>
ParamsActivo<Real> preparar_activo(const Activo& a) {
    ParamsActivo<Real> pa{};
    const Modelo& m = a.modelo;
    pa.tipo = (int)m.tipo;
    pa.n_extra = ruidos_extra(m.tipo);
    // Se simula el log-rendimiento x = ln(S / nivel_inicial), que empieza cerca de
    // 0: en precisión simple (GPU) los incrementos diarios se representan mucho
    // mejor que sumándolos a ln S. Las dinámicas solo dependen de diferencias de
    // x (x - x_ancla), así que el cambio de origen no altera nada más.
    const double ln_ref = std::log(a.nivel_inicial);
    pa.x0 = (Real)(std::log(a.spot) - ln_ref);
    pa.x_ancla = (Real)(std::log(a.spot_ancla > 0.0 ? a.spot_ancla : a.spot) - ln_ref);
    pa.s_ref_inv = Real(1);
    pa.q = (Real)a.dividendos;
    pa.rho_c = Real(1);
    pa.vol_max = Real(3);
    pa.n_fac = 0;
    pa.tab = -1;
    pa.tab_paso = 0;
    pa.ranura = -1;
    auto saltos = [&]() {
        double lam = m["lambda"], mj = m["media_salto"], sj = m["vol_salto"];
        pa.lambda = (Real)lam;
        pa.mu_j = (Real)mj;
        pa.sigma_j = (Real)sj;
        pa.comp_j = (Real)(lam * (std::exp(mj + 0.5 * sj * sj) - 1.0));
    };
    auto heston = [&]() {
        double vi = m["vol_inicial"], vl = m["vol_largo_plazo"], r = m["rho"];
        pa.v0 = (Real)(vi * vi);
        pa.theta = (Real)(vl * vl);
        pa.kappa = (Real)m["kappa"];
        pa.xi = (Real)m["xi"];
        pa.rho = (Real)r;
        pa.rho_c = (Real)std::sqrt(std::max(0.0, 1.0 - r * r));
    };
    switch (m.tipo) {
    case TipoModelo::BlackScholes:
        pa.sigma = (Real)m["volatilidad"];
        break;
    case TipoModelo::Heston:
        heston();
        break;
    case TipoModelo::Merton:
        pa.sigma = (Real)m["volatilidad"];
        saltos();
        break;
    case TipoModelo::Bates:
        heston();
        saltos();
        break;
    case TipoModelo::VolLocal:
        pa.sigma = (Real)m["sigma0"];
        pa.alpha_lv = (Real)m["alpha"];
        pa.beta_m1 = (Real)(m["beta"] - 1.0);
        pa.vol_max = (Real)m["vol_max"];
        break;
    case TipoModelo::SABR: {
        double r = m["rho"];
        pa.sigma = (Real)m["alpha"];
        pa.beta_m1 = (Real)(m["beta"] - 1.0);
        pa.nu = (Real)m["nu"];
        pa.rho = (Real)r;
        pa.rho_c = (Real)std::sqrt(std::max(0.0, 1.0 - r * r));
        pa.vol_max = (Real)m["vol_max"];
    } break;
    case TipoModelo::NIG: {
        double s = m["sigma"], th = m["theta"], k = m["kappa"];
        pa.sigma = (Real)s;
        pa.theta_nig = (Real)th;
        pa.kappa_nig = (Real)k;
        // ω = -ψ(-i): E[exp(θG + σ√G Z)] = exp((t/κ)(1 - √(1 - σ²κ - 2θκ)))
        pa.omega_nig = (Real)(-(1.0 - std::sqrt(1.0 - s * s * k - 2.0 * th * k)) / k);
    } break;
    case TipoModelo::RoughBergomi: {
        const double vi = m["vol_inicial"], r = m["rho"];
        pa.xi0 = (Real)(vi * vi);
        pa.v_ini = pa.xi0;
        pa.eta = (Real)m["eta"];
        pa.rho = (Real)r;
        pa.rho_c = (Real)std::sqrt(std::max(0.0, 1.0 - r * r));
    } break;
    case TipoModelo::RoughHeston:
        heston();             // v0, θ, κ, ξ, ρ (la tabla por pasos fija v_ini)
        pa.v_ini = pa.v0;
        break;
    case TipoModelo::FMLS: {
        const double al = m["alpha"], sg = m["volatilidad"], PI = 3.14159265358979323846;
        const double esc = sg / std::sqrt(2.0), ta = std::tan(PI * al / 2.0);
        pa.sigma = (Real)sg;
        pa.alpha_st = (Real)al;
        pa.inv_alpha_st = (Real)(1.0 / al);
        pa.cms_b = (Real)(std::atan(-ta) / al);
        pa.cms_s = (Real)std::pow(1.0 + ta * ta, 1.0 / (2.0 * al));
        pa.escala_st = (Real)esc;
        pa.mu_st = (Real)(std::pow(esc, al) / std::cos(PI * al / 2.0));
    } break;
    }
    return pa;
}

// --------------------------------------------------------------------------- //
// Tabla por pasos de un subyacente rugoso para la rejilla dt[0..n-1] (que     //
// empieza en t = 0). El núcleo se aproxima con frac::nodos_nucleo en          //
// [min dt, Σ dt]; cada paso guarda 3·n_fac + 3 valores (ver paso_modelo).      //
// Fija n_fac, tab, tab_paso y v_ini.                                          //
// --------------------------------------------------------------------------- //
template<typename Real>
void preparar_rugoso(ParamsActivo<Real>& pa, const Modelo& m, const std::vector<double>& dt, std::vector<Real>& tabla) {
    const double H = m["hurst"];
    const int fac = std::max(3, (int)std::lround(m["factores"]));
    double T = 0.0, dtmin = 1e300;
    for (double d : dt) {
        T += d;
        dtmin = std::min(dtmin, d);
    }
    const frac::NodosNucleo nn = frac::nodos_nucleo(H, fac, dtmin, T);
    const int nf = (int)nn.x.size();
    pa.n_fac = nf;
    pa.tab_paso = 3 * nf + 3;
    pa.tab = (int)tabla.size();
    const double a = H + 0.5;
    const size_t n = dt.size();
    auto wf = [](double x, double d) { // (1 - e^{-xd})/(xd)
        const double z = x * d;
        return z < 1e-12 ? 1.0 - 0.5 * z : -std::expm1(-z) / z;
    };
    tabla.reserve(tabla.size() + n * (size_t)pa.tab_paso);
    if (m.tipo == TipoModelo::RoughBergomi) {
        const double eta = m["eta"], vi = m["vol_inicial"];
        const double s2H = std::sqrt(2.0 * H), fa = s2H * std::tgamma(a); // a_j = √(2H)·Γ(α)·c_j
        const double gfac = std::sqrt(std::max(0.0, 1.0 - 2.0 * H / (a * a)));
        double t = 0.0;
        for (size_t k = 0; k < n; ++k) {
            const double d = dt[k];
            t += d;
            for (int j = 0; j < nf; ++j) tabla.push_back((Real)(fa * nn.c[(size_t)j] * std::exp(-nn.x[(size_t)j] * d)));
            for (int j = 0; j < nf; ++j) tabla.push_back((Real)std::exp(-nn.x[(size_t)j] * d));
            for (int j = 0; j < nf; ++j) tabla.push_back((Real)wf(nn.x[(size_t)j], d));
            tabla.push_back((Real)(s2H * std::pow(d, H - 0.5) / a));
            tabla.push_back((Real)(std::pow(d, H) * gfac));
            tabla.push_back((Real)(std::log(vi * vi) - 0.5 * eta * eta * std::pow(t, 2.0 * H)));
        }
    } else {
        const double g2 = std::tgamma(a + 2.0);
        for (size_t k = 0; k < n; ++k) {
            const double d = dt[k], dn = (k + 1 < n) ? dt[k + 1] : d;
            double A1 = 0.0;
            for (int j = 0; j < nf; ++j) tabla.push_back((Real)std::exp(-nn.x[(size_t)j] * d));
            for (int j = 0; j < nf; ++j) tabla.push_back((Real)wf(nn.x[(size_t)j], d));
            for (int j = 0; j < nf; ++j) {
                const double cw = nn.c[(size_t)j] * wf(nn.x[(size_t)j], dn);
                tabla.push_back((Real)cw);
                A1 += cw * wf(nn.x[(size_t)j], d);
            }
            tabla.push_back((Real)(std::pow(dn, a) / g2));      // kd' = Δ'^α/Γ(α+2)
            tabla.push_back((Real)A1);                         // A1' = Σ c_j w_j(Δ') w_j(Δ)
            tabla.push_back((Real)(std::pow(d, a - 1.0) / g2)); // A0 = Δ^{α-1}/Γ(α+2)
        }
        const double v0 = std::pow(m["vol_inicial"], 2.0), th = std::pow(m["vol_largo_plazo"], 2.0);
        pa.v_ini = (Real)(v0 + m["kappa"] * (th - v0) * std::pow(dt.empty() ? 0.0 : dt[0], a) / g2);
    }
}

template<typename Real>
std::unique_ptr<PlanSim<Real>> construir_plan(const Producto& p) {
    auto plan = std::make_unique<PlanSim<Real>>();
    PlanSim<Real>& P = *plan;
    SimProducto<Real>& v = P.vista;
    const int n = (int)p.activos.size();
    const int nobs = p.n_obs();

    // Subyacentes y ruido por paso
    v.n_act = n;
    int off = n;
    for (int i = 0; i < n; ++i) {
        P.act.push_back(preparar_activo<Real>(p.activos[i]));
        v.off_extra[i] = off;
        off += P.act.back().n_extra;
    }
    v.dim_paso = off;
    if (n > 1)
        for (double c : p.chol) P.chol.push_back((Real)c);
    v.cesta = p.cesta;
    for (double w : p.pesos) P.pesos.push_back((Real)w);

    // Rejilla temporal
    const double ppa = pasos_por_anio_efectivos(p);
    P.pasos_por_anio = ppa;
    std::vector<double> nodos = {0.0};
    double t_prev = 0.0;
    for (int j = 0; j < nobs; ++j) {
        double delta = p.t_obs[j] - t_prev;
        int m = (ppa > 0.0) ? std::max(1, (int)std::ceil(delta * ppa - 1e-9)) : 1;
        for (int s = 1; s <= m; ++s) {
            nodos.push_back(s == m ? p.t_obs[j] : t_prev + delta * s / m);
            P.obs_paso.push_back(s == m ? j : -1);
        }
        t_prev = p.t_obs[j];
    }
    const int npasos = (int)nodos.size() - 1;
    v.n_pasos = npasos;
    // Subyacentes rugosos: tabla por pasos y ranura de estado
    {
        std::vector<double> dts((size_t)npasos);
        for (int k = 0; k < npasos; ++k) dts[(size_t)k] = nodos[(size_t)k + 1] - nodos[(size_t)k];
        int ranura = 0;
        for (int i = 0; i < n; ++i) {
            if (!modelo_rugoso(p.activos[(size_t)i].modelo.tipo)) continue;
            preparar_rugoso(P.act[(size_t)i], p.activos[(size_t)i].modelo, dts, P.tab_frac);
            P.act[(size_t)i].ranura = ranura++;
        }
    }
    for (int k = 0; k < npasos; ++k) {
        double a = nodos[k], b = nodos[k + 1];
        double Ia = p.curva.integral(a), Ib = p.curva.integral(b);
        P.dt.push_back((Real)(b - a));
        P.sqdt.push_back((Real)std::sqrt(b - a));
        P.rdt.push_back((Real)(Ib - Ia));
        P.t_ini.push_back((Real)a);
        P.rcum_ini.push_back((Real)Ia);
        P.df_fin.push_back((Real)std::exp(-Ib));
    }

    // Observaciones
    v.n_obs = nobs;
    for (int j = 0; j < nobs; ++j) {
        P.t_obs.push_back((Real)p.t_obs[j]);
        P.df_obs.push_back((Real)p.curva.df(p.t_obs[j]));
    }
    v.T = (Real)p.T;
    v.df_T = (Real)p.curva.df(p.T);

    // Cupones
    v.n_cup = (int)p.cupones.size();
    v.hay_rango = 0;
    for (int c = 0; c < v.n_cup; ++c) {
        const Cupon& cu = p.cupones[c];
        v.cup_tipo[c] = cu.tipo;
        v.cup_mem[c] = cu.memoria ? 1 : 0;
        v.cup_inf[c] = (Real)cu.inf;
        v.cup_sup[c] = (Real)cu.sup;
        v.mem_ini[c] = (Real)(c < (int)p.memoria_inicial.size() ? p.memoria_inicial[c] : 0.0);
        if (cu.tipo == CUP_RANGO) v.hay_rango = 1;
        for (int j = 0; j < nobs; ++j) {
            P.cup_imp.push_back((Real)cu.importe[j]);
            P.cup_bar.push_back((Real)cu.barrera[j]);
        }
    }

    // Autocancelación
    if (p.hay_autocall) {
        for (int j = 0; j < nobs; ++j) {
            P.ac_activa.push_back(p.ac_activa[j]);
            P.ac_bar.push_back((Real)p.ac_barrera[j]);
            P.ac_imp.push_back((Real)p.ac_importe[j]);
        }
    }

    // Barreras
    v.ki_obs = p.ki_obs;
    v.ki_nivel = (Real)p.ki_nivel;
    v.ki_bgk = p.ki_bgk ? 1 : 0;
    v.ki_ini = p.ki_inicial ? 1 : 0;
    v.ko_obs = p.ko_obs;
    v.ko_nivel = (Real)p.ko_nivel;
    v.ko_bgk = p.ko_bgk ? 1 : 0;
    v.ko_imp = (Real)p.ko_importe;
    v.ko_pago_venc = p.ko_pago_venc ? 1 : 0;
    v.monitor_paso = (p.ki_obs >= OBS_DIARIA || p.ko_obs >= OBS_DIARIA || v.hay_rango) ? 1 : 0;

    // Amortización final
    v.rend_medio = p.rend_medio ? 1 : 0;
    v.cliquet = 0;
    v.cliq_suelo_loc = (Real)(-HD_INFINITO);
    v.cliq_techo_loc = (Real)HD_INFINITO;
    for (const Bloque& b : p.bloques) {
        BloqueSim<Real> bs{};
        bs.tipo = b.tipo;
        bs.cond = b.cond;
        bs.importe = (Real)b.importe;
        bs.part = (Real)b.part;
        bs.strike = (Real)b.strike;
        bs.suelo = (Real)b.suelo;
        bs.techo = (Real)b.techo;
        bs.ini = b.formula_ini;
        bs.len = b.formula_len;
        P.bloques.push_back(bs);
        if (b.tipo == BLQ_CLIQUET) {
            v.cliquet = 1;
            v.cliq_suelo_loc = (Real)b.suelo_local;
            v.cliq_techo_loc = (Real)b.techo_local;
        }
    }
    v.n_bloques = (int)P.bloques.size();
    for (const auto& ins : p.formulas) P.formulas.push_back(InstrFormula<Real>{ins.op, (Real)ins.valor});

    P.enlazar();
    return plan;
}

} // namespace est
