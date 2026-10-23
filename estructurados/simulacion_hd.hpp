#pragma once
// ---------------------------------------------------------------------------- //
// Núcleo de simulación de productos estructurados (__host__ __device__).       //
//                                                                              //
// Una trayectoria = una llamada a simular_trayectoria(): avanza todos los      //
// subyacentes por la rejilla temporal (que incluye exactamente las fechas de   //
// observación), vigila las barreras en cada paso si hace falta y, en cada      //
// fecha de observación, aplica en este orden:                                  //
//   1. barrera de capital (knock-in) y barrera desactivante (knock-out)        //
//   2. cupones (fijo, condicional con o sin memoria, range accrual)            //
//   3. acumuladores (rendimiento medio, cliquet)                               //
//   4. autocancelación; si no la hay y es la última fecha, amortización final  //
// Todos los importes se expresan en tanto por uno del nominal y se descuentan  //
// a hoy con la curva de tipos determinista.                                   //
//                                                                              //
// El mismo código lo ejecutan el motor CPU (motor_cpu.cpp, Real=double) y el   //
// kernel CUDA (motor_gpu.cu, Real=float o double).                            //
// ---------------------------------------------------------------------------- //

#include "../modelos_hd.hpp"

namespace est {

inline constexpr int MAX_ACT_HD     = 16;  // subyacentes por producto
inline constexpr int MAX_CUP_HD     = 4;   // patas de cupón
inline constexpr int MAX_BLOQUES_HD = 16;  // componentes de la amortización final

// Códigos enteros (se guardan en estructuras POD que van a la GPU)
enum : int { CESTA_PEOR = 0, CESTA_MEJOR = 1, CESTA_MEDIA = 2, CESTA_PONDERADA = 3 };
enum : int { OBS_NINGUNA = 0, OBS_VENCIMIENTO = 1, OBS_FECHAS = 2, OBS_DIARIA = 3, OBS_CONTINUA = 4 };
enum : int { CUP_FIJO = 0, CUP_CONDICIONAL = 1, CUP_RANGO = 2 };
enum : int { BLQ_FIJO = 0, BLQ_LINEAL = 1, BLQ_CALL = 2, BLQ_PUT = 3,
             BLQ_DIGITAL = 4, BLQ_DIGITAL_PUT = 5, BLQ_CLIQUET = 6, BLQ_FORMULA = 7 };
enum : int { COND_SIEMPRE = 0, COND_CON_BARRERA = 1, COND_SIN_BARRERA = 2 };

// ------------------------------------------------------------------------ //
// Fórmulas de pago: una expresión en P ("max(P - 1, 0)^2", ...) se compila  //
// en host (expresion.cpp) a instrucciones de una máquina de pila que se     //
// evalúa aquí, igual en CPU y en GPU. La profundidad de la pila se         //
// comprueba al compilar, así que la evaluación no necesita comprobaciones. //
// ------------------------------------------------------------------------ //
enum : int {
    FX_CONST = 0, FX_P,                                  // apilan un valor
    FX_NEG, FX_ABS, FX_EXP, FX_LOG, FX_RAIZ,             // unarias
    FX_SUMA, FX_RESTA, FX_MUL, FX_DIV, FX_POT, FX_MAX, FX_MIN,   // binarias
    FX_MAYOR_IGUAL, FX_MAYOR, FX_MENOR_IGUAL, FX_MENOR, FX_IGUAL, FX_DISTINTO,
    FX_SI                                                // ternaria: si(c, a, b)
};
inline constexpr int MAX_PILA_FORMULA = 32;

template<typename Real>
struct InstrFormula {
    int  op;
    Real valor;   // solo FX_CONST
};

template<typename Real>
HD_INLINE Real evaluar_formula(const InstrFormula<Real>* cod, int n, Real P) {
    Real pila[MAX_PILA_FORMULA];
    int sp = 0;
    for (int i = 0; i < n; ++i) {
        const int op = cod[i].op;
        if (op == FX_CONST) { pila[sp++] = cod[i].valor; continue; }
        if (op == FX_P)     { pila[sp++] = P; continue; }
        if (op <= FX_RAIZ) {
            Real& a = pila[sp - 1];
            switch (op) {
                case FX_NEG:  a = -a; break;
                case FX_ABS:  a = hd::abs_(a); break;
                case FX_EXP:  a = hd::exp_(a); break;
                case FX_LOG:  a = hd::log_(a); break;
                default:      a = hd::sqrt_(a); break;
            }
            continue;
        }
        if (op == FX_SI) {
            const Real b = pila[--sp], a = pila[--sp];
            Real& c = pila[sp - 1];
            c = (c != Real(0)) ? a : b;
            continue;
        }
        const Real b = pila[--sp];
        Real& a = pila[sp - 1];
        switch (op) {
            case FX_SUMA:        a = a + b; break;
            case FX_RESTA:       a = a - b; break;
            case FX_MUL:         a = a * b; break;
            case FX_DIV:         a = a / b; break;
            case FX_POT:         a = hd::pow_(a, b); break;
            case FX_MAX:         a = hd::max_(a, b); break;
            case FX_MIN:         a = hd::min_(a, b); break;
            case FX_MAYOR_IGUAL: a = (a >= b) ? Real(1) : Real(0); break;
            case FX_MAYOR:       a = (a > b) ? Real(1) : Real(0); break;
            case FX_MENOR_IGUAL: a = (a <= b) ? Real(1) : Real(0); break;
            case FX_MENOR:       a = (a < b) ? Real(1) : Real(0); break;
            case FX_IGUAL:       a = (a == b) ? Real(1) : Real(0); break;
            default:             a = (a != b) ? Real(1) : Real(0); break;
        }
    }
    return sp > 0 ? pila[0] : Real(0);
}

// Componente de la amortización final (fracción del nominal), función del
// rendimiento final P de la cesta:
//   FIJO        importe
//   LINEAL      part · clamp(P, suelo, techo)
//   CALL        part · max(min(P, techo) - strike, 0)
//   PUT         part · max(strike - max(P, suelo), 0)
//   DIGITAL     importe · 1{P ≥ strike}
//   DIGITAL_PUT importe · 1{P < strike}
//   CLIQUET     part · clamp(Σ_j clamp(P_j/P_{j-1} - 1, suelo_loc, techo_loc), suelo, techo)
//   FORMULA     f(P), instrucciones [ini, ini+len) de SimProducto::formulas
template<typename Real>
struct BloqueSim {
    int  tipo;
    int  cond;     // COND_*: solo cuenta si se cumple la condición sobre la barrera de capital
    Real importe;
    Real part;
    Real strike;
    Real suelo;
    Real techo;
    int  ini, len; // FORMULA
};

// Vista POD del producto ya "compilado" y de su rejilla temporal. Los punteros
// apuntan a memoria de host (motor CPU) o de dispositivo (motor GPU).
template<typename Real>
struct SimProducto {
    // Subyacentes
    int n_act;
    int dim_paso;                  // normales por paso = n_act + Σ n_extra
    int off_extra[MAX_ACT_HD];     // posición del primer ruido extra de cada activo en el paso
    const ParamsActivo<Real>* act; // n_act
    const Real* chol;              // n_act×n_act triangular inferior (fila principal) o nullptr
    int cesta;                     // CESTA_*
    const Real* pesos;             // n_act (solo CESTA_PONDERADA)

    // Rejilla temporal: pasos k = 0..n_pasos-1 de [t_ini[k], t_ini[k]+dt[k]]
    int n_pasos;
    const Real* dt;
    const Real* sqdt;
    const Real* rdt;               // ∫ r en el paso
    const Real* t_ini;
    const Real* rcum_ini;          // ∫_0^{t_ini} r
    const Real* df_fin;            // factor de descuento al final del paso
    const int*  obs_paso;          // índice de la observación que cae al final del paso, o -1
    int monitor_paso;              // 1 si hay que evaluar la cesta en todos los pasos
    int hay_rango;                 // 1 si alguna pata de cupón es range accrual

    // Calendario de observación
    int n_obs;
    const Real* t_obs;
    const Real* df_obs;
    Real T;
    Real df_T;

    // Cupones (n_cup patas; importes y barreras por fecha, fila = pata)
    int  n_cup;
    int  cup_tipo[MAX_CUP_HD];
    int  cup_mem[MAX_CUP_HD];
    Real cup_inf[MAX_CUP_HD];
    Real cup_sup[MAX_CUP_HD];
    Real mem_ini[MAX_CUP_HD];      // cupones ya acumulados en memoria (productos vivos)
    const Real* cup_imp;           // n_cup × n_obs
    const Real* cup_bar;           // n_cup × n_obs (HD_INFINITO = esa fecha no paga)

    // Autocancelación (nullptr si el producto no es autocancelable)
    const int*  ac_activa;         // n_obs
    const Real* ac_bar;            // n_obs
    const Real* ac_imp;            // n_obs: importe amortizado (1 + prima)

    // Barrera de capital (knock-in, hacia abajo) y desactivante (knock-out, hacia arriba)
    int  ki_obs;  Real ki_nivel; int ki_bgk; int ki_ini;
    int  ko_obs;  Real ko_nivel; int ko_bgk; Real ko_imp; int ko_pago_venc;

    // Amortización final
    int rend_medio;                // 1: P final = media de las observaciones
    int n_bloques;
    const BloqueSim<Real>* bloques;
    const InstrFormula<Real>* formulas; // código de los componentes FORMULA (o nullptr)
    int  cliquet;
    Real cliq_suelo_loc, cliq_techo_loc;
};

// Resultado de una trayectoria (fracciones del nominal, valor presente)
template<typename Real>
struct ResTray {
    Real pv;       // total
    Real pv_cup;   // cupones
    Real pv_ant;   // amortización anticipada (autocancelación / desactivación)
    Real pv_fin;   // amortización a vencimiento
    Real vida;     // instante en que termina el producto
    Real perdida;  // 1 - amortización final, si es < 1
    int  ki, ko, hubo_perdida;
};

// ------------------------------------------- //
// Fuentes de ruido normal: Philox (MC) y Sobol //
// ------------------------------------------- //

// MC: la normal de la dimensión `dim` de la trayectoria `idx` sale de
// Philox4x32-10 con contador (dim/4, idx_lo, idx_hi, flujo) y clave (semilla):
// cada llamada a Philox sirve 4 dimensiones consecutivas.
template<typename Real>
struct RuidoPhilox {
    uint32_t k0, k1, c1, c2, c3;
    uint32_t bloque;
    uint32_t cache[4];

    HD_INLINE void iniciar(uint32_t semilla, uint64_t idx, uint32_t flujo) {
        k0 = semilla; k1 = 0x6D2B79F5u ^ (semilla * 0x9E3779B9u);
        c1 = (uint32_t)idx; c2 = (uint32_t)(idx >> 32); c3 = flujo;
        bloque = 0xFFFFFFFFu;
    }
    HD_INLINE Real normal(uint32_t dim) {
        uint32_t b = dim >> 2;
        if (b != bloque) {
            hd::Philox4 r = hd::philox4x32_10(b, c1, c2, c3, k0, k1);
            cache[0] = r.v[0]; cache[1] = r.v[1]; cache[2] = r.v[2]; cache[3] = r.v[3];
            bloque = b;
        }
        return hd::inv_norm_cdf(hd::uniforme24<Real>(cache[dim & 3u]));
    }
};

// QMC: punto `idx` de la secuencia de Sobol con el scrambling de Hong-Hickernell
// de la réplica (sal). Las dimensiones que exceden la tabla de vectores
// directores se completan con Philox (flujo propio de la réplica).
template<typename Real>
struct RuidoSobol {
    const uint32_t* dv;   // vectores directores: [n_dims_sobol][32]
    uint32_t n_dims_sobol;
    uint32_t idx;
    uint32_t sal_base;    // splitmix32(sal de la réplica)
    RuidoPhilox<Real> relleno;

    HD_INLINE void iniciar(const uint32_t* dirvec, uint32_t ndims, uint32_t indice,
                           uint32_t sal, uint32_t semilla, uint32_t replica) {
        dv = dirvec; n_dims_sobol = ndims; idx = indice;
        sal_base = hd::splitmix32(sal);
        relleno.iniciar(semilla ^ 0x51ED270Bu, (uint64_t)indice, 1u + replica);
    }
    HD_INLINE Real normal(uint32_t dim) {
        if (dim >= n_dims_sobol) return relleno.normal(dim - n_dims_sobol);
        uint32_t raw = hd::sobol32(dv + 32u * dim, idx);
        uint32_t s = hd::hh_scramble(raw, sal_base ^ dim);
        return hd::inv_norm_cdf(hd::uniforme24<Real>(s));
    }
};

// Registro que no guarda nada (para trayectorias de las que solo interesa el precio)
struct RegistroNulo {
    HD_INLINE void autocancelacion(int) {}
    HD_INLINE void cupon(int) {}
};

// ------------------------------- //
// Cesta y amortización final       //
// ------------------------------- //

template<typename Real>
HD_INLINE Real agregar_cesta(int tipo, const Real* p, int n, const Real* pesos) {
    Real r = p[0];
    switch (tipo) {
    case CESTA_PEOR:  for (int i = 1; i < n; ++i) r = hd::min_(r, p[i]); break;
    case CESTA_MEJOR: for (int i = 1; i < n; ++i) r = hd::max_(r, p[i]); break;
    case CESTA_MEDIA: for (int i = 1; i < n; ++i) r += p[i]; r /= (Real)n; break;
    default:          r = Real(0); for (int i = 0; i < n; ++i) r += pesos[i] * p[i]; break;
    }
    return r;
}

template<typename Real>
HD_INLINE Real amortizacion_final(const SimProducto<Real>& sp, Real P, int ki, Real cliq) {
    Real A = Real(0);
    for (int b = 0; b < sp.n_bloques; ++b) {
        const BloqueSim<Real>& bl = sp.bloques[b];
        if (bl.cond == COND_CON_BARRERA && !ki) continue;
        if (bl.cond == COND_SIN_BARRERA && ki) continue;
        switch (bl.tipo) {
        case BLQ_FIJO:        A += bl.importe; break;
        case BLQ_LINEAL:      A += bl.part * hd::clamp_(P, bl.suelo, bl.techo); break;
        case BLQ_CALL:        A += bl.part * hd::max_(hd::min_(P, bl.techo) - bl.strike, Real(0)); break;
        case BLQ_PUT:         A += bl.part * hd::max_(bl.strike - hd::max_(P, bl.suelo), Real(0)); break;
        case BLQ_DIGITAL:     if (P >= bl.strike) A += bl.importe; break;
        case BLQ_DIGITAL_PUT: if (P < bl.strike) A += bl.importe; break;
        case BLQ_CLIQUET:     A += bl.part * hd::clamp_(cliq, bl.suelo, bl.techo); break;
        case BLQ_FORMULA:     A += evaluar_formula(sp.formulas + bl.ini, bl.len, P); break;
        default: break;
        }
    }
    return A;
}

// Cesta con cada rendimiento ajustado por la corrección de continuidad BGK:
// signo = -1 acerca la barrera inferior (knock-in), +1 la superior (knock-out).
template<typename Real>
HD_INLINE Real cesta_bgk(const SimProducto<Real>& sp, const Real* p, const Real* x, const Real* v,
                         Real t1, Real rc1, Real sq, Real signo, Real* tmp) {
    for (int i = 0; i < sp.n_act; ++i) {
        Real vol = vol_instantanea(sp.act[i], x[i], v[i], t1, rc1);
        tmp[i] = p[i] * hd::exp_(signo * Real(BGK_BETA_HD) * vol * sq);
    }
    return agregar_cesta(sp.cesta, tmp, sp.n_act, sp.pesos);
}

// ------------------------------------------------------------------------ //
// Simulación completa de una trayectoria                                    //
// ------------------------------------------------------------------------ //
template<typename Real, class Ruido, class Registro>
HD void simular_trayectoria(const SimProducto<Real>& sp, Ruido& rz, Registro& reg, ResTray<Real>& out) {
    const int n = sp.n_act;
    Real x[MAX_ACT_HD], v[MAX_ACT_HD], z[MAX_ACT_HD], p[MAX_ACT_HD];
    for (int i = 0; i < n; ++i) {
        x[i] = sp.act[i].x0;
        v[i] = estado_aux_inicial(sp.act[i]);
    }

    int ki = sp.ki_ini, ko = 0;
    Real mem[MAX_CUP_HD];
    int dentro[MAX_CUP_HD];
    for (int c = 0; c < sp.n_cup; ++c) { mem[c] = sp.mem_ini[c]; dentro[c] = 0; }
    int total_rango = 0;
    Real suma_P = Real(0);
    int n_P = 0;
    Real cliq = Real(0), P_prev = Real(1);
    Real pv_cup = Real(0), pv_ant = Real(0), pv_fin = Real(0);
    Real vida = sp.T, perdida = Real(0);
    int hubo_perdida = 0;
    bool vivo = true;

    for (int k = 0; k < sp.n_pasos && vivo; ++k) {
        const Real dt = sp.dt[k], sq = sp.sqdt[k], rd = sp.rdt[k];
        const Real t0 = sp.t_ini[k], rc = sp.rcum_ini[k];
        const uint32_t base = (uint32_t)k * (uint32_t)sp.dim_paso;

        // Normales principales y correlación w = L·z (in situ, de la última fila a la primera)
        for (int i = 0; i < n; ++i) z[i] = rz.normal(base + (uint32_t)i);
        if (sp.chol) {
            for (int i = n - 1; i >= 0; --i) {
                Real s = Real(0);
                for (int jj = 0; jj <= i; ++jj) s += sp.chol[i * n + jj] * z[jj];
                z[i] = s;
            }
        }
        // Paso de cada subyacente con sus normales extra
        for (int i = 0; i < n; ++i) {
            const ParamsActivo<Real>& a = sp.act[i];
            const uint32_t eb = base + (uint32_t)sp.off_extra[i];
            Real e0 = Real(0), e1 = Real(0), e2 = Real(0);
            if (a.n_extra > 0) e0 = rz.normal(eb);
            if (a.n_extra > 1) e1 = rz.normal(eb + 1u);
            if (a.n_extra > 2) e2 = rz.normal(eb + 2u);
            paso_modelo(a, x[i], v[i], t0, dt, sq, rd, rc, z[i], e0, e1, e2);
        }

        const int j = sp.obs_paso[k];
        if (!sp.monitor_paso && j < 0) continue;

        for (int i = 0; i < n; ++i) p[i] = hd::exp_(x[i]) * sp.act[i].s_ref_inv;

        // Vigilancia en cada paso: barreras diarias/continuas y range accrual
        if (sp.monitor_paso) {
            const Real t1 = t0 + dt, rc1 = rc + rd;
            if (!ki && sp.ki_obs >= OBS_DIARIA) {
                Real P = (sp.ki_obs == OBS_CONTINUA && sp.ki_bgk)
                       ? cesta_bgk(sp, p, x, v, t1, rc1, sq, Real(-1), z)
                       : agregar_cesta(sp.cesta, p, n, sp.pesos);
                if (P <= sp.ki_nivel) ki = 1;
            }
            if (sp.ko_obs >= OBS_DIARIA) {
                Real P = (sp.ko_obs == OBS_CONTINUA && sp.ko_bgk)
                       ? cesta_bgk(sp, p, x, v, t1, rc1, sq, Real(1), z)
                       : agregar_cesta(sp.cesta, p, n, sp.pesos);
                if (P >= sp.ko_nivel) {
                    ko = 1; vivo = false; vida = t1;
                    pv_ant += sp.ko_imp * (sp.ko_pago_venc ? sp.df_T : sp.df_fin[k]);
                    break;
                }
            }
            if (sp.hay_rango) {
                Real P = agregar_cesta(sp.cesta, p, n, sp.pesos);
                ++total_rango;
                for (int c = 0; c < sp.n_cup; ++c)
                    if (sp.cup_tipo[c] == CUP_RANGO && P >= sp.cup_inf[c] && P <= sp.cup_sup[c])
                        ++dentro[c];
            }
        }
        if (j < 0) continue;

        // ---------------- Fecha de observación j ---------------- //
        const Real P = agregar_cesta(sp.cesta, p, n, sp.pesos);
        const Real df = sp.df_obs[j];
        const bool ultima = (j == sp.n_obs - 1);

        if (!ki && (sp.ki_obs == OBS_FECHAS || (sp.ki_obs == OBS_VENCIMIENTO && ultima)) &&
            P <= sp.ki_nivel)
            ki = 1;
        if (sp.ko_obs == OBS_FECHAS && P >= sp.ko_nivel) {
            ko = 1; vivo = false; vida = sp.t_obs[j];
            pv_ant += sp.ko_imp * (sp.ko_pago_venc ? sp.df_T : df);
            break;
        }

        // Cupones
        Real pago = Real(0);
        int pagado = 0;
        for (int c = 0; c < sp.n_cup; ++c) {
            const Real imp = sp.cup_imp[c * sp.n_obs + j];
            if (sp.cup_tipo[c] == CUP_FIJO) {
                if (imp != Real(0)) { pago += imp; pagado = 1; }
            } else if (sp.cup_tipo[c] == CUP_CONDICIONAL) {
                if (P >= sp.cup_bar[c * sp.n_obs + j]) {
                    Real a = imp + mem[c];
                    if (a != Real(0)) { pago += a; pagado = 1; }
                    mem[c] = Real(0);
                } else if (sp.cup_mem[c]) {
                    mem[c] += imp;
                }
            } else { // CUP_RANGO
                if (total_rango > 0 && imp != Real(0) && dentro[c] > 0) {
                    pago += imp * (Real)dentro[c] / (Real)total_rango;
                    pagado = 1;
                }
                dentro[c] = 0;
            }
        }
        total_rango = 0;
        if (pagado) {
            pv_cup += pago * df;
            reg.cupon(j);
        }

        // Acumuladores
        suma_P += P;
        ++n_P;
        if (sp.cliquet) {
            cliq += hd::clamp_(P / P_prev - Real(1), sp.cliq_suelo_loc, sp.cliq_techo_loc);
            P_prev = P;
        }

        // Autocancelación o amortización final
        if (sp.ac_activa && sp.ac_activa[j] && P >= sp.ac_bar[j]) {
            pv_ant += sp.ac_imp[j] * df;
            vivo = false;
            vida = sp.t_obs[j];
            reg.autocancelacion(j);
        } else if (ultima) {
            const Real Pf = sp.rend_medio ? suma_P / (Real)n_P : P;
            const Real A = amortizacion_final(sp, Pf, ki, cliq);
            pv_fin += A * df;
            if (A < Real(1) - Real(1e-6)) { hubo_perdida = 1; perdida = Real(1) - A; }
            vivo = false;
        }
    }

    out.pv = pv_cup + pv_ant + pv_fin;
    out.pv_cup = pv_cup;
    out.pv_ant = pv_ant;
    out.pv_fin = pv_fin;
    out.vida = vida;
    out.perdida = perdida;
    out.ki = ki;
    out.ko = ko;
    out.hubo_perdida = hubo_perdida;
}

} // namespace est
