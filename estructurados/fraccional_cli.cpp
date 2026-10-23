// estructurados fraccional: modelos fraccionarios y sus métodos numéricos.
//
//   fraccional [modelos]                 catálogo de modelos y métodos
//   fraccional precio <modelo> [k=v ...] el mismo precio por todos los métodos del modelo
//   fraccional sonrisa <modelo> [k=v ...] volatilidades implícitas y skew ATM por plazo
//   fraccional fbm --hurst H              trayectorias del fBm y comprobación de covarianzas
//   fraccional nucleo --hurst H           aproximación markoviana del núcleo fraccionario

#include "fraccional_cli.hpp"
#include "vanilla.hpp"
#include "../fraccional/fraccional.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace est {

namespace {

using Reloj = std::chrono::steady_clock;
double segundos(Reloj::time_point t0) { return std::chrono::duration<double>(Reloj::now() - t0).count(); }

size_t ancho_visual(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) n += ((c & 0xC0) != 0x80);
    return n;
}
std::string col(std::string s, size_t w) {
    const size_t v = ancho_visual(s);
    if (v < w) s += std::string(w - v, ' ');
    return s;
}
std::string dcha(std::string s, size_t w) {
    const size_t v = ancho_visual(s);
    if (v < w) s = std::string(w - v, ' ') + s;
    return s;
}
std::string fijo(double x, int dec) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", dec, x);
    return b;
}
std::string pct(double x, int dec) { return fijo(100.0 * x, dec) + "%"; }
std::string tiempo(double s) { return s < 1e-3 ? "<1 ms" : (s < 1.0 ? fmt(1e3 * s, 0) + " ms" : fmt(s, 2) + " s"); }

// "1/252", "0.004", "20%"
double real_o_fraccion(const std::string& s) {
    const size_t b = s.find('/');
    bool ok = false;
    if (b != std::string::npos) {
        bool ok2 = false;
        const double a = texto_a_real(s.substr(0, b), &ok), c = texto_a_real(s.substr(b + 1), &ok2);
        if (!ok || !ok2 || c == 0) throw std::runtime_error("número no válido: " + s);
        return a / c;
    }
    const double v = texto_a_real(s, &ok);
    if (!ok) throw std::runtime_error("número no válido: " + s);
    return v;
}

struct Args {
    std::vector<std::string> pos;
    std::map<std::string, std::string> op;
    std::set<std::string> flags;
    std::map<std::string, double> prm; // parámetros del modelo (k=v)

    bool tiene(const std::string& k) const { return op.count(k) || flags.count(k); }
    std::string texto(const std::string& k, const std::string& def = "") const {
        auto it = op.find(k);
        return it == op.end() ? def : it->second;
    }
    double real(const std::string& k, double def) const {
        auto it = op.find(k);
        return it == op.end() ? def : real_o_fraccion(it->second);
    }
    double plazo(const std::string& k, double def) const {
        auto it = op.find(k);
        if (it == op.end()) return def;
        bool ok = false;
        const double v = texto_a_plazo(it->second, &ok);
        if (!ok || !(v > 0)) throw std::runtime_error("plazo no válido en " + k + ": " + it->second);
        return v;
    }
};

void leer_pares(const std::string& t, std::map<std::string, double>& m) {
    size_t i = 0;
    while (i < t.size()) {
        const size_t c = t.find(',', i);
        const std::string par = t.substr(i, c == std::string::npos ? std::string::npos : c - i);
        const size_t e = par.find('=');
        if (e == std::string::npos) throw std::runtime_error("se esperaba clave=valor en \"" + par + "\"");
        m[par.substr(0, e)] = real_o_fraccion(par.substr(e + 1));
        if (c == std::string::npos) break;
        i = c + 1;
    }
}

Args leer_args(const std::vector<std::string>& a, size_t desde) {
    static const std::set<std::string> banderas = {"--put", "--call", "--todas"};
    Args r;
    for (size_t i = desde; i < a.size(); ++i) {
        const std::string& k = a[i];
        if (k.rfind("--", 0) == 0) {
            if (banderas.count(k)) { r.flags.insert(k); continue; }
            if (i + 1 >= a.size()) throw std::runtime_error("falta el valor de la opción " + k);
            if (k == "--param" || k == "--params") leer_pares(a[++i], r.prm);
            else r.op[k] = a[++i];
        } else if (k.find('=') != std::string::npos) {
            leer_pares(k, r.prm);
        } else {
            r.pos.push_back(k);
        }
    }
    return r;
}

// ---------------------------------------------------------------- //
// Modelos: los del catálogo del motor y dos solo del módulo        //
// ---------------------------------------------------------------- //
enum class Familia { Catalogo, FBS, Subdif };

struct ModeloFrac {
    Familia fam = Familia::Catalogo;
    Modelo cat;
    frac::FBS fbs;
    frac::Subdifusivo sub;
    std::string nombre;
};

std::string normal(std::string s) {
    for (auto& c : s) c = (c == '-' || c == ' ') ? '_' : (char)std::tolower((unsigned char)c);
    return s;
}

ModeloFrac leer_modelo(const std::string& nombre0, const std::map<std::string, double>& prm) {
    ModeloFrac M;
    const std::string nombre = normal(nombre0);
    auto tomar = [&](std::initializer_list<const char*> claves, double& destino) {
        for (const char* k : claves) {
            auto it = prm.find(k);
            if (it != prm.end()) { destino = it->second; return true; }
        }
        return false;
    };
    auto comprobar_claves = [&](std::initializer_list<const char*> validas) {
        for (const auto& [k, v] : prm) {
            bool ok = false;
            for (const char* x : validas) ok |= (k == x);
            if (!ok) {
                std::string l;
                for (const char* x : validas) l += std::string(l.empty() ? "" : ", ") + x;
                throw std::runtime_error("parámetro \"" + k + "\" desconocido para " + M.nombre + " (válidos: " + l + ")");
            }
        }
    };
    if (nombre == "fbs" || nombre == "bs_fraccional" || nombre == "fbs_mixto" || nombre == "mixto" ||
        nombre == "black_scholes_fraccional") {
        M.fam = Familia::FBS;
        const bool mixto = nombre == "fbs_mixto" || nombre == "mixto";
        M.nombre = mixto ? "Black-Scholes fraccionario mixto" : "Black-Scholes fraccionario";
        M.fbs.H = mixto ? 0.75 : 0.6;
        M.fbs.sigma = mixto ? 0.15 : 0.0;
        M.fbs.sigma_h = mixto ? 0.10 : 0.20;
        tomar({"hurst", "h", "H"}, M.fbs.H);
        tomar({"sigma", "volatilidad", "vol"}, M.fbs.sigma);
        tomar({"sigma_h", "vol_fraccional"}, M.fbs.sigma_h);
        comprobar_claves({"hurst", "h", "H", "sigma", "volatilidad", "vol", "sigma_h", "vol_fraccional"});
        if (!(M.fbs.H > 0 && M.fbs.H < 1)) throw std::runtime_error("fBS: hace falta 0 < H < 1");
        return M;
    }
    if (nombre == "subdifusivo" || nombre == "subdif" || nombre == "bs_subdifusivo") {
        M.fam = Familia::Subdif;
        M.nombre = "Black-Scholes subdifusivo";
        tomar({"alpha", "alfa"}, M.sub.alpha);
        tomar({"sigma", "volatilidad", "vol"}, M.sub.sigma);
        comprobar_claves({"alpha", "alfa", "sigma", "volatilidad", "vol"});
        if (!(M.sub.alpha > 0 && M.sub.alpha <= 1)) throw std::runtime_error("subdifusivo: hace falta 0 < alpha ≤ 1");
        return M;
    }
    const DefModelo* d = buscar_modelo(nombre0);
    if (!d)
        throw std::runtime_error("modelo \"" + nombre0 + "\" desconocido (rough_heston, rough_bergomi, fmls, fbs, "
                                 "fbs_mixto, subdifusivo o cualquiera de 'estructurados modelos')");
    M.nombre = d->nombre;
    M.cat.tipo = d->tipo;
    for (const auto& p : d->params) M.cat.valores.push_back(p.defecto);
    for (const auto& [k, v] : prm) {
        bool hecho = false;
        for (size_t i = 0; i < d->params.size() && !hecho; ++i) {
            const DefParam& p = d->params[i];
            bool es = (p.clave == k);
            for (const auto& al : p.alias) es |= (al == k);
            if (es) { M.cat.valores[i] = v; hecho = true; }
            else if (!p.es_vol_de.empty() && p.es_vol_de == k) { M.cat.valores[i] = std::sqrt(std::max(v, 0.0)); hecho = true; }
        }
        if (!hecho) {
            std::string l;
            for (const auto& p : d->params) l += (l.empty() ? "" : ", ") + p.clave;
            throw std::runtime_error("parámetro \"" + k + "\" desconocido para " + d->nombre + " (válidos: " + l + ")");
        }
    }
    for (size_t i = 0; i < d->params.size(); ++i) {
        const DefParam& p = d->params[i];
        const double v = M.cat.valores[i];
        if (v < p.min || v > p.max || (p.min_abierto && v <= p.min))
            throw std::runtime_error("\"" + p.clave + "\" = " + fmt(v, 6) + " fuera de rango (" + p.descripcion + ")");
    }
    return M;
}

std::string resumen(const ModeloFrac& M) {
    std::ostringstream os;
    if (M.fam == Familia::FBS)
        os << M.nombre << " (hurst=" << fmt(M.fbs.H, 3) << ", sigma=" << porcentaje(M.fbs.sigma) << ", sigma_h="
           << porcentaje(M.fbs.sigma_h) << ")";
    else if (M.fam == Familia::Subdif)
        os << M.nombre << " (alpha=" << fmt(M.sub.alpha, 3) << ", volatilidad=" << porcentaje(M.sub.sigma) << ")";
    else
        os << M.cat.resumen();
    return os.str();
}

Activo activo_de(const Modelo& m, double S, double q) {
    Activo a;
    a.nombre = "S";
    a.spot = a.nivel_inicial = S;
    a.dividendos = q;
    a.modelo = m;
    return a;
}

Curva curva_plana(double r) {
    Curva c;
    c.t = {1.0};
    c.r = {r};
    return c;
}

// ---------------------------------------------------------------- //
// fraccional modelos                                               //
// ---------------------------------------------------------------- //
int modelos() {
    std::cout <<
R"T(Modelos fraccionarios y sus métodos numéricos

 Volatilidad rugosa (núcleo K(t) = t^{H-1/2}/Γ(H+1/2), H < 1/2; skew ATM ~ T^{H-1/2})
  rough_heston   Heston con núcleo fraccionario (El Euch-Rosenbaum 2019)              [motor]
                 · Fourier: ecuación de Riccati fraccionaria D^α h = F(h) con la regla
                   trapezoidal fraccionaria implícita (cuadrática resuelta en cada paso)
                 · Monte Carlo: varianza forward media de cada paso por QE (Andersen),
                   núcleo con factores markovianos y corrección de martingala
  rough_bergomi  V = ξ0·exp(ηY - ½η²t^{2H}), Y de Volterra (Bayer-Friz-Gatheral 2016)  [motor]
                 · esquema híbrido de Bennedsen-Lunde-Pakkanen (convolución por FFT)
                 · esquema híbrido-markoviano del motor (paso cercano exacto + factores)

 Saltos estables (derivada fraccionaria en espacio)
  fmls           log-estable de momentos finitos, β = -1 (Carr-Wu 2003)              [motor]
                 · Fourier (función característica cerrada)
                 · EDP fraccionaria en espacio: Grünwald desplazado y ponderado (WSGD,
                   2º orden) + Crank-Nicolson con arranque de Rannacher; barreras
                 · Monte Carlo exacto (Chambers-Mallows-Stuck)

 Ruido fraccionario en el precio
  fbs, fbs_mixto Black-Scholes fraccionario (Necula 2002) y mixto (Cheridito 2001)
                 · fórmula cerrada con varianza total σ²T + σ_H²T^{2H}
                 · asiáticas: geométrica cerrada; Monte Carlo con trayectorias del fBm
                   (Davies-Harte por FFT o Cholesky) y variable de control

 Derivada fraccionaria en tiempo
  subdifusivo    Black-Scholes en tiempo operacional (Magdziarz 2009), α ∈ (0, 1]
                 · cuadratura de la subordinación (representación de Kanter)
                 · EDP de Caputo: esquema L1 en malla graduada + Richardson
                 · Monte Carlo exacto del tiempo operacional

 [motor]: disponible en los productos estructurados ("modelo": {"tipo": ...}), en la
 réplica estática y en la calibración. Más detalles en estructurados/FRACCIONAL.md.

 Ejemplos:
   estructurados fraccional precio fmls alpha=1.5 volatilidad=25%
   estructurados fraccional precio rough_heston hurst=0.1 --vencimiento 3M --strike 95
   estructurados fraccional sonrisa rough_bergomi hurst=0.07 eta=1.9 rho=-0.9
   estructurados fraccional precio subdifusivo alpha=0.7 --put
   estructurados fraccional precio fbs_mixto hurst=0.8 --asiatica 12
   estructurados fraccional fbm --hurst 0.3 --csv fbm.csv
   estructurados fraccional nucleo --hurst 0.1 --factores 17
)T";
    return 0;
}

// ---------------------------------------------------------------- //
// fraccional precio                                                //
// ---------------------------------------------------------------- //
struct Fila {
    std::string metodo;
    double precio = 0, error = -1, seg = 0;
};

int precio(const Args& A) {
    if (A.pos.empty()) throw std::runtime_error("indique el modelo: estructurados fraccional precio <modelo> [k=v ...]");
    const ModeloFrac M = leer_modelo(A.pos[0], A.prm);
    const double S = A.real("--spot", 100.0), K = A.real("--strike", S), T = A.plazo("--vencimiento", 1.0);
    const double r = A.real("--tipo", 0.03), q = A.real("--dividendos", 0.0);
    const bool call = !A.tiene("--put");
    const double B = A.real("--barrera", 0.0);
    const long long N = (long long)A.real("--trayectorias", 200000);
    const double ppa = A.real("--pasos", 252.0);
    const int n_asia = (int)A.real("--asiatica", 0.0);
    const std::string pay = call ? "call" : "put";
    std::cout << "\n " << resumen(M) << "\n S = " << fmt(S, 2) << ", K = " << fmt(K, 2) << ", T = " << fmt(T, 4)
              << " años, r = " << porcentaje(r) << ", q = " << porcentaje(q) << "  ·  " << pay << " europea";
    if (B > 0) std::cout << " con barrera desactivante " << fmt(B, 2) << " (continua)";
    if (n_asia > 0) std::cout << "; asiática con " << n_asia << " fechas";
    std::cout << "\n\n";

    std::vector<Fila> filas;
    auto medir = [&](const std::string& nom, auto&& f) {
        const auto t0 = Reloj::now();
        Fila fl;
        fl.metodo = nom;
        f(fl);
        fl.seg = segundos(t0);
        filas.push_back(fl);
    };
    const double F = S * std::exp((r - q) * T), DF = std::exp(-r * T);

    if (M.fam == Familia::Catalogo) {
        const TipoModelo t = M.cat.tipo;
        const Activo act = activo_de(M.cat, S, q);
        const Curva cv = curva_plana(r);
        if (B > 0) {
            if (t != TipoModelo::FMLS) throw std::runtime_error("--barrera solo está disponible (por EDP) con fmls");
            frac::FMLS p{M.cat["alpha"], M.cat["volatilidad"]};
            for (auto [nx, nt] : {std::pair{800, 150}, std::pair{1600, 300}})
                medir("EDP fraccionaria WSGD-CN (" + std::to_string(nx) + "×" + std::to_string(nt) + ")",
                      [&](Fila& f) { f.precio = frac::fmls_edp(p, call, S, K, T, r, q, B, {nx, nt}); });
        } else {
            PreciosVanilla pv(act, cv);
            if (!pv.es_mc()) {
                std::string nom = pv.metodo();
                if (nom.rfind("Fourier", 0) == 0)
                    nom = t == TipoModelo::RoughHeston ? "Fourier: Riccati fraccionaria (trapezoidal implícita)"
                                                      : "Fourier (Lewis, función característica)";
                medir(nom, [&](Fila& f) { f.precio = call ? pv.call(T, K) : pv.put(T, K); });
            }
            if (t == TipoModelo::FMLS) {
                frac::FMLS p{M.cat["alpha"], M.cat["volatilidad"]};
                for (auto [nx, nt] : {std::pair{600, 100}, std::pair{1200, 200}})
                    medir("EDP fraccionaria WSGD-CN (" + std::to_string(nx) + "×" + std::to_string(nt) + ")",
                          [&](Fila& f) { f.precio = frac::fmls_edp(p, call, S, K, T, r, q, 0.0, {nx, nt}); });
                medir("Monte Carlo exacto (Chambers-Mallows-Stuck)", [&](Fila& f) {
                    frac::PrecioMC m = frac::fmls_mc(p, call, S, K, T, r, q, N, 17);
                    f.precio = m.precio;
                    f.error = m.error;
                });
            }
            if (t == TipoModelo::RoughBergomi) {
                frac::RoughBergomi p;
                p.H = M.cat["hurst"];
                p.xi0 = std::pow(M.cat["vol_inicial"], 2.0);
                p.eta = M.cat["eta"];
                p.rho = M.cat["rho"];
                const int pasos = std::max(4, (int)std::ceil(T * ppa - 1e-9));
                medir("esquema híbrido BLP con FFT (" + std::to_string(pasos) + " pasos)", [&](Fila& f) {
                    auto R = frac::rbergomi_hibrido(p, T, pasos, N, {K / F}, 23);
                    // c(K) = DF·F·c_1(K/F); la put por paridad
                    const double c = DF * F * R[0].precio;
                    f.precio = call ? c : c - DF * (F - K);
                    f.error = DF * F * R[0].error;
                });
            }
            OpcionesMCVanilla o;
            o.trayectorias = N;
            o.pasos_por_anio = ppa;
            medir(std::string("Monte Carlo del motor (") +
                      (modelo_exacto(t) ? "paso exacto" : std::to_string((int)ppa) + " pasos/año") + ")",
                  [&](Fila& f) {
                      PreciosVanilla mc(act, cv, o, true);
                      f.precio = call ? mc.call(T, K) : mc.put(T, K);
                      f.error = mc.error_mc(T, K, call);
                  });
        }
    } else if (M.fam == Familia::FBS) {
        if (n_asia <= 0) {
            medir("fórmula cerrada (varianza σ²T + σ_H²T^{2H})",
                  [&](Fila& f) { f.precio = frac::fbs_precio(M.fbs, call, S, K, T, r, q); });
        } else {
            medir("asiática geométrica: fórmula cerrada",
                  [&](Fila& f) { f.precio = frac::fbs_asiatica_geometrica(M.fbs, call, S, K, T, r, q, n_asia); });
            for (auto m : {frac::GeneradorFBM::Metodo::DaviesHarte, frac::GeneradorFBM::Metodo::Cholesky}) {
                const std::string nm = m == frac::GeneradorFBM::Metodo::DaviesHarte ? "Davies-Harte" : "Cholesky";
                frac::ResultadoAsiaticaMC R;
                medir("asiática geométrica: Monte Carlo (fBm por " + nm + ")", [&](Fila& f) {
                    R = frac::fbs_asiatica_mc(M.fbs, call, S, K, T, r, q, n_asia, N, 29, m);
                    f.precio = R.geometrica.precio;
                    f.error = R.geometrica.error;
                });
                filas.push_back({"asiática aritmética: MC (" + nm + ") con control geométrico", R.aritmetica.precio,
                                 R.aritmetica.error, 0.0});
            }
        }
    } else {
        if (B > 0 || n_asia > 0) throw std::runtime_error("el modelo subdifusivo solo valora aquí opciones europeas");
        medir("cuadratura de la subordinación (Kanter)",
              [&](Fila& f) { f.precio = frac::subdif_precio(M.sub, call, S, K, T, r, q); });
        medir("EDP de Caputo, esquema L1 + Richardson (800×300)",
              [&](Fila& f) { f.precio = frac::subdif_edp(M.sub, call, S, K, T, r, q, 800, 300); });
        medir("Monte Carlo exacto del tiempo operacional", [&](Fila& f) {
            frac::PrecioMC m = frac::subdif_mc(M.sub, call, S, K, T, r, q, N, 31);
            f.precio = m.precio;
            f.error = m.error;
        });
        std::cout << " Tiempo operacional medio E[E_T] = " << fijo(frac::tiempo_operacional_medio(M.sub.alpha, T), 4)
                  << " años (" << fmt(T, 4) << " de calendario)\n\n";
    }

    std::cout << " " << col("Método", 62) << dcha("Precio", 12) << dcha("Error est.", 12) << dcha("Vol. impl.", 12)
              << dcha("Tiempo", 10) << "\n " << std::string(108, '-') << "\n";
    const int dec = S >= 10 ? 4 : 6;
    for (const auto& f : filas) {
        bool ok = false;
        double iv = (B > 0 || n_asia > 0) ? NAN : vol_implicita(f.precio, call, S, K, T, r, q, &ok);
        std::cout << " " << col(f.metodo, 62) << dcha(fijo(f.precio, dec), 12)
                  << dcha(f.error >= 0 ? fijo(f.error, dec) : "-", 12) << dcha(ok ? pct(iv, 3) : "-", 12)
                  << dcha(f.seg > 0 ? tiempo(f.seg) : "", 10) << "\n";
    }
    std::cout << "\n";
    return 0;
}

// ---------------------------------------------------------------- //
// fraccional sonrisa                                               //
// ---------------------------------------------------------------- //
std::vector<double> lista_plazos(const std::string& s) {
    std::vector<double> v;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t c = s.find(',', i);
        const std::string x = s.substr(i, c == std::string::npos ? std::string::npos : c - i);
        bool ok = false;
        const double t = texto_a_plazo(x, &ok);
        if (!ok || !(t > 0)) throw std::runtime_error("plazo no válido: " + x);
        v.push_back(t);
        if (c == std::string::npos) break;
        i = c + 1;
    }
    std::sort(v.begin(), v.end());
    return v;
}

int sonrisa(const Args& A) {
    if (A.pos.empty()) throw std::runtime_error("indique el modelo: estructurados fraccional sonrisa <modelo> [k=v ...]");
    const ModeloFrac M = leer_modelo(A.pos[0], A.prm);
    const double S = A.real("--spot", 100.0), r = A.real("--tipo", 0.0), q = A.real("--dividendos", 0.0);
    const std::vector<double> Ts = lista_plazos(A.texto("--vencimientos", "1M,3M,6M,1A,2A"));
    // Strikes relativos al forward: min:max:paso
    double kmin = 0.8, kmax = 1.2, kpaso = 0.05;
    {
        const std::string s = A.texto("--strikes", "80%:120%:5%");
        const size_t a1 = s.find(':'), a2 = s.find(':', a1 + 1);
        if (a1 == std::string::npos || a2 == std::string::npos) throw std::runtime_error("--strikes se escribe min:max:paso (K/F)");
        kmin = real_o_fraccion(s.substr(0, a1));
        kmax = real_o_fraccion(s.substr(a1 + 1, a2 - a1 - 1));
        kpaso = real_o_fraccion(s.substr(a2 + 1));
        if (!(kmin > 0 && kmax > kmin && kpaso > 0)) throw std::runtime_error("--strikes no válido");
    }
    std::vector<double> mon;
    for (double m = kmin; m <= kmax + 1e-9; m += kpaso) mon.push_back(m);
    const long long N = (long long)A.real("--trayectorias", 200000);
    const double ppa = A.real("--pasos", 252.0);

    std::unique_ptr<PreciosVanilla> pv;
    const Curva cv = curva_plana(r);
    if (M.fam == Familia::Catalogo) {
        OpcionesMCVanilla o;
        o.trayectorias = N;
        o.pasos_por_anio = ppa;
        pv = std::make_unique<PreciosVanilla>(activo_de(M.cat, S, q), cv, o);
        pv->preparar(Ts);
    }
    auto precio_otm = [&](double T, double K) { // opción fuera del dinero respecto al forward
        const double F = S * std::exp((r - q) * T);
        const bool c = K >= F;
        if (pv) return std::make_pair(c ? pv->call(T, K) : pv->put(T, K), c);
        if (M.fam == Familia::FBS) return std::make_pair(frac::fbs_precio(M.fbs, c, S, K, T, r, q), c);
        return std::make_pair(frac::subdif_precio(M.sub, c, S, K, T, r, q), c);
    };
    auto vol = [&](double T, double K) {
        auto [p, c] = precio_otm(T, K);
        bool ok = false;
        const double v = vol_implicita(p, c, S, K, T, r, q, &ok);
        return ok ? v : NAN;
    };
    const auto t0 = Reloj::now();
    std::cout << "\n " << resumen(M) << "\n Volatilidades implícitas (strike relativo al forward)";
    if (pv) std::cout << " · " << pv->metodo();
    std::cout << "\n\n " << col("Plazo", 9);
    for (double m : mon) std::cout << dcha(pct(m, 0), 9);
    std::cout << dcha("skew ATM", 11) << "\n " << std::string(9 + 9 * mon.size() + 11, '-') << "\n";
    std::vector<double> lt, ls;
    for (double T : Ts) {
        const double F = S * std::exp((r - q) * T);
        std::cout << " " << col(fijo(T, 3), 9);
        for (double m : mon) {
            const double v = vol(T, m * F);
            std::cout << dcha(std::isfinite(v) ? pct(v, 2) : "-", 9);
        }
        // Skew ATM: ∂σ/∂k en k = ln(K/F) = 0, diferencias centradas con h ∝ σ√T
        const double atm = vol(T, F);
        const double h = std::max(0.01, 0.25 * (std::isfinite(atm) ? atm : 0.2) * std::sqrt(T));
        const double sk = (vol(T, F * std::exp(h)) - vol(T, F * std::exp(-h))) / (2.0 * h);
        std::cout << dcha(std::isfinite(sk) ? fijo(sk, 4) : "-", 11) << "\n";
        if (std::isfinite(sk) && std::fabs(sk) > 1e-6) {
            lt.push_back(std::log(T));
            ls.push_back(std::log(std::fabs(sk)));
        }
    }
    if (lt.size() >= 3) {
        double mt = 0, ms = 0;
        for (size_t i = 0; i < lt.size(); ++i) { mt += lt[i]; ms += ls[i]; }
        mt /= lt.size();
        ms /= lt.size();
        double sxy = 0, sxx = 0;
        for (size_t i = 0; i < lt.size(); ++i) { sxy += (lt[i] - mt) * (ls[i] - ms); sxx += (lt[i] - mt) * (lt[i] - mt); }
        const double b = sxy / sxx;
        std::cout << "\n Skew ATM en log-log: |ψ(T)| ∝ T^b con b = " << fijo(b, 3) << " (b + 1/2 = " << fijo(b + 0.5, 3)
                  << ")\n Con volatilidad rugosa ψ(T) ~ T^{H-1/2} cuando T → 0 (b → H - 1/2); en Heston clásico el skew\n"
                     " tiende a una constante en plazos cortos (b → 0). Lejos de T = 0 la reversión a la media\n"
                     " y los términos de orden superior inclinan la recta.\n";
    }
    std::cout << "\n (" << tiempo(segundos(t0)) << ")\n\n";
    return 0;
}

// ---------------------------------------------------------------- //
// fraccional fbm                                                   //
// ---------------------------------------------------------------- //
int fbm(const Args& A) {
    const double H = A.real("--hurst", 0.3), T = A.plazo("--horizonte", 1.0);
    const int n = (int)A.real("--pasos", 252.0);
    const long long M = (long long)A.real("--trayectorias", 20000);
    const std::string met = normal(A.texto("--metodo", "davies_harte"));
    frac::GeneradorFBM::Metodo m;
    if (met == "davies_harte" || met == "dh" || met == "fft") m = frac::GeneradorFBM::Metodo::DaviesHarte;
    else if (met == "cholesky") m = frac::GeneradorFBM::Metodo::Cholesky;
    else throw std::runtime_error("--metodo debe ser davies_harte o cholesky");
    const auto t0 = Reloj::now();
    frac::GeneradorFBM g(H, n, T, m);
    const double t_prep = segundos(t0);
    std::mt19937_64 gen((uint64_t)A.real("--semilla", 1));
    std::vector<double> B;
    // Momentos empíricos: Var(B_T), Var(B_{T/2}), Cov(B_{T/2}, B_T), autocorrelación de los
    // incrementos a retardo 1 y variograma E|B_{t+kΔ} - B_t|² para estimar H
    const int nl = std::min(16, n / 4);
    std::vector<double> vario((size_t)std::max(nl, 1) + 1, 0.0);
    double vT = 0, vh = 0, cov = 0, a1 = 0, a0 = 0;
    std::ofstream csv;
    const std::string ruta = A.texto("--csv");
    const int n_csv = (int)A.real("--guardar", 10);
    std::vector<std::vector<double>> guardadas;
    const auto t1 = Reloj::now();
    for (long long it = 0; it < M; ++it) {
        g.generar(gen, B);
        vT += B[(size_t)n] * B[(size_t)n];
        vh += B[(size_t)n / 2] * B[(size_t)n / 2];
        cov += B[(size_t)n / 2] * B[(size_t)n];
        for (int k = 0; k + 1 < n; ++k) {
            const double d0 = B[(size_t)k + 1] - B[(size_t)k], d1 = B[(size_t)k + 2] - B[(size_t)k + 1];
            a1 += d0 * d1;
            a0 += d0 * d0;
        }
        for (int l = 1; l <= nl; ++l) {
            double s = 0;
            for (int k = 0; k + l <= n; ++k) s += (B[(size_t)k + l] - B[(size_t)k]) * (B[(size_t)k + l] - B[(size_t)k]);
            vario[(size_t)l] += s / (n - l + 1);
        }
        if (!ruta.empty() && (int)guardadas.size() < n_csv) guardadas.push_back(B);
    }
    const double t_gen = segundos(t1);
    const double th = std::pow(T, 2 * H), thh = std::pow(T / 2, 2 * H);
    std::cout << "\n Movimiento browniano fraccionario: H = " << fmt(H, 3) << ", " << n << " pasos en [0, " << fmt(T, 3)
              << "], " << M << " trayectorias, método "
              << (g.metodo() == frac::GeneradorFBM::Metodo::DaviesHarte ? "Davies-Harte (inmersión circulante + FFT)"
                                                                         : "Cholesky")
              << "\n preparación " << tiempo(t_prep) << ", generación " << fijo(1e6 * t_gen / M, 1)
              << " µs por trayectoria\n\n"
              << " " << col("Magnitud", 40) << dcha("Simulado", 12) << dcha("Teórico", 12) << "\n " << std::string(64, '-')
              << "\n";
    auto fila = [](const std::string& k, double a, double b) {
        std::cout << " " << col(k, 40) << dcha(fijo(a, 5), 12) << dcha(fijo(b, 5), 12) << "\n";
    };
    fila("Var(B_T) = T^{2H}", vT / M, th);
    fila("Var(B_{T/2})", vh / M, thh);
    fila("Cov(B_{T/2}, B_T)", cov / M, 0.5 * (th + thh - thh));
    fila("correlación de incrementos (retardo 1)", a1 / a0, std::pow(2.0, 2 * H - 1) - 1);
    if (nl >= 2) {
        double mx = 0, my = 0;
        for (int l = 1; l <= nl; ++l) { mx += std::log((double)l); my += std::log(vario[(size_t)l]); }
        mx /= nl;
        my /= nl;
        double sxy = 0, sxx = 0;
        for (int l = 1; l <= nl; ++l) {
            const double x = std::log((double)l) - mx, y = std::log(vario[(size_t)l]) - my;
            sxy += x * y;
            sxx += x * x;
        }
        fila("H estimado por el variograma", 0.5 * sxy / sxx, H);
    }
    if (!ruta.empty()) {
        std::ofstream f(ruta);
        if (!f) throw std::runtime_error("no se puede escribir " + ruta);
        f << "t";
        for (size_t j = 0; j < guardadas.size(); ++j) f << ",B" << j + 1;
        f << "\n";
        for (int k = 0; k <= n; ++k) {
            f << T * k / n;
            for (const auto& b : guardadas) f << "," << b[(size_t)k];
            f << "\n";
        }
        std::cout << "\n " << guardadas.size() << " trayectorias guardadas en " << ruta << "\n";
    }
    std::cout << "\n";
    return 0;
}

// ---------------------------------------------------------------- //
// fraccional nucleo                                                //
// ---------------------------------------------------------------- //
int nucleo(const Args& A) {
    const double H = A.real("--hurst", 0.1), T = A.plazo("--horizonte", 1.0), dt = A.real("--dt", 1.0 / 252);
    const int fac = (int)A.real("--factores", 17);
    if (!(H > 0 && H <= 0.5)) throw std::runtime_error("--hurst debe estar en (0, 0.5]");
    const frac::NodosNucleo nn = frac::nodos_nucleo(H, fac, dt, T);
    std::cout << "\n Núcleo K(t) = t^{H-1/2}/Γ(H+1/2) con H = " << fmt(H, 3) << " ≈ Σ c_j·exp(-x_j·t) en [" << fmt(dt, 5)
              << ", " << fmt(T, 3) << "]\n (medida de Bernstein μ(dx) = x^{-α}dx/(Γ(α)Γ(1-α)): un nodo de masa en "
                 "[0, x_min] y Gauss-Legendre de 2 puntos en paneles de ln x)\n\n"
              << " " << dcha("j", 4) << dcha("x_j", 16) << dcha("c_j", 16) << "\n " << std::string(36, '-') << "\n";
    for (size_t j = 0; j < nn.x.size(); ++j) {
        char b1[32], b2[32];
        std::snprintf(b1, sizeof b1, "%.6e", nn.x[j]);
        std::snprintf(b2, sizeof b2, "%.6e", nn.c[j]);
        std::cout << " " << dcha(std::to_string(j), 4) << dcha(b1, 16) << dcha(b2, 16) << "\n";
    }
    std::cout << "\n " << col("t", 12) << dcha("K(t)", 14) << dcha("aproximación", 16) << dcha("error rel.", 12) << "\n "
              << std::string(54, '-') << "\n";
    for (int i = 0; i <= 6; ++i) {
        const double t = dt * std::pow(T / dt, i / 6.0);
        double s = 0;
        for (size_t j = 0; j < nn.x.size(); ++j) s += nn.c[j] * std::exp(-nn.x[j] * t);
        const double k = frac::nucleo(H, t);
        char e[32];
        std::snprintf(e, sizeof e, "%.2e", std::fabs(s / k - 1));
        std::cout << " " << col(fijo(t, 5), 12) << dcha(fijo(k, 6), 14) << dcha(fijo(s, 6), 16) << dcha(e, 12) << "\n";
    }
    char e[32];
    std::snprintf(e, sizeof e, "%.2e", frac::error_nodos(nn, H, dt, T));
    std::cout << "\n Error relativo máximo en [dt, T]: " << e << "  (" << nn.x.size() << " factores)\n\n";
    return 0;
}

} // namespace

void uso_fraccional(std::ostream& os) {
    os << "  estructurados fraccional [modelos]                  Modelos fraccionarios y sus métodos numéricos\n"
          "  estructurados fraccional precio <modelo> [k=v ...]  Mismo precio por todos los métodos del modelo\n"
          "        --spot S --strike K --vencimiento T --tipo r --dividendos q --put --barrera B (fmls)\n"
          "        --asiatica N (fbs) --trayectorias N --pasos P\n"
          "  estructurados fraccional sonrisa <modelo> [k=v ...] Sonrisa y skew ATM por plazo (ley de potencia)\n"
          "        --vencimientos 1M,3M,1A --strikes 80%:120%:5% (K/F)\n"
          "  estructurados fraccional fbm --hurst H [--pasos n --horizonte T --trayectorias N\n"
          "        --metodo davies_harte|cholesky --csv trayectorias.csv]\n"
          "  estructurados fraccional nucleo --hurst H [--factores n --dt 1/252 --horizonte T]\n";
}

int cmd_fraccional(const std::vector<std::string>& a) {
    const std::string sub = a.empty() ? "modelos" : a[0];
    const Args A = leer_args(a, 1);
    if (sub == "modelos" || sub == "ayuda") return modelos();
    if (sub == "precio" || sub == "precios") return precio(A);
    if (sub == "sonrisa" || sub == "smile") return sonrisa(A);
    if (sub == "fbm") return fbm(A);
    if (sub == "nucleo" || sub == "núcleo") return nucleo(A);
    throw std::runtime_error("subcomando desconocido: fraccional " + sub + " (modelos, precio, sonrisa, fbm o nucleo)");
}

} // namespace est
