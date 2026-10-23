// Tests del motor de productos estructurados (CPU, sin GPU).
//
// Cada test valora un producto con el motor CPU (Monte Carlo con Philox) y lo
// compara con un precio de referencia analítico o semianalítico
// (referencias.hpp). Criterio: |MC - referencia| <= 4·error_estándar + tolerancia
// (la tolerancia cubre el sesgo de discretización temporal cuando lo hay).
// Los casos deterministas (volatilidad 0, autocancelación segura...) se exigen
// exactos. Uso: ./test_estructurados [rapido]

#include "estructurados/calibracion.hpp"
#include "estructurados/cotizaciones.hpp"
#include "estructurados/expresion.hpp"
#include "estructurados/historico.hpp"
#include "estructurados/mercado.hpp"
#include "estructurados/optimizacion.hpp"
#include "estructurados/perfil.hpp"
#include "estructurados/plan.hpp"
#include "estructurados/plantillas.hpp"
#include "estructurados/producto.hpp"
#include "estructurados/replicacion.hpp"
#include "estructurados/valoracion.hpp"
#include "estructurados/vanilla.hpp"
#include "referencias.hpp"

#include <fstream>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace est;

namespace {

int g_ok = 0, g_fallos = 0;
double g_escala = 1.0; // "rapido": menos trayectorias

void resultado(bool ok, const std::string& nombre, const std::string& detalle) {
    // relleno por anchura visual (los caracteres UTF-8 con tilde ocupan 2 bytes)
    size_t vis = 0;
    for (unsigned char c : nombre) vis += ((c & 0xC0) != 0x80);
    std::string nom = nombre + std::string(vis < 58 ? 58 - vis : 1, ' ');
    std::printf("[%s] %s %s\n", ok ? " OK  " : "FALLO", nom.c_str(), detalle.c_str());
    (ok ? g_ok : g_fallos)++;
}

void comparar(const std::string& nombre, double mc, double se, double ref, double tol_abs) {
    double dif = mc - ref;
    bool ok = std::fabs(dif) <= 4.0 * se + tol_abs;
    char buf[256];
    std::snprintf(buf, sizeof buf, "MC=%.6f ±%.6f  ref=%.6f  dif=%+.6f (%.1f σ)", mc, se, ref, dif,
                  se > 0 ? dif / se : 0.0);
    resultado(ok, nombre, buf);
}

void exacto(const std::string& nombre, double x, double ref, double tol = 1e-9) {
    char buf[200];
    std::snprintf(buf, sizeof buf, "valor=%.10f  esperado=%.10f", x, ref);
    resultado(std::fabs(x - ref) <= tol, nombre, buf);
}

std::string num(double x) { return json::numero_a_texto(x); }

Producto compilar_texto(const std::string& s) { return compilar(json::parsear(s)); }

ResultadoValoracion mc(const Producto& p, long long N, int semilla = 11) {
    OpcionesValoracion o;
    o.metodo = "mc";
    o.dispositivo = "cpu";
    o.trayectorias = std::max<long long>(2000, (long long)(N * g_escala));
    o.semilla = semilla;
    return valorar(p, o);
}

// Producto de un solo subyacente (spot 100 = nivel inicial)
std::string un_activo(const std::string& modelo, double q, double T, double r, const std::string& resto,
                      const std::string& obs = "") {
    return "{\"vencimiento\": " + num(T) + ", \"mercado\": {\"tipo_interes\": " + num(r) + "}," +
           "\"subyacentes\": [{\"nombre\": \"X\", \"spot\": 100, \"dividendos\": " + num(q) +
           ", \"modelo\": " + modelo + "}]," + (obs.empty() ? "" : obs + ",") + resto + "}";
}

const std::string BS25 = R"({"tipo": "black_scholes", "volatilidad": 0.25})";

// ------------------------------------------------------------------------- //

void test_deterministas() {
    // Cupón cero: el único pago es el nominal a vencimiento
    {
        Producto p = compilar_texto(un_activo(BS25, 0.0, 2.0, 0.03, R"("amortizacion_final": [{"tipo": "fijo"}])"));
        auto r = mc(p, 5000);
        exacto("Cupón cero = e^{-rT}", r.precio, std::exp(-0.06), 1e-12);
    }
    // Bono con cupón fijo semestral (5% anual) y curva de tipos
    {
        std::string s = R"({"vencimiento": "3A", "mercado": {"curva": [["1A", "2%"], ["3A", "3%"]]},
            "subyacentes": [{"modelo": {"tipo": "bs", "volatilidad": "20%"}}],
            "observaciones": {"frecuencia": "semestral"},
            "cupones": {"tipo": "fijo", "tasa_anual": "5%"},
            "amortizacion_final": [{"tipo": "fijo", "importe": "100%"}]})";
        Producto p = compilar_texto(s);
        double ref = 0.0;
        for (int j = 1; j <= 6; ++j) ref += 0.025 * p.curva.df(0.5 * j);
        ref += p.curva.df(3.0);
        auto r = mc(p, 5000);
        exacto("Bono con cupones fijos y curva interpolada", r.precio, ref, 1e-12);
    }
    // Autocancelación segura (barrera 1%): amortiza siempre en la primera fecha
    {
        Producto p = compilar_texto(un_activo(BS25, 0.0, 1.0, 0.04,
            R"("autocancelacion": {"barrera": "1%", "prima": "2%"}, "amortizacion_final": [{"tipo": "fijo"}])",
            R"("observaciones": "trimestral")"));
        auto r = mc(p, 5000);
        exacto("Autocancelación segura en la 1ª fecha", r.precio, 1.02 * std::exp(-0.04 * 0.25), 1e-12);
        exacto("  ... con vida media = 0.25 años", r.vida_media, 0.25, 1e-12);
    }
    // Cupón condicional con memoria, trayectoria determinista (σ = 0, q < 0):
    // P(1) = e^{0.05} = 1.051 < 1.07 (no paga, memoriza); P(2) = e^{0.10} = 1.105 (paga 2 cupones)
    for (bool memoria : {true, false}) {
        std::string cup = std::string(R"("cupones": {"importe": "3%", "barrera": "107%", "memoria": )") +
                          (memoria ? "true" : "false") + "}, ";
        Producto p = compilar_texto(un_activo(R"({"tipo": "bs", "volatilidad": 0})", -0.05, 2.0, 0.0,
            cup + R"("amortizacion_final": [{"tipo": "fijo"}])", R"("observaciones": "anual")"));
        auto r = mc(p, 2000);
        exacto(memoria ? "Cupón condicional CON memoria (determinista)" : "Cupón condicional SIN memoria (determinista)",
               r.precio, memoria ? 1.06 : 1.03, 1e-9);
    }
    // Range accrual diario determinista: P_t = e^{0.1 t} está en [100%, 105%] mientras t ≤ ln(1.05)/0.1
    {
        Producto p = compilar_texto(un_activo(R"({"tipo": "bs", "volatilidad": 0})", -0.1, 1.0, 0.0,
            R"("cupones": {"tipo": "rango", "tasa_anual": "10%", "rango": ["100%", "105%"]},
               "amortizacion_final": [{"tipo": "fijo"}])", R"("observaciones": "anual")"));
        auto r = mc(p, 2000);
        int dentro = 0;
        for (int k = 1; k <= 252; ++k) dentro += (std::exp(0.1 * k / 252.0) <= 1.05);
        exacto("Range accrual diario (determinista)", r.precio, 1.0 + 0.10 * dentro / 252.0, 1e-9);
    }
    // Cliquet determinista: rendimientos anuales e^{0.05}-1 = 5.13% topados al 4%, 3 años
    {
        Producto p = compilar_texto(un_activo(R"({"tipo": "bs", "volatilidad": 0})", -0.05, 3.0, 0.0,
            R"("amortizacion_final": [{"tipo": "fijo"},
               {"tipo": "cliquet", "suelo_local": "0%", "techo_local": "4%", "suelo_global": "0%"}])",
            R"("observaciones": "anual")"));
        auto r = mc(p, 2000);
        exacto("Cliquet con techo local (determinista)", r.precio, 1.12, 1e-9);
    }
    // Rendimiento final medio (asiático) determinista: media de e^{0.05 t}, t = 1..4
    {
        Producto p = compilar_texto(un_activo(R"({"tipo": "bs", "volatilidad": 0})", -0.05, 4.0, 0.0,
            R"("amortizacion_final": {"rendimiento": "medio", "componentes": [{"tipo": "lineal"}]})",
            R"("observaciones": "anual")"));
        double m = 0;
        for (int t = 1; t <= 4; ++t) m += std::exp(0.05 * t) / 4.0;
        auto r = mc(p, 2000);
        exacto("Rendimiento final medio (determinista)", r.precio, m, 1e-9);
    }
}

void test_black_scholes() {
    const double r = 0.03, q = 0.02, s = 0.25, T = 1.5;
    {
        Producto p = compilar_texto(un_activo(BS25, q, T, r, R"("amortizacion_final": [{"tipo": "call", "strike": "95%"}])"));
        auto res = mc(p, 400000);
        comparar("Call europea Black-Scholes", res.precio, res.error, ref::bs_precio(true, 100, 95, T, r, q, s) / 100, 0);
    }
    {
        Producto p = compilar_texto(un_activo(BS25, q, T, r, R"("amortizacion_final": [{"tipo": "put", "strike": "110%"}])"));
        auto res = mc(p, 400000);
        comparar("Put europea Black-Scholes", res.precio, res.error, ref::bs_precio(false, 100, 110, T, r, q, s) / 100, 0);
    }
    {
        Producto p = compilar_texto(un_activo(BS25, q, T, r,
            R"("amortizacion_final": [{"tipo": "digital", "strike": "105%", "importe": "100%"}])"));
        auto res = mc(p, 400000);
        comparar("Digital Black-Scholes", res.precio, res.error, ref::bs_digital(true, 100, 105, T, r, q, s), 0);
    }
    // Reverse convertible con barrera europea del 70%: 1 - 1{P≤0.7}(1-P)
    {
        Producto p = compilar_texto(un_activo(BS25, q, T, r,
            R"("barrera_capital": {"nivel": "70%"},
               "amortizacion_final": [{"tipo": "fijo"}, {"tipo": "put_vendida", "condicion": "barrera_tocada"}])"));
        auto res = mc(p, 400000);
        double ref = std::exp(-r * T) - (ref::bs_precio(false, 100, 70, T, r, q, s) / 100 +
                                         0.30 * ref::bs_digital(false, 100, 70, T, r, q, s));
        comparar("Reverse convertible, barrera europea 70%", res.precio, res.error, ref, 0);
    }
    // Put down-and-in con barrera continua (corrección BGK) frente a Reiner-Rubinstein
    {
        Producto p = compilar_texto(un_activo(BS25, 0.01, 1.0, 0.03,
            R"("barrera_capital": {"nivel": "80%", "observacion": "continua"},
               "amortizacion_final": [{"tipo": "put", "condicion": "barrera_tocada"}])"));
        auto res = mc(p, 200000);
        comparar("Put down-and-in continua (BGK, 252 pasos/año)", res.precio, res.error,
                 ref::bs_put_down_in(100, 100, 80, 1.0, 0.03, 0.01, 0.25) / 100, 0.0006);
    }
    // Shark fin: capital + call, desactivante continua al 130% que paga 105% a vencimiento
    {
        const double sg = 0.2, rr = 0.03, qq = 0.01, H = 1.3;
        Producto p = compilar_texto(un_activo(R"({"tipo": "bs", "volatilidad": 0.2})", qq, 1.0, rr,
            R"("barrera_desactivante": {"nivel": "130%", "observacion": "continua", "importe": "105%"},
               "amortizacion_final": [{"tipo": "fijo"}, {"tipo": "call"}])"));
        auto res = mc(p, 200000);
        double nu = rr - qq - 0.5 * sg * sg, h = std::log(H);
        double p_ko = ref::norm_cdf((-h + nu) / sg) + std::exp(2 * nu * h / (sg * sg)) * ref::norm_cdf((-h - nu) / sg);
        double ref = std::exp(-rr) * (1.0 + 0.05 * p_ko) + ref::bs_call_up_out(100, 100, 130, 1.0, rr, qq, sg) / 100;
        comparar("Shark fin (KO continua, rebate a vencimiento)", res.precio, res.error, ref, 0.0008);
    }
}

void test_modelos() {
    const double r = 0.03, q = 0.01, T = 1.0;
    const std::string call = R"("amortizacion_final": [{"tipo": "call"}])";
    {
        Producto p = compilar_texto(un_activo(
            R"({"tipo": "merton", "volatilidad": 0.15, "lambda": 0.8, "media_salto": -0.12, "vol_salto": 0.2})",
            q, T, r, call));
        auto res = mc(p, 400000);
        comparar("Merton: call frente a la serie de Poisson", res.precio, res.error,
                 ref::merton_precio(true, 100, 100, T, r, q, 0.15, 0.8, -0.12, 0.2) / 100, 0);
    }
    {
        Producto p = compilar_texto(un_activo(
            R"({"tipo": "heston", "vol_inicial": 0.2, "kappa": 1.5, "vol_largo_plazo": 0.22, "xi": 0.5, "rho": -0.7})",
            q, T, r, call));
        auto res = mc(p, 200000);
        comparar("Heston: call frente a Fourier (truncamiento completo)", res.precio, res.error,
                 ref::precio_fourier(true, ref::fc_heston(0.04, 1.5, 0.0484, 0.5, -0.7, T), 100, 100, T, r, q) / 100,
                 0.0015);
    }
    {
        Producto p = compilar_texto(un_activo(
            R"({"tipo": "bates", "vol_inicial": 0.2, "kappa": 1.5, "vol_largo_plazo": 0.22, "xi": 0.5, "rho": -0.7,
                "lambda": 0.5, "media_salto": -0.1, "vol_salto": 0.15})", q, T, r, call));
        auto res = mc(p, 200000);
        comparar("Bates: call frente a Fourier", res.precio, res.error,
                 ref::precio_fourier(true, ref::fc_bates(0.04, 1.5, 0.0484, 0.5, -0.7, 0.5, -0.1, 0.15, T),
                                     100, 100, T, r, q) / 100, 0.0015);
    }
    {
        Producto p = compilar_texto(un_activo(R"({"tipo": "nig", "sigma": 0.2, "theta": -0.1, "kappa": 0.2})", q, T, r,
            R"("amortizacion_final": [{"tipo": "call", "strike": "90%"}])"));
        auto res = mc(p, 400000);
        comparar("NIG: call frente a Fourier", res.precio, res.error,
                 ref::precio_fourier(true, ref::fc_nig(0.2, -0.1, 0.2, T), 100, 90, T, r, q) / 100, 0);
    }
    {
        Producto p = compilar_texto(un_activo(R"({"tipo": "sabr", "alpha": 0.2, "beta": 0.7, "nu": 0.4, "rho": -0.5})",
            q, T, r, call));
        auto res = mc(p, 200000);
        comparar("SABR: call frente a la aproximación de Hagan", res.precio, res.error,
                 ref::sabr_precio(true, 100, 100, T, r, q, 0.2, 0.7, -0.5, 0.4) / 100, 0.002);
    }
    {
        // Vol local con β = 1: σ(t) = σ0·e^{-αt} determinista -> Black-Scholes con varianza integrada
        double s0 = 0.3, a = 0.5;
        double sef = s0 * std::sqrt((1.0 - std::exp(-2 * a * T)) / (2 * a * T));
        Producto p = compilar_texto(un_activo(R"({"tipo": "vol_local", "sigma0": 0.3, "alpha": 0.5, "beta": 1})",
            q, T, r, call));
        auto res = mc(p, 200000);
        comparar("Vol local con β=1 (vol temporal determinista)", res.precio, res.error,
                 ref::bs_precio(true, 100, 100, T, r, q, sef) / 100, 0.0005);
    }
    // Martingala: E[e^{-rT} S_T / S_0] = e^{-qT} para todos los modelos
    const char* modelos[] = {
        R"({"tipo": "bs", "volatilidad": 0.3})",
        R"({"tipo": "heston", "vol_inicial": 0.25, "kappa": 2, "vol_largo_plazo": 0.2, "xi": 0.8, "rho": -0.8})",
        R"({"tipo": "merton", "volatilidad": 0.1, "lambda": 2, "media_salto": -0.2, "vol_salto": 0.3})",
        R"({"tipo": "bates", "vol_inicial": 0.2, "kappa": 1, "vol_largo_plazo": 0.2, "xi": 0.6, "rho": -0.5, "lambda": 1, "media_salto": -0.15, "vol_salto": 0.2})",
        R"({"tipo": "vol_local", "sigma0": 0.25, "alpha": 0.3, "beta": 0.5})",
        R"({"tipo": "sabr", "alpha": 0.25, "beta": 0.5, "nu": 0.6, "rho": -0.6})",
        R"({"tipo": "nig", "sigma": 0.25, "theta": -0.2, "kappa": 0.5})"};
    for (const char* m : modelos) {
        Producto p = compilar_texto(un_activo(m, 0.02, 2.0, 0.04, R"("amortizacion_final": [{"tipo": "lineal"}])"));
        auto res = mc(p, 200000);
        comparar("Martingala (forward) " + p.activos[0].modelo.resumen().substr(0, 18), res.precio, res.error,
                 std::exp(-0.02 * 2.0), 0.0005);
    }
}

void test_cestas() {
    // Digital worst-of / best-of de dos activos Black-Scholes correlacionados
    const double r = 0.03, T = 1.0, rho = 0.4, s1 = 0.2, s2 = 0.3, q1 = 0.01, q2 = 0.02, K = 0.9;
    auto cesta = [&](const char* tipo) {
        return std::string(R"({"vencimiento": 1, "mercado": {"tipo_interes": 0.03},
            "subyacentes": [{"nombre": "A", "spot": 50, "dividendos": 0.01, "modelo": {"tipo": "bs", "volatilidad": 0.2}},
                            {"nombre": "B", "spot": 80, "dividendos": 0.02, "modelo": {"tipo": "bs", "volatilidad": 0.3}}],
            "correlacion": 0.4, "cesta": ")") + tipo +
            R"(", "amortizacion_final": [{"tipo": "digital", "strike": 0.9, "importe": 1}]})";
    };
    double d1 = (std::log(1.0 / K) + (r - q1 - 0.5 * s1 * s1) * T) / (s1 * std::sqrt(T));
    double d2 = (std::log(1.0 / K) + (r - q2 - 0.5 * s2 * s2) * T) / (s2 * std::sqrt(T));
    double df = std::exp(-r * T);
    {
        auto res = mc(compilar_texto(cesta("peor")), 400000);
        comparar("Digital worst-of (normal bivariante, ρ=0.4)", res.precio, res.error,
                 df * ref::normal_bivariante(d1, d2, rho), 0);
    }
    {
        auto res = mc(compilar_texto(cesta("mejor")), 400000);
        comparar("Digital best-of (normal bivariante, ρ=0.4)", res.precio, res.error,
                 df * (1.0 - ref::normal_bivariante(-d1, -d2, rho)), 0);
    }
    // Tres activos idénticos con correlación 1: el worst-of es el activo individual
    {
        std::string s = R"({"vencimiento": 1, "mercado": {"tipo_interes": 0.03},
            "subyacentes": [{"nombre": "A", "modelo": {"tipo": "bs", "volatilidad": 0.25}},
                            {"nombre": "B", "modelo": {"tipo": "bs", "volatilidad": 0.25}},
                            {"nombre": "C", "modelo": {"tipo": "bs", "volatilidad": 0.25}}],
            "correlacion": 1, "amortizacion_final": [{"tipo": "call"}]})";
        auto res = mc(compilar_texto(s), 200000);
        comparar("Worst-of de 3 activos idénticos con ρ=1", res.precio, res.error,
                 ref::bs_precio(true, 100, 100, 1, 0.03, 0, 0.25) / 100, 1e-9);
    }
}

void test_qmc() {
    const uint32_t* dv = nullptr;
    uint32_t nd = 0;
    if (!direcciones_sobol(&dv, &nd)) {
        std::printf("[ --  ] QMC en CPU no disponible (compilado sin cuRAND): se omite\n");
        return;
    }
    // Call BS: QMC con 16 réplicas frente a la fórmula cerrada
    Producto p = compilar_texto(un_activo(BS25, 0.02, 1.5, 0.03, R"("amortizacion_final": [{"tipo": "call", "strike": "95%"}])"));
    OpcionesValoracion o;
    o.metodo = "qmc";
    o.dispositivo = "cpu";
    o.trayectorias = (long long)(1 << 16);
    auto res = valorar(p, o);
    comparar("QMC (Sobol+HH, 16 réplicas): call Black-Scholes", res.precio, res.error,
             ref::bs_precio(true, 100, 95, 1.5, 0.03, 0.02, 0.25) / 100, 0);
    // Worst-of de dos activos con QMC (dimensión 2): mismo resultado que la normal bivariante
    std::string s = R"({"vencimiento": 1, "mercado": {"tipo_interes": 0.03},
        "subyacentes": [{"nombre": "A", "dividendos": 0.01, "modelo": {"tipo": "bs", "volatilidad": 0.2}},
                        {"nombre": "B", "dividendos": 0.02, "modelo": {"tipo": "bs", "volatilidad": 0.3}}],
        "correlacion": -0.3, "amortizacion_final": [{"tipo": "digital", "strike": 1.0, "importe": 1}]})";
    auto r2 = valorar(compilar_texto(s), o);
    double d1 = (0.03 - 0.01 - 0.02) / 0.2, d2 = (0.03 - 0.02 - 0.045) / 0.3;
    comparar("QMC: digital worst-of (ρ=-0.3)", r2.precio, r2.error,
             std::exp(-0.03) * ref::normal_bivariante(d1, d2, -0.3), 0);
}

void test_float_vs_double() {
    // La GPU simula en float por defecto: el mismo flujo Philox en float y en
    // double debe dar prácticamente el mismo precio (mismas trayectorias).
    Producto p = compilar_texto(R"({"vencimiento": "2A", "mercado": {"tipo_interes": "3%"},
        "subyacentes": [{"nombre": "A", "dividendos": "2%", "modelo": {"tipo": "heston", "vol_inicial": 0.2, "kappa": 1.5,
                          "vol_largo_plazo": 0.2, "xi": 0.6, "rho": -0.7}},
                        {"nombre": "B", "modelo": {"tipo": "merton", "volatilidad": 0.15, "lambda": 1, "media_salto": -0.1, "vol_salto": 0.1}}],
        "correlacion": 0.5, "observaciones": "trimestral",
        "cupones": {"importe": "2%", "barrera": "75%", "memoria": true},
        "autocancelacion": {"barrera": "100%", "desde": 2},
        "barrera_capital": {"nivel": "60%", "observacion": "continua"},
        "amortizacion_final": [{"tipo": "fijo"}, {"tipo": "put_vendida", "condicion": "barrera_tocada"}]})");
    auto pf = construir_plan<float>(p);
    auto pd = construir_plan<double>(p);
    const int N = (int)(20000 * g_escala) + 1000;
    double sf = 0, sd = 0, maxdif = 0;
    RegistroNulo reg;
    for (int i = 0; i < N; ++i) {
        RuidoPhilox<float> rf;
        rf.iniciar(5u, (uint64_t)i, 0u);
        RuidoPhilox<double> rd;
        rd.iniciar(5u, (uint64_t)i, 0u);
        ResTray<float> a{};
        ResTray<double> b{};
        simular_trayectoria(pf->vista, rf, reg, a);
        simular_trayectoria(pd->vista, rd, reg, b);
        sf += a.pv;
        sd += b.pv;
        maxdif = std::max(maxdif, std::fabs((double)a.pv - b.pv));
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "media float=%.6f double=%.6f |dif|=%.2e (máx. por trayectoria %.2e)", sf / N, sd / N,
                  std::fabs(sf - sd) / N, maxdif);
    resultado(std::fabs(sf - sd) / N < 2e-4, "Precisión simple (GPU) frente a doble (CPU)", buf);
}

// Solo con GPU: el motor GPU y el CPU recorren exactamente las mismas
// trayectorias (mismo Philox / mismo Sobol), así que en doble precisión deben
// coincidir salvo redondeos y en simple precisión casi.
void test_gpu_frente_a_cpu() {
    std::vector<std::pair<std::string, Producto>> casos;
    casos.push_back({"Phoenix worst-of (Heston + BS)",
                     cargar(std::string(MC_DIR_FUENTES) + "/productos/phoenix_worst_of.json")});
    casos.push_back({"Cesta de 3 (Bates, NIG, SABR) con KO y rango", compilar_texto(R"({
        "vencimiento": "2A", "mercado": {"curva": [["1A", "2%"], ["2A", "3%"]]},
        "subyacentes": [
          {"nombre": "A", "dividendos": "1%", "modelo": {"tipo": "bates", "vol_inicial": 0.2, "kappa": 2, "vol_largo_plazo": 0.2, "xi": 0.5, "rho": -0.6, "lambda": 1, "media_salto": -0.1, "vol_salto": 0.1}},
          {"nombre": "B", "modelo": {"tipo": "nig", "sigma": 0.2, "theta": -0.1, "kappa": 0.3}},
          {"nombre": "C", "spot": 50, "nivel_inicial": 52, "modelo": {"tipo": "sabr", "alpha": 0.25, "beta": 0.6, "nu": 0.5, "rho": -0.4}}],
        "correlacion": 0.4, "cesta": "media", "observaciones": "trimestral",
        "cupones": [{"tipo": "rango", "tasa_anual": "6%", "rango": ["90%", "110%"]}, {"importe": "1%", "barrera": "95%", "memoria": true}],
        "barrera_desactivante": {"nivel": "125%", "observacion": "continua", "importe": "103%"},
        "barrera_capital": {"nivel": "70%", "observacion": "diaria"},
        "amortizacion_final": [{"tipo": "fijo"}, {"tipo": "put_vendida", "condicion": "barrera_tocada"},
                               {"tipo": "cliquet", "techo_local": "3%", "suelo_global": 0}]})")});
    std::string desc;
    if (!gpu_disponible(&desc)) {
        std::printf("[ --  ] GPU no disponible (%s): se omite la comparación GPU/CPU\n", desc.c_str());
        return;
    }
    std::printf("        GPU: %s\n", desc.c_str());
    const long long N = 1 << 16;
    for (auto& c : casos) {
        Producto& p = c.second;
        auto cpu = crear_motor_cpu(p, 77u, 0);
        auto g64 = crear_motor_gpu(p, 77u, true);
        auto g32 = crear_motor_gpu(p, 77u, false);
        Acumulado a, b, f;
        a.iniciar(p.n_obs()); b.iniciar(p.n_obs()); f.iniciar(p.n_obs());
        cpu->mc(0, N, a);
        g64->mc(0, N, b);
        g32->mc(0, N, f);
        double mc = a.s_pv / N, md = b.s_pv / N, mf = f.s_pv / N;
        char buf[256];
        std::snprintf(buf, sizeof buf, "CPU=%.8f GPU(double)=%.8f GPU(float)=%.8f", mc, md, mf);
        resultado(std::fabs(mc - md) < 2e-6 && std::fabs(mc - mf) < 5e-4, "GPU = CPU (MC, mismas trayectorias): " + c.first, buf);
        double dif_ac = 0;
        for (int j = 0; j < p.n_obs(); ++j) dif_ac = std::max(dif_ac, std::fabs(a.n_ac[j] - b.n_ac[j]) / N);
        std::snprintf(buf, sizeof buf, "máx. |ΔP(autocancela)| = %.2e", dif_ac);
        resultado(dif_ac < 1e-4, "  ... contadores por fecha GPU = CPU", buf);
        if (cpu->admite_qmc()) {
            Acumulado q1, q2;
            q1.iniciar(p.n_obs()); q2.iniciar(p.n_obs());
            cpu->qmc(3, 0, 8192, q1);
            g64->qmc(3, 0, 8192, q2);
            std::snprintf(buf, sizeof buf, "CPU=%.8f GPU=%.8f", q1.s_pv / q1.n, q2.s_pv / q2.n);
            resultado(std::fabs(q1.s_pv / q1.n - q2.s_pv / q2.n) < 2e-6, "  ... QMC (réplica 3) GPU = CPU", buf);
        }
    }
    // Valoración completa adaptativa en GPU frente a la fórmula cerrada
    Producto p = compilar_texto(un_activo(BS25, 0.02, 1.5, 0.03, R"("amortizacion_final": [{"tipo": "call", "strike": "95%"}])"));
    OpcionesValoracion o;
    o.dispositivo = "gpu";
    o.metodo = "mc";
    o.error_objetivo = 0.0001;
    auto r = valorar(p, o);
    comparar("GPU MC adaptativo: call Black-Scholes", r.precio, r.error,
             ref::bs_precio(true, 100, 95, 1.5, 0.03, 0.02, 0.25) / 100, 0);
    o.metodo = "qmc";
    r = valorar(p, o);
    comparar("GPU QMC adaptativo: call Black-Scholes", r.precio, r.error,
             ref::bs_precio(true, 100, 95, 1.5, 0.03, 0.02, 0.25) / 100, 0);
}

void test_json_y_errores() {
    bool ok = false;
    resultado(std::fabs(texto_a_real("60%", &ok) - 0.6) < 1e-15 && ok, "Lectura de \"60%\"", "");
    resultado(std::fabs(texto_a_real("25pb") - 0.0025) < 1e-15, "Lectura de \"25pb\"", "");
    resultado(std::fabs(texto_a_real("0,5") - 0.5) < 1e-15, "Lectura de \"0,5\" (coma decimal)", "");
    resultado(std::fabs(texto_a_plazo("18M") - 1.5) < 1e-15, "Plazo \"18M\" = 1.5 años", "");
    resultado(std::fabs(texto_a_plazo("2A6M") - 2.5) < 1e-15, "Plazo \"2A6M\" = 2.5 años", "");
    resultado(std::fabs(texto_a_plazo("3 años") - 3.0) < 1e-15, "Plazo \"3 años\"", "");
    resultado(texto_a_frecuencia("trimestral") == 4 && texto_a_frecuencia("6M") == 2, "Frecuencias", "");
    int d1 = 0, d2 = 0;
    resultado(texto_a_fecha("2026-10-23", d1) && texto_a_fecha("23/10/2026", d2) && d1 == d2,
              "Fechas AAAA-MM-DD y DD/MM/AAAA", "");
    // Fechas absolutas con fecha de valoración: ACT/365
    {
        Producto p = compilar_texto(R"({"fecha_valoracion": "2026-01-01", "mercado": {"tipo_interes": 0.03},
            "subyacentes": [{"modelo": {"tipo": "bs", "volatilidad": 0.2}}],
            "observaciones": {"fechas": ["2026-07-02", "2027-01-01"]},
            "amortizacion_final": [{"tipo": "fijo"}]})");
        resultado(std::fabs(p.t_obs[0] - 182.0 / 365.0) < 1e-12 && std::fabs(p.T - 1.0) < 1e-12,
                  "Calendario con fechas absolutas (ACT/365)", "");
    }
    // Producto vivo descrito con su calendario completo: la fecha pasada se omite,
    // las listas pueden ser completas o solo de fechas futuras y la prima
    // acumulada cuenta las observaciones desde la emisión (2ª observación -> 2 × 5%).
    for (const char* barreras : {R"(["100%", "1%", "1%"])", R"(["1%", "1%"])"}) {
        Producto p = compilar_texto(std::string(R"({"fecha_valoracion": "2026-01-01", "mercado": {"tipo_interes": 0.02},
            "subyacentes": [{"modelo": {"tipo": "bs", "volatilidad": 0.2}}],
            "observaciones": {"fechas": ["2025-07-01", "2026-07-02", "2027-01-01"]},
            "autocancelacion": {"barrera": )") + barreras + R"(, "prima": "5%", "acumulativa": true},
            "amortizacion_final": [{"tipo": "fijo"}]})");
        auto r = mc(p, 2000);
        exacto(std::string("Producto vivo: fechas pasadas omitidas (lista de ") +
                   (std::string(barreras).size() > 14 ? "3" : "2") + " barreras)",
               r.precio, 1.10 * std::exp(-0.02 * 182.0 / 365.0), 1e-9);
    }
    // Ida y vuelta: escribir y volver a leer da el mismo documento
    {
        json::Valor d = json::leer_fichero(std::string(MC_DIR_FUENTES) + "/productos/phoenix_worst_of.json");
        std::string a = json::escribir(d);
        std::string b = json::escribir(json::parsear(a));
        resultado(a == b, "JSON: escribir -> leer -> escribir es idempotente", "");
    }
    // Comentarios y comas finales tolerados al leer
    {
        bool bien = true;
        try { json::parsear("{ // comentario\n \"a\": [1, 2, ], /* otro */ \"b\": \"x\", }"); }
        catch (...) { bien = false; }
        resultado(bien, "JSON: admite comentarios y comas finales", "");
    }
    // Errores con línea y columna
    auto espera_error = [](const std::string& nombre, const std::string& texto, const std::string& frag) {
        std::string msg;
        try { compilar_texto(texto); } catch (const std::exception& e) { msg = e.what(); }
        bool encontrado = !msg.empty() && msg.find(frag) != std::string::npos;
        resultado(encontrado, nombre, encontrado ? "" : ("mensaje: " + msg));
    };
    espera_error("Error: JSON mal formado indica la línea", "{\n \"a\": 1,\n \"b\" 2\n}", "línea 3");
    espera_error("Error: falta el tipo de interés", R"({"vencimiento": 1, "subyacentes": [{"modelo": {"tipo": "bs", "volatilidad": 0.2}}],
        "amortizacion_final": [{"tipo": "fijo"}]})", "tipo de interés");
    espera_error("Error: modelo desconocido", R"({"vencimiento": 1, "mercado": {"tipo_interes": 0}, "subyacentes": [{"modelo": {"tipo": "garch"}}],
        "amortizacion_final": [{"tipo": "fijo"}]})", "Modelos disponibles");
    espera_error("Error: parámetro de modelo ausente", R"({"vencimiento": 1, "mercado": {"tipo_interes": 0}, "subyacentes": [{"modelo": {"tipo": "heston", "kappa": 1}}],
        "amortizacion_final": [{"tipo": "fijo"}]})", "vol_inicial");
    espera_error("Error: NIG sin martingala", R"({"vencimiento": 1, "mercado": {"tipo_interes": 0}, "subyacentes": [{"modelo": {"tipo": "nig", "sigma": 1.5, "theta": 1, "kappa": 1}}],
        "amortizacion_final": [{"tipo": "fijo"}]})", "martingala");
    espera_error("Error: correlación no semidefinida positiva", R"({"vencimiento": 1, "mercado": {"tipo_interes": 0},
        "subyacentes": [{"nombre": "A", "modelo": {"tipo": "bs", "volatilidad": 0.2}}, {"nombre": "B", "modelo": {"tipo": "bs", "volatilidad": 0.2}},
                        {"nombre": "C", "modelo": {"tipo": "bs", "volatilidad": 0.2}}],
        "correlacion": {"A/B": 0.9, "A/C": 0.9, "B/C": -0.9}, "amortizacion_final": [{"tipo": "fijo"}]})", "semidefinida");
    espera_error("Error: fechas no crecientes", R"({"mercado": {"tipo_interes": 0}, "subyacentes": [{"modelo": {"tipo": "bs", "volatilidad": 0.2}}],
        "observaciones": {"fechas": ["1A", "6M"]}, "amortizacion_final": [{"tipo": "fijo"}]})", "crecientes");
    // Aviso con sugerencia ante una errata en una clave
    {
        Producto p = compilar_texto(R"({"vencimiento": 1, "mercado": {"tipo_interes": 0}, "subyacentes": [{"modelo": {"tipo": "bs", "volatilidad": 0.2}}],
            "barrera_capital": {"nivel": 0.6, "obsrevacion": "continua"}, "amortizacion_final": [{"tipo": "fijo"}]})");
        bool hay = false;
        for (const auto& a : p.avisos) hay |= (a.find("quería decir \"observacion\"") != std::string::npos);
        resultado(hay, "Aviso: sugiere la clave correcta ante una errata", "");
    }
}

// ------------------------------------------------------------------------- //
// Pagos a medida (fórmula y curva), réplica estática y precios vanilla       //
// ------------------------------------------------------------------------- //

void test_formulas_y_curvas() {
    // La misma call escrita como bloques, como curva y como fórmula: con la misma
    // semilla las trayectorias son idénticas, así que los precios coinciden exactamente
    const std::string modelo = R"J({"tipo": "heston", "vol_inicial": 0.2, "kappa": 1.5, "vol_largo_plazo": 0.22, "xi": 0.5, "rho": -0.6})J";
    auto precio = [&](const std::string& comp) {
        return mc(compilar_texto(un_activo(modelo, 0.01, 1.0, 0.03, "\"amortizacion_final\": " + comp)), 20000).precio;
    };
    const double base = precio(R"J([{"tipo": "fijo"}, {"tipo": "call", "strike": "100%", "participacion": "80%"}, {"tipo": "digital", "strike": "120%", "importe": "5%"}])J");
    exacto("Curva por puntos = bloques equivalentes",
           precio(R"J([{"tipo": "curva", "puntos": [["0%", "100%"], ["100%", "100%"], ["120%", "116%"], ["120%", "121%"], ["200%", "185%"]], "extrapolacion": "lineal"}])J"),
           base, 1e-12);
    exacto("Fórmula = bloques equivalentes",
           precio(R"J([{"tipo": "formula", "expresion": "100% + 80% * max(P - 1, 0) + 5% * (P >= 1.2)"}])J"), base, 1e-12);
    exacto("Fórmula con atajos call() y digital()",
           precio(R"J([{"tipo": "formula", "expresion": "1 + 0.8 * call(100%) + 5% * digital(120%)"}])J"), base, 1e-12);
    exacto("Fórmula con S (precio) en un solo subyacente",
           precio(R"J([{"tipo": "formula", "expresion": "1 + 0.8 * max(S - 100, 0) / 100 + 0.05 * si(S >= 120, 1, 0)"}])J"), base, 1e-12);

    // Evaluación directa: precedencias y funciones
    auto f = [](const std::string& t, double P) { return evaluar(compilar_formula(t), P); };
    exacto("Fórmula: -P^2 = -(P^2) y potencias a la derecha", f("-P^2 + 2^3^2", 3.0), -9.0 + 512.0);
    exacto("Fórmula: clamp, abs, si, porcentajes y pb", f("clamp(P, 50%, 150%) + abs(P - 2) + si(P < 1, 1, 25pb)", 2.0),
           1.5 + 0.0 + 0.0025);
    exacto("Fórmula: max y min con varios argumentos", f("max(1, P, 3) - min(P, 0.5, 2)", 2.0), 3.0 - 0.5);
    auto error_formula = [](const std::string& t, const std::string& frag) {
        std::string msg;
        try { Formula fo = compilar_formula(t); comprobar_finita(fo); } catch (const ErrorFormula& e) { msg = e.what(); }
        resultado(msg.find(frag) != std::string::npos, "Error de fórmula: " + t, msg.empty() ? "sin error" : "");
    };
    error_formula("max(P - 1, 0", "se esperaba");
    error_formula("2P", "falta un operador");
    error_formula("maxi(P, 1)", "desconocida");
    error_formula("log(P - 1)", "no da un valor finito");
    error_formula("S - 1", "un único subyacente");
    error_formula("max(P)", "al menos 2");

    // Validación de respuestas del asistente (fórmula y lista de puntos)
    {
        Pregunta qf{"f", "", "", TipoDato::Formula, "", {}, ""};
        Pregunta qp{"p", "", "", TipoDato::Puntos, "", {}, ""};
        std::string can, err;
        bool a = normalizar_respuesta(qf, "100% + 2*call(110%) - 2*call(130%)", can, err);
        bool b = !normalizar_respuesta(qf, "max(P 1)", can, err) && !normalizar_respuesta(qf, "log(P-1)", can, err);
        bool c = normalizar_respuesta(qp, "0:90 90:90 100:100 130:130 130:135 200:135", can, err) &&
                 can == "0%:90% 90%:90% 100%:100% 130%:130% 130%:135% 200%:135%";
        bool d = !normalizar_respuesta(qp, "100:100 90:90", can, err) && !normalizar_respuesta(qp, "100-100", can, err);
        resultado(a && b && c && d, "Asistente: valida fórmulas y listas de puntos", "");
        // Plantillas nuevas: generan productos válidos
        bool ok = true;
        for (const char* id : {"pago_a_medida", "pago_por_puntos"}) {
            const Plantilla* pl = buscar_plantilla(id);
            try { compilar(construir_producto(*pl, comunes_por_defecto(*pl), {})); } catch (...) { ok = false; }
        }
        resultado(ok, "Plantillas pago_a_medida y pago_por_puntos", "");
    }
}

void test_replica_estatica() {
    // Réplica exacta de una call: su valor es Black-Scholes
    {
        Producto p = compilar_texto(un_activo(BS25, 0.02, 1.5, 0.03, R"J("nominal": 100, "amortizacion_final": [{"tipo": "call", "strike": "110%"}])J"));
        Replica r = replicar(p, OpcionesReplica{});
        exacto("Réplica de una call = Black-Scholes", r.valor, ref::bs_precio(true, 100, 110, 1.5, 0.03, 0.02, 0.25), 1e-9);
        resultado(r.exacta && r.lineas.size() == 1 && r.lineas[0].instrumento == "Call", "Réplica de una call: una sola call", "");
    }
    // Reverse convertible con barrera a vencimiento y digital: réplica exacta frente a la fórmula cerrada
    {
        const double S = 100, r_ = 0.03, q = 0.01, T = 1.0, s = 0.25, B = 0.7;
        Producto p = compilar_texto(un_activo(BS25, q, T, r_, R"J("nominal": 100, "barrera_capital": {"nivel": "70%", "observacion": "vencimiento"},
            "amortizacion_final": [{"tipo": "fijo"}, {"tipo": "put_vendida", "strike": "100%", "condicion": "barrera_tocada"},
                                   {"tipo": "digital", "strike": "120%", "importe": "5%"}])J"));
        Replica rp = replicar(p, OpcionesReplica{});
        // A = 1 - (1-P)·1{P≤B} + 5%·1{P≥1.2}: E[(1-P)1{P≤B}] = Put(K=B) + (1-B)·Digital_put(B)
        double ref = std::exp(-r_ * T) - (ref::bs_precio(false, S, B * S, T, r_, q, s) / S +
                                         (1 - B) * ref::bs_digital(false, S, B * S, T, r_, q, s)) +
                     0.05 * ref::bs_digital(true, S, 120, T, r_, q, s);
        exacto("Réplica: reverse convertible con barrera europea", rp.valor / 100.0, ref, 1e-9);
        resultado(rp.exacta && rp.error_max < 1e-12, "Réplica exacta: error del pago nulo", "");
        auto m = mc(p, 400000);
        comparar("Réplica frente a simulación (BS)", m.precio, m.error, rp.valor / 100.0, 0.0);
    }
    // Heston con cupones fijos y condicionales sin memoria: digitales en cada fecha (Fourier)
    {
        Producto p = compilar_texto(un_activo(R"J({"tipo": "heston", "vol_inicial": 0.2, "kappa": 2, "vol_largo_plazo": 0.22, "xi": 0.6, "rho": -0.7})J",
            0.025, 1.0, 0.03, R"J("nominal": 100, "observaciones": {"frecuencia": "trimestral"},
            "cupones": [{"tipo": "fijo", "tasa_anual": "4%"}, {"tipo": "condicional", "importe": "1%", "barrera": "90%"}],
            "barrera_capital": {"nivel": "70%", "observacion": "vencimiento"},
            "amortizacion_final": [{"tipo": "fijo"}, {"tipo": "put_vendida", "strike": "100%", "condicion": "barrera_tocada"}],
            "simulacion": {"pasos_por_anio": 250})J"));
        Replica rp = replicar(p, OpcionesReplica{});
        auto m = mc(p, 400000);
        comparar("Réplica (Fourier) frente a simulación (Heston, cupones)", m.precio, m.error, rp.valor / 100.0, 0.0006);
    }
    // Fórmula suave: la rejilla converge al valor exacto de E[(P-1)²]
    {
        const double r_ = 0.03, q = 0.015, s = 0.2, T = 1.0;
        Producto p = compilar_texto(un_activo(R"J({"tipo": "black_scholes", "volatilidad": 0.2})J", q, T, r_,
            R"J("nominal": 1, "amortizacion_final": [{"tipo": "formula", "expresion": "100% + 50% * (P - 1)^2"}])J"));
        double EP = std::exp((r_ - q) * T), EP2 = std::exp(2 * (r_ - q) * T + s * s * T);
        double ref = std::exp(-r_ * T) * (1 + 0.5 * (EP2 - 2 * EP + 1));
        OpcionesReplica o;
        o.paso = 0.01;
        o.p_min = 0.01;
        o.p_max = 4.0;
        Replica rp = replicar(p, o);
        exacto("Réplica en rejilla de 1% de un pago cuadrático", rp.valor, ref, 2e-5);
    }
    // No replicables: autocancelable, barrera continua, memoria
    {
        auto motivo = [](const std::string& texto) {
            std::string m;
            es_replicable(compilar_texto(texto), &m);
            return m;
        };
        std::string a = motivo(un_activo(BS25, 0, 1, 0.02, R"J("observaciones": {"frecuencia": "semestral"}, "autocancelacion": {"barrera": "100%"}, "amortizacion_final": [{"tipo": "fijo"}])J"));
        std::string b = motivo(un_activo(BS25, 0, 1, 0.02, R"J("barrera_capital": {"nivel": "60%", "observacion": "continua"}, "amortizacion_final": [{"tipo": "fijo"}, {"tipo": "put_vendida", "condicion": "barrera_tocada"}])J"));
        resultado(a.find("autocancelable") != std::string::npos && b.find("continua") != std::string::npos,
                  "Réplica: detecta los productos que dependen de la trayectoria", "");
    }
}

void test_vanilla_y_vol_implicita() {
    Activo a;
    a.nombre = "X";
    a.spot = a.nivel_inicial = 100;
    a.dividendos = 0.01;
    Curva c;
    c.t = {1.0};
    c.r = {0.03};
    // Fourier con control Black-Scholes frente a la integral de referencia
    a.modelo = compilar_texto(un_activo(R"J({"tipo": "heston", "vol_inicial": 0.2, "kappa": 1.5, "vol_largo_plazo": 0.25, "xi": 0.7, "rho": -0.7})J",
                                        0.01, 1, 0.03, R"J("amortizacion_final": [{"tipo": "fijo"}])J")).activos[0].modelo;
    {
        PreciosVanilla pv(a, c);
        double peor = 0, peor_put = 0, peor_dig = 0;
        for (double T : {0.1, 1.0, 3.0})
            for (double K : {60.0, 90.0, 100.0, 115.0, 160.0}) {
                double v0 = 0.04, th = 0.0625;
                auto phi = ref::fc_heston(v0, 1.5, th, 0.7, -0.7, T);
                double cr = ref::precio_fourier(true, phi, 100, K, T, 0.03, 0.01);
                double pr = ref::precio_fourier(false, phi, 100, K, T, 0.03, 0.01);
                peor = std::max(peor, std::fabs(pv.call(T, K) - cr));
                peor_put = std::max(peor_put, std::fabs(pv.put(T, K) - pr));
                double h = 1e-4 * K;
                double dig_fd = (ref::precio_fourier(true, phi, 100, K - h, T, 0.03, 0.01) -
                                 ref::precio_fourier(true, phi, 100, K + h, T, 0.03, 0.01)) / (2 * h);
                peor_dig = std::max(peor_dig, std::fabs(pv.digital(T, K) - dig_fd));
            }
        char buf[160];
        std::snprintf(buf, sizeof buf, "máx |dif| call %.2e, put %.2e, digital %.2e", peor, peor_put, peor_dig);
        resultado(peor < 1e-7 && peor_put < 1e-7 && peor_dig < 1e-7, "Fourier (Lewis + control BS): Heston", buf);
    }
    // Monte Carlo vanilla (forzado) frente a Black-Scholes
    {
        a.modelo = compilar_texto(un_activo(BS25, 0.01, 1, 0.03, R"J("amortizacion_final": [{"tipo": "fijo"}])J")).activos[0].modelo;
        OpcionesMCVanilla om;
        om.trayectorias = (long long)std::max(20000.0, 400000 * g_escala);
        PreciosVanilla pv(a, c, om, true);
        pv.preparar({0.5, 2.0});
        double peor = 0;
        for (double T : {0.5, 2.0})
            for (double K : {80.0, 100.0, 125.0}) {
                double e = std::fabs(pv.call(T, K) - ref::bs_precio(true, 100, K, T, 0.03, 0.01, 0.25)) / (0.5 + K * 0.0);
                peor = std::max(peor, e);
                double paridad = pv.call(T, K) - pv.put(T, K) - (100 * std::exp(-0.01 * T) - K * std::exp(-0.03 * T));
                peor = std::max(peor, std::fabs(paridad) * 1e3);
            }
        resultado(peor < 0.3 * std::sqrt(0.1 / g_escala) + 1e-9, "Vanilla por Monte Carlo: Black-Scholes y paridad", "");
    }
    // Volatilidad implícita: ida y vuelta en todo el plano strike/plazo, calls y puts
    {
        double peor = 0;
        int fallos = 0;
        for (double T : {0.02, 0.25, 1.0, 5.0})
            for (double K : {30.0, 70.0, 95.0, 100.0, 105.0, 140.0, 300.0})
                for (double s : {0.05, 0.2, 0.8})
                    for (bool call : {true, false}) {
                        double pr = ref::bs_precio(call, 100, K, T, 0.03, 0.01, s);
                        // El valor temporal (la opción fuera del dinero equivalente) debe ser
                        // apreciable frente al precio: si no, el precio no contiene la volatilidad
                        double F = 100 * std::exp(0.02 * T);
                        double otm = ref::bs_precio(K > F, 100, K, T, 0.03, 0.01, s);
                        if (otm < 1e-10 * std::max(pr, 1.0)) continue;
                        bool ok = false;
                        double v = vol_implicita(pr, call, 100, K, T, 0.03, 0.01, &ok);
                        if (!ok) { ++fallos; continue; }
                        peor = std::max(peor, std::fabs(v - s));
                    }
        char buf[120];
        std::snprintf(buf, sizeof buf, "máx |error| %.2e, fallos %d", peor, fallos);
        resultado(peor < 1e-6 && fallos == 0, "Volatilidad implícita: ida y vuelta", buf);
        bool ok = true;
        vol_implicita(0.0, true, 100, 120, 1, 0.03, 0, &ok);
        bool ok2 = true;
        vol_implicita(200.0, true, 100, 120, 1, 0.03, 0, &ok2);
        resultado(!ok && !ok2, "Volatilidad implícita: rechaza precios imposibles", "");
    }
}

void test_cadenas_csv() {
    // Formato de Yahoo Finance (tipo deducido del símbolo, fechas) y formato español
    // (punto y coma, coma decimal, plazos y volatilidad en puntos porcentuales)
    {
        std::ofstream f("prueba_cadena_yf.csv");
        f << "# fecha: 2026-01-01\n# spot: 5600\n"
             "contractSymbol,strike,lastPrice,bid,ask,volume,openInterest,impliedVolatility,expiration\n"
             "SPXW270101P05000000,5000,100.5,100,101,12,500,0.25,2027-01-01\n"
             "SPXW270101C06000000,6000,80.0,79,81,,300,0.18,2027-01-01\n"
             "SPXW270101C06500000,6500,30.0,29.5,30.5,25,200,0.17,2027-01-01\n";
    }
    {
        std::ofstream f("prueba_cadena_es.csv");
        f << "# subyacente: SX5E\nvencimiento;tipo;strike;precio;volumen;vol_implicita\n"
             "6M;put;4500;95,25;10;22,5\n6M;call;5000;80,5;0;18\n1A;call;5200;120;7;19,1%\n";
    }
    CadenaOpciones yf = leer_cadena_csv("prueba_cadena_yf.csv");
    CadenaOpciones es = leer_cadena_csv("prueba_cadena_es.csv");
    bool ok_yf = yf.cot.size() == 3 && !yf.cot[0].call && yf.cot[1].call && std::fabs(yf.cot[0].T - 1.0) < 1e-12 &&
                 std::fabs(yf.cot[0].referencia() - 100.5) < 1e-12 && yf.cot[1].volumen < 0 && yf.spot == 5600;
    bool ok_es = es.cot.size() == 3 && std::fabs(es.cot[0].T - 0.5) < 1e-12 && std::fabs(es.cot[0].precio - 95.25) < 1e-12 &&
                 std::fabs(es.cot[0].vol - 0.225) < 1e-12 && std::fabs(es.cot[2].vol - 0.191) < 1e-12;
    resultado(ok_yf, "Cadena CSV: formato de Yahoo Finance", "");
    resultado(ok_es, "Cadena CSV: punto y coma, coma decimal y plazos", "");
    auto st = strikes_fuera_del_dinero(yf, 1.0, 5600, 10, 0, 1e9);
    resultado(st.size() == 2 && st[0].K == 5000 && st[1].K == 6500, "Cadena: filtro fuera del dinero y volumen mínimo", "");
    std::remove("prueba_cadena_yf.csv");
    std::remove("prueba_cadena_es.csv");

    // Réplica con strikes cotizados: exacta en cada strike
    {
        std::ofstream f("prueba_cadena_k.csv");
        f << "vencimiento,tipo,strike,precio\n";
        for (int K = 60; K <= 160; K += 10) f << "1A," << (K < 100 ? "put" : "call") << "," << K << ",1\n";
    }
    CadenaOpciones ck = leer_cadena_csv("prueba_cadena_k.csv");
    std::remove("prueba_cadena_k.csv");
    Producto p = compilar_texto(un_activo(BS25, 0, 1, 0.02, R"J("amortizacion_final": [{"tipo": "formula", "expresion": "1 + 0.5 * (P - 1)^2"}])J"));
    OpcionesReplica o;
    o.cadena = &ck;
    Replica r = replicar(p, o);
    // Pago de la cartera reconstruido a partir de sus líneas (pesos en P)
    auto cartera = [&](double P) {
        double v = 0;
        for (const auto& l : r.lineas) {
            if (l.instrumento == "Bono cupón cero") v += l.peso;
            else if (l.instrumento.rfind("Subyacente", 0) == 0) v += l.peso * P;
            else if (l.instrumento == "Put") v += l.peso * std::max(l.k - P, 0.0);
            else if (l.instrumento == "Call") v += l.peso * std::max(P - l.k, 0.0);
        }
        return v;
    };
    double peor = 0;
    for (int K = 60; K <= 160; K += 10) {
        if (K == 100) continue; // en el dinero: no se usa (solo opciones fuera del dinero)
        double P = K / 100.0;
        peor = std::max(peor, std::fabs(cartera(P) - (1 + 0.5 * (P - 1) * (P - 1))));
    }
    char buf[80];
    std::snprintf(buf, sizeof buf, "máx |dif| = %.2e", peor);
    resultado(peor < 1e-12 && r.lineas.size() == 11, "Réplica con strikes cotizados: exacta en los strikes", buf);
}

// ------------------------------------------------------------------------- //
// Calibración                                                                //
// ------------------------------------------------------------------------- //

// Superficie de volatilidades generada con un modelo conocido (precios exactos)
DatosMercado superficie_sintetica(const Modelo& mod, double ruido_pb = 0.0) {
    DatosMercado m;
    m.subyacente = "SINT";
    m.fuente = "superficie sintética";
    m.spot = 100;
    m.r = 0.03;
    m.q = 0.01;
    Activo a;
    a.spot = a.nivel_inicial = 100;
    a.dividendos = m.q;
    a.modelo = mod;
    Curva c;
    c.t = {1.0};
    c.r = {m.r};
    OpcionesMCVanilla omc;
    omc.trayectorias = 400000;
    PreciosVanilla pv(a, c, omc);
    uint64_t estado = 12345;
    auto normal = [&]() { // Box-Muller con un generador lineal congruente (reproducible)
        auto u = [&]() { estado = estado * 6364136223846793005ULL + 1442695040888963407ULL; return ((estado >> 11) + 0.5) / 9007199254740992.0; };
        double u1 = u(), u2 = u();
        return std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    };
    for (double T : {0.25, 0.5, 1.0, 2.0}) {
        VencimientoMercado v;
        v.T = T;
        v.etiqueta = num(T);
        v.DF = std::exp(-m.r * T);
        v.F = 100 * std::exp((m.r - m.q) * T);
        for (double k : {0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35}) {
            PuntoMercado p;
            p.v = (int)m.vencimientos.size();
            p.T = T;
            p.K = 100 * k;
            p.call = p.K >= v.F;
            double pr = p.call ? pv.call_fwd(T, p.K, v.F, v.DF) : pv.put_fwd(T, p.K, v.F, v.DF);
            p.vol = vol_implicita_fwd(pr, p.call, v.F, p.K, T, v.DF) + ruido_pb * 1e-4 * normal();
            p.precio = black(p.call, v.F, p.K, T, v.DF, p.vol);
            p.vega = vega_black(v.F, p.K, T, v.DF, p.vol);
            v.puntos.push_back((int)m.puntos.size());
            m.puntos.push_back(p);
        }
        m.vencimientos.push_back(v);
    }
    return m;
}

Modelo modelo_con(const std::string& tipo, std::initializer_list<std::pair<const char*, double>> ps) {
    Modelo mod = modelo_por_defecto(buscar_modelo(tipo)->tipo);
    for (auto& [k, v] : ps) mod[k] = v;
    return mod;
}

void test_optimizadores() {
    // Rosenbrock: r = (10·(y - x²), 1 - x), mínimo en (1, 1)
    opt::FuncResiduos ros = [](const std::vector<double>& x, std::vector<double>& r) {
        r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]};
        return true;
    };
    opt::Resultado a = opt::levenberg_marquardt(ros, {-1.2, 1.0});
    resultado(std::fabs(a.x[0] - 1) < 1e-8 && std::fabs(a.x[1] - 1) < 1e-8, "Levenberg-Marquardt: Rosenbrock", a.motivo);
    opt::FuncObjetivo fr = [](const std::vector<double>& x) {
        return 100 * std::pow(x[1] - x[0] * x[0], 2) + std::pow(1 - x[0], 2);
    };
    opt::Resultado b = opt::nelder_mead(fr, {-1.2, 1.0});
    resultado(std::fabs(b.x[0] - 1) < 1e-4 && std::fabs(b.x[1] - 1) < 1e-4, "Nelder-Mead: Rosenbrock", b.motivo);
    // Rastrigin 2D (muchos mínimos locales): la evolución diferencial encuentra el global
    opt::FuncObjetivo ras = [](const std::vector<double>& x) {
        double s = 20;
        for (double v : x) s += v * v - 10 * std::cos(6.283185307179586 * v);
        return s;
    };
    opt::OpcionesDE ode;
    ode.generaciones = 300;
    opt::Resultado c = opt::evolucion_diferencial(ras, {-5.12, -5.12}, {5.12, 5.12}, ode);
    resultado(c.f < 1e-6, "Evolución diferencial: Rastrigin (mínimo global)", "f = " + num(c.f));
}

void test_calibracion() {
    struct Caso { const char* nombre; Modelo mod; double tol; };
    std::vector<Caso> casos = {
        {"black_scholes", modelo_con("black_scholes", {{"volatilidad", 0.27}}), 1e-6},
        {"heston", modelo_con("heston", {{"vol_inicial", 0.18}, {"kappa", 2.5}, {"vol_largo_plazo", 0.22}, {"xi", 0.7}, {"rho", -0.75}}), 1e-3},
        {"merton", modelo_con("merton", {{"volatilidad", 0.15}, {"lambda", 0.4}, {"media_salto", -0.12}, {"vol_salto", 0.18}}), 1e-3},
        {"nig", modelo_con("nig", {{"sigma", 0.2}, {"theta", -0.15}, {"kappa", 0.4}}), 1e-3},
        {"sabr", modelo_con("sabr", {{"alpha", 0.21}, {"beta", 0.7}, {"nu", 0.6}, {"rho", -0.45}}), 1e-4},
    };
    for (const auto& c : casos) {
        DatosMercado m = superficie_sintetica(c.mod);
        OpcionesCalibracion o;
        o.modelo = c.nombre;
        ResultadoCalibracion r = calibrar(m, o);
        double peor = 0;
        const auto& d = def_modelo(c.mod.tipo);
        for (size_t i = 0; i < d.params.size(); ++i)
            peor = std::max(peor, std::fabs(r.modelo.valores[i] - c.mod.valores[i]) / std::max(0.1, std::fabs(c.mod.valores[i])));
        char buf[120];
        std::snprintf(buf, sizeof buf, "RMSE %.1e pb, máx. error relativo de parámetros %.1e", 1e4 * r.rmse, peor);
        resultado(peor < c.tol && r.rmse < 1e-5, std::string("Calibración: recupera ") + d.nombre, buf);
    }
    // Con ruido de 20 pb: los parámetros quedan dentro de 4 errores estándar
    {
        Modelo verdad = casos[1].mod;
        DatosMercado m = superficie_sintetica(verdad, 20.0);
        OpcionesCalibracion o;
        o.modelo = "heston";
        ResultadoCalibracion r = calibrar(m, o);
        bool ok = r.rmse > 1e-4 && r.rmse < 4e-3;
        std::string det;
        for (const auto& p : r.params) {
            if (!p.libre) continue;
            double z = (p.valor - verdad[p.clave]) / p.error;
            ok &= p.error > 0 && std::fabs(z) < 4;
            det += p.clave + " " + num(std::round(z * 10) / 10) + "σ  ";
        }
        resultado(ok, "Calibración con ruido: Heston dentro de su error estándar", det);
    }
    // SVI: recupera una superficie SVI exacta
    {
        DatosMercado m = superficie_sintetica(casos[1].mod);
        const double sp[5] = {0.02, 0.15, -0.5, 0.05, 0.2};
        for (auto& v : m.vencimientos)
            for (int i : v.puntos) {
                PuntoMercado& p = m.puntos[(size_t)i];
                double k = std::log(p.K / v.F), d = k - sp[3];
                double w = sp[0] * v.T + sp[1] * v.T * (sp[2] * d + std::sqrt(d * d + sp[4] * sp[4]));
                p.vol = std::sqrt(w / v.T);
                p.precio = black(p.call, v.F, p.K, v.T, v.DF, p.vol);
                p.vega = vega_black(v.F, p.K, v.T, v.DF, p.vol);
            }
        OpcionesCalibracion o;
        o.modelo = "svi";
        ResultadoCalibracion r = calibrar(m, o);
        bool ok = r.svi.size() == 4 && r.rmse < 1e-6;
        for (size_t v = 0; v < r.svi.size(); ++v) {
            double T = m.vencimientos[v].T;
            ok &= std::fabs(r.svi[v].rho - sp[2]) < 1e-3 && std::fabs(r.svi[v].m - sp[3]) < 1e-3 &&
                  std::fabs(r.svi[v].b - sp[1] * T) < 1e-3 && r.svi[v].g_min > 0;
        }
        resultado(ok, "SVI: recupera los parámetros y no detecta arbitraje", "RMSE " + num(r.rmse));
    }
    // Volatilidad local por Monte Carlo con números comunes (solo en el modo completo)
    if (g_escala >= 1.0) {
        Modelo verdad = modelo_con("vol_local", {{"sigma0", 0.2}, {"alpha", 0.3}, {"beta", 0.4}});
        DatosMercado m = superficie_sintetica(verdad);
        OpcionesCalibracion o;
        o.modelo = "vol_local";
        o.trayectorias_mc = 40000;
        ResultadoCalibracion r = calibrar(m, o);
        resultado(std::fabs(r.modelo["sigma0"] - 0.2) < 0.01 && std::fabs(r.modelo["beta"] - 0.4) < 0.1,
                  "Calibración por Monte Carlo: volatilidad local", "sigma0 " + num(r.modelo["sigma0"]) + ", beta " + num(r.modelo["beta"]));
    }
}

void test_mercado_y_paridad() {
    // Cadena con calls y puts exactos de Black-Scholes con r = 4%, q = 1%: la paridad
    // devuelve exactamente el forward y el descuento
    {
        std::ofstream f("prueba_paridad.csv");
        f << "# spot: 100\nvencimiento,tipo,strike,precio\n";
        for (double T : {0.5, 1.0})
            for (int K = 80; K <= 120; K += 5)
                for (bool call : {true, false})
                {
                    char buf[64];
                    std::snprintf(buf, sizeof buf, "%.17g", ref::bs_precio(call, 100, K, T, 0.04, 0.01, 0.2 + 0.001 * (100 - K)));
                    f << (T == 0.5 ? "6M" : "1A") << "," << (call ? "call" : "put") << "," << K << "," << buf << "\n";
                }
    }
    DatosMercado m = cargar_mercado("prueba_paridad.csv");
    std::remove("prueba_paridad.csv");
    bool ok = m.vencimientos.size() == 2;
    for (const auto& v : m.vencimientos) {
        ok &= v.forward_implicito && std::fabs(v.DF - std::exp(-0.04 * v.T)) < 1e-9 &&
              std::fabs(v.F - 100 * std::exp(0.03 * v.T)) < 1e-7;
    }
    double peor = 0;
    for (const auto& p : m.puntos) peor = std::max(peor, std::fabs(p.vol - (0.2 + 0.001 * (100 - p.K))));
    resultado(ok && peor < 1e-9, "Mercado: forward y descuento implícitos por paridad put-call", "");
    resultado(std::all_of(m.puntos.begin(), m.puntos.end(), [&](const PuntoMercado& p) {
                  return p.call == (p.K >= m.vencimientos[(size_t)p.v].F); }),
              "Mercado: se usa la opción fuera del dinero de cada strike", "");
    // Fichero JSON con una superficie (strikes relativos y un hueco)
    {
        std::ofstream f("prueba_superficie.json");
        f << R"J({"spot": 4800, "tipo_interes": "3%", "dividendos": "2%",
                 "superficie": {"vencimientos": ["6M", "1A"], "strikes": ["90%", "100%", "110%"],
                                "volatilidades": [["22%", "20%", null], ["21.5%", "20%", "19%"]]}})J";
    }
    DatosMercado s = cargar_mercado("prueba_superficie.json");
    std::remove("prueba_superficie.json");
    resultado(s.puntos.size() == 5 && std::fabs(s.puntos[0].K - 4320) < 1e-9 && std::fabs(s.puntos[0].vol - 0.22) < 1e-12 &&
              std::fabs(s.vencimientos[1].F - 4800 * std::exp(0.01)) < 1e-9,
              "Mercado: superficie JSON con strikes en % y huecos", "");
}

void test_historico() {
    // Serie diaria simulada (Philox): GBM y Merton con saltos frecuentes
    auto serie = [](double s, double lam, double mj, double sj, int n, uint32_t sem) {
        SerieHistorica h;
        h.ruta = "simulada";
        double p = 100, dt = 1.0 / 252, kap = std::exp(mj + 0.5 * sj * sj) - 1;
        RuidoPhilox<double> rz;
        h.precios.push_back(p);
        h.dias.push_back(-1);
        for (int i = 0; i < n; ++i) {
            rz.iniciar(sem, (uint64_t)i, 3u);
            double z = rz.normal(0), u = hd::norm_cdf(rz.normal(1)), zj = rz.normal(2);
            int k = poisson_inv(lam * dt, u);
            double r = (0.05 - 0.5 * s * s - lam * kap) * dt + s * std::sqrt(dt) * z + k * mj + std::sqrt((double)k) * sj * zj;
            p *= std::exp(r);
            h.precios.push_back(p);
            h.dias.push_back(-1);
        }
        return h;
    };
    {
        ResultadoHistorico r = estimar_historico(serie(0.2, 0, 0, 0, 5000, 7), "auto");
        double z = (r.bs[1].valor - 0.2) / r.bs[1].error;
        resultado(std::fabs(z) < 4 && r.modelo.tipo == TipoModelo::BlackScholes && std::fabs(r.hurst_rend - 0.5) < 0.08,
                  "Histórico: GBM (volatilidad, sin saltos, H ≈ 0.5)",
                  "vol " + num(r.bs[1].valor) + ", p " + num(r.p_valor) + ", H " + num(r.hurst_rend));
    }
    {
        ResultadoHistorico r = estimar_historico(serie(0.12, 25, -0.03, 0.01, 5000, 9), "auto");
        bool ok = r.modelo.tipo == TipoModelo::Merton && r.p_valor < 1e-6 && std::fabs(r.merton[1].valor - 0.12) < 0.02 &&
                  std::fabs(r.merton[2].valor - 25) < 8 && std::fabs(r.merton[3].valor + 0.03) < 0.008;
        resultado(ok, "Histórico: Merton (detecta y estima los saltos)",
                  "σ " + num(std::round(r.merton[1].valor * 1e4) / 1e4) + ", λ " + num(std::round(r.merton[2].valor * 100) / 100) +
                      ", μJ " + num(std::round(r.merton[3].valor * 1e4) / 1e4));
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "rapido") == 0) g_escala = 0.1;
    std::printf("== Tests del motor de productos estructurados (CPU) ==\n");
    test_json_y_errores();
    test_formulas_y_curvas();
    test_replica_estatica();
    test_vanilla_y_vol_implicita();
    test_cadenas_csv();
    test_optimizadores();
    test_mercado_y_paridad();
    test_calibracion();
    test_historico();
    test_deterministas();
    test_black_scholes();
    test_modelos();
    test_cestas();
    test_qmc();
    test_float_vs_double();
    test_gpu_frente_a_cpu();
    std::printf("\n%d correctos, %d fallos\n", g_ok, g_fallos);
    return g_fallos == 0 ? 0 : 1;
}
