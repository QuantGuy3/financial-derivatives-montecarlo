// Tests del motor de productos estructurados (CPU, sin GPU).
//
// Cada test valora un producto con el motor CPU (Monte Carlo con Philox) y lo
// compara con un precio de referencia analítico o semianalítico
// (referencias.hpp). Criterio: |MC - referencia| <= 4·error_estándar + tolerancia
// (la tolerancia cubre el sesgo de discretización temporal cuando lo hay).
// Los casos deterministas (volatilidad 0, autocancelación segura...) se exigen
// exactos. Uso: ./test_estructurados [rapido]

#include "estructurados/plan.hpp"
#include "estructurados/producto.hpp"
#include "estructurados/valoracion.hpp"
#include "referencias.hpp"

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

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "rapido") == 0) g_escala = 0.1;
    std::printf("== Tests del motor de productos estructurados (CPU) ==\n");
    test_json_y_errores();
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
