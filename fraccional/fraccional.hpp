#pragma once
// ---------------------------------------------------------------------------- //
// Modelos fraccionarios y sus métodos numéricos (namespace frac).              //
//                                                                              //
// Rugosidad en la volatilidad (núcleo K(t) = t^{H-1/2}/Γ(H+1/2), H < 1/2):     //
//   rough Heston   función característica por la ecuación de Riccati          //
//                  fraccionaria (regla trapezoidal fraccionaria implícita,     //
//                  Adams-Moulton de producto) y simulación con varianza        //
//                  forward y QE sobre la aproximación markoviana del núcleo    //
//   rough Bergomi  esquema híbrido de Bennedsen-Lunde-Pakkanen (convolución    //
//                  por FFT) y esquema híbrido-markoviano (motor)               //
// Saltos estables (derivada fraccionaria en espacio):                          //
//   FMLS           (Carr-Wu 2003) función característica, EDP fraccionaria    //
//                  en espacio (Grünwald desplazado y ponderado + Crank-        //
//                  Nicolson) y muestreo exacto de Chambers-Mallows-Stuck       //
// Ruido fraccionario en el precio:                                             //
//   fBS / mixto    Black-Scholes fraccionario (Necula 2002) y mixto            //
//                  (Cheridito 2001): fórmulas cerradas y simulación del fBm    //
//                  (Davies-Harte por FFT o Cholesky)                           //
// Derivada fraccionaria en tiempo:                                             //
//   subdifusivo    (Magdziarz 2009) Black-Scholes en tiempo operacional        //
//                  E_t (inversa del subordinador α-estable): cuadratura de la  //
//                  representación de Kanter, EDP de Caputo con el esquema L1   //
//                  en malla graduada y Monte Carlo exacto                      //
// ---------------------------------------------------------------------------- //

#include <complex>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace frac {

using cd = std::complex<double>;

// ============================================================ //
// Funciones especiales, cuadratura y núcleo fraccionario        //
// ============================================================ //

// Mittag-Leffler de un parámetro en el semieje negativo: E_α(-x), x ≥ 0, 0 < α ≤ 1
// (serie para x pequeño; representación integral de Gorenflo-Mainardi si no)
double mittag_leffler(double alpha, double x);

// Integral de f en [a, b] con Gauss-Kronrod 7-15 adaptativo (b puede ser +∞)
double integrar(const std::function<double(double)>& f, double a, double b, double tol = 1e-11);

// Núcleo fraccionario K(t) = t^{α-1}/Γ(α), α = H + 1/2
double nucleo(double H, double t);

// Aproximación markoviana (suma de exponenciales) del núcleo:
//   K(t) ≈ Σ_j c_j·e^{-x_j t}  en  [dt_min, T]
// a partir de K(t) = ∫ e^{-xt} μ(dx), μ(dx) = x^{-α} dx / (Γ(α)Γ(1-α)): un nodo
// de masa para [0, x_min] y Gauss-Legendre de 2 puntos en paneles de ln x hasta
// x_max = 30/dt_min. "factores" = número total de nodos (impar: 1 + 2·paneles).
// Con H = 1/2 el núcleo es 1: un único nodo x = 0, c = 1.
struct NodosNucleo {
    std::vector<double> x, c;
};
NodosNucleo nodos_nucleo(double H, int factores, double dt_min, double T);
// Error relativo máximo de la aproximación en [dt_min, T]
double error_nodos(const NodosNucleo& n, double H, double dt_min, double T);

// ============================================================ //
// FFT                                                           //
// ============================================================ //

// FFT compleja in situ (tamaño potencia de 2); inversa = true divide por n
void fft(std::vector<cd>& a, bool inversa);

// ============================================================ //
// Movimiento browniano fraccionario                             //
// ============================================================ //

// Autocovarianza del ruido gaussiano fraccionario de incrementos unitarios:
// γ(k) = ½(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})
double autocov_fgn(double H, long k);

// Trayectorias de B^H en t_k = k·T/n (k = 0..n, B^H_0 = 0).
//  DaviesHarte: inmersión circulante de la covarianza del ruido fraccionario
//               (exacta, O(n log n) por trayectoria; Davies-Harte 1987, Wood-Chan 1994)
//  Cholesky:    factorización de la covarianza (exacta, O(n²) por trayectoria)
class GeneradorFBM {
public:
    enum class Metodo { DaviesHarte, Cholesky };
    GeneradorFBM(double H, int n, double T, Metodo m = Metodo::DaviesHarte);
    Metodo metodo() const { return metodo_; }
    int pasos() const { return n_; }
    void generar(std::mt19937_64& gen, std::vector<double>& B) const;

private:
    double H_, T_;
    int n_;
    Metodo metodo_;
    std::vector<double> raiz_;   // Davies-Harte: √(λ_k / M)
    std::vector<double> L_;      // Cholesky: triangular inferior n×n
};

// ============================================================ //
// Rough Heston (El Euch-Rosenbaum 2019)                         //
//   dS/S = √V dB,  V_t = V0 + ∫K(t-s)κ(θ - V_s)ds + ∫K(t-s)ξ√V_s dW_s   //
// ============================================================ //
struct RoughHeston {
    double H = 0.1, V0 = 0.04, theta = 0.04, kappa = 0.3, xi = 0.3, rho = -0.7;
};

// φ(u) = E[exp(iu·X_T)], X_T = ln(S_T/S_0) - (r-q)T:
//   φ = exp(κθ·∫h + V0·∫F(h)),  D^α h = F(h) = -½(u²+iu) + (iρξu - κ)h + ½ξ²h²,  h(0) = 0
// La ecuación de Riccati fraccionaria se integra con la regla trapezoidal
// fraccionaria implícita (Adams-Moulton de producto, N pasos) resolviendo la
// ecuación cuadrática de cada paso de forma exacta: estable aunque |u| sea grande.
cd fc_rough_heston(const RoughHeston& p, cd u, double T, int N = 128);

// Curva de varianza forward E[V_t] = θ + (V0 - θ)·E_α(-κ t^α)
double varianza_forward_rh(const RoughHeston& p, double t);

// ============================================================ //
// Rough Bergomi (Bayer-Friz-Gatheral 2016)                      //
//   V_t = ξ0·exp(η Y_t - ½η² t^{2H}),  Y_t = √(2H)∫(t-s)^{H-1/2} dW_s   //
// ============================================================ //
struct RoughBergomi {
    double H = 0.1, xi0 = 0.04, eta = 1.9, rho = -0.9;
};

struct PrecioMC {
    double precio = 0, error = 0;   // media y error estándar
};

// Calls sobre un forward unitario (S_0 = 1, r = q = 0) con el esquema híbrido de
// Bennedsen, Lunde y Pakkanen (κ = 1: paso cercano exacto, resto con los puntos
// óptimos b_k) y la suma de convolución por FFT; antitéticas.
std::vector<PrecioMC> rbergomi_hibrido(const RoughBergomi& p, double T, int pasos, long long trayectorias,
                                       const std::vector<double>& strikes, uint64_t semilla = 1);

// ============================================================ //
// FMLS: Finite Moment Log Stable (Carr-Wu 2003)                 //
//   ln S_T = ln S_0 + (r - q + μ)T + L_T,  L α-estable con β = -1  //
// ============================================================ //
struct FMLS {
    double alpha = 1.5;   // índice de estabilidad α ∈ (1, 2]
    double sigma = 0.2;   // "volatilidad": escala estable σ/√2 (con α = 2 es Black-Scholes con σ)
    double escala() const;          // σ/√2
    double mu() const;              // corrección de martingala por año: (σ/√2)^α·sec(πα/2)
};
cd fc_fmls(const FMLS& p, cd u, double T);
// Incremento de L en dt por Chambers-Mallows-Stuck con β = -1, a partir de una
// uniforme u ∈ (0,1) (ángulo) y una exponencial W de media 1
double cms_fmls(const FMLS& p, double dt, double u, double W);

// EDP fraccionaria en espacio (x = ln S, τ = T - t):
//   V_τ = (r - q + μ)V_x + κ_α·D^α V - rV,  κ_α = -(σ/√2)^α sec(πα/2) > 0
// con D^α la derivada de Weyl por la izquierda, discretizada con el operador de
// Grünwald desplazado y ponderado (WSGD, 2º orden; Tian-Zhou-Deng 2015) y
// Crank-Nicolson con arranque de Rannacher. Los valores por debajo de la malla
// son 0 (call) o la barrera desactivante está en el borde inferior. Las puts
// europeas salen por paridad.
struct OpcionesEDP {
    int nx = 1200;   // nodos en espacio
    int nt = 200;    // pasos en tiempo
};
double fmls_edp(const FMLS& p, bool call, double S0, double K, double T, double r, double q,
                double barrera_abajo = 0.0, const OpcionesEDP& o = {});
PrecioMC fmls_mc(const FMLS& p, bool call, double S0, double K, double T, double r, double q, long long n,
                 uint64_t semilla = 1);

// ============================================================ //
// Black-Scholes fraccionario y mixto                            //
//   S_t = S_0·exp((r-q)t - ½(σ²t + σ_H² t^{2H}) + σW_t + σ_H B^H_t)  //
// (Necula 2002 con σ = 0; Cheridito 2001 / Sun 2013 el mixto)   //
// ============================================================ //
struct FBS {
    double H = 0.6, sigma = 0.0, sigma_h = 0.2;
    double varianza(double t) const;     // σ²t + σ_H² t^{2H}
};
double fbs_precio(const FBS& p, bool call, double S0, double K, double T, double r, double q);
// Asiática geométrica discreta (n_obs fechas equiespaciadas): fórmula cerrada
double fbs_asiatica_geometrica(const FBS& p, bool call, double S0, double K, double T, double r, double q, int n_obs);
// Asiática por Monte Carlo con trayectorias del fBm; la aritmética usa la
// geométrica como variable de control
struct ResultadoAsiaticaMC {
    PrecioMC geometrica, aritmetica;
};
ResultadoAsiaticaMC fbs_asiatica_mc(const FBS& p, bool call, double S0, double K, double T, double r, double q,
                                    int n_obs, long long n, uint64_t semilla = 1,
                                    GeneradorFBM::Metodo m = GeneradorFBM::Metodo::DaviesHarte);

// ============================================================ //
// Black-Scholes subdifusivo (Magdziarz 2009)                    //
//   F_t·exp(σB(E_t) - ½σ²E_t), E_t = inf{τ: U_α(τ) > t}      //
// ============================================================ //
struct Subdifusivo {
    double alpha = 0.8;   // α ∈ (0, 1]; α = 1 es Black-Scholes
    double sigma = 0.2;
};
// E_T = T^α·(W/A(πV))^{1-α} (representación de Kanter): precio = E[Black(σ²E_T)]
// con Gauss-Legendre en V y la regla trapezoidal en ln W
double subdif_precio(const Subdifusivo& p, bool call, double S0, double K, double T, double r, double q);
// EDP de Caputo en el log-forward: ∂_t^α c = ½σ²(c_yy - c_y), esquema L1 en malla
// graduada t_n = T(n/N)^γ y diferencias centradas implícitas en y, con
// extrapolación de Richardson en t (orden 2-α) y en y (orden 2)
double subdif_edp(const Subdifusivo& p, bool call, double S0, double K, double T, double r, double q,
                  int ny = 800, int nt = 300);
PrecioMC subdif_mc(const Subdifusivo& p, bool call, double S0, double K, double T, double r, double q, long long n,
                   uint64_t semilla = 1);
double tiempo_operacional_medio(double alpha, double T); // E[E_T] = T^α/Γ(1+α)

// Precio de Black en forma forward (útil en todo el módulo)
double black(bool call, double F, double K, double DF, double varianza_total);

} // namespace frac
