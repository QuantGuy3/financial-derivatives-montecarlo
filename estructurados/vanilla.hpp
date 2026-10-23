#pragma once
// ---------------------------------------------------------------------------- //
// Precios de opciones europeas (call, put, digital) de un subyacente con el    //
// modelo que tenga asignado en el producto. Es la pieza común de la réplica    //
// estática (valorar la cartera de opciones) y de la calibración.               //
//                                                                              //
// Método según el modelo:                                                      //
//   black_scholes      fórmula cerrada                                         //
//   merton             serie de Poisson de precios Black-Scholes               //
//   heston, bates, nig Fourier (Lewis 2001) con la función característica      //
//                      evaluada una sola vez por vencimiento y Black-Scholes   //
//                      como variable de control (precisión en las alas)        //
//   sabr               fórmula de Hagan et al. (2002) (aproximación)           //
//   vol_local y cualquier otro: Monte Carlo con el mismo paso de modelo que el //
//                      motor de estructurados (números aleatorios comunes,     //
//                      antitéticas y ajuste del forward)                       //
// Los tipos de la curva se tratan como deterministas: r_ef(T) = ∫_0^T r / T.   //
// ---------------------------------------------------------------------------- //

#include "producto.hpp"

#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace est {

struct OpcionesMCVanilla {
    long long trayectorias = 200000; // incluidas las antitéticas
    double pasos_por_anio = 252.0;   // modelos con error de discretización
    uint32_t semilla = 12345u;
    int hilos = 0;                   // 0 = todos los núcleos
};

// Integral de Lewis con variable de control Black-Scholes, precalculada para un
// vencimiento: Δ(k) = (1/π)∫_0^∞ Re[e^{iuk}(φ(u-i/2) - φ_BS(u-i/2))]/(u²+1/4) du
class IntegralLewis {
public:
    using FuncCar = std::function<std::complex<double>(std::complex<double>)>;
    // φ es la función característica de X_T = ln(S_T/S_0) - (r-q)·T (φ(-i) = 1)
    IntegralLewis(const FuncCar& phi, double T);
    double sigma_ref() const { return sigma_ref_; }
    // Precio de la call y de la digital (paga 1 si S_T ≥ K)
    double call(double S, double K, double r, double q) const;
    double digital(double S, double K, double r, double q) const;
    size_t nodos() const { return u_.size(); }

private:
    double T_ = 0.0, sigma_ref_ = 0.2;
    std::vector<double> u_, cr_, ci_; // nodos y pesos complejos (ya incluyen 1/(u²+1/4))
    void delta(double k, double* d, double* dprima) const;
};

class PreciosVanilla {
public:
    // forzar_mc: valora todo por Monte Carlo (coherencia exacta con el motor)
    PreciosVanilla(const Activo& a, const Curva& c, OpcionesMCVanilla mc = {}, bool forzar_mc = false);
    ~PreciosVanilla();
    PreciosVanilla(const PreciosVanilla&) = delete;
    PreciosVanilla& operator=(const PreciosVanilla&) = delete;

    double call(double T, double K);
    double put(double T, double K);
    // Digital "cash-or-nothing" que paga 1 en T: si S_T ≥ K (call) o si S_T < K (put)
    double digital(double T, double K, bool call = true);
    double forward(double T) const;
    double df(double T) const;
    double spot() const { return S0_; }
    double r_efectivo(double T) const;
    // Declara de una vez los vencimientos que se van a pedir (el Monte Carlo los
    // simula juntos en una sola pasada)
    void preparar(const std::vector<double>& vencimientos);
    std::string metodo() const;
    bool es_mc() const { return metodo_ == Metodo::MC; }

private:
    enum class Metodo { BS, MERTON, FOURIER, HAGAN, MC };
    Metodo metodo_;
    Activo act_;
    Curva curva_;
    double S0_, q_;
    OpcionesMCVanilla mc_;
    std::map<double, std::unique_ptr<IntegralLewis>> lewis_;
    struct Muestra;                       // trayectorias simuladas a un vencimiento
    std::map<double, std::unique_ptr<Muestra>> muestras_;

    const IntegralLewis& lewis(double T);
    const Muestra& muestra(double T);
    void simular(const std::vector<double>& Ts);
    IntegralLewis::FuncCar func_car(double T) const;
};

// ------------------------------------------------------------------------ //
// Volatilidad implícita de Black-Scholes (Newton protegido por bisección,  //
// en la forma normalizada de Jäckel; en las alas se itera sobre ln precio). //
// ok = false si el precio no es compatible con ninguna volatilidad.        //
// ------------------------------------------------------------------------ //
double vol_implicita(double precio, bool call, double S, double K, double T, double r, double q, bool* ok = nullptr);

} // namespace est
