#pragma once
// Precios de referencia analíticos (tiempo continuo o monitorización discreta exacta) de los
// modelos/payoffs que los admiten. Sirven para dibujar la línea de referencia de la GUI y como
// verdad de terreno en los tests; los ejemplos del repositorio usan en cambio un MC grande.
//
// IMPORTANTE: los motores simulan con esquemas de Euler con paso h = T/n_steps, así que el
// resultado simulado difiere de la referencia en un sesgo O(h) (visible a n_steps pequeño).

#include "models.hpp"
#include "payoffs.hpp"

#include <optional>
#include <string>

struct ReferencePrice {
    double value = 0.0;
    std::string kind;    // texto legible: de dónde sale la referencia
};

// Call europea bajo GBM con deriva mu y descuento a tasa r: e^{-rT}·E[(S_T-K)+].
// Con mu = r es Black-Scholes.
double gbm_call_drift(double S0, double K, double T, double mu, double r, double sigma);

// Call europea bajo Heston con deriva mu y descuento a tasa r (inversión de Fourier de Heston 1993
// con la formulación "little trap" de Albrecher et al.): e^{-rT}·E[(S_T-K)+].
double heston_call(const HestonParams& h, double K, double r);

// Referencia disponible para (modelo, payoff), si existe:
//   GBM + europea               Black-Scholes con deriva mu
//   GBM + Asian geométrica      fórmula del lognormal con n_steps fechas (n_steps > 0)
//   Heston + europea            inversión de Fourier
std::optional<ReferencePrice> reference_price(const ModelVariant& model, const PayoffVariant& payoff,
                                              int n_steps);
