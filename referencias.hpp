#pragma once
// ---------------------------------------------------------------------------- //
// Precios de referencia analíticos y semianalíticos (CPU, doble precisión).    //
//                                                                              //
// Sirven para validar los motores Monte Carlo (vanilla y estructurados):      //
//  - Black-Scholes con dividendos: call/put, digitales, put down-and-in        //
//    continua (Reiner-Rubinstein), call up-and-out continua                   //
//  - Merton (serie de Poisson de precios Black-Scholes)                        //
//  - Fourier (fórmula de Lewis 2001) con la función característica de         //
//    Black-Scholes, Merton, Heston ("little trap" de Albrecher et al.),       //
//    Bates y NIG                                                               //
//  - SABR: volatilidad implícita aproximada de Hagan et al. (2002)            //
//  - Normal bivariante (algoritmo de Genz) para digitales worst-of            //
// Todos los tipos y dividendos son continuos.                                 //
// ---------------------------------------------------------------------------- //

#include <complex>
#include <functional>

namespace ref {

double norm_cdf(double x);
double norm_pdf(double x);

// P(X < a, Y < b) con (X, Y) normales estándar de correlación rho
double normal_bivariante(double a, double b, double rho);

double bs_precio(bool call, double S, double K, double T, double r, double q, double sigma);
// Digital "cash-or-nothing" que paga 1
double bs_digital(bool call, double S, double K, double T, double r, double q, double sigma);
// Put down-and-in con barrera H < S vigilada de forma continua
double bs_put_down_in(double S, double K, double H, double T, double r, double q, double sigma);
// Call up-and-out con barrera H > S y H > K vigilada de forma continua (sin rebate)
double bs_call_up_out(double S, double K, double H, double T, double r, double q, double sigma);

// Merton (1976): serie de Black-Scholes con saltos lognormales
double merton_precio(bool call, double S, double K, double T, double r, double q, double sigma,
                     double lambda, double mu_j, double sigma_j);

// Función característica de X_T = ln(S_T/S_0) - (r-q)·T (cumple φ(-i) = 1)
using FuncCar = std::function<std::complex<double>(std::complex<double>)>;
FuncCar fc_black_scholes(double sigma, double T);
FuncCar fc_merton(double sigma, double lambda, double mu_j, double sigma_j, double T);
FuncCar fc_heston(double v0, double kappa, double theta, double xi, double rho, double T);
FuncCar fc_bates(double v0, double kappa, double theta, double xi, double rho,
                 double lambda, double mu_j, double sigma_j, double T);
FuncCar fc_nig(double sigma, double theta, double kappa, double T);

// Precio por Fourier (Lewis 2001); la put sale por paridad put-call
double precio_fourier(bool call, const FuncCar& phi, double S, double K, double T, double r, double q);

// SABR: vol implícita de Hagan sobre el forward F con α "absoluto" (unidades F^(1-β))
double sabr_vol_hagan(double F, double K, double T, double alpha, double beta, double rho, double nu);
// Precio con la convención del motor: α es la vol en el forward inicial (α_abs = α·F^(1-β))
double sabr_precio(bool call, double S, double K, double T, double r, double q,
                   double alpha, double beta, double rho, double nu);

} // namespace ref
