#pragma once
// ---------------------------------------------------------------------------- //
// Calibración de modelos a cotizaciones de opciones europeas.                  //
//                                                                              //
// Problema: min_θ Σ_i w_i·(σ_modelo(T_i, K_i; θ) - σ_mercado,i)²                //
// sobre volatilidades implícitas (puntos de volatilidad), que pesan igual      //
// opciones de precios muy distintos. Cada precio del modelo sale del mismo     //
// servicio que valora las réplicas (vanilla.hpp): Fourier de Lewis para        //
// Heston, Bates y NIG, serie de Merton, Hagan para SABR y Monte Carlo con      //
// números aleatorios comunes para la volatilidad local (y para cualquier       //
// modelo que no tenga fórmula).                                               //
//                                                                              //
// Los parámetros se mueven en un espacio sin restricciones (transformación     //
// logística a su rango) con Levenberg-Marquardt desde varios puntos de         //
// partida, evolución diferencial (búsqueda global) + LM, o Nelder-Mead.        //
// En la solución se estiman los errores estándar y las correlaciones de los    //
// parámetros a partir de JᵀJ: dicen qué parámetros fijan de verdad los datos.  //
//                                                                              //
// "svi" ajusta en cada vencimiento la parametrización SVI de Gatheral de la    //
// varianza total y comprueba el arbitraje de mariposa y de calendario.         //
// ---------------------------------------------------------------------------- //

#include "mercado.hpp"
#include "producto.hpp"

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace est {

struct OpcionesCalibracion {
    std::string modelo;                       // clave del catálogo (o alias) o "svi"
    std::map<std::string, double> fijos;      // parámetros que no se calibran
    std::vector<std::string> liberar;         // parámetros fijos por defecto que se calibran (p. ej. beta de SABR)
    std::map<std::string, double> iniciales;  // punto de partida
    const Modelo* partida = nullptr;          // modelo actual del producto (punto de partida)
    std::string optimizador = "auto";         // auto | lm | de | nm
    int arranques = 4;                        // puntos de partida de LM
    std::string pesos = "iguales";            // iguales | vega | horquilla
    long long trayectorias_mc = 20000;        // modelos valorados por Monte Carlo
    double pasos_mc = 100.0;
    uint32_t semilla = 777u;
    std::ostream* progreso = nullptr;
};

struct ParamCalibrado {
    std::string clave, descripcion;
    double valor = 0, inicial = 0, error = -1, lo = 0, hi = 0; // error < 0: no disponible
    bool libre = true, porcentaje = false;
};

struct AjusteSVI {
    double a = 0, b = 0, rho = 0, m = 0, sigma = 0;
    double rmse = 0;
    double g_min = 0;          // mínimo de la función de densidad g(k) (< 0: arbitraje de mariposa)
    double w(double k) const;  // varianza total
};

struct ResultadoCalibracion {
    std::string modelo_clave, modelo_nombre, metodo_precios, optimizador, motivo;
    Modelo modelo;
    std::vector<ParamCalibrado> params;
    std::vector<std::string> libres;          // claves de los parámetros calibrados
    std::vector<double> correlacion;          // entre los libres (n×n)
    double rmse = 0, rmse_inicial = 0, max_err = 0; // en volatilidad (tanto por uno)
    int i_max = -1;
    std::vector<double> vol_modelo, precio_modelo;  // por punto
    std::vector<double> rmse_venc;            // por vencimiento
    double dentro_horquilla = -1;             // fracción con la vol del modelo dentro de [bid, ask]
    int evaluaciones = 0, iteraciones = 0;
    double segundos = 0;
    std::vector<std::string> notas;
    std::vector<AjusteSVI> svi;               // modelo "svi": un ajuste por vencimiento
};

ResultadoCalibracion calibrar(const DatosMercado& m, const OpcionesCalibracion& o);
void imprimir_calibracion(std::ostream& os, const DatosMercado& m, const ResultadoCalibracion& r);
json::Valor modelo_a_json(const Modelo& mod);
json::Valor calibracion_a_json(const DatosMercado& m, const ResultadoCalibracion& r);
void guardar_ajuste_csv(const std::string& ruta, const DatosMercado& m, const ResultadoCalibracion& r);
Modelo modelo_por_defecto(TipoModelo t);

} // namespace est
