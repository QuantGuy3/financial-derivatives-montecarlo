#pragma once
// ---------------------------------------------------------------------------- //
// Calibración histórica: parámetros estimados con una serie de precios del      //
// subyacente (medida real P), como alternativa o complemento a la calibración  //
// a opciones (medida riesgo neutro Q).                                         //
//                                                                              //
//  - Black-Scholes: volatilidad y tendencia de los log-rendimientos (MV)        //
//  - Merton: difusión con saltos por máxima verosimilitud (mezcla de Poisson   //
//    de normales), con contraste de razón de verosimilitudes frente a BS       //
//  - Exponente de Hurst de los rendimientos y de la log-volatilidad realizada   //
//    (métodos de los modelos fraccionarios; ver fraccional/)                   //
//                                                                              //
// Los parámetros de volatilidad, saltos y rugosidad se trasladan a la valoración//
// (la tendencia no: bajo Q la deriva es r - q). Hay que tener presente que la   //
// volatilidad histórica no es la implícita: la diferencia es la prima de riesgo //
// de volatilidad.                                                              //
// ---------------------------------------------------------------------------- //

#include "producto.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace est {

struct SerieHistorica {
    std::string nombre, ruta;
    std::vector<int> dias;          // fecha de cada precio (días civiles) o -1 si no hay fechas
    std::vector<double> precios;
};

// CSV con una columna de fecha (fecha/date) y una de precio (cierre, close, adj close,
// precio, valor...); si no hay fecha, cada fila es una observación diaria
SerieHistorica leer_serie_csv(const std::string& ruta);

struct EstimacionParam { std::string clave; double valor = 0, error = -1; bool porcentaje = false; };

struct ResultadoHistorico {
    int n = 0;                      // rendimientos
    double anios = 0, obs_por_anio = 0;
    double media = 0, vol = 0, asimetria = 0, curtosis = 0; // anualizadas (media, vol) y exceso de curtosis
    double loglik_bs = 0;
    std::vector<EstimacionParam> bs; // mu, volatilidad
    bool merton_ok = false;
    double loglik_merton = 0, razon_verosimilitud = 0, p_valor = 1;
    std::vector<EstimacionParam> merton;
    // Exponentes de Hurst (rendimientos y log-volatilidad realizada); < 0: no calculado
    double hurst_rend = -1, hurst_rend_error = -1;
    double hurst_vol = -1, hurst_vol_error = -1;
    int ventana_vol = 0;
    std::vector<std::string> notas;
    Modelo modelo;                  // el modelo elegido (BS o Merton), listo para un producto
};

ResultadoHistorico estimar_historico(const SerieHistorica& s, const std::string& modelo = "auto");
void imprimir_historico(std::ostream& os, const SerieHistorica& s, const ResultadoHistorico& r);

} // namespace est
