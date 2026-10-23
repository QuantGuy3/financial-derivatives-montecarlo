#pragma once
// ---------------------------------------------------------------------------- //
// Datos de mercado para calibrar: cotizaciones de opciones europeas de un      //
// subyacente agrupadas por vencimiento, con su forward y su descuento.         //
//                                                                              //
// Fuentes:                                                                     //
//  - fichero JSON de mercado con una superficie de volatilidades (rejilla      //
//    vencimientos × strikes) o una lista de cotizaciones (precio o vol)        //
//  - cadena de opciones en CSV (cotizaciones.hpp), p. ej. la que descarga      //
//    herramientas/descargar_cadena.py                                          //
//                                                                              //
// Forward y descuento de cada vencimiento: con el tipo de interés y los        //
// dividendos dados o, si hay calls y puts del mismo strike, implícitos por la  //
// paridad put-call (C - P = DF·(F - K): regresión en los strikes cercanos).    //
// De cada strike se usa la opción fuera del dinero, la más líquida y la que    //
// mejor fija la volatilidad.                                                   //
// ---------------------------------------------------------------------------- //

#include "producto.hpp"

#include <string>
#include <vector>

namespace est {

struct PuntoMercado {
    int v = 0;                // índice del vencimiento
    double T = 0, K = 0;      // K en unidades del subyacente
    bool call = true;         // instrumento (fuera del dinero)
    double precio = 0;        // precio de mercado (medio)
    double vol = 0;           // volatilidad implícita de mercado
    double vol_bid = -1, vol_ask = -1;
    double vega = 0;          // ∂precio/∂σ (para pesos y residuos)
    double peso = 1;          // peso dado en el fichero
    double volumen = -1;
};

struct VencimientoMercado {
    double T = 0;
    std::string etiqueta;
    double DF = 1, F = 0;     // descuento y forward
    bool forward_implicito = false;
    std::vector<int> puntos;  // índices en DatosMercado::puntos
};

struct DatosMercado {
    std::string subyacente, fuente, fecha;
    double spot = 0;
    double r = 0, q = 0;      // tipos planos de referencia (los que se dieron o se infirieron)
    std::vector<VencimientoMercado> vencimientos;
    std::vector<PuntoMercado> puntos;
    std::vector<std::string> avisos;
    int descartados = 0;      // cotizaciones descartadas por los filtros

    double vol_atm(int v) const; // volatilidad interpolada en K = F
};

struct OpcionesMercado {
    int dias_valoracion = -1;         // fecha para vencimientos con fecha (si no está en el fichero)
    double spot = -1;                 // fuerza el spot
    double tipo = -1e9, dividendos = -1e9; // fuerzan r y q
    std::string forward = "auto";     // auto (paridad si se puede) | dado | paridad
    double T_min = 7.0 / 365.0, T_max = 1e9;
    double mon_min = 0.0, mon_max = 1e9; // K/F admitido
    double volumen_min = 0.0;
    bool incluir_dentro_del_dinero = false;
};

// Carga un fichero .json (mercado) o .csv (cadena de opciones)
DatosMercado cargar_mercado(const std::string& ruta, const OpcionesMercado& o = {});

// Black en forma forward y su vega
double black(bool call, double F, double K, double T, double DF, double vol);
double vega_black(double F, double K, double T, double DF, double vol);
// Volatilidad implícita en forma forward (NAN si no existe)
double vol_implicita_fwd(double precio, bool call, double F, double K, double T, double DF);

} // namespace est
