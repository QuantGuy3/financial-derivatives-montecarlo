#pragma once
// ---------------------------------------------------------------------------- //
// Cadenas de opciones cotizadas (CSV): para replicar un pago con los strikes   //
// que de verdad cotizan y para calibrar modelos.                               //
//                                                                              //
// El CSV lleva una cabecera con el nombre de cada columna. Se reconocen los     //
// nombres en español y los de Yahoo Finance / yfinance (option_chain):         //
//   vencimiento | expiration, expiry      fecha AAAA-MM-DD, plazo (3M) o años  //
//   tipo | type, cp, right                call/put (C/P); si falta, se deduce  //
//                                         del símbolo (contractSymbol)         //
//   strike                                en unidades del subyacente           //
//   bid, ask | compra, venta              horquilla                            //
//   precio | mid, price                   precio medio                         //
//   ultimo | last, lastPrice              último cruce                         //
//   vol_implicita | impliedVolatility, iv volatilidad implícita (0.25 o 25%)    //
//   volumen | volume                      contratos negociados en el día       //
//   interes_abierto | openInterest        posiciones abiertas                  //
//   peso | weight                         peso en la calibración               //
// El separador puede ser coma, punto y coma o tabulador (con punto y coma se   //
// admite la coma decimal). Las líneas que empiezan por '#' son comentarios;    //
// "# clave: valor" aporta datos generales: spot, fecha, subyacente,            //
// tipo_interes, dividendos, fuente.                                            //
// ---------------------------------------------------------------------------- //

#include <string>
#include <vector>

namespace est {

struct Cotizacion {
    double T = 0.0;              // años hasta el vencimiento
    std::string vencimiento;     // tal como venía en el fichero
    bool call = true;
    double K = 0.0;
    double bid = -1, ask = -1, ultimo = -1, precio = -1; // -1: no disponible
    double vol = -1;             // volatilidad implícita (fracción), -1 si no hay
    double volumen = -1, interes_abierto = -1;
    double peso = 1.0;
    int linea = 0;               // línea del fichero (para los mensajes)

    // Precio de referencia: medio de la horquilla si es válida; si no, "precio" o el último
    double referencia() const;
    bool tiene_precio() const { return referencia() > 0; }
};

struct CadenaOpciones {
    std::string subyacente, fecha, fuente, ruta;
    double spot = -1, tipo_interes = -1e9, dividendos = -1e9; // -1 / -1e9: no vienen en el fichero
    int dias_valoracion = -1;   // fecha de los datos en días civiles (si se conoce)
    std::vector<Cotizacion> cot;
    std::vector<std::string> avisos;

    std::vector<double> vencimientos() const; // distintos, ordenados
    // Vencimiento de la cadena más cercano a T (con su diferencia en años)
    double vencimiento_cercano(double T, double* diferencia = nullptr) const;
};

// dias_valoracion >= 0 fija la fecha de valoración para convertir las fechas de
// vencimiento en años; si es < 0 se usa la "# fecha:" del fichero.
CadenaOpciones leer_cadena_csv(const std::string& ruta, int dias_valoracion = -1);

// Strikes fuera del dinero respecto a S0 de un vencimiento: puts con K < S0 y calls
// con K > S0, con volumen ≥ volumen_min (si el fichero trae volumen) y precio.
struct StrikeCotizado { double K; bool call; const Cotizacion* q; };
std::vector<StrikeCotizado> strikes_fuera_del_dinero(const CadenaOpciones& c, double T, double S0,
                                                     double volumen_min, double K_min, double K_max);

} // namespace est
