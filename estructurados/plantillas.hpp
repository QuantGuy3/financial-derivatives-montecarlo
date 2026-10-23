#pragma once
// ---------------------------------------------------------------------------- //
// Plantillas de productos estructurados habituales.                            //
//                                                                              //
// Cada plantilla es una lista de preguntas (con valor por defecto y ayuda) y   //
// una función que, con las respuestas, escribe las secciones propias del       //
// producto en el JSON (calendario, cupones, autocancelación, barreras y        //
// amortización final). La usan tanto el asistente interactivo como el comando  //
// "plantilla" (que toma todos los valores por defecto).                        //
// ---------------------------------------------------------------------------- //

#include "json.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace est {

enum class TipoDato {
    Real,        // número
    Porcentaje,  // en el asistente, "70" significa 70%
    PctOpcional, // porcentaje o "no" (sin límite)
    Plazo,       // "3A", "18M", 2.5
    Entero,
    SiNo,
    Opcion,      // una de `opciones`
    Texto,
    Frecuencia,  // mensual, trimestral, semestral, anual, "3M"...
    Observacion, // vencimiento, fechas, diaria, continua
    Formula,     // expresión en P: "100% + 50% * max(P - 1, 0)"
    Puntos       // curva P:pago, p. ej. "0:90 90:90 100:100 130:135"
};

struct Pregunta {
    std::string clave;
    std::string texto;
    std::string defecto;
    TipoDato tipo = TipoDato::Porcentaje;
    std::string ayuda;
    std::vector<std::string> opciones;
    std::string solo_si; // "clave=valor": se pregunta solo si otra respuesta tiene ese valor
};

using Respuestas = std::map<std::string, std::string>;

struct SubyacenteDef {
    std::string nombre;
    double spot;
    std::string dividendos;
    std::string vol;
};

struct Plantilla {
    std::string id;
    std::string titulo;
    std::string descripcion;
    std::vector<SubyacenteDef> subyacentes_def;
    std::string vencimiento_def;
    std::vector<Pregunta> preguntas;
    std::function<void(json::Valor& doc, const Respuestas& r)> construir;
    std::function<std::string(const Respuestas& r)> describir;
    bool nominal_es_spot = false; // opción vanilla: nominal = spot (precio por acción)
};

const std::vector<Plantilla>& plantillas();
const Plantilla* buscar_plantilla(const std::string& id);

// Datos comunes que se piden para cualquier plantilla
struct DatosComunes {
    std::string nombre, descripcion, moneda = "EUR";
    double nominal = 1000.0;
    std::string vencimiento = "3A";
    std::string tipo_interes = "3%";
    json::Valor subyacentes = json::Valor::lista();
    json::Valor correlacion;   // nulo si hay un solo subyacente
    std::string cesta = "peor";
    json::Valor simulacion;    // nulo = valores por defecto
};

DatosComunes comunes_por_defecto(const Plantilla& p);
// Completa con los valores por defecto las respuestas que falten y genera el producto
json::Valor construir_producto(const Plantilla& p, const DatosComunes& c, const Respuestas& r);

// Normaliza una respuesta de texto según su tipo; devuelve false y un mensaje si no es válida.
// En las preguntas de porcentaje un número sin "%" se interpreta en puntos porcentuales.
bool normalizar_respuesta(const Pregunta& q, const std::string& texto, std::string& canonico, std::string& error);
// Valor JSON de una respuesta ya normalizada
json::Valor a_json(TipoDato t, const std::string& canonico);
// Modelo por defecto (Black-Scholes) con la volatilidad dada
json::Valor modelo_black_scholes(const std::string& vol);

} // namespace est
