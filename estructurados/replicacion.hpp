#pragma once
// ---------------------------------------------------------------------------- //
// Réplica estática de un producto con bonos, el subyacente y opciones          //
// europeas (Carr y Madan 1998; la idea de "dibuja un pago y replícalo" de      //
// TooManyOptions, github.com/Serranegra/TooManyOptions, licencia MIT).         //
//                                                                              //
// Un producto es replicable estáticamente si cada flujo depende solo del       //
// precio del subyacente en su propia fecha: un subyacente, sin              //
// autocancelación ni barrera desactivante, barrera de capital (si la hay)       //
// observada solo a vencimiento, sin rendimiento medio ni cliquet, y cupones     //
// fijos o condicionales sin memoria (cada uno es una digital en su fecha).      //
//                                                                              //
// Métodos para la amortización final A(P):                                     //
//   exacta     pago lineal a trozos: opciones en los nodos, error nulo         //
//   rejilla    pago con fórmulas: interpolación en una rejilla de strikes      //
//   cotizados  strikes de una cadena real (filtro de volumen, fuera del dinero)//
//              con pesos por interpolación (exacta en los strikes) o por       //
//              curvatura A''(K)·ΔK (la cuadratura de Carr-Madan)               //
// Cada pieza se valora con el modelo del subyacente (vanilla.hpp) o con el     //
// precio de mercado de la cadena.                                             //
// ---------------------------------------------------------------------------- //

#include "cotizaciones.hpp"
#include "producto.hpp"
#include "vanilla.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace est {

struct OpcionesReplica {
    double paso = 0.05;                   // rejilla de strikes (en P) para pagos con fórmulas
    double p_min = -1, p_max = -1;        // rango de la rejilla/medición (< 0: automático)
    std::string pesos = "interpolacion";  // o "curvatura" (solo con cadena y pagos suaves)
    const CadenaOpciones* cadena = nullptr;
    double volumen_min = 0.0;
    double tolerancia_vencimiento = 10.0 / 365.0; // diferencia admitida con la cadena
    bool precios_mc = false;              // valorar las opciones por Monte Carlo
    OpcionesMCVanilla mc;
};

struct LineaReplica {
    std::string instrumento;  // Bono cupón cero, Subyacente, Put, Call, Digital call, Digital put
    std::string origen;       // "amortización final", "cupón 3 (1A6M)"
    double T = 0;
    double k = -1;            // strike en P (tanto por uno del nivel inicial); -1 si no aplica
    double K = -1;            // strike en unidades del subyacente
    double peso = 0;          // tanto por uno del nominal por unidad de pago (en P)
    double cantidad = 0;      // contratos (opciones sobre 1 unidad), unidades de subyacente o nominal del bono
    double precio = 0;        // precio de 1 contrato
    double valor = 0;         // cantidad × precio
    double bid = -1, ask = -1;// precio de mercado (si viene de la cadena)
    bool de_mercado = false;
};

struct Replica {
    std::string metodo, metodo_precios;
    double P0 = 1, S_ref = 1, nominal = 1;
    std::vector<LineaReplica> lineas;
    double valor = 0;                 // moneda
    bool con_horquilla = false;       // hay bid/ask de todas las opciones de mercado
    double valor_bid = 0, valor_ask = 0; // vender / comprar la cartera cruzando la horquilla
    bool exacta = false;
    double error_max = 0, error_rms = 0, p_lo = 0, p_hi = 0; // error del pago final (tanto por uno del nominal)
    std::vector<double> P, objetivo, replicado;              // perfil a vencimiento
    std::vector<std::string> notas;
};

bool es_replicable(const Producto& p, std::string* motivo);
Replica replicar(const Producto& p, const OpcionesReplica& o);
void imprimir_replica(std::ostream& os, const Producto& p, const Replica& r);
void guardar_replica_csv(const std::string& ruta, const Replica& r);       // cartera
void guardar_perfil_csv(const std::string& ruta, const Replica& r);        // pago objetivo vs réplica
json::Valor replica_a_json(const Producto& p, const Replica& r);

} // namespace est
