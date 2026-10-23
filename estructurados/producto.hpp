#pragma once
// ---------------------------------------------------------------------------- //
// Definición de productos estructurados.                                       //
//                                                                              //
// El producto se describe en un documento JSON (ver productos/*.json y         //
// estructurados/FORMATO.md). compilar() lo valida y lo convierte en un        //
// `Producto`: calendario expandido, importes y barreras por fecha, Cholesky de //
// la correlación, etc. El JSON es la "fuente de verdad" (lo que se guarda en   //
// disco); el Producto es lo que se simula.                                     //
// ---------------------------------------------------------------------------- //

#include "json.hpp"
#include "simulacion_hd.hpp"

#include <cmath>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace est {

struct ErrorProducto : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ----------------------------------------- //
// Catálogo de modelos y de sus parámetros   //
// ----------------------------------------- //

struct DefParam {
    std::string clave;        // nombre en el JSON
    std::vector<std::string> alias;
    std::string descripcion;  // texto para el asistente / listados
    double defecto;           // valor propuesto en el asistente y plantillas
    bool porcentaje;          // se muestra/pide en % (p. ej. volatilidades)
    double min, max;          // rango válido
    bool min_abierto;         // true: debe ser > min (si no, >= min)
    // Si != "", la clave guarda una volatilidad y el modelo usa su cuadrado con este nombre
    std::string es_vol_de;
};

struct DefModelo {
    TipoModelo tipo;
    std::string clave;                 // "black_scholes", "heston", ...
    std::vector<std::string> alias;    // "bs", "gbm", ...
    std::string nombre;                // nombre legible
    std::string descripcion;
    std::vector<DefParam> params;
    std::vector<std::string> claves_vol; // parámetros que se desplazan al calcular la vega
};

const std::vector<DefModelo>& catalogo_modelos();
const DefModelo& def_modelo(TipoModelo t);
const DefModelo* buscar_modelo(const std::string& nombre); // por clave o alias

struct Modelo {
    TipoModelo tipo = TipoModelo::BlackScholes;
    std::vector<double> valores;       // alineados con def_modelo(tipo).params
    double operator[](const std::string& clave) const;
    double& operator[](const std::string& clave);
    std::string resumen() const;       // "Heston (vol_inicial=20%, kappa=1.5, ...)"
};

struct Activo {
    std::string nombre;
    double spot = 100.0;
    double nivel_inicial = 100.0;
    double dividendos = 0.0;
    Modelo modelo;
    // Spot de anclaje de la vol local / SABR. Por defecto el propio spot; al
    // calcular deltas se mueve el spot pero no el ancla (superficie fija en S).
    double spot_ancla = 0.0; // <= 0: usar spot
};

struct Cupon {
    int tipo = CUP_FIJO;           // CUP_*
    std::vector<double> importe;   // por fecha de observación (fracción del nominal)
    std::vector<double> barrera;   // por fecha (HD_INFINITO = no observa)
    bool memoria = false;
    double inf = 0.0, sup = HD_INFINITO; // range accrual
};

struct Bloque {
    int tipo = BLQ_FIJO;
    int cond = COND_SIEMPRE;
    double importe = 0.0, part = 1.0, strike = 1.0;
    double suelo = -HD_INFINITO, techo = HD_INFINITO;
    double suelo_local = -HD_INFINITO, techo_local = HD_INFINITO; // cliquet
    int formula_ini = 0, formula_len = 0; // FORMULA: código en Producto::formulas
    std::string descripcion;       // texto legible ("put vendida a strike 100%"...)
    // Un componente "curva" del JSON se descompone de forma exacta en varios bloques
    // elementales (fijo + lineal + calls + digitales); solo el primero se muestra en
    // la ficha, con la descripción de la curva.
    bool oculto = false;
    int componente = -1;           // índice del componente del JSON del que procede
};

struct ConfigSimulacion {
    std::string metodo = "auto";      // auto | mc | qmc
    std::string dispositivo = "auto"; // auto | gpu | cpu
    double error_objetivo = 0.0005;   // error estándar objetivo, fracción del nominal (0,05 %)
    double pasos_por_anio = -1.0;     // < 0: automático
    uint32_t semilla = 2024u;
    long long trayectorias = 0;       // > 0: número fijo de trayectorias (sin adaptación)
    long long trayectorias_max = 200000000LL;
    int replicas_qmc = 16;
    bool doble_precision = false;     // GPU: simular en double (por defecto float)
    int hilos = 0;                    // CPU: 0 = todos los núcleos
};

struct Curva {
    std::vector<double> t, r;         // tipos cero continuos; 1 punto = plano
    double tipo(double tt) const;
    double df(double tt) const { return std::exp(-tipo(tt) * tt); }
    double integral(double tt) const { return tipo(tt) * tt; } // ∫_0^t r(s) ds
};

struct Producto {
    std::string nombre, descripcion, moneda = "EUR", plantilla;
    double nominal = 100.0;
    double T = 1.0;
    Curva curva;
    std::vector<Activo> activos;
    std::vector<double> correlacion;  // n×n
    std::vector<double> chol;         // n×n triangular inferior
    int cesta = CESTA_PEOR;
    std::vector<double> pesos;
    std::vector<double> t_obs;
    std::vector<std::string> etiqueta_obs; // fecha tal como se escribió (o plazo)
    std::vector<Cupon> cupones;
    bool hay_autocall = false;
    std::vector<int> ac_activa;
    std::vector<double> ac_barrera, ac_importe;
    int ki_obs = OBS_NINGUNA; double ki_nivel = 0.0; bool ki_bgk = true;
    int ko_obs = OBS_NINGUNA; double ko_nivel = HD_INFINITO; bool ko_bgk = true;
    double ko_importe = 1.0; bool ko_pago_venc = true;
    bool rend_medio = false;
    std::vector<Bloque> bloques;
    std::vector<InstrFormula<double>> formulas; // código de los componentes "formula"
    std::vector<std::string> textos_formula;    // expresión original de cada componente "formula"
    bool ki_inicial = false;
    std::vector<double> memoria_inicial;  // por pata de cupón
    ConfigSimulacion sim;
    std::vector<std::string> avisos;      // advertencias de validación (no bloqueantes)

    int n_obs() const { return (int)t_obs.size(); }
    int indice_cliquet() const;           // bloque cliquet o -1
    bool usa_barrera_capital() const { return ki_obs != OBS_NINGUNA; }
    std::string nombre_cesta() const;
};

// Valida y expande el documento JSON. Lanza ErrorProducto con un mensaje que
// indica la clave y la línea/columna del problema.
Producto compilar(const json::Valor& doc);
Producto cargar(const std::string& ruta, json::Valor* doc = nullptr);

// Utilidades de lectura (también las usa el asistente)
double texto_a_real(const std::string& s, bool* ok = nullptr);   // "60%", "0.6", "1e-3"
double texto_a_plazo(const std::string& s, bool* ok = nullptr);  // "18M", "3A", "2Y6M", "1.5"
int    texto_a_frecuencia(const std::string& s);                  // "trimestral" -> 4, 0 si no
bool   texto_a_fecha(const std::string& s, int& dias);            // "2026-10-23" -> días civiles
std::string porcentaje(double x, int dec = 2);                    // 0.6 -> "60%"
std::string fmt(double x, int dec);                               // número con dec decimales

// Ficha descriptiva (term sheet) del producto, con el perfil de pago final
void imprimir_ficha(std::ostream& os, const Producto& p);

// Pasos por año efectivos que usará la simulación (resuelve el modo automático)
double pasos_por_anio_efectivos(const Producto& p);

} // namespace est
