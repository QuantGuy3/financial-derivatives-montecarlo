#pragma once
// ---------------------------------------------------------------------------- //
// Optimizadores para la calibración (sin dependencias externas).               //
//                                                                              //
//  levenberg_marquardt  mínimos cuadrados no lineales: min ½·Σ r_i(x)²         //
//                       (jacobiano por diferencias finitas, escalado de         //
//                       Marquardt y regla de actualización de Nielsen)         //
//  nelder_mead          símplex sin derivadas (parámetros adaptativos de       //
//                       Gao y Han), robusto ante objetivos con ruido            //
//  evolucion_diferencial búsqueda global DE/rand/1/bin, reproducible (semilla)  //
//                                                                              //
// Todos trabajan sin restricciones: las cotas de los parámetros se imponen     //
// con transformaciones (Transformacion) en la capa de calibración.             //
// ---------------------------------------------------------------------------- //

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace est::opt {

// Residuos r(x); devuelve false si x no es admisible (p. ej. viola una restricción)
using FuncResiduos = std::function<bool(const std::vector<double>& x, std::vector<double>& r)>;
// Función objetivo escalar; devuelve +inf (o NaN) si x no es admisible
using FuncObjetivo = std::function<double(const std::vector<double>& x)>;

struct Resultado {
    std::vector<double> x;
    double f = 0.0;                 // ½·Σr² (LM) o el valor del objetivo
    int iteraciones = 0, evaluaciones = 0;
    bool convergido = false;
    std::string motivo;             // por qué se detuvo
    std::vector<double> jtj;        // JᵀJ en la solución (LM; n×n), para errores estándar
    std::vector<double> residuos;   // residuos en la solución (LM)
};

struct OpcionesLM {
    int max_iter = 200;
    double paso_jacobiano = 1e-6;   // relativo (con suelo absoluto del mismo valor)
    double tol_f = 1e-12;           // reducción relativa de ½Σr²
    double tol_x = 1e-10;           // tamaño relativo del paso
    double tol_g = 1e-12;           // norma del gradiente
    double tau = 1e-3;              // amortiguación inicial (relativa a max diag JᵀJ)
};
Resultado levenberg_marquardt(const FuncResiduos& f, std::vector<double> x0, const OpcionesLM& o = {});

struct OpcionesNM {
    int max_eval = 4000;
    double paso_inicial = 0.25;     // tamaño del símplex inicial
    double tol_f = 1e-12, tol_x = 1e-9;
};
Resultado nelder_mead(const FuncObjetivo& f, std::vector<double> x0, const OpcionesNM& o = {});

struct OpcionesDE {
    int poblacion = 0;              // 0 = max(15, 10·n)
    int generaciones = 150;
    double F = 0.6, CR = 0.9;
    uint32_t semilla = 20240617u;
    double tol_f = 1e-12;           // parada si la dispersión de la población es menor
};
// Busca en la caja [lo, hi] (en el espacio transformado)
Resultado evolucion_diferencial(const FuncObjetivo& f, const std::vector<double>& lo, const std::vector<double>& hi,
                                const OpcionesDE& o = {});

// Resuelve A·x = b (A simétrica definida positiva, n×n por filas) por Cholesky.
// Devuelve false si A no es definida positiva.
bool resolver_cholesky(std::vector<double> A, std::vector<double> b, int n, std::vector<double>& x);
// Inversa de una matriz simétrica definida positiva (false si no lo es)
bool invertir_spd(const std::vector<double>& A, int n, std::vector<double>& inv);

} // namespace est::opt
