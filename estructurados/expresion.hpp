#pragma once
// ---------------------------------------------------------------------------- //
// Fórmulas de pago escritas como texto, p. ej.                                 //
//     "100% + 50% * max(P - 1, 0)^2"        "si(P >= 120%, 130%, P)"           //
//     "call(100%) - call(130%) + digital(80%) * 5%"                            //
// compilar_formula() las traduce a las instrucciones de pila que evalúa        //
// evaluar_formula() (simulacion_hd.hpp) en CPU y en GPU.                       //
//                                                                              //
// Sintaxis                                                                     //
//   números    1.5   2e-3   100%   25pb (puntos básicos)                       //
//   variables  P (rendimiento) y, si se da una escala, S = P·escala (precio)   //
//   operadores + - * / ^   comparaciones < <= > >= == != (valen 1 o 0)          //
//   funciones  max(a,b,...) min(a,b,...) abs exp log|ln sqrt|raiz pow|pot      //
//              si|if(c,a,b) clamp|acotar(x,lo,hi)                              //
//              call(k) = max(P-k,0)   put(k) = max(k-P,0)                       //
//              digital(k) = 1{P>=k}   digital_put(k) = 1{P<k}                  //
// ---------------------------------------------------------------------------- //

#include "simulacion_hd.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace est {

struct ErrorFormula : std::runtime_error {
    size_t posicion; // carácter (desde 0) donde se detectó el problema
    ErrorFormula(const std::string& m, size_t pos) : std::runtime_error(m), posicion(pos) {}
};

struct Formula {
    std::string texto;
    std::vector<InstrFormula<double>> codigo;
    int profundidad = 0; // máximo de elementos en la pila
    bool usa_P = false;  // false: la fórmula es una constante
};

// Compila la expresión. Si escala_S > 0 también se admite la variable S = P·escala_S.
Formula compilar_formula(const std::string& texto, double escala_S = 0.0);

inline double evaluar(const Formula& f, double P) {
    return evaluar_formula(f.codigo.data(), (int)f.codigo.size(), P);
}

// Comprueba que la fórmula es finita en P ∈ [0, p_max]; si no, lanza ErrorFormula
// indicando el primer P problemático.
void comprobar_finita(const Formula& f, double p_max = 10.0);

// Lista de funciones admitidas (para mensajes de ayuda)
const char* ayuda_formulas();

} // namespace est
