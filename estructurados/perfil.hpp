#pragma once
// ---------------------------------------------------------------------------- //
// Pagos a vencimiento como funciones lineales a trozos de P (con saltos) y su  //
// descomposición estática en bono, subyacente, puts, calls y digitales.        //
//                                                                              //
// Para una función A(P) lineal a trozos, con cambios de pendiente Δs_k y saltos //
// J_k en los nodos K_k, el desarrollo alrededor de P0 (Carr y Madan, 1998) es  //
// EXACTO:                                                                      //
//   A(P) = A(P0⁻) + A'(P0⁻)·(P - P0)                                           //
//        + Σ_{K_k < P0} Δs_k·(K_k - P)⁺ + Σ_{K_k ≥ P0} Δs_k·(P - K_k)⁺          //
//        - Σ_{K_k < P0} J_k·1{P < K_k}  + Σ_{K_k ≥ P0} J_k·1{P ≥ K_k}           //
// (para una función suave, Δs_k ≈ A''(K_k)·ΔK: la fórmula integral de Carr-    //
// Madan discretizada).                                                         //
// ---------------------------------------------------------------------------- //

#include "producto.hpp"

#include <functional>
#include <string>
#include <vector>

namespace est {

struct NodoPerfil {
    double p;   // rendimiento del nodo
    double izq; // límite por la izquierda
    double der; // valor en el nodo y por la derecha (izq != der: salto)
};

struct PerfilLineal {
    std::vector<NodoPerfil> nodos; // p estrictamente creciente
    double pend_izq = 0.0;         // pendiente antes del primer nodo
    double pend_der = 0.0;         // pendiente después del último

    double operator()(double P) const;
    // Pendiente del tramo que empieza en el nodo i (después del último: pend_der)
    double pendiente_tras(size_t i) const;
};

struct Descomposicion {
    double P0 = 1.0;
    double base = 0.0;      // A(P0⁻): importe fijo a vencimiento
    double pendiente = 0.0; // A'(P0⁻): unidades de P (del subyacente)
    struct Pieza { double k, w; };
    std::vector<Pieza> puts, calls;       // w = cambio de pendiente en k
    std::vector<Pieza> dig_puts, dig_calls; // w = salto en k (las digitales put llevan -J)
};

Descomposicion descomponer(const PerfilLineal& f, double P0);

// Perfil exacto de la amortización final de un producto (función de P final),
// con la barrera de capital observada solo a vencimiento si la hay. Devuelve
// false si algún componente no es lineal a trozos (fórmulas o cliquet).
bool perfil_exacto(const Producto& p, PerfilLineal* out);

// Pago de la amortización final del producto en función de P final (incluye
// la barrera de capital observada a vencimiento y las fórmulas).
class EvaluadorFinal {
public:
    explicit EvaluadorFinal(const Producto& p);
    double operator()(double P) const;     // con la barrera resuelta por P
    double con_barrera(double P, int ki) const;
private:
    std::vector<BloqueSim<double>> bloques_;
    SimProducto<double> sp_{};
    int ki_obs_ = OBS_NINGUNA;
    double ki_nivel_ = 0.0;
    bool ki_inicial_ = false;
};

// Interpola g en los nodos dados (p creciente); los saltos de g entre nodos se
// convierten en rampas. Fuera de los nodos se prolongan los tramos extremos.
PerfilLineal interpolar(const std::function<double(double)>& g, const std::vector<double>& nodos);

} // namespace est
