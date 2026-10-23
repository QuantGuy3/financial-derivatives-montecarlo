#pragma once
// ---------------------------------------------------------------------------- //
// Valoración de productos estructurados.                                       //
//                                                                              //
// Un `Motor` simula trayectorias de un producto concreto (CPU multihilo o GPU) //
// y acumula sus resultados. valorar() decide cuántas trayectorias hacen falta: //
//  - MC  (Philox): piloto + N = Var/error² trayectorias, error = desv./√N       //
//  - QMC (Sobol con scrambling de Hong-Hickernell, R réplicas): se duplica el   //
//        número de puntos por réplica hasta que el error estándar entre        //
//        réplicas baja del objetivo (mismo criterio que run_qmc_cuda).         //
// Las griegas se calculan por diferencias finitas con números aleatorios       //
// comunes (misma semilla y mismo número de trayectorias en cada revaluación). //
// ---------------------------------------------------------------------------- //

#include "producto.hpp"

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace est {

// Sumas acumuladas de un conjunto de trayectorias
struct Acumulado {
    double n = 0, s_pv = 0, s_pv2 = 0, s_cup = 0, s_ant = 0, s_fin = 0, s_vida = 0, s_perd = 0;
    double n_ki = 0, n_ko = 0, n_perd = 0;
    std::vector<double> n_ac, n_cup; // por fecha de observación

    void iniciar(int nobs) {
        *this = Acumulado();
        n_ac.assign((size_t)nobs, 0.0);
        n_cup.assign((size_t)nobs, 0.0);
    }
    void combinar(const Acumulado& o) {
        n += o.n; s_pv += o.s_pv; s_pv2 += o.s_pv2; s_cup += o.s_cup; s_ant += o.s_ant;
        s_fin += o.s_fin; s_vida += o.s_vida; s_perd += o.s_perd;
        n_ki += o.n_ki; n_ko += o.n_ko; n_perd += o.n_perd;
        if (n_ac.size() < o.n_ac.size()) { n_ac.resize(o.n_ac.size(), 0.0); n_cup.resize(o.n_cup.size(), 0.0); }
        for (size_t j = 0; j < o.n_ac.size(); ++j) { n_ac[j] += o.n_ac[j]; n_cup[j] += o.n_cup[j]; }
    }
    template<typename Real>
    void sumar(const ResTray<Real>& r) {
        double pv = (double)r.pv;
        n += 1; s_pv += pv; s_pv2 += pv * pv;
        s_cup += (double)r.pv_cup; s_ant += (double)r.pv_ant; s_fin += (double)r.pv_fin;
        s_vida += (double)r.vida; s_perd += (double)r.perdida;
        n_ki += r.ki; n_ko += r.ko; n_perd += r.hubo_perdida;
    }
};

class Motor {
public:
    virtual ~Motor() = default;
    virtual std::string descripcion() const = 0;
    virtual bool admite_qmc() const = 0;
    virtual bool es_gpu() const = 0;
    // MC: trayectorias de índice [desde, hasta) (cada índice tiene su propio flujo Philox)
    virtual void mc(long long desde, long long hasta, Acumulado& acc) = 0;
    // QMC: puntos [desde, hasta) de la secuencia de Sobol de la réplica r
    virtual void qmc(int replica, long long desde, long long hasta, Acumulado& acc) = 0;
};

std::unique_ptr<Motor> crear_motor_cpu(const Producto& p, uint32_t semilla, int hilos);
// Motor GPU (solo si el proyecto se compiló con CUDA y hay una GPU visible)
bool gpu_disponible(std::string* descripcion = nullptr);
std::unique_ptr<Motor> crear_motor_gpu(const Producto& p, uint32_t semilla, bool doble_precision);
// Tabla de vectores directores de Sobol (Joe-Kuo, vía cuRAND en host) si está disponible
bool direcciones_sobol(const uint32_t** dv, uint32_t* ndims);

struct OpcionesValoracion {
    // Cada campo, si se deja en su valor "sin fijar", toma el del bloque "simulacion" del producto
    std::string metodo;          // "", auto, mc, qmc
    std::string dispositivo;     // "", auto, gpu, cpu
    double error_objetivo = -1;  // fracción del nominal
    long long trayectorias = -1;
    long long trayectorias_max = -1;
    long long semilla = -1;
    int replicas_qmc = -1;
    int hilos = -1;
    int doble_precision = -1;    // -1 sin fijar, 0 float, 1 double
    double pasos_por_anio = -2;  // -2 sin fijar, -1 automático
};

struct ResultadoValoracion {
    double precio = 0, error = 0;  // fracción del nominal
    long long trayectorias = 0;
    double tiempo_s = 0;
    std::string metodo, motor;
    int replicas = 0;
    long long puntos_por_replica = 0;
    int pasos = 0;
    double pasos_por_anio = 0;
    long long dimension = 0;
    uint32_t semilla = 0;
    bool objetivo_alcanzado = true;
    double error_objetivo = 0;
    // Descomposición y estadísticas
    double vp_cupones = 0, vp_anticipada = 0, vp_final = 0, vida_media = 0;
    double prob_ki = 0, prob_ko = 0, prob_perdida = 0, perdida_media = 0;
    std::vector<double> prob_autocall, prob_cupon;
};

struct Griegas {
    std::vector<std::string> nombres;
    std::vector<double> delta;      // ∂V/∂S_i (en moneda por unidad de subyacente)
    std::vector<double> delta_pct;  // variación del precio (% nominal) por +1% del subyacente
    std::vector<double> gamma_pct;  // variación de delta_pct por +1% del subyacente
    std::vector<double> vega;       // variación (% nominal) por +1 punto de volatilidad
    double rho = 0;                 // variación (% nominal) por +1% (100 pb) de tipos
    double salto = 0.01;            // desplazamiento relativo del spot usado
};

// Resuelve las opciones efectivas (producto + línea de comandos)
ConfigSimulacion configuracion_efectiva(const Producto& p, const OpcionesValoracion& o);
ResultadoValoracion valorar(const Producto& p, const OpcionesValoracion& o, std::ostream* progreso = nullptr);
Griegas calcular_griegas(const Producto& p, const OpcionesValoracion& o, const ResultadoValoracion& base,
                         std::ostream* progreso = nullptr);

void imprimir_resultado(std::ostream& os, const Producto& p, const ResultadoValoracion& r);
void imprimir_griegas(std::ostream& os, const Producto& p, const Griegas& g);
json::Valor resultado_a_json(const Producto& p, const ResultadoValoracion& r, const Griegas* g);

} // namespace est
