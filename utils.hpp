#pragma once
#include "models.hpp"
#include "payoffs.hpp"
#include <vector>
#include <string>
#include <functional>
#include <cstdint>


// ------------------------------//
// Precios con fórmula analítica //
// ------------------------------//

double bs_call(double S0, double K, double T, double r, double sigma);

// Precio analítico (sin descuento) de la Asian geométrica discreta bajo GBM
double geom_asian_analytic(double S0, double K, double T, double mu,
                           double sigma, int n_steps);


// ------------------------------------//
// Precomputación del Brownian Bridge  //
// ------------------------------------//

struct BBData {
    int N;
    double T;
    std::vector<int>    map_idx;
    std::vector<int>    left_idx;
    std::vector<int>    right_idx;
    std::vector<double> weight_left;
    std::vector<double> weight_right;
    std::vector<double> std_dev;
};

// N debe ser potencia de 2
BBData bb_precompute(int N, double T);

// Transforma normales Z_(n_sim×N) en incrementos brownianos dW_(n_sim×N) usando BB
void bb_apply(const BBData& bb, double* Z, double* dW, int n_sim);


// --------------------------------//
// Precomputación PCA (usa Eigen)  //
// --------------------------------//

struct PCAData {
    int m;
    double T;
    // dW_(n_sim×m) = Z_(n_sim×m) @ M_pca^T
    std::vector<double> M_pca;
    std::vector<float>  M_pca_f32; // versión float32 para la GPU
};

PCAData pca_compute(int m, double T);


// ------------------------------- //
// Estimación de c1 por Richardson //
// ------------------------------- //

using SimFn = std::function<double(int n_steps, long long n_paths, unsigned seed)>;

// Devuelve c1 tal que sesgo(h) ≈ c1 · h  (extrapolación de Richardson de dos niveles)
double estimar_c1_richardson(SimFn sim_fn, double T, int M_rich, int N_pilot, unsigned seed = 0);


// ---------------------------- //
// Actualizar media y varianza  //
// ---------------------------- //

struct RunningStats {
    double mean = 0.0;
    double M2   = 0.0;
    long long n = 0;

    void update(double x) {
        ++n;
        double delta = x - mean;
        mean += delta / n;
        M2   += delta * (x - mean);
    }

    double variance()  const { return n > 1 ? M2 / (n - 1) : 0.0; }
    // Desviación típica de la propia magnitud acumulada (p.ej. de los precios
    // de R repeticiones), NO del error de su media.
    double stddev()     const { return std::sqrt(variance()); }
    // Error estándar de la media de las n muestras acumuladas. Usado en la
    // TAREA 2 tanto para el std_error "intra-corrida" clásico (n = trayectorias)
    // como, reutilizando la misma clase, para el std_error "entre repeticiones"
    // de la media de precio (n = R repeticiones, cada una con semilla propia):
    // std_error_entre_reps = stddev(precios de las R corridas) / sqrt(R).
    // Se prefiere reportar ESTE último en la tabla en vez de promediar los
    // std_error intra-corrida, porque stddev/√R sí captura la variabilidad
    // real observada entre corridas (incluye cualquier fuente de varianza que
    // el estimador intra-corrida no modele, p.ej. la propia aleatoriedad de L
    // en MLMC), mientras que promediar std_error intra-corrida solo repetiría,
    // con un poco de suavizado, la misma cifra que ya da una sola corrida.
    double std_error() const { return n > 0 ? std::sqrt(variance() / n) : 0.0; }
};


// ------------------- //
// Tabla de resultados //
// ------------------- //

struct TableRow {
    std::string method;
    double      price;
    double      std_error;
    long long   n_samples;
    double      time_s;
    // TAREA 2 (repeticiones para significancia estadística): número de
    // repeticiones EFECTIVAS del método (R, con semillas distintas) que se
    // promediaron para obtener esta fila. R=1 (el valor por defecto) reproduce
    // el comportamiento anterior de una sola corrida, así que ningún sitio que
    // siga construyendo un TableRow con el agregado de 5 campos de siempre
    // necesita cambiar nada.
    int n_reps = 1;
};

void print_table(const std::vector<TableRow>& rows, double price_ref,
                 double epsilon, const std::string& example_name);


// ------------------------------------------------------------- //
// Barrido de precision con corte por metodo (correccion al       //
// mecanismo de repeticiones original de la TAREA 2)              //
// ------------------------------------------------------------- //
//
// Diseno corregido por el usuario tras revisar la primera version:
//   - Cada EJEMPLO recorre internamente una lista de niveles de eps cada vez
//     mas finos (antes: el eps se fijaba una unica vez desde argv[1] y solo
//     se hacia una tabla; ahora argv[1], si se da, es el eps MAS FINO que se
//     intenta, y el barrido interno va desde un eps grueso fijo hasta ese
//     limite).
//   - Para cada METODO, en cada nivel de eps, se ejecuta primero UNA sola
//     corrida y se mide su tiempo. El presupuesto T_BUDGET_S=30.0 se aplica
//     a ESA corrida individual, nunca a la suma de repeticiones: si tarda
//     mas de 30s se marca "CORTADO" en ese nivel y en todos los siguientes
//     para ESE metodo (se asume que a un eps mas fino tardara mas, no tiene
//     sentido malgastar tiempo comprobandolo), pero los demas metodos siguen
//     su propio barrido de forma independiente.
//   - Si la corrida individual entra en el presupuesto, SI se repite (con
//     semilla distinta cada vez) hasta R_MAX=30 veces, SIN limite adicional
//     sobre el tiempo TOTAL de esas repeticiones (30 corridas de 2s = 60s
//     esta bien, eso no es un "fallo" de presupuesto). Se reporta la media,
//     la desviacion tipica ENTRE repeticiones / sqrt(R) (RunningStats::
//     std_error) y el R efectivo, igual que en el diseno anterior.
//
// El formato de impresion es compatible con el parser gen_informe.py (que ya
// esperaba un bloque "### ejemplo", lineas "--- nivel N eps=... wall=...s ---",
// "Referencia: ..." y filas de metodo de 6 columnas numericas + SI/NO; ver
// row_re en gen_informe.py, actualizado para aceptar la columna R final de
// forma opcional). Un metodo cortado en un nivel simplemente no imprime fila
// ese nivel (se marca con una linea "CORTADO" aparte que row_re no matchea),
// lo que gen_informe.py ya interpreta como "--" al reconstruir la matriz.

