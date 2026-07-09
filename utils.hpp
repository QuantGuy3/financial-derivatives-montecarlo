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
};

void print_table(const std::vector<TableRow>& rows, double price_ref,
                 double epsilon, const std::string& example_name);
