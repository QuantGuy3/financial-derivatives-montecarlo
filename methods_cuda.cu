// Implementación CUDA de métodos Monte Carlo para valoración de derivados financieros.
// Objetivo: NVIDIA A-100 (sm_80). Requiere CUDA 11+, cuBLAS, cuRAND, CUB.

#include "methods_cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// --------------- //
// Errores de CUDA //
// --------------- //

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e) \
                                     + " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } while (0)

#define CURAND_CHECK(call) \
    do { \
        curandStatus_t _s = (call); \
        if (_s != CURAND_STATUS_SUCCESS) \
            throw std::runtime_error("cuRAND error " + std::to_string((int)_s) \
                                     + " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t _s = (call); \
        if (_s != CUBLAS_STATUS_SUCCESS) \
            throw std::runtime_error("cuBLAS error at " __FILE__ ":" \
                                     + std::to_string(__LINE__)); \
    } while (0)


// ----------------- //
// Memoria constante //
// ----------------- //

enum class PayoffKind : int { European=0, Asian=1, GeomAsian=2, Lookback=3, Barrier=4, Basket=5 };
static constexpr int N_PAYOFFS = 6;

enum class ModelKind : int { GBM=0, Heston=1, Dupire=2, MultiDupire=3 };
static constexpr int N_MODELS = 4;

struct KernelParams {
    // Común a todos los modelos
    float S0, mu, T, h, discount;

    // GBM / Dupire
    float sigma;

    // Heston
    float kappa, theta, xi, rho, v0;
    float L_hes[4]; // Cholesky 2×2 fila-mayor: [0]=1, [1]=0, [2]=rho, [3]=sqrt(1-rho²)

    // Dupire local
    float sigma0, alpha, beta_d;

    // Payoff
    float K, B, r;
    PayoffKind payoff_kind;

    // Corrección BGK (Lookback, Barrier)
    bool  bgk_on;
    float sigma_bgk;

    // MLMC
    int M_refine;

    // Importance Sampling
    float z_star;

    // Variable de control
    float beta_cv, E_ctrl;

    // Multi-activo
    int n_assets;
};

__constant__ KernelParams c_p;

static constexpr int BLOCK_SIZE = 256;


// ---------------- //
// Auxiliares Euler //
// ---------------- //

// dS = μ·S·dt + σ·S·dW
__device__ __forceinline__
float d_euler_gbm(float S, float dw) {
    return S + c_p.mu * S * c_p.h + c_p.sigma * S * dw;
}

// dS = μ·S·dt + √v·S·dW₁;  dv = κ(θ-v)dt + ξ·√v·dW₂  (Milstein exacto en v)
__device__ __forceinline__
void d_euler_heston(float& S, float& V, float dw1, float dw2) {
    float Vp   = fmaxf(V, 0.0f);
    float sqVp = sqrtf(Vp);
    S = S + c_p.mu * S * c_p.h + sqVp * S * dw1;
    float em = expf(-c_p.kappa * c_p.h);
    V = c_p.theta + em * (V - c_p.theta) + c_p.xi * sqVp * dw2;
}

// dS = μ·S·dt + σ_loc(S,t)·S·dW,  σ_loc = σ₀·exp(-α·t)·(S/S₀)^(β-1)
__device__ __forceinline__
float d_euler_dupire(float S, float dw, float t) {
    float sigma_loc = c_p.sigma0 * expf(-c_p.alpha * t) * powf(S / c_p.S0, c_p.beta_d - 1.0f);
    return S + c_p.mu * S * c_p.h + sigma_loc * S * dw;
}


// ------------------------------ //
// Auxiliares para path-dependent //
// ------------------------------ //

// Valor inicial del acumulador según el tipo de payoff
template<PayoffKind PK>
__device__ __forceinline__
float d_running_init() {
    if constexpr (PK == PayoffKind::Lookback) return c_p.S0;
    else return 0.0f;
}

// Actualiza el acumulador tras cada paso de Euler
template<PayoffKind PK>
__device__ __forceinline__
void d_running_update(float& running, float S) {
    if constexpr (PK == PayoffKind::Asian)     running += S;
    else if constexpr (PK == PayoffKind::GeomAsian) running += logf(S);
    else if constexpr (PK == PayoffKind::Lookback)  running = fminf(running, S);
    else if constexpr (PK == PayoffKind::Barrier)   running = fmaxf(running, S);
}

// Indica si el payoff necesita toda la trayectoria
template<PayoffKind PK>
__device__ __forceinline__
constexpr bool d_is_path_dep() {
    return PK == PayoffKind::Asian     ||
           PK == PayoffKind::GeomAsian ||
           PK == PayoffKind::Lookback  ||
           PK == PayoffKind::Barrier;
}

// Aplica corrección BGK al acumulador (si corresponde)
template<PayoffKind PK>
__device__ __forceinline__
void d_apply_bgk(float& running, float h_step) {
    if (!c_p.bgk_on) return;
    float corr = expf(-BGK_BETA_F * c_p.sigma_bgk * sqrtf(h_step));
    if constexpr (PK == PayoffKind::Lookback) running *= corr;
    else if constexpr (PK == PayoffKind::Barrier) running /= corr;
}

// Payoff terminal
template<PayoffKind PK>
__device__ __forceinline__
float d_payoff(float S_T, float running, int n_steps) {
    if constexpr (PK == PayoffKind::European)
        return fmaxf(S_T - c_p.K, 0.0f) * c_p.discount;
    else if constexpr (PK == PayoffKind::Asian)
        return fmaxf(running / n_steps - c_p.K, 0.0f);
    else if constexpr (PK == PayoffKind::GeomAsian)
        return fmaxf(expf(running / n_steps) - c_p.K, 0.0f);
    else if constexpr (PK == PayoffKind::Lookback)
        return S_T - running;
    else if constexpr (PK == PayoffKind::Barrier)
        return (running >= c_p.B) ? 0.0f : fmaxf(S_T - c_p.K, 0.0f) * c_p.discount;
    else // Basket: el caller pasa la media de activos como S_T
        return fmaxf(S_T - c_p.K, 0.0f) * c_p.discount;
}


// ------------------------------- //
// Macros de reducción CUB         //
// ------------------------------- //

#define CUB_REDUCE2(val0, val1, d_s0, d_s1) \
    do { \
        using BR = cub::BlockReduce<double, BLOCK_SIZE>; \
        __shared__ typename BR::TempStorage _ts; \
        double _b; \
        _b = BR(_ts).Sum(val0); if (threadIdx.x==0) atomicAdd(d_s0, _b); \
        __syncthreads(); \
        _b = BR(_ts).Sum(val1); if (threadIdx.x==0) atomicAdd(d_s1, _b); \
    } while (0)

#define CUB_REDUCE4(val0, val1, val2, val3, d_s0, d_s1, d_s2, d_s3) \
    do { \
        using BR = cub::BlockReduce<double, BLOCK_SIZE>; \
        __shared__ typename BR::TempStorage _ts; \
        double _b; \
        _b = BR(_ts).Sum(val0); if (threadIdx.x==0) atomicAdd(d_s0, _b); \
        __syncthreads(); \
        _b = BR(_ts).Sum(val1); if (threadIdx.x==0) atomicAdd(d_s1, _b); \
        __syncthreads(); \
        _b = BR(_ts).Sum(val2); if (threadIdx.x==0) atomicAdd(d_s2, _b); \
        __syncthreads(); \
        _b = BR(_ts).Sum(val3); if (threadIdx.x==0) atomicAdd(d_s3, _b); \
    } while (0)


// ----------------------- //
// Un paso de Euler genérico //
// ----------------------- //

template<ModelKind MK>
__device__ __forceinline__
void d_step(float& S, float& V, const float* __restrict__ d_dW,
            int k, int p, int N_paths) {
    if constexpr (MK == ModelKind::GBM) {
        S = d_euler_gbm(S, d_dW[k * N_paths + p]);
    } else if constexpr (MK == ModelKind::Dupire) {
        S = d_euler_dupire(S, d_dW[k * N_paths + p], k * c_p.h);
    } else if constexpr (MK == ModelKind::Heston) {
        // d_dW contiene Z*sqrt_h (normales escaladas, sin correlacionar);
        // aplicamos Cholesky L para obtener los incrementos correlacionados.
        float z1  = d_dW[(k * 2 + 0) * N_paths + p];
        float z2  = d_dW[(k * 2 + 1) * N_paths + p];
        float dw1 = z1;                                          // L[0][0]=1, L[0][1]=0
        float dw2 = c_p.L_hes[2] * z1 + c_p.L_hes[3] * z2;    // L[1][0]=rho, L[1][1]=sqrt(1-rho²)
        d_euler_heston(S, V, dw1, dw2);
    }
    // MultiDupire se maneja en kernel_multi_dupire; no llega aquí
}


// ------------------------------------------ //
// Kernel MC de un único nivel (GBM/Heston/Dupire) //
// ------------------------------------------ //

// d_sums[2] = {sum_Y, sum_Y²}
template<ModelKind MK, PayoffKind PK>
__global__ void kernel_mc(
    const float* __restrict__ d_dW,
    double* d_sums,
    int N_paths, int N_steps)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float S       = (p < N_paths) ? c_p.S0 : 0.0f;
    float V       = (p < N_paths) ? c_p.v0 : 0.0f;
    float running = d_running_init<PK>();

    if (p < N_paths) {
        for (int k = 0; k < N_steps; k++) {
            d_step<MK>(S, V, d_dW, k, p, N_paths);
            if constexpr (d_is_path_dep<PK>())
                d_running_update<PK>(running, S);
        }
        if constexpr (d_is_path_dep<PK>())
            d_apply_bgk<PK>(running, c_p.h);
    }
    double Y  = (p < N_paths) ? (double)d_payoff<PK>(S, running, N_steps) : 0.0;
    double Y2 = Y * Y;
    CUB_REDUCE2(Y, Y2, d_sums, d_sums + 1);
}


// ------------------------------------------ //
// Kernel MLMC de nivel l (trayectorias acopladas) //
// ------------------------------------------ //

// d_sums[4] = {sum_delta, sum_delta², sum_fino, sum_fino²}
template<ModelKind MK, PayoffKind PK>
__global__ void kernel_mlmc(
    const float* __restrict__ d_Z,
    double* d_sums,
    int N_paths, int N_fine, int N_coarse, int M,
    float h_fine, float h_coarse, float sqrt_h_fine)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float Sf = (p < N_paths) ? c_p.S0 : 0.0f;
    float Vf = (p < N_paths) ? c_p.v0 : 0.0f;
    float Sc = Sf, Vc = Vf;
    float run_f = d_running_init<PK>(), run_c = run_f;
    float acc1 = 0.0f, acc2 = 0.0f;
    int coarse_k = 0;

    if (p < N_paths) {
        for (int k = 0; k < N_fine; k++) {
            // Paso fino
            if constexpr (MK == ModelKind::Heston) {
                float z1  = d_Z[(k * 2 + 0) * N_paths + p];
                float z2  = d_Z[(k * 2 + 1) * N_paths + p];
                float dw1 = z1 * sqrt_h_fine;
                float dw2 = (c_p.L_hes[2] * z1 + c_p.L_hes[3] * z2) * sqrt_h_fine;
                acc1 += dw1; acc2 += dw2;
                d_euler_heston(Sf, Vf, dw1, dw2);
            } else {
                float z  = d_Z[k * N_paths + p];
                float dw = z * sqrt_h_fine;
                acc1 += dw;
                if constexpr (MK == ModelKind::GBM)
                    Sf = d_euler_gbm(Sf, dw);
                else if constexpr (MK == ModelKind::Dupire)
                    Sf = d_euler_dupire(Sf, dw, k * h_fine);
                if constexpr (d_is_path_dep<PK>())
                    d_running_update<PK>(run_f, Sf);
            }

            // Paso grueso (cada M pasos finos): usar h_coarse para el drift
            if ((k + 1) % M == 0) {
                if constexpr (MK == ModelKind::Heston) {
                    float Vcp  = fmaxf(Vc, 0.0f);
                    float sqVcp = sqrtf(Vcp);
                    Sc = Sc + c_p.mu * Sc * h_coarse + sqVcp * Sc * acc1;
                    float emc = expf(-c_p.kappa * h_coarse);
                    Vc = c_p.theta + emc * (Vc - c_p.theta) + c_p.xi * sqVcp * acc2;
                    acc1 = acc2 = 0.0f;
                } else {
                    if constexpr (MK == ModelKind::GBM)
                        Sc = Sc + c_p.mu * Sc * h_coarse + c_p.sigma * Sc * acc1;
                    else if constexpr (MK == ModelKind::Dupire) {
                        float sigma_loc = c_p.sigma0 * expf(-c_p.alpha * (coarse_k * h_coarse))
                                        * powf(Sc / c_p.S0, c_p.beta_d - 1.0f);
                        Sc = Sc + c_p.mu * Sc * h_coarse + sigma_loc * Sc * acc1;
                    }
                    if constexpr (d_is_path_dep<PK>())
                        d_running_update<PK>(run_c, Sc);
                    acc1 = 0.0f;
                }
                ++coarse_k;
            }
        }
        // Corrección BGK con pasos distintos para nivel fino y grueso
        if constexpr (d_is_path_dep<PK>()) {
            d_apply_bgk<PK>(run_f, h_fine);
            d_apply_bgk<PK>(run_c, h_coarse);
        }
    }

    float pf = (p < N_paths) ? d_payoff<PK>(Sf, run_f, N_fine) : 0.0f;
    float pc = (p < N_paths && N_coarse > 0) ? d_payoff<PK>(Sc, run_c, N_coarse) : 0.0f;
    double dY = (double)(pf - pc), dY2 = dY * dY;
    double yf = (double)pf, yf2 = yf * yf;
    CUB_REDUCE4(dY, dY2, yf, yf2, d_sums, d_sums+1, d_sums+2, d_sums+3);
}


// ----------------------------------- //
// Kernel Dupire multi-activo terminal //
// ----------------------------------- //

// d_dW: [N_pasos × n_activos × N_caminos], layout dW[paso*n*N + activo*N + camino]
__global__ void kernel_multi_dupire(
    const float* __restrict__ d_dW,
    double* d_sums,
    int N_paths, int N_steps, int n_assets)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float S_mean = 0.0f;

    if (p < N_paths) {
        float running_mean = 0.0f;
        for (int a = 0; a < n_assets; a++) {
            float S = c_p.S0;
            for (int k = 0; k < N_steps; k++) {
                float dw = d_dW[(k * n_assets + a) * N_paths + p];
                float t  = k * c_p.h;
                S = d_euler_dupire(S, dw, t);
            }
            running_mean += S;
        }
        S_mean = running_mean / n_assets;
    }
    double Y  = (p < N_paths) ? (double)fmaxf(S_mean - c_p.K, 0.0f) * c_p.discount : 0.0;
    double Y2 = Y * Y;
    CUB_REDUCE2(Y, Y2, d_sums, d_sums + 1);
}


// ---------------------------------------------------------------------- //
// Kernel de transformada Brownian Bridge                                  //
// d_Z_in:     [N_pasos × N_caminos] normales en orden Sobol              //
// d_dW_out:   [N_pasos × N_caminos] incrementos brownianos en orden tpo. //
// d_W_scratch: [(N+1) × N_caminos] espacio de trabajo                    //
// ---------------------------------------------------------------------- //

__global__ void kernel_bb_transform(
    const float* __restrict__ d_Z_in,
    float*       __restrict__ d_dW_out,
    float*       __restrict__ d_W_scratch,
    const int*   __restrict__ d_map_idx,
    const int*   __restrict__ d_left_idx,
    const int*   __restrict__ d_right_idx,
    const float* __restrict__ d_wl,
    const float* __restrict__ d_wr,
    const float* __restrict__ d_std_dev,
    int N, int N_paths)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (p >= N_paths) return;

    // Punto terminal W[N] y frontera izquierda W[0] = 0
    d_W_scratch[d_map_idx[0] * N_paths + p] = d_std_dev[0] * d_Z_in[0 * N_paths + p];
    d_W_scratch[0 * N_paths + p] = 0.0f;

    for (int step = 1; step < N; step++) {
        int   m  = d_map_idx[step];
        int   l  = d_left_idx[step];
        int   r  = d_right_idx[step];
        float wl = d_wl[step], wr = d_wr[step], sd = d_std_dev[step];
        float z  = d_Z_in[step * N_paths + p];
        d_W_scratch[m * N_paths + p] = wl * d_W_scratch[l * N_paths + p]
                                     + wr * d_W_scratch[r * N_paths + p]
                                     + sd * z;
    }
    // dW[k] = W[k+1] - W[k]
    for (int k = 0; k < N; k++)
        d_dW_out[k * N_paths + p] =
            d_W_scratch[(k + 1) * N_paths + p] - d_W_scratch[k * N_paths + p];
}


// ---------------------------------------------------------------------- //
// Kernels de variable de control                                          //
// ---------------------------------------------------------------------- //

// Ej. 9: Asian aritmética (principal) + Asian geométrica (control), misma trayectoria GBM
// d_sums[4] = {sum_main, sum_main², sum_cv, sum_cv²}
__global__ void kernel_gbm_asian_cv(
    const float* __restrict__ d_dW,
    double* d_sums,
    int N_paths, int N_steps)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float S = (p < N_paths) ? c_p.S0 : 0.0f;
    float arith_sum = 0.0f, log_sum = 0.0f;

    if (p < N_paths) {
        for (int k = 0; k < N_steps; k++) {
            float dw = d_dW[k * N_paths + p];
            S = d_euler_gbm(S, dw);
            arith_sum += S;
            log_sum   += logf(S);
        }
    }
    float Y_arith = (p < N_paths) ? fmaxf(arith_sum / N_steps - c_p.K, 0.0f) : 0.0f;
    float Y_geom  = (p < N_paths) ? fmaxf(expf(log_sum / N_steps) - c_p.K, 0.0f) : 0.0f;
    // Guarda (p < N_paths): los hilos de relleno deben aportar 0 a TODOS los
    // acumuladores. Sin esta guarda, Y_cv = -beta·(0 - E_ctrl) = beta·E_ctrl
    // para los hilos sobrantes, sesgando la media cuando N_paths no es múltiplo
    // de BLOCK_SIZE.
    float Y_cv    = (p < N_paths) ? (Y_arith - c_p.beta_cv * (Y_geom - c_p.E_ctrl)) : 0.0f;

    using BR = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage ts;
    double v;
    // [0..3]: stats del main y del CV (para el run principal)
    v = BR(ts).Sum((double)Y_arith);            if (threadIdx.x==0) atomicAdd(d_sums+0, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_arith*Y_arith));  if (threadIdx.x==0) atomicAdd(d_sums+1, v); __syncthreads();
    v = BR(ts).Sum((double)Y_cv);               if (threadIdx.x==0) atomicAdd(d_sums+2, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_cv*Y_cv));        if (threadIdx.x==0) atomicAdd(d_sums+3, v); __syncthreads();
    // [4..6]: stats del control para calcular beta_opt en el piloto
    v = BR(ts).Sum((double)Y_geom);             if (threadIdx.x==0) atomicAdd(d_sums+4, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_geom*Y_geom));    if (threadIdx.x==0) atomicAdd(d_sums+5, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_arith*Y_geom));   if (threadIdx.x==0) atomicAdd(d_sums+6, v);
}

// Ej. 10: Dupire europeo (principal) + GBM (control), mismo Z para ambos modelos
__global__ void kernel_dupire_gbm_cv(
    const float* __restrict__ d_dW,
    double* d_sums,
    int N_paths, int N_steps)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float S_dup = (p < N_paths) ? c_p.S0 : 0.0f;
    float S_gbm = S_dup;

    if (p < N_paths) {
        for (int k = 0; k < N_steps; k++) {
            float dw = d_dW[k * N_paths + p];
            float t  = k * c_p.h;
            S_dup = d_euler_dupire(S_dup, dw, t);
            S_gbm = d_euler_gbm(S_gbm, dw);
        }
    }
    float Y_dup = (p < N_paths) ? fmaxf(S_dup - c_p.K, 0.0f) * c_p.discount : 0.0f;
    float Y_gbm = (p < N_paths) ? fmaxf(S_gbm - c_p.K, 0.0f) * c_p.discount : 0.0f;
    // Guarda (p < N_paths): sin ella los hilos de relleno aportan beta·E_ctrl a Y_cv.
    float Y_cv  = (p < N_paths) ? (Y_dup - c_p.beta_cv * (Y_gbm - c_p.E_ctrl)) : 0.0f;

    using BR = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage ts;
    double v;
    // [0..3]: stats del main y del CV
    v = BR(ts).Sum((double)Y_dup);              if (threadIdx.x==0) atomicAdd(d_sums+0, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_dup*Y_dup));      if (threadIdx.x==0) atomicAdd(d_sums+1, v); __syncthreads();
    v = BR(ts).Sum((double)Y_cv);               if (threadIdx.x==0) atomicAdd(d_sums+2, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_cv*Y_cv));        if (threadIdx.x==0) atomicAdd(d_sums+3, v); __syncthreads();
    // [4..6]: stats del control para calcular beta_opt en el piloto
    v = BR(ts).Sum((double)Y_gbm);              if (threadIdx.x==0) atomicAdd(d_sums+4, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_gbm*Y_gbm));      if (threadIdx.x==0) atomicAdd(d_sums+5, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_dup*Y_gbm));      if (threadIdx.x==0) atomicAdd(d_sums+6, v);
}


// ---------------------------------------------------------------------- //
// Kernel de Importance Sampling (Ej. 11, GBM, call muy fuera del dinero) //
// Z desplazada: Z' = Z + z_star;  LR = exp(-z_star·ΣZ_k - N·z_star²/2)  //
// ---------------------------------------------------------------------- //

__global__ void kernel_is_gbm(
    const float* __restrict__ d_Z,
    double* d_sums,
    int N_paths, int N_steps)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float S     = (p < N_paths) ? c_p.S0 : 0.0f;
    float z_sum = 0.0f;

    if (p < N_paths) {
        for (int k = 0; k < N_steps; k++) {
            float z  = d_Z[k * N_paths + p];
            z_sum   += z;
            float dw = (z + c_p.z_star) * sqrtf(c_p.h);
            S = d_euler_gbm(S, dw);
        }
    }
    float lr     = (p < N_paths) ? expf(-c_p.z_star * z_sum
                                        - 0.5f * c_p.z_star * c_p.z_star * N_steps) : 1.0f;
    float payoff = (p < N_paths) ? fmaxf(S - c_p.K, 0.0f) * c_p.discount * lr : 0.0f;
    double Y = (double)payoff, Y2 = Y * Y;
    CUB_REDUCE2(Y, Y2, d_sums, d_sums + 1);
}


// ------------------------------------ //
// Auxiliares para escalar y convertir  //
// ------------------------------------ //

// Escala d[i] *= scalar en dispositivo
__global__ void kernel_scale(float* d, float scalar, long long n) {
    long long i = (long long)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) d[i] *= scalar;
}

// Convierte float32 a float16 en dispositivo
__global__ void kernel_cast_f32_to_f16(const float* src, __half* dst, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}


// ------------------------------------------- //
// Tablas de despacho [modelo × payoff]         //
// ------------------------------------------- //

using McKernelFn   = void(*)(const float*, double*, int, int);
using MlmcKernelFn = void(*)(const float*, double*, int, int, int, int, float, float, float);

template<ModelKind MK>
struct McRow {
    static constexpr McKernelFn fns[N_PAYOFFS] = {
        &kernel_mc<MK, PayoffKind::European>,
        &kernel_mc<MK, PayoffKind::Asian>,
        &kernel_mc<MK, PayoffKind::GeomAsian>,
        &kernel_mc<MK, PayoffKind::Lookback>,
        &kernel_mc<MK, PayoffKind::Barrier>,
        &kernel_mc<MK, PayoffKind::Basket>,
    };
};

template<ModelKind MK>
struct MlmcRow {
    static constexpr MlmcKernelFn fns[N_PAYOFFS] = {
        &kernel_mlmc<MK, PayoffKind::European>,
        &kernel_mlmc<MK, PayoffKind::Asian>,
        &kernel_mlmc<MK, PayoffKind::GeomAsian>,
        &kernel_mlmc<MK, PayoffKind::Lookback>,
        &kernel_mlmc<MK, PayoffKind::Barrier>,
        &kernel_mlmc<MK, PayoffKind::Basket>,
    };
};

static const McKernelFn MC_TABLE[N_MODELS][N_PAYOFFS] = {
    { McRow<ModelKind::GBM>::fns[0],        McRow<ModelKind::GBM>::fns[1],
      McRow<ModelKind::GBM>::fns[2],        McRow<ModelKind::GBM>::fns[3],
      McRow<ModelKind::GBM>::fns[4],        McRow<ModelKind::GBM>::fns[5] },
    { McRow<ModelKind::Heston>::fns[0],     McRow<ModelKind::Heston>::fns[1],
      McRow<ModelKind::Heston>::fns[2],     McRow<ModelKind::Heston>::fns[3],
      McRow<ModelKind::Heston>::fns[4],     McRow<ModelKind::Heston>::fns[5] },
    { McRow<ModelKind::Dupire>::fns[0],     McRow<ModelKind::Dupire>::fns[1],
      McRow<ModelKind::Dupire>::fns[2],     McRow<ModelKind::Dupire>::fns[3],
      McRow<ModelKind::Dupire>::fns[4],     McRow<ModelKind::Dupire>::fns[5] },
    { McRow<ModelKind::MultiDupire>::fns[0], McRow<ModelKind::MultiDupire>::fns[1],
      McRow<ModelKind::MultiDupire>::fns[2], McRow<ModelKind::MultiDupire>::fns[3],
      McRow<ModelKind::MultiDupire>::fns[4], McRow<ModelKind::MultiDupire>::fns[5] },
};

static const MlmcKernelFn MLMC_TABLE[N_MODELS][N_PAYOFFS] = {
    { MlmcRow<ModelKind::GBM>::fns[0],     MlmcRow<ModelKind::GBM>::fns[1],
      MlmcRow<ModelKind::GBM>::fns[2],     MlmcRow<ModelKind::GBM>::fns[3],
      MlmcRow<ModelKind::GBM>::fns[4],     MlmcRow<ModelKind::GBM>::fns[5] },
    { MlmcRow<ModelKind::Heston>::fns[0],  MlmcRow<ModelKind::Heston>::fns[1],
      MlmcRow<ModelKind::Heston>::fns[2],  MlmcRow<ModelKind::Heston>::fns[3],
      MlmcRow<ModelKind::Heston>::fns[4],  MlmcRow<ModelKind::Heston>::fns[5] },
    { MlmcRow<ModelKind::Dupire>::fns[0],  MlmcRow<ModelKind::Dupire>::fns[1],
      MlmcRow<ModelKind::Dupire>::fns[2],  MlmcRow<ModelKind::Dupire>::fns[3],
      MlmcRow<ModelKind::Dupire>::fns[4],  MlmcRow<ModelKind::Dupire>::fns[5] },
    { MlmcRow<ModelKind::GBM>::fns[0],     MlmcRow<ModelKind::GBM>::fns[1],
      MlmcRow<ModelKind::GBM>::fns[2],     MlmcRow<ModelKind::GBM>::fns[3],
      MlmcRow<ModelKind::GBM>::fns[4],     MlmcRow<ModelKind::GBM>::fns[5] },
};


// ---------------------------- //
// DeviceBBData y DevicePCAData //
// ---------------------------- //

struct DeviceBBData {
    int    N;
    int*   d_map_idx   = nullptr;
    int*   d_left_idx  = nullptr;
    int*   d_right_idx = nullptr;
    float* d_wl        = nullptr;
    float* d_wr        = nullptr;
    float* d_std_dev   = nullptr;
};

struct DevicePCAData {
    int     m;
    __half* d_M_pca_f16 = nullptr; // Matriz PCA [m×m] en fp16
};

DeviceBBData* bb_upload(const BBData& bb) {
    auto* d = new DeviceBBData;
    d->N = bb.N;
    int N = bb.N;
    CUDA_CHECK(cudaMalloc(&d->d_map_idx,   N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->d_left_idx,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->d_right_idx, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->d_wl,        N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d->d_wr,        N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d->d_std_dev,   N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d->d_map_idx,   bb.map_idx.data(),   N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_left_idx,  bb.left_idx.data(),  N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_right_idx, bb.right_idx.data(), N*sizeof(int), cudaMemcpyHostToDevice));

    std::vector<float> wl(N), wr(N), sd(N);
    for (int i = 0; i < N; i++) {
        wl[i] = (float)bb.weight_left[i];
        wr[i] = (float)bb.weight_right[i];
        sd[i] = (float)bb.std_dev[i];
    }
    CUDA_CHECK(cudaMemcpy(d->d_wl,      wl.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_wr,      wr.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_std_dev, sd.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

void bb_free(DeviceBBData* d) {
    if (!d) return;
    cudaFree(d->d_map_idx); cudaFree(d->d_left_idx); cudaFree(d->d_right_idx);
    cudaFree(d->d_wl); cudaFree(d->d_wr); cudaFree(d->d_std_dev);
    delete d;
}

DevicePCAData* pca_upload(const PCAData& pca) {
    auto* d = new DevicePCAData;
    d->m = pca.m;
    int sz = pca.m * pca.m;
    std::vector<__half> hf(sz);
    for (int i = 0; i < sz; i++) hf[i] = __float2half(pca.M_pca_f32[i]);
    CUDA_CHECK(cudaMalloc(&d->d_M_pca_f16, sz * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d->d_M_pca_f16, hf.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    return d;
}

void pca_free(DevicePCAData* d) {
    if (!d) return;
    cudaFree(d->d_M_pca_f16);
    delete d;
}


// ----------------------------------------- //
// Auxiliares internos — construir c_p       //
// ----------------------------------------- //

using Clock = std::chrono::high_resolution_clock;

static KernelParams make_params(const ModelVariant& model, const PayoffVariant& payoff,
                                int n_steps) {
    KernelParams p{};
    p.payoff_kind = PayoffKind::European;

    // Payoff
    std::visit([&](const auto& pv) {
        using T = std::decay_t<decltype(pv)>;
        if constexpr (std::is_same_v<T, European>) {
            p.payoff_kind = PayoffKind::European;
            p.K = (float)pv.K; p.r = (float)pv.r;
            p.discount = (float)std::exp(-pv.r * pv.T);
        }
        if constexpr (std::is_same_v<T, Asian>) {
            p.payoff_kind = PayoffKind::Asian; p.K = (float)pv.K; p.discount = 1.0f;
        }
        if constexpr (std::is_same_v<T, GeomAsian>) {
            p.payoff_kind = PayoffKind::GeomAsian; p.K = (float)pv.K; p.discount = 1.0f;
        }
        if constexpr (std::is_same_v<T, Lookback>) {
            p.payoff_kind = PayoffKind::Lookback; p.discount = 1.0f;
            p.bgk_on = true; p.sigma_bgk = (float)pv.sigma;
        }
        if constexpr (std::is_same_v<T, Barrier>) {
            p.payoff_kind = PayoffKind::Barrier;
            p.K = (float)pv.K; p.B = (float)pv.B; p.r = (float)pv.r;
            p.discount = (float)std::exp(-pv.r * pv.T);
            p.bgk_on = true; p.sigma_bgk = (float)pv.sigma;
        }
        if constexpr (std::is_same_v<T, Basket>) {
            p.payoff_kind = PayoffKind::Basket;
            p.K = (float)pv.K; p.r = (float)pv.r;
            p.discount = (float)std::exp(-pv.r * pv.T);
            p.n_assets = pv.n_assets;
        }
    }, payoff);

    // Modelo
    std::visit([&](const auto& m) {
        using T = std::decay_t<decltype(m)>;
        p.T = (float)m.T;
        p.h = p.T / n_steps;
        if constexpr (std::is_same_v<T, GBMParams>) {
            p.S0 = (float)m.S0; p.mu = (float)m.mu; p.sigma = (float)m.sigma;
        }
        if constexpr (std::is_same_v<T, HestonParams>) {
            p.S0 = (float)m.S0; p.mu = (float)m.mu;
            p.kappa = (float)m.kappa; p.theta = (float)m.theta;
            p.xi    = (float)m.xi;    p.rho   = (float)m.rho;
            p.v0    = (float)m.v0;
            p.L_hes[0] = (float)m.L[0][0]; p.L_hes[1] = (float)m.L[0][1];
            p.L_hes[2] = (float)m.L[1][0]; p.L_hes[3] = (float)m.L[1][1];
        }
        if constexpr (std::is_same_v<T, DupireLocalParams>) {
            p.S0 = (float)m.S0; p.mu = (float)m.mu;
            p.sigma0 = (float)m.sigma0; p.alpha = (float)m.alpha; p.beta_d = (float)m.beta_d;
            p.sigma  = (float)m.sigma0; // sigma_eff para el kernel GBM de variable de control
        }
        if constexpr (std::is_same_v<T, MultiDupireParams>) {
            p.S0 = (float)(m.S0.empty() ? 100.0 : m.S0[0]);
            p.mu = (float)m.mu;
            p.sigma0 = (float)m.sigma0; p.alpha = (float)m.alpha; p.beta_d = (float)m.beta_d;
            p.n_assets = m.n;
        }
    }, model);

    return p;
}

// Identifica el tipo de modelo (para despacho de kernel)
static ModelKind model_kind(const ModelVariant& mv) {
    return std::visit([](const auto& m) -> ModelKind {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, GBMParams>)          return ModelKind::GBM;
        if constexpr (std::is_same_v<T, HestonParams>)        return ModelKind::Heston;
        if constexpr (std::is_same_v<T, DupireLocalParams>)   return ModelKind::Dupire;
        if constexpr (std::is_same_v<T, MultiDupireParams>)   return ModelKind::MultiDupire;
        return ModelKind::GBM;
    }, mv);
}

// Identifica el tipo de payoff (índice para la tabla de despacho)
static int payoff_idx(const PayoffVariant& pv) {
    return std::visit([](const auto& p) -> int {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, European>)  return 0;
        if constexpr (std::is_same_v<T, Asian>)     return 1;
        if constexpr (std::is_same_v<T, GeomAsian>) return 2;
        if constexpr (std::is_same_v<T, Lookback>)  return 3;
        if constexpr (std::is_same_v<T, Barrier>)   return 4;
        if constexpr (std::is_same_v<T, Basket>)    return 5;
        return 0;
    }, pv);
}

// Lanza el kernel MC de un nivel; el caller debe sincronizar después
static void launch_mc_kernel(ModelKind mk, int pidx, bool is_multi,
                             const float* d_dW, double* d_sums,
                             int N_paths, int N_steps, int n_assets) {
    int blocks = (N_paths + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (is_multi) {
        kernel_multi_dupire<<<blocks, BLOCK_SIZE>>>(d_dW, d_sums, N_paths, N_steps, n_assets);
    } else {
        MC_TABLE[(int)mk][pidx]<<<blocks, BLOCK_SIZE>>>(d_dW, d_sums, N_paths, N_steps);
    }
}

// Fase 1 PCA con Tensor Cores (F16→F32): dW = M_pca[D×D] @ Z[D×N]
static void phase1_pca(cublasHandle_t cublas,
                       const __half* d_M_pca, const __half* d_Z_f16,
                       float* d_dW_f32, int D, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, D, D,
        &alpha,
        d_Z_f16,  CUDA_R_16F, N,
        d_M_pca,  CUDA_R_16F, D,
        &beta,
        d_dW_f32, CUDA_R_32F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


// -------------------------------------------- //
// run_mc_fixed_impl — bloque constructor interno //
// -------------------------------------------- //

static std::pair<double,double> run_mc_fixed_impl(
    const ModelVariant& model, const PayoffVariant& payoff,
    int n_steps, long long n_paths, unsigned seed,
    NoiseMode mode        = NoiseMode::Raw,
    DeviceBBData*  dev_bb  = nullptr,
    DevicePCAData* dev_pca = nullptr,
    cublasHandle_t cublas  = nullptr)
{
    n_paths = (n_paths + 1) & ~1LL;  // cuRAND exige count par
    int       d_noise  = model_noise_dim(model);
    long long D        = (long long)n_steps * d_noise;
    ModelKind mk       = model_kind(model);
    bool      path_dep = payoff_needs_full_path(payoff);
    int       pidx     = payoff_idx(payoff);
    bool      is_multi = (mk == ModelKind::MultiDupire);
    int       n_assets = is_multi ? std::get<MultiDupireParams>(model).n : 1;

    KernelParams kp = make_params(model, payoff, n_steps);
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    float*  d_Z    = nullptr;
    float*  d_dW   = nullptr;
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Z,    D * n_paths * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW,   D * n_paths * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 4 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 4 * sizeof(double)));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateNormal(gen, d_Z, D * n_paths, 0.0f, 1.0f));
    curandDestroyGenerator(gen);

    float     sqrt_h      = sqrtf(kp.h);
    long long total_elems = D * n_paths;
    int scale_blk = (int)((total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (mode == NoiseMode::Raw) {
        CUDA_CHECK(cudaMemcpy(d_dW, d_Z, total_elems*sizeof(float), cudaMemcpyDeviceToDevice));
        kernel_scale<<<scale_blk, BLOCK_SIZE>>>(d_dW, sqrt_h, total_elems);
    } else if (mode == NoiseMode::BrownianBridge && dev_bb) {
        float* d_W_scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&d_W_scratch, (n_steps + 1) * n_paths * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_W_scratch, 0, (n_steps+1)*n_paths*sizeof(float)));
        int blk = ((int)n_paths + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_bb_transform<<<blk, BLOCK_SIZE>>>(
            d_Z, d_dW, d_W_scratch,
            dev_bb->d_map_idx, dev_bb->d_left_idx, dev_bb->d_right_idx,
            dev_bb->d_wl, dev_bb->d_wr, dev_bb->d_std_dev,
            n_steps, (int)n_paths);
        cudaFree(d_W_scratch);
    } else if (mode == NoiseMode::PCA && dev_pca && cublas) {
        __half* d_Z_f16 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Z_f16, total_elems * sizeof(__half)));
        kernel_cast_f32_to_f16<<<scale_blk, BLOCK_SIZE>>>(d_Z, d_Z_f16, (int)total_elems);
        phase1_pca(cublas, dev_pca->d_M_pca_f16, d_Z_f16, d_dW, (int)D, (int)n_paths);
        cudaFree(d_Z_f16);
    }

    launch_mc_kernel(mk, pidx, is_multi, d_dW, d_sums, (int)n_paths, n_steps, n_assets);
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_sums[4] = {};
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums, 4*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_Z); cudaFree(d_dW); cudaFree(d_sums);

    double mean = h_sums[0] / n_paths;
    double var  = std::max(0.0, (h_sums[1]/n_paths - mean*mean)) / n_paths;
    return {mean, var};
}


// ================= //
// run_mc_cuda       //
// ================= //

MCResult run_mc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                     double eps, int n_steps, const MCConfig& cfg) {
    auto t0 = Clock::now();

    auto [mu_pilot, var_pilot] = run_mc_fixed_impl(model, payoff, n_steps,
                                                    cfg.pilot_n, cfg.seed);
    double sample_var = var_pilot * cfg.pilot_n;
    long long N_needed = (long long)std::ceil(2.0 * sample_var / (eps * eps));
    N_needed = std::max(N_needed, (long long)cfg.pilot_n);

    double    sum_Y  = 0.0, sum_Y2 = 0.0;
    long long N_done = 0;
    unsigned  cur_seed = cfg.seed + 1;

    while (N_done < N_needed) {
        long long batch = std::min((long long)cfg.N_batch, N_needed - N_done);
        auto [bm, bv] = run_mc_fixed_impl(model, payoff, n_steps, batch, cur_seed++);
        sum_Y  += bm * batch;
        sum_Y2 += (bv * batch + bm * bm) * batch;
        N_done += batch;
    }

    double mean = sum_Y / N_done;
    double var  = std::max(0.0, sum_Y2 / N_done - mean * mean) / N_done;
    double t_s  = std::chrono::duration<double>(Clock::now() - t0).count();
    return {mean, std::sqrt(var), N_done, t_s};
}


// ================= //
// run_mlmc_cuda     //
// ================= //

MCResult run_mlmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                       double eps, const MLMCConfig& cfg) {
    auto t0 = Clock::now();

    int       M        = cfg.M;
    int       max_L    = cfg.max_L;
    bool      path_dep = payoff_needs_full_path(payoff);
    ModelKind mk       = model_kind(model);
    int       pidx     = payoff_idx(payoff);
    bool      is_multi = (mk == ModelKind::MultiDupire);
    int       n_assets = is_multi ? std::get<MultiDupireParams>(model).n : 1;
    double    T        = model_T(model);

    int L = 2;
    std::vector<double>    E_l(max_L+1, 0.0), V_l(max_L+1, 0.0);
    std::vector<long long> N_l(max_L+1, 0);

    // Ejecuta n_paths en el nivel l del MLMC; devuelve {sum_dY, sum_dY², sum_f, sum_f²}
    auto run_level = [&](int l, long long n_paths, unsigned seed) -> std::array<double,4> {
        n_paths = (n_paths + 1) & ~1LL;  // cuRAND exige count par
        int   N_fine   = (l == 0) ? 1 : (int)std::round(std::pow(M, l));
        int   N_coarse = (l == 0) ? 0 : N_fine / M;
        float h_fine   = (float)(T / N_fine);
        float h_coarse = (float)(T / std::max(N_coarse, 1));
        float sqrt_hf  = sqrtf(h_fine);
        int   d_noise  = model_noise_dim(model);
        long long D    = (long long)N_fine * d_noise;

        KernelParams kp = make_params(model, payoff, N_fine);
        kp.M_refine = M;
        CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

        // Sampling por lotes: acota la memoria a D·batch_cap y, crucialmente,
        // evita que (int)n_paths desborde cuando N_opt supera 2.1e9 a eps pequeño
        // (causaba "illegal memory access" en el kernel MLMC). El kernel acumula
        // en d_sums vía atomicAdd, así que basta no resetear entre lotes.
        const long long MAX_BATCH_FLOATS = 1LL << 26;  // ~256 MB de Z por lote
        long long batch_cap = std::max(2LL, MAX_BATCH_FLOATS / std::max(D, 1LL));
        batch_cap = (batch_cap + 1) & ~1LL;
        batch_cap = std::min(batch_cap, n_paths);

        float*  d_Z    = nullptr;
        double* d_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Z,    D * batch_cap * sizeof(float) + 2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sums, 4 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sums, 0, 4 * sizeof(double)));

        curandGenerator_t gen;
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

        long long done = 0;
        while (done < n_paths) {
            long long batch = std::min(batch_cap, n_paths - done);
            long long gen_count = (D * batch + 1) & ~1LL;  // cuRAND exige count par
            CURAND_CHECK(curandGenerateNormal(gen, d_Z, gen_count, 0.0f, 1.0f));
            int blocks = (int)((batch + BLOCK_SIZE - 1) / BLOCK_SIZE);
            MLMC_TABLE[(int)mk][pidx]<<<blocks, BLOCK_SIZE>>>(
                d_Z, d_sums, (int)batch, N_fine, N_coarse, M, h_fine, h_coarse, sqrt_hf);
            done += batch;
        }
        curandDestroyGenerator(gen);
        CUDA_CHECK(cudaDeviceSynchronize());

        double hs[4] = {};
        CUDA_CHECK(cudaMemcpy(hs, d_sums, 4*sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_Z); cudaFree(d_sums);
        return {hs[0], hs[1], hs[2], hs[3]};
    };

    // Piloto inicial
    unsigned seed_ctr = 42u;
    for (int l = 0; l <= L; l++) {
        auto s = run_level(l, cfg.pilot_n, seed_ctr++);
        long long np = cfg.pilot_n;
        double em = s[0] / np;
        V_l[l] = std::max(0.0, s[1]/np - em*em);
        E_l[l] = em;
        N_l[l] = np;
    }

    // Bucle adaptativo MLMC (Giles 2008, Teorema 1)
    bool converged = false;
    int  iter = 0;
    while (!converged && L <= max_L && iter++ < 50) {
        double sum_term = 0.0;
        for (int l = 0; l <= L; l++)
            sum_term += std::sqrt(V_l[l] * std::pow((double)M, l));

        std::vector<long long> N_opt(L+1);
        for (int l = 0; l <= L; l++) {
            double C_l = std::pow((double)M, l);
            N_opt[l] = (long long)std::ceil(2.0/(eps*eps) * std::sqrt(V_l[l]/C_l) * sum_term);
            N_opt[l] = std::max(N_opt[l], 100LL);
        }

        for (int l = 0; l <= L; l++) {
            if (N_opt[l] > N_l[l]) {
                long long extra = N_opt[l] - N_l[l];
                auto s = run_level(l, extra, seed_ctr++);
                long long N_new   = N_l[l] + extra;
                double total_sum  = E_l[l] * N_l[l] + s[0];
                double total_s2   = (V_l[l] + E_l[l]*E_l[l]) * N_l[l] + s[1];
                E_l[l] = total_sum / N_new;
                V_l[l] = std::max(0.0, total_s2/N_new - E_l[l]*E_l[l]);
                N_l[l] = N_new;
            }
        }

        // Criterio de convergencia: sesgo estimado < eps/sqrt(2)
        double bias_est = std::abs(E_l[L]) / (M - 1);
        if (L >= 1) bias_est = std::max(bias_est, std::abs(E_l[L-1]) * M / (M*M - M));
        converged = (bias_est < eps / std::sqrt(2.0));

        if (!converged && L < max_L) {
            L++;
            auto s = run_level(L, cfg.pilot_n, seed_ctr++);
            long long np = cfg.pilot_n;
            double em = s[0] / np;
            V_l[L] = std::max(0.0, s[1]/np - em*em);
            E_l[L] = em;
            N_l[L] = np;
        }
    }

    // Estimador final: suma telescópica Σ E_l
    double price = 0.0, var_sum = 0.0;
    long long N_total = 0;
    for (int l = 0; l <= L; l++) {
        price   += E_l[l];
        var_sum += (N_l[l] > 0 ? V_l[l] / N_l[l] : 0.0);
        N_total += N_l[l];
    }
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {price, std::sqrt(var_sum), N_total, t_s};
}


// ================= //
// run_qmc_cuda      //
// ================= //

MCResult run_qmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                      double eps, int n_steps,
                      const QMCConfig& cfg, NoiseMode mode,
                      DeviceBBData* dev_bb, DevicePCAData* dev_pca) {
    auto t0 = Clock::now();
    int       R       = cfg.R;
    int       d_noise = model_noise_dim(model);
    long long D       = (long long)n_steps * d_noise;
    ModelKind mk      = model_kind(model);
    int       pidx    = payoff_idx(payoff);
    bool      is_multi = (mk == ModelKind::MultiDupire);
    int       n_assets = is_multi ? std::get<MultiDupireParams>(model).n : 1;

    cublasHandle_t cublas = nullptr;
    if (mode == NoiseMode::PCA) { CUBLAS_CHECK(cublasCreate(&cublas)); }

    long long N_per_replica = 64;
    double var_of_means = 1e30, grand_mean = 0.0;
    long long total_N = 0;
    std::vector<double> replica_means(R);

    // Lote de puntos por réplica: acota la memoria a D·chunk_cap sin limitar el
    // numero total de puntos. Antes habia un techo (MAX_ALLOC) que cortaba el
    // bucle y dejaba el QMC sin converger a eps finos; con batching desaparece.
    bool use_sobol = (D <= D_MAX_SOBOL);
    const long long MAX_CHUNK_FLOATS = 64LL * 1024 * 1024;  // ~256 MB de Z por lote
    long long chunk_cap = std::max((long long)BLOCK_SIZE, MAX_CHUNK_FLOATS / std::max(D, 1LL));
    chunk_cap = (chunk_cap + 1) & ~1LL;

    for (int doublings = 0; doublings < cfg.max_doublings; doublings++) {
        for (int r = 0; r < R; r++) {
            // Una réplica = una secuencia Sobol independiente (offset r*N_per_replica).
            // El kernel MC acumula en d_sums vía atomicAdd, así que la media de la
            // réplica se trocea en lotes de puntos sin resetear d_sums entre ellos.
            double* d_sums = nullptr;
            CUDA_CHECK(cudaMalloc(&d_sums, 4*sizeof(double)));
            CUDA_CHECK(cudaMemset(d_sums, 0, 4*sizeof(double)));

            long long done = 0;
            while (done < N_per_replica) {
                long long chunk     = std::min(chunk_cap, N_per_replica - done);
                long long gen_count = (D * chunk + 1) & ~1LL;  // cuRAND exige count par

                float* d_Z_r = nullptr;
                CUDA_CHECK(cudaMalloc(&d_Z_r, gen_count * sizeof(float) + 2*sizeof(float)));

                curandGenerator_t gen;
                if (use_sobol) {
                    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
                    CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(gen, (unsigned)D));
                    CURAND_CHECK(curandSetGeneratorOffset(gen,
                        (unsigned long long)(r * N_per_replica + done)));
                } else {
                    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
                    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42u + r));
                    CURAND_CHECK(curandSetGeneratorOffset(gen, (unsigned long long)done * D));
                }
                CURAND_CHECK(curandGenerateNormal(gen, d_Z_r, gen_count, 0.0f, 1.0f));
                curandDestroyGenerator(gen);

                float* d_dW_r = nullptr;
                CUDA_CHECK(cudaMalloc(&d_dW_r, D * chunk * sizeof(float)));

                if (mode == NoiseMode::BrownianBridge && dev_bb) {
                    float* d_W_scratch = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_W_scratch, (n_steps+1)*chunk*sizeof(float)));
                    CUDA_CHECK(cudaMemset(d_W_scratch, 0, (n_steps+1)*chunk*sizeof(float)));
                    int blk = ((int)chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    kernel_bb_transform<<<blk, BLOCK_SIZE>>>(
                        d_Z_r, d_dW_r, d_W_scratch,
                        dev_bb->d_map_idx, dev_bb->d_left_idx, dev_bb->d_right_idx,
                        dev_bb->d_wl, dev_bb->d_wr, dev_bb->d_std_dev,
                        n_steps, (int)chunk);
                    cudaFree(d_W_scratch);
                } else if (mode == NoiseMode::PCA && dev_pca) {
                    __half* d_Z_f16 = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_Z_f16, D * chunk * sizeof(__half)));
                    int blk2 = ((int)(D*chunk)+BLOCK_SIZE-1)/BLOCK_SIZE;
                    kernel_cast_f32_to_f16<<<blk2,BLOCK_SIZE>>>(d_Z_r, d_Z_f16, (int)(D*chunk));
                    phase1_pca(cublas, dev_pca->d_M_pca_f16, d_Z_f16, d_dW_r, (int)D, (int)chunk);
                    cudaFree(d_Z_f16);
                } else {
                    // Modo Raw: escalar Z → dW
                    CUDA_CHECK(cudaMemcpy(d_dW_r, d_Z_r, D*chunk*sizeof(float), cudaMemcpyDeviceToDevice));
                    KernelParams kp = make_params(model, payoff, n_steps);
                    kernel_scale<<<((int)(D*chunk)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
                        d_dW_r, sqrtf(kp.h), D*chunk);
                }

                KernelParams kp = make_params(model, payoff, n_steps);
                CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));
                launch_mc_kernel(mk, pidx, is_multi, d_dW_r, d_sums, (int)chunk, n_steps, n_assets);
                CUDA_CHECK(cudaDeviceSynchronize());

                cudaFree(d_dW_r);
                cudaFree(d_Z_r);
                done += chunk;
            }

            double hs[4] = {};
            CUDA_CHECK(cudaMemcpy(hs, d_sums, 4*sizeof(double), cudaMemcpyDeviceToHost));
            replica_means[r] = hs[0] / N_per_replica;
            cudaFree(d_sums);
        }

        double m = 0.0;
        for (double v : replica_means) m += v;
        m /= R;
        double v = 0.0;
        for (double rv : replica_means) v += (rv - m) * (rv - m);
        v /= (R - 1);
        var_of_means = v / R;
        grand_mean   = m;
        total_N      = (long long)R * N_per_replica;

        if (var_of_means < eps * eps / 2.0) break;
        N_per_replica *= 2;
    }

    if (cublas) cublasDestroy(cublas);
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {grand_mean, std::sqrt(var_of_means), total_N, t_s};
}


// ================== //
// run_mlqmc_cuda     //
// ================== //

MCResult run_mlqmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                        double eps,
                        const MLMCConfig& ml_cfg, const QMCConfig& qmc_cfg,
                        NoiseMode mode,
                        std::vector<DeviceBBData*>  bb_list,
                        std::vector<DevicePCAData*> pca_list) {
    auto t0 = Clock::now();

    int       M        = ml_cfg.M;
    int       max_L    = ml_cfg.max_L;
    double    T        = model_T(model);
    ModelKind mk       = model_kind(model);
    int       pidx     = payoff_idx(payoff);

    int L = 2;
    std::vector<double>    E_l(max_L+1, 0.0), V_l(max_L+1, 0.0);
    std::vector<long long> N_l(max_L+1, 0);

    auto run_qmc_level = [&](int l, long long N_per_rep, unsigned) -> std::pair<double,double> {
        int   N_fine   = (l == 0) ? 1 : (int)std::round(std::pow(M, l));
        int   N_coarse = (l == 0) ? 0 : N_fine / M;
        float h_fine   = (float)(T / N_fine);
        float h_coarse = (float)(T / std::max(N_coarse, 1));
        float sqrt_hf  = sqrtf(h_fine);
        int   d_noise  = model_noise_dim(model);
        long long D    = (long long)N_fine * d_noise;

        int R = qmc_cfg.R;
        std::vector<double> rmeans(R);
        bool use_sobol = (D <= D_MAX_SOBOL);
        long long gen_count = ((D * N_per_rep) + 1) & ~1LL; // cuRAND exige count par

        for (int r = 0; r < R; r++) {
            float* d_Z = nullptr;
            CUDA_CHECK(cudaMalloc(&d_Z, gen_count * sizeof(float)));

            curandGenerator_t gen;
            if (use_sobol) {
                CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
                CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(gen, (unsigned)D));
                CURAND_CHECK(curandSetGeneratorOffset(gen, (unsigned long long)r * N_per_rep));
            } else {
                CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
                CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42u + r));
            }
            CURAND_CHECK(curandGenerateNormal(gen, d_Z, gen_count, 0.0f, 1.0f));
            curandDestroyGenerator(gen);

            KernelParams kp = make_params(model, payoff, N_fine);
            kp.M_refine = M;
            CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

            double* d_sums = nullptr;
            CUDA_CHECK(cudaMalloc(&d_sums, 4*sizeof(double)));
            CUDA_CHECK(cudaMemset(d_sums, 0, 4*sizeof(double)));

            int blocks = ((int)N_per_rep + BLOCK_SIZE - 1) / BLOCK_SIZE;
            MLMC_TABLE[(int)mk][pidx]<<<blocks, BLOCK_SIZE>>>(
                d_Z, d_sums, (int)N_per_rep, N_fine, N_coarse, M, h_fine, h_coarse, sqrt_hf);
            CUDA_CHECK(cudaDeviceSynchronize());

            double hs[4] = {};
            CUDA_CHECK(cudaMemcpy(hs, d_sums, 4*sizeof(double), cudaMemcpyDeviceToHost));
            rmeans[r] = hs[0] / N_per_rep;

            cudaFree(d_Z); cudaFree(d_sums);
        }

        double m = 0.0;
        for (double v : rmeans) m += v;
        m /= R;
        double v = 0.0;
        for (double rv : rmeans) v += (rv - m)*(rv - m);
        v = (R > 1) ? v / (R - 1) : 0.0;
        return {m, v};
    };

    long long N_pilot = ml_cfg.pilot_n;
    unsigned  seed_ctr = 42u;
    for (int l = 0; l <= L; l++) {
        auto [em, vv] = run_qmc_level(l, N_pilot, seed_ctr++);
        E_l[l] = em; V_l[l] = vv; N_l[l] = N_pilot;
    }

    bool converged = false;
    int  iter = 0;
    while (!converged && L <= max_L && iter++ < 50) {
        double sum_term = 0.0;
        for (int l = 0; l <= L; l++)
            sum_term += std::sqrt(V_l[l] * std::pow((double)M, l));

        for (int l = 0; l <= L; l++) {
            double    C_l   = std::pow((double)M, l);
            long long N_opt = (long long)std::ceil(2.0/(eps*eps) * std::sqrt(V_l[l]/C_l) * sum_term);
            N_opt = std::max(N_opt, 100LL);
            if (N_opt > N_l[l]) {
                auto [em, vv] = run_qmc_level(l, N_opt - N_l[l], seed_ctr++);
                double wold = (double)N_l[l], wnew = (double)(N_opt - N_l[l]);
                E_l[l] = (E_l[l]*wold + em*wnew) / (wold + wnew);
                V_l[l] = vv;
                N_l[l] = N_opt;
            }
        }

        double bias_est = std::abs(E_l[L]) / std::max(M - 1, 1);
        if (L >= 1) bias_est = std::max(bias_est, std::abs(E_l[L-1]) * M / std::max(M*(M-1), 1));
        converged = (bias_est < eps / std::sqrt(2.0));

        if (!converged && L < max_L) {
            L++;
            auto [em, vv] = run_qmc_level(L, N_pilot, seed_ctr++);
            E_l[L] = em; V_l[L] = vv; N_l[L] = N_pilot;
        }
    }

    double price = 0.0, var_sum = 0.0;
    long long N_total = 0;
    for (int l = 0; l <= L; l++) {
        price   += E_l[l];
        var_sum += V_l[l] / N_l[l];
        N_total += N_l[l];
    }
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {price, std::sqrt(var_sum), N_total, t_s};
}


// ================== //
// Variables de control //
// ================== //

CVPilot cv_pilot(const ModelVariant& main_model,
                 const ModelVariant& ctrl_model,
                 const PayoffVariant& main_payoff,
                 const PayoffVariant& ctrl_payoff,
                 double E_ctrl, int n_steps,
                 int N_pilot, unsigned seed) {
    ModelKind mk_main = model_kind(main_model);
    KernelParams kp = make_params(main_model, main_payoff, n_steps);
    if (mk_main == ModelKind::Dupire)
        kp.sigma = (float)std::get<DupireLocalParams>(main_model).sigma0;
    kp.E_ctrl  = (float)E_ctrl;
    kp.beta_cv = 0.0f;  // beta=0 → Y_cv = Y_main, pero el kernel también acumula Y_ctrl
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    float*  d_dW   = nullptr;
    double* d_sums = nullptr;
    long long D = (long long)n_steps;
    CUDA_CHECK(cudaMalloc(&d_dW,   D * N_pilot * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 7 * sizeof(double)));  // 7 acumuladores
    CUDA_CHECK(cudaMemset(d_sums, 0, 7*sizeof(double)));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateNormal(gen, d_dW, D * N_pilot, 0.0f, 1.0f));
    curandDestroyGenerator(gen);
    kernel_scale<<<((int)(D*N_pilot)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        d_dW, sqrtf(kp.h), D*N_pilot);

    int blocks = (N_pilot + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (mk_main == ModelKind::GBM)
        kernel_gbm_asian_cv<<<blocks, BLOCK_SIZE>>>(d_dW, d_sums, N_pilot, n_steps);
    else
        kernel_dupire_gbm_cv<<<blocks, BLOCK_SIZE>>>(d_dW, d_sums, N_pilot, n_steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    double hs[7] = {};
    CUDA_CHECK(cudaMemcpy(hs, d_sums, 7*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_dW); cudaFree(d_sums);

    double N  = (double)N_pilot;
    double Ea = hs[0]/N,  Ea2 = hs[1]/N;
    double Eg = hs[4]/N,  Eg2 = hs[5]/N,  Eag = hs[6]/N;
    double var_main = std::max(0.0, Ea2 - Ea*Ea);
    double var_ctrl = std::max(1e-30, Eg2 - Eg*Eg);
    double cov      = Eag - Ea*Eg;
    // beta óptimo: beta* = Cov(Y_main, Y_ctrl) / Var(Y_ctrl)
    double beta     = std::clamp(cov / var_ctrl, 0.0, 5.0);
    double var_cv   = std::max(0.0, var_main - cov*cov/var_ctrl);
    return {beta, var_main, var_cv};
}

MCResult run_mc_cv_cuda(const ModelVariant& main_model,
                        const ModelVariant& ctrl_model,
                        const PayoffVariant& main_payoff,
                        const PayoffVariant& ctrl_payoff,
                        double E_ctrl, double beta,
                        double eps, int n_steps,
                        const MCConfig& cfg) {
    auto t0 = Clock::now();
    ModelKind mk_main = model_kind(main_model);

    KernelParams kp = make_params(main_model, main_payoff, n_steps);
    kp.beta_cv = (float)beta;
    kp.E_ctrl  = (float)E_ctrl;
    if (mk_main == ModelKind::Dupire)
        kp.sigma = (float)std::get<DupireLocalParams>(main_model).sigma0;
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    long long D = (long long)n_steps;
    float*  d_dW   = nullptr;
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dW,   D * cfg.pilot_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 7 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 7*sizeof(double)));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cfg.seed));
    CURAND_CHECK(curandGenerateNormal(gen, d_dW, D*cfg.pilot_n, 0.0f, 1.0f));
    curandDestroyGenerator(gen);
    kernel_scale<<<((int)(D*cfg.pilot_n)+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
        d_dW, sqrtf(kp.h), D*cfg.pilot_n);

    int blk0 = (cfg.pilot_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (mk_main == ModelKind::GBM)
        kernel_gbm_asian_cv<<<blk0,BLOCK_SIZE>>>(d_dW, d_sums, cfg.pilot_n, n_steps);
    else
        kernel_dupire_gbm_cv<<<blk0,BLOCK_SIZE>>>(d_dW, d_sums, cfg.pilot_n, n_steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    double hs[7] = {};
    CUDA_CHECK(cudaMemcpy(hs, d_sums, 7*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_dW); cudaFree(d_sums);

    double mu_cv  = hs[2] / cfg.pilot_n;
    double var_cv = std::max(0.0, hs[3]/cfg.pilot_n - mu_cv*mu_cv);
    long long N_needed = (long long)std::ceil(2.0 * var_cv / (eps*eps));
    N_needed = std::max(N_needed, (long long)cfg.pilot_n);
    N_needed = (N_needed + 1) & ~1LL;  // cuRAND exige count par

    // Run principal por lotes (como run_mc_cuda): acotamos la memoria a
    // D·N_batch en vez de D·N_needed, que a alta precisión desborda la VRAM.
    // El kernel acumula en d_sums vía atomicAdd, así que basta no resetear
    // d_sums entre lotes. +2 floats de holgura por el redondeo a count par.
    long long batch_cap = std::min(N_needed, (long long)cfg.N_batch);
    CUDA_CHECK(cudaMalloc(&d_dW,   D * batch_cap * sizeof(float) + 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 7 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 7*sizeof(double)));

    curandGenerator_t gen2;
    CURAND_CHECK(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen2, cfg.seed + 100));

    long long N_done = 0;
    while (N_done < N_needed) {
        long long batch = std::min(batch_cap, N_needed - N_done);
        long long gen_count = (D * batch + 1) & ~1LL;  // cuRAND exige count par
        CURAND_CHECK(curandGenerateNormal(gen2, d_dW, gen_count, 0.0f, 1.0f));
        kernel_scale<<<((int)(D*batch)+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
            d_dW, sqrtf(kp.h), D*batch);
        int blk = ((int)batch + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (mk_main == ModelKind::GBM)
            kernel_gbm_asian_cv<<<blk,BLOCK_SIZE>>>(d_dW, d_sums, (int)batch, n_steps);
        else
            kernel_dupire_gbm_cv<<<blk,BLOCK_SIZE>>>(d_dW, d_sums, (int)batch, n_steps);
        N_done += batch;
    }
    curandDestroyGenerator(gen2);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hs, d_sums, 7*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_dW); cudaFree(d_sums);

    double mean_cv   = hs[2] / N_done;
    double var_final = std::max(0.0, hs[3]/N_done - mean_cv*mean_cv) / N_done;
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {mean_cv, std::sqrt(var_final), N_done, t_s};
}

MCResult run_qmc_cv_cuda(const ModelVariant& main_model,
                         const ModelVariant& ctrl_model,
                         const PayoffVariant& main_payoff,
                         const PayoffVariant& ctrl_payoff,
                         double E_ctrl, double beta,
                         double eps, int n_steps,
                         const QMCConfig& cfg) {
    auto t0 = Clock::now();
    ModelKind mk_main = model_kind(main_model);

    KernelParams kp = make_params(main_model, main_payoff, n_steps);
    kp.beta_cv = (float)beta;
    kp.E_ctrl  = (float)E_ctrl;
    if (mk_main == ModelKind::Dupire)
        kp.sigma = (float)std::get<DupireLocalParams>(main_model).sigma0;

    int R = cfg.R;
    long long N_per_rep = 512;  // mínimo suficiente para estimar varianza inter-réplica
    double var_of_means = 1e30, grand_mean = 0.0;
    long long total_N = 0;
    std::vector<double> rmeans(R);

    for (int doublings = 0; doublings < cfg.max_doublings; doublings++) {
        for (int r = 0; r < R; r++) {
            long long D = (long long)n_steps;
            float*  d_dW   = nullptr;
            double* d_sums = nullptr;
            CUDA_CHECK(cudaMalloc(&d_dW,   D * N_per_rep * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sums, 7*sizeof(double)));
            CUDA_CHECK(cudaMemset(d_sums, 0, 7*sizeof(double)));

            curandGenerator_t gen;
            CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
            CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(gen, (unsigned)D));
            CURAND_CHECK(curandSetGeneratorOffset(gen, (unsigned long long)r * N_per_rep));
            CURAND_CHECK(curandGenerateNormal(gen, d_dW, D*N_per_rep, 0.0f, 1.0f));
            curandDestroyGenerator(gen);

            CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));
            kernel_scale<<<((int)(D*N_per_rep)+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
                d_dW, sqrtf(kp.h), D*N_per_rep);

            int blk = ((int)N_per_rep+BLOCK_SIZE-1)/BLOCK_SIZE;
            if (mk_main == ModelKind::GBM)
                kernel_gbm_asian_cv<<<blk,BLOCK_SIZE>>>(d_dW, d_sums, (int)N_per_rep, n_steps);
            else
                kernel_dupire_gbm_cv<<<blk,BLOCK_SIZE>>>(d_dW, d_sums, (int)N_per_rep, n_steps);
            CUDA_CHECK(cudaDeviceSynchronize());

            double hs[7] = {};
            CUDA_CHECK(cudaMemcpy(hs, d_sums, 7*sizeof(double), cudaMemcpyDeviceToHost));
            rmeans[r] = hs[2] / N_per_rep;

            cudaFree(d_dW); cudaFree(d_sums);
        }

        double m = 0.0;
        for (double v : rmeans) m += v;
        m /= R;
        double v = 0.0;
        for (double rv : rmeans) v += (rv - m)*(rv - m);
        v /= (R - 1);
        var_of_means = v / R;
        grand_mean   = m;
        total_N      = (long long)R * N_per_rep;

        if (var_of_means < eps*eps / 2.0) break;
        N_per_rep *= 2;
    }

    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {grand_mean, std::sqrt(var_of_means), total_N, t_s};
}


// ==================== //
// Importance Sampling  //
// ==================== //

MCResult run_is_cuda(const GBMParams& model, const European& payoff,
                     double z_star, double eps, const MCConfig& cfg) {
    auto t0 = Clock::now();

    KernelParams kp{};
    kp.S0  = (float)model.S0;  kp.mu = (float)model.mu;  kp.sigma = (float)model.sigma;
    kp.T   = (float)model.T;
    kp.K   = (float)payoff.K;  kp.r  = (float)payoff.r;
    kp.discount    = (float)std::exp(-payoff.r * payoff.T);
    kp.payoff_kind = PayoffKind::European;

    int n_steps = std::max(4, (int)std::ceil(model.T / eps));
    // Escalar z_star al nivel de paso: el shift terminal z* se distribuye en n_steps pasos
    kp.z_star = (float)(z_star / std::sqrt((double)n_steps));
    kp.h = kp.T / n_steps;
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    long long D = (long long)n_steps;
    float*  d_Z    = nullptr;
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Z,    D * cfg.pilot_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 4*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 4*sizeof(double)));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cfg.seed));
    CURAND_CHECK(curandGenerateNormal(gen, d_Z, D * cfg.pilot_n, 0.0f, 1.0f));
    curandDestroyGenerator(gen);

    int blk0 = (cfg.pilot_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_is_gbm<<<blk0, BLOCK_SIZE>>>(d_Z, d_sums, cfg.pilot_n, n_steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    double hs[4] = {};
    CUDA_CHECK(cudaMemcpy(hs, d_sums, 4*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_Z); cudaFree(d_sums);

    double mu_is  = hs[0] / cfg.pilot_n;
    double var_is = std::max(0.0, hs[1]/cfg.pilot_n - mu_is*mu_is);
    long long N_needed = (long long)std::ceil(var_is / (eps * eps));
    N_needed = std::max(N_needed, (long long)cfg.pilot_n);
    N_needed = (N_needed + 1) & ~1LL;  // cuRAND exige count par

    CUDA_CHECK(cudaMalloc(&d_Z,    D * N_needed * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 4*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 4*sizeof(double)));

    curandGenerator_t gen2;
    CURAND_CHECK(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen2, cfg.seed + 1));
    CURAND_CHECK(curandGenerateNormal(gen2, d_Z, D*N_needed, 0.0f, 1.0f));
    curandDestroyGenerator(gen2);

    int blk = ((int)N_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_is_gbm<<<blk, BLOCK_SIZE>>>(d_Z, d_sums, (int)N_needed, n_steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hs, d_sums, 4*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_Z); cudaFree(d_sums);

    double mean  = hs[0] / N_needed;
    double var_f = std::max(0.0, hs[1]/N_needed - mean*mean) / N_needed;
    double t_s   = std::chrono::duration<double>(Clock::now() - t0).count();
    return {mean, std::sqrt(var_f), N_needed, t_s};
}


// Envoltorio público de run_mc_fixed_impl (utilizado desde los ejemplos vía SimFn)
std::pair<double, double> run_mc_fixed(const ModelVariant& model,
                                       const PayoffVariant& payoff,
                                       int n_steps, long long n_paths,
                                       unsigned seed) {
    return run_mc_fixed_impl(model, payoff, n_steps, n_paths, seed);
}
