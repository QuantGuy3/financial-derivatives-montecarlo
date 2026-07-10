// Implementación CUDA de métodos Monte Carlo para valoración de derivados financieros.
// Objetivo: NVIDIA A-100 (sm_80). Requiere CUDA 11+, cuBLAS, cuRAND, CUB.

#include "methods_cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// El DEVICE es la GPU, el HOST es la CPU + RAM.

// ----------------------------------------------------------------------------------//
// Sobre los streams en CUDA                                                         //
//                                                                                   //
// Un stream en CUDA es una cola ordenada de operaciones (lanzamientos de kernels,   //
// copias de memoria, llamadas a cuRAND/cuBLAS, etc.) que la GPU se compromete a     //
// ejecutar en el orden en que se han encolado, pero de forma asíncrona respecto     //
// a la CPU: la CPU lanza la operación y continúa inmediatamente, sin esperar a que  //
// termine.                                                                          //
// En nuestro trabajo, se usa para calcular múltiples niveles a la vez               //
// ----------------------------------------------------------------------------------//

// ----------------------------------------------------------------------------------
// Resumen de las funciones de librería CUDA empleadas en este fichero:
//
//
//
//  cudaMalloc(&ptr, bytes)        
//
//                                 Reserva memoria en la GPU (memoria global).
//
//  cudaFree(ptr)                  
//                                 Libera memoria previamente reservada con cudaMalloc.
//
//  cudaMemcpy(dst, src, bytes, kind)
//
//                                 Copia bytes entre CPU y GPU (o GPU-GPU). 'kind' indica
//                                 la dirección: HostToDevice, DeviceToHost, DeviceToDevice.
//
//  cudaMemcpyAsync(dst, src, bytes, kind, stream)
//
//                                 Como cudaMemcpy, pero encolada en 'stream' sin
//                                 bloquear la CPU; hay que sincronizar el stream (o el
//                                 dispositivo) antes de leer el resultado en el host.
//
//  cudaMemcpyToSymbol(sym, &val, bytes)
//
//                                 Copia datos desde la CPU a una variable __constant__
//                                 de la GPU (aquí, la estructura KernelParams c_p).
//
//  cudaMemset(ptr, val, bytes)    
//                                 
//                                 Pone a 'val' (normalmente 0) un bloque de memoria GPU;
//                                 se usa para inicializar los acumuladores d_sums.
//
//  cudaMemsetAsync(ptr, val, bytes, stream)
//
//                                 Como cudaMemset, pero encolada en 'stream'.
//
//  cudaMemGetInfo(&free, &total)  
//                                  
//                                 Devuelve la memoria libre y total de la GPU en este
//                                 momento; se usa para dimensionar lotes dinámicamente
//                                 en vez de con un tope fijo (ver gpu_free_bytes()).
//
//  cudaDeviceSynchronize()        
//                                 
//                                 Bloquea la CPU hasta que todos los kernels lanzados
//                                 en TODOS los streams hayan terminado; necesario antes
//                                 de leer resultados si no se usan streams explícitos.
//
//  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
//
//                                 Crea un stream de ejecución independiente del stream
//                                 por defecto (legacy stream 0); el flag NonBlocking
//                                 evita que se serialice implícitamente con él, lo cual
//                                 es necesario para que varios streams se solapen de
//                                 verdad (ver run_mlmc_cuda/run_mlqmc_cuda).
//
//  cudaStreamSynchronize(stream)  
//                              
//                                 Bloquea la CPU hasta que el trabajo encolado en ese
//                                 stream (y solo ese) haya terminado.
//
//  cudaStreamDestroy(stream)      
//              
//                                 Libera los recursos asociados a un stream.
//
//  cudaGetErrorString(err)        
// 
//                                 Traduce un código de error de CUDA a texto legible
//                                 (usado dentro de CUDA_CHECK para lanzar excepciones).
//
//  curandCreateGenerator(&gen, tipo)
//
//                                 Crea un generador de números aleatorios en GPU; el
//                                 tipo determina si son pseudoaleatorios (XORWOW) o
//                                 cuasialeatorios (Sobol con o sin scrambling).
//
//  curandSetStream(gen, stream)   
// 
//                                 Asocia el generador a un stream: toda generación
//                                 posterior con 'gen' se encola ahí en vez de en el
//                                 stream por defecto. Debe llamarse antes de fijar la
//                                 semilla/offset, pues esa llamada ya lanza trabajo en
//                                 la GPU (inicialización de estado del generador).
//
//  curandSetPseudoRandomGeneratorSeed(gen, seed) / curandSetGeneratorOffset(gen, off)
//                                 
//                                 Fijan la semilla o el punto de partida dentro de la
//                                 secuencia, para poder generar tramos disjuntos.
//
//  curandSetQuasiRandomGeneratorDimensions(gen, D)
//
//                                 Fija la dimensión de la sucesión de Sobol generada.
//
//  curandGenerateNormal(gen, ptr, n, mu, sigma)
//
//                                 Rellena un array de la GPU con n números N(mu,sigma).
//
//  curandDestroyGenerator(gen)
//
//                                  Libera los recursos asociados al generador.
//
//  curandGetDirectionVectors32(&ptr, set)
//
//                                 Devuelve, en memoria de host, la tabla de vectores
//                                 directores de Sobol (Joe-Kuo 2008) SIN escramblear,
//                                 usada para generar con la API de dispositivo en vez
//                                 de la de host y aplicar un scrambling propio por
//                                 réplica (ver kernel_gen_hh_scrambled_sobol_normal más
//                                 abajo: matrices triangulares de Hong-Hickernell en vez
//                                 del scramble_c fijo/opaco de CURAND_RNG_QUASI_-
//                                 SCRAMBLED_SOBOL32).
//
//  curand_init(direction_vectors, offset, &state) / curand(&state)
//
//                                 Versión de dispositivo (una llamada por hilo) del
//                                 generador Sobol SIN escramblear: curand_init fija la
//                                 posición dentro de la secuencia y curand() devuelve el
//                                 entero de 32 bits crudo (el vector de dígitos), sin
//                                 pasar por ninguna transformación de distribución.
//
//  cublasCreate(&handle) / cublasDestroy(handle)
//
//                                 Crean/destruyen el contexto de cuBLAS necesario para
//                                 lanzar productos matriciales acelerados.
//
//  cublasGemmEx(handle, opA, opB, m, n, k, &alpha, A, tA, lda, B, tB, ldb, &beta, C, tC,
//               ldc, computeType, algo)
//
//                                 Producto de matrices C = alpha*A*B + beta*C, con tipos
//                                 de dato mixtos (aquí fp16 de entrada, fp32 de salida)
//                                 aprovechando los Tensor Cores de la GPU; se usa para
//                                 aplicar la transformación PCA a todas las trayectorias
//                                 de un lote en una sola llamada.
// ----------------------------------------------------------------------------------


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
    bool bgk_on;
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

// Redondea x a la potencia de 2 más cercana por abajo (x>=1).
static long long pow2_floor(long long x) {
    long long p = 1;
    while (p * 2 <= x) p *= 2;
    return p;
}

// Memoria libre de la GPU ahora mismo, con cierto margen de seguridad.
static long long gpu_free_bytes() {
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    constexpr double FRACCION_LIBRE = 0.8;
    return (long long)((double)free_b * FRACCION_LIBRE);
}


// ---------------- //
// Auxiliares Euler //
// ---------------- //

// dS = μ·S·dt + σ·S·dW.
__device__ __forceinline__
float d_euler_gbm(float S, float dw, float h) {
    return S + c_p.mu * S * h + c_p.sigma * S * dw;
}

// dS = μ·S·dt + √v·S·dW₁; dv = κ(θ-v)dt + ξ·√v·dW₂ (Milstein exacto en v)
__device__ __forceinline__
void d_euler_heston(float& S, float& V, float dw1, float dw2, float h) {
    float Vp = fmaxf(V, 0.0f);
    float sqVp = sqrtf(Vp);
    S = S + c_p.mu * S * h + sqVp * S * dw1;
    float em = expf(-c_p.kappa * h);
    V = c_p.theta + em * (V - c_p.theta) + c_p.xi * sqVp * dw2;
}

// dS = μ·S·dt + σ_loc(S,t)·S·dW, σ_loc = σ₀·exp(-α·t)·(S/S₀)^(β-1)
__device__ __forceinline__
float d_euler_dupire(float S, float dw, float t, float h) {
    float sigma_loc = c_p.sigma0 * expf(-c_p.alpha * t) * powf(S / c_p.S0, c_p.beta_d - 1.0f);
    return S + c_p.mu * S * h + sigma_loc * S * dw;
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
    if constexpr (PK == PayoffKind::Asian) running += S;
    else if constexpr (PK == PayoffKind::GeomAsian) running += logf(S);
    else if constexpr (PK == PayoffKind::Lookback) running = fminf(running, S);
    else if constexpr (PK == PayoffKind::Barrier) running = fmaxf(running, S);
}

// Indica si el payoff necesita toda la trayectoria
template<PayoffKind PK>
__device__ __forceinline__
constexpr bool d_is_path_dep() {
    return PK == PayoffKind::Asian ||
           PK == PayoffKind::GeomAsian ||
           PK == PayoffKind::Lookback ||
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
// Macros de reducción CUB //
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
        S = d_euler_gbm(S, d_dW[k * N_paths + p], c_p.h);
    } else if constexpr (MK == ModelKind::Dupire) {
        S = d_euler_dupire(S, d_dW[k * N_paths + p], k * c_p.h, c_p.h);
    } else if constexpr (MK == ModelKind::Heston) {
        // d_dW contiene Z*sqrt_h (normales escaladas, sin correlacionar);
        // aplicamos Cholesky L para obtener los incrementos correlacionados.
        float z1 = d_dW[(k * 2 + 0) * N_paths + p];
        float z2 = d_dW[(k * 2 + 1) * N_paths + p];
        float dw1 = z1; // L[0][0]=1, L[0][1]=0
        float dw2 = c_p.L_hes[2] * z1 + c_p.L_hes[3] * z2; // L[1][0]=rho, L[1][1]=sqrt(1-rho²)
        d_euler_heston(S, V, dw1, dw2, c_p.h);
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
    float S = (p < N_paths) ? c_p.S0 : 0.0f;
    float V = (p < N_paths) ? c_p.v0 : 0.0f;
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
    double Y = (p < N_paths) ? (double)d_payoff<PK>(S, running, N_steps) : 0.0;
    double Y2 = Y * Y;
    CUB_REDUCE2(Y, Y2, d_sums, d_sums + 1);
}


// ------------------------------------------ //
// Kernel MLMC de nivel l (trayectorias acopladas) //
// ------------------------------------------ //

// d_sums[4] = {sum_delta, sum_delta², sum_fino, sum_fino²}
template<ModelKind MK, PayoffKind PK>
__global__ void kernel_mlmc(
    const float* __restrict__ d_in,
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
                float z1 = d_in[(k * 2 + 0) * N_paths + p];
                float z2 = d_in[(k * 2 + 1) * N_paths + p];
                float dw1 = z1 * sqrt_h_fine;
                float dw2 = (c_p.L_hes[2] * z1 + c_p.L_hes[3] * z2) * sqrt_h_fine;
                acc1 += dw1; acc2 += dw2;
                d_euler_heston(Sf, Vf, dw1, dw2, h_fine);
            } else {
                float dw = d_in[k * N_paths + p];
                acc1 += dw;
                if constexpr (MK == ModelKind::GBM)
                    Sf = d_euler_gbm(Sf, dw, h_fine);
                else if constexpr (MK == ModelKind::Dupire)
                    Sf = d_euler_dupire(Sf, dw, k * h_fine, h_fine);
                if constexpr (d_is_path_dep<PK>())
                    d_running_update<PK>(run_f, Sf);
            }

            // Paso grueso (cada M pasos finos): usar h_coarse para el drift
            if ((k + 1) % M == 0) {
                if constexpr (MK == ModelKind::Heston) {
                    float Vcp = fmaxf(Vc, 0.0f);
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

// d_dW: [N_pasos × n_activos × N_caminos], dW[paso*n*N + activo*N + camino]
__global__ void kernel_multi_dupire(
    const float* __restrict__ d_dW,
    const float* __restrict__ d_S0,
    double* d_sums,
    int N_paths, int N_steps, int n_assets)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float S_mean = 0.0f;

    if (p < N_paths) {
        float running_mean = 0.0f;
        for (int a = 0; a < n_assets; a++) {
            float S = d_S0 ? d_S0[a] : c_p.S0;
            for (int k = 0; k < N_steps; k++) {
                float dw = d_dW[(k * n_assets + a) * N_paths + p];
                float t = k * c_p.h;
                S = d_euler_dupire(S, dw, t, c_p.h);
            }
            running_mean += S;
        }
        S_mean = running_mean / n_assets;
    }
    double Y = (p < N_paths) ? (double)fmaxf(S_mean - c_p.K, 0.0f) * c_p.discount : 0.0;
    double Y2 = Y * Y;
    CUB_REDUCE2(Y, Y2, d_sums, d_sums + 1);
}


// ---------------------------------------------------------------------- //
// Kernel de transformada Brownian Bridge //
// d_Z_in: [N_pasos × N_caminos] normales en orden Sobol //
// d_dW_out: [N_pasos × N_caminos] incrementos brownianos en orden tpo. //
// d_W_scratch: [(N+1) × N_caminos] espacio de trabajo //
// ---------------------------------------------------------------------- //

__global__ void kernel_bb_transform(
    const float* __restrict__ d_Z_in,
    float* __restrict__ d_dW_out,
    float* __restrict__ d_W_scratch,
    const int* __restrict__ d_map_idx,
    const int* __restrict__ d_left_idx,
    const int* __restrict__ d_right_idx,
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
        int m = d_map_idx[step];
        int l = d_left_idx[step];
        int r = d_right_idx[step];
        float wl = d_wl[step], wr = d_wr[step], sd = d_std_dev[step];
        float z = d_Z_in[step * N_paths + p];
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
// Kernels de variable de control //
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
            S = d_euler_gbm(S, dw, c_p.h);
            arith_sum += S;
            log_sum += logf(S);
        }
    }
    float Y_arith = (p < N_paths) ? fmaxf(arith_sum / N_steps - c_p.K, 0.0f) : 0.0f;
    float Y_geom = (p < N_paths) ? fmaxf(expf(log_sum / N_steps) - c_p.K, 0.0f) : 0.0f;
    // Guarda (p < N_paths): los hilos de relleno deben aportar 0 a todos los
    // acumuladores. Sin esto, Y_cv = -beta·(0 - E_ctrl) = beta·E_ctrl
    // para los hilos sobrantes, sesgando la media cuando N_paths no es múltiplo
    // de BLOCK_SIZE.
    float Y_cv = (p < N_paths) ? (Y_arith - c_p.beta_cv * (Y_geom - c_p.E_ctrl)) : 0.0f;

    using BR = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage ts;
    double v;
    // [0..3]: stats del main y del CV (para el run principal)
    v = BR(ts).Sum((double)Y_arith); if (threadIdx.x==0) atomicAdd(d_sums+0, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_arith*Y_arith)); if (threadIdx.x==0) atomicAdd(d_sums+1, v); __syncthreads();
    v = BR(ts).Sum((double)Y_cv); if (threadIdx.x==0) atomicAdd(d_sums+2, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_cv*Y_cv)); if (threadIdx.x==0) atomicAdd(d_sums+3, v); __syncthreads();
    // [4..6]: stats del control para calcular beta_opt en el piloto
    v = BR(ts).Sum((double)Y_geom); if (threadIdx.x==0) atomicAdd(d_sums+4, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_geom*Y_geom)); if (threadIdx.x==0) atomicAdd(d_sums+5, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_arith*Y_geom)); if (threadIdx.x==0) atomicAdd(d_sums+6, v);
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
            float t = k * c_p.h;
            S_dup = d_euler_dupire(S_dup, dw, t, c_p.h);
            S_gbm = d_euler_gbm(S_gbm, dw, c_p.h);
        }
    }
    float Y_dup = (p < N_paths) ? fmaxf(S_dup - c_p.K, 0.0f) * c_p.discount : 0.0f;
    float Y_gbm = (p < N_paths) ? fmaxf(S_gbm - c_p.K, 0.0f) * c_p.discount : 0.0f;
    // Guarda (p < N_paths): sin ella los hilos de relleno aportan beta·E_ctrl a Y_cv.
    float Y_cv = (p < N_paths) ? (Y_dup - c_p.beta_cv * (Y_gbm - c_p.E_ctrl)) : 0.0f;

    using BR = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage ts;
    double v;
    // [0..3]: stats del main y del CV
    v = BR(ts).Sum((double)Y_dup); if (threadIdx.x==0) atomicAdd(d_sums+0, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_dup*Y_dup)); if (threadIdx.x==0) atomicAdd(d_sums+1, v); __syncthreads();
    v = BR(ts).Sum((double)Y_cv); if (threadIdx.x==0) atomicAdd(d_sums+2, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_cv*Y_cv)); if (threadIdx.x==0) atomicAdd(d_sums+3, v); __syncthreads();
    // [4..6]: stats del control para calcular beta_opt en el piloto
    v = BR(ts).Sum((double)Y_gbm); if (threadIdx.x==0) atomicAdd(d_sums+4, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_gbm*Y_gbm)); if (threadIdx.x==0) atomicAdd(d_sums+5, v); __syncthreads();
    v = BR(ts).Sum((double)(Y_dup*Y_gbm)); if (threadIdx.x==0) atomicAdd(d_sums+6, v);
}


// ---------------------------------------------------------------------- //
// Kernel de Importance Sampling (Ej. 11, GBM, call)                      //
// Z' = Z + z_star;  LR = exp(-z_star·ΣZ_k - N·z_star²/2)                 //
// ---------------------------------------------------------------------- //

__global__ void kernel_is_gbm(
    const float* __restrict__ d_Z,
    double* d_sums,
    int N_paths, int N_steps)
{
    int p = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float S = (p < N_paths) ? c_p.S0 : 0.0f;
    float z_sum = 0.0f;

    if (p < N_paths) {
        for (int k = 0; k < N_steps; k++) {
            float z = d_Z[k * N_paths + p];
            z_sum += z;
            float dw = (z + c_p.z_star) * sqrtf(c_p.h);
            S = d_euler_gbm(S, dw, c_p.h);
        }
    }
    float lr = (p < N_paths) ? expf(-c_p.z_star * z_sum
                                        - 0.5f * c_p.z_star * c_p.z_star * N_steps) : 1.0f;
    float payoff = (p < N_paths) ? fmaxf(S - c_p.K, 0.0f) * c_p.discount * lr : 0.0f;
    double Y = (double)payoff, Y2 = Y * Y;
    CUB_REDUCE2(Y, Y2, d_sums, d_sums + 1);
}


// ------------------------------------ //
// Auxiliares para escalar y convertir //
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


// ---------------------------------------------------------------------------- //
// Generación scrambled-Sobol con matrices triangulares de Hong-Hickernell,     //
// scrambling independiente por réplica (observación 3.19 / §2.4 de la memoria) //
//                                                                              //
// La implementación nativa de CUDA del scrambling de Owen nos ha resultado     //
// insuficiente. Por ello, en vez de apoyarse en el scrambling interno de       //
// cuRAND, aquí se genera                                                       // 
// el Sobol sin scrambling (API de dispositivo, curandStateSobol32_t) y se      //
// aplica a mano, por réplica y dimensión, una matriz triangular inferior       //
// aleatoria invertible sobre F_2 más un desplazamiento digital aditivo —       //
// exactamente la construcción del teorema 2.20 (Hong-Hickernell, alg. 823 de   //
// ACM): preserva la estructura de red-(t,m,s) y cuesta O(M) operaciones sobre  //
// F_2 por dimensión y punto (M=32 bits). Además, lo implementamos con __popc   //
// para que                                                                     //
// cada producto fila·vector sobre F_2 sea una única instrucción de hardware.   //
//                                                                              //
// La matriz L y el desplazamiento e de la réplica r en la dimensión dim no se  //
// almacenan. En su lugar, se deriva en el momento, en cada hilo, a partir de   //
// (replica_salt,                                                               //
// dim, fila) mediante el mezclador splitmix32. Esto evita reservar y llenar    //
// una tabla de matrices por (réplica, dimensión), potencialmente varios GB    //
// para R y D grandes, a cambio de recomputar unos 32 hashes de 32 bits por     //
// punto, argumentamos que es un coste marginal frente al resto de cálculos     // 
// ---------------------------------------------------------------------------- //

// Función hash "slitmix32"
__host__ __device__ __forceinline__ unsigned int splitmix32(unsigned int x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x21F0AAADu;
    x = (x ^ (x >> 15)) * 0x735A2D97u;
    x = x ^ (x >> 15);
    return x;
}

// Vectores directores de Sobol sin scrambling (Joe-Kuo 2008), subidos una
// única vez a memoria de dispositivo la primera vez que se necesitan y
// reutilizados el resto de la ejecución.
static curandDirectionVectors32_t* g_d_sobol_dirvectors = nullptr;

static void ensure_sobol_dirvectors_uploaded() {
    if (g_d_sobol_dirvectors) return; // ya subidas en una llamada anterior
    curandDirectionVectors32_t* h_dirvectors = nullptr;
    CURAND_CHECK(curandGetDirectionVectors32(&h_dirvectors,
        CURAND_DIRECTION_VECTORS_32_JOEKUO6));
    CUDA_CHECK(cudaMalloc(&g_d_sobol_dirvectors,
        (size_t)D_MAX_SOBOL * sizeof(curandDirectionVectors32_t)));
    CUDA_CHECK(cudaMemcpy(g_d_sobol_dirvectors, h_dirvectors,
        (size_t)D_MAX_SOBOL * sizeof(curandDirectionVectors32_t), cudaMemcpyHostToDevice));
}

// Hace scrambling a cada punto de Sobol aplicándole una matriz triangular inferior aleatoria 
// sobre F_2 (diagonal en 1 para garantizar que sea invertible, preservando así la 
// estructura de red del Sobol) más un desplazamiento digital adicional (teorema 2.20 
// (Hong-Hickernell, alg. 823 de ACM))
__device__ __forceinline__ unsigned int hh_scramble(unsigned int raw, unsigned int base) {
    unsigned int scrambled = 0;
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        // fila por fila hace:
        unsigned int row_seed = splitmix32(base ^ (unsigned int)k); // Genera 0 o 1 pseudoaleatoriamente
        unsigned int diag_bit = 1u << (31 - k); // 1 en diagonal
        unsigned int top_mask = (k == 0) ? 0u : (0xFFFFFFFFu << (32 - k)); // Para que sea triang inferior
        unsigned int row = diag_bit | (row_seed & top_mask); // Fila final.
        unsigned int bit = __popc(row & raw) & 1u; // Producto escalar en F_2 reducida a ser O(1) (contar número de 1s)
        scrambled |= (bit << (31 - k)); // Coloca el bit en su posición
    }
    unsigned int shift = splitmix32(base ^ 0xA5A5A5A5u); // Desplazamiento
    return scrambled ^ shift; // Se aplica el desplazamiento
}

// Genera D*chunk normales N(0,1) de una secuencia de Sobol habiéndole hecho scrambling
// con matrices de Hong-Hickernell.
__global__ void kernel_gen_hh_scrambled_sobol_normal(
    const curandDirectionVectors32_t* __restrict__ dirvectors,
    unsigned int replica_salt,
    unsigned int offset,
    float* __restrict__ out,
    long long total, int chunk)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dim = (int)(idx / chunk);
    int p   = (int)(idx - (long long)dim * chunk);

    // Punto de Sobol (sin scrambling) en esta dimensión y posición.
    unsigned int dv[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) dv[i] = dirvectors[dim].v[i];
    curandStateSobol32_t state;
    curand_init(dv, offset + (unsigned int)p, &state);
    unsigned int raw = curand(&state);

    unsigned int base = splitmix32(replica_salt) ^ (unsigned int)dim;
    unsigned int scrambled = hh_scramble(raw, base);

    // Uniforme en (0,1].
    float u = ((float)scrambled + 1.0f) * (1.0f / 4294967296.0f);
    out[idx] = normcdfinvf(u);
}

// Kernel para generar normales sobol. Número de la secuencia 'offser64' con scrambling
// 'replica_salt'.
static void gen_scrambled_sobol_normal_replica(
    float* d_out, long long D, long long chunk, unsigned long long offset64,
    unsigned int replica_salt, cudaStream_t stream = 0)
{
    ensure_sobol_dirvectors_uploaded();
    if (offset64 > (unsigned long long)UINT_MAX)
        throw std::runtime_error("gen_scrambled_sobol_normal_replica: offset "
            + std::to_string(offset64) + " excede el límite de 32 bits de curand_init "
            + "(API de dispositivo); reduzca N o max_doublings.");
    long long total = D * chunk;
    int blk = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_gen_hh_scrambled_sobol_normal<<<blk, BLOCK_SIZE, 0, stream>>>(
        g_d_sobol_dirvectors, replica_salt,
        (unsigned int)offset64, d_out, total, (int)chunk);
    CUDA_CHECK(cudaGetLastError());
}


// ------------------------------------------- //
// Tablas [modelo × payoff]                    //
// ------------------------------------------- //

using McKernelFn = void(*)(const float*, double*, int, int);
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
    int N;
    int* d_map_idx = nullptr;
    int* d_left_idx = nullptr;
    int* d_right_idx = nullptr;
    float* d_wl = nullptr;
    float* d_wr = nullptr;
    float* d_std_dev = nullptr;
};

struct DevicePCAData {
    int m;
    __half* d_M_pca_f16 = nullptr; // Matriz PCA [m×m] en fp16
};

DeviceBBData* bb_upload(const BBData& bb) {
    auto* d = new DeviceBBData;
    d->N = bb.N;
    int N = bb.N;
    CUDA_CHECK(cudaMalloc(&d->d_map_idx, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->d_left_idx, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->d_right_idx, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->d_wl, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d->d_wr, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d->d_std_dev, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d->d_map_idx, bb.map_idx.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_left_idx, bb.left_idx.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_right_idx, bb.right_idx.data(), N*sizeof(int), cudaMemcpyHostToDevice));

    std::vector<float> wl(N), wr(N), sd(N);
    for (int i = 0; i < N; i++) {
        wl[i] = (float)bb.weight_left[i];
        wr[i] = (float)bb.weight_right[i];
        sd[i] = (float)bb.std_dev[i];
    }
    CUDA_CHECK(cudaMemcpy(d->d_wl, wl.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_wr, wr.data(), N*sizeof(float), cudaMemcpyHostToDevice));
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
// Auxiliares internos — construir c_p //
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
            p.xi = (float)m.xi; p.rho = (float)m.rho;
            p.v0 = (float)m.v0;
            p.L_hes[0] = (float)m.L[0][0]; p.L_hes[1] = (float)m.L[0][1];
            p.L_hes[2] = (float)m.L[1][0]; p.L_hes[3] = (float)m.L[1][1];
        }
        if constexpr (std::is_same_v<T, DupireLocalParams>) {
            p.S0 = (float)m.S0; p.mu = (float)m.mu;
            p.sigma0 = (float)m.sigma0; p.alpha = (float)m.alpha; p.beta_d = (float)m.beta_d;
            p.sigma = (float)m.sigma0; // sigma_eff para el kernel GBM de variable de control
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

// Identifica el tipo de modelo
static ModelKind model_kind(const ModelVariant& mv) {
    return std::visit([](const auto& m) -> ModelKind {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, GBMParams>) return ModelKind::GBM;
        if constexpr (std::is_same_v<T, HestonParams>) return ModelKind::Heston;
        if constexpr (std::is_same_v<T, DupireLocalParams>) return ModelKind::Dupire;
        if constexpr (std::is_same_v<T, MultiDupireParams>) return ModelKind::MultiDupire;
        return ModelKind::GBM;
    }, mv);
}

// Identifica el tipo de payoff
static int payoff_idx(const PayoffVariant& pv) {
    return std::visit([](const auto& p) -> int {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, European>) return 0;
        if constexpr (std::is_same_v<T, Asian>) return 1;
        if constexpr (std::is_same_v<T, GeomAsian>) return 2;
        if constexpr (std::is_same_v<T, Lookback>) return 3;
        if constexpr (std::is_same_v<T, Barrier>) return 4;
        if constexpr (std::is_same_v<T, Basket>) return 5;
        return 0;
    }, pv);
}

// Lanza el kernel MC de un nivel
static void launch_mc_kernel(ModelKind mk, int pidx, bool is_multi,
                             const float* d_dW, double* d_sums,
                             int N_paths, int N_steps, int n_assets,
                             const float* d_S0_multi = nullptr) {
    int blocks = (N_paths + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (is_multi) {
        kernel_multi_dupire<<<blocks, BLOCK_SIZE>>>(d_dW, d_S0_multi, d_sums, N_paths, N_steps, n_assets);
    } else {
        MC_TABLE[(int)mk][pidx]<<<blocks, BLOCK_SIZE>>>(d_dW, d_sums, N_paths, N_steps);
    }
}

// Sube a la GPU S0 por activo (float) y, si !uncorrelated, la Cholesky L (n×n, fila
// principal, float) de una cesta. uncorrelated==true evita subir/usar L: cada activo
// ya se simula con ruido independiente, que es exactamente rho=I.
struct DeviceMultiData {
    float* d_S0 = nullptr;
    float* d_L = nullptr; // nullptr si uncorrelated
    int n = 0;
};
static DeviceMultiData multi_upload(const MultiDupireParams& mp) {
    if ((int)mp.S0.size() < mp.n)
        throw std::runtime_error("multi_upload: MultiDupireParams.S0 tiene "
            + std::to_string(mp.S0.size()) + " elementos, se esperaban " + std::to_string(mp.n));
    if (!mp.uncorrelated && !mp.L.empty() && (long long)mp.L.size() < (long long)mp.n * mp.n)
        throw std::runtime_error("multi_upload: MultiDupireParams.L tiene "
            + std::to_string(mp.L.size()) + " elementos, se esperaban " + std::to_string((long long)mp.n*mp.n));

    DeviceMultiData d;
    d.n = mp.n;
    std::vector<float> S0f(mp.S0.begin(), mp.S0.begin() + mp.n);
    CUDA_CHECK(cudaMalloc(&d.d_S0, d.n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d.d_S0, S0f.data(), d.n*sizeof(float), cudaMemcpyHostToDevice));
    if (!mp.uncorrelated && !mp.L.empty()) {
        std::vector<float> Lf(mp.L.begin(), mp.L.begin() + (long long)mp.n*mp.n);
        CUDA_CHECK(cudaMalloc(&d.d_L, (long long)d.n * d.n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d.d_L, Lf.data(), (long long)d.n*d.n*sizeof(float), cudaMemcpyHostToDevice));
    }
    return d;
}
static void multi_free(DeviceMultiData& d) {
    if (d.d_S0) cudaFree(d.d_S0);
    if (d.d_L)  cudaFree(d.d_L);
    d.d_S0 = d.d_L = nullptr;
}

// Correlaciona (dW_in -> dW_out) las N_steps matrices de incrementos
// [n_assets × N_paths] (fila principal) vía Cholesky: dW_out_k = L · dW_in_k para cada
// paso k. Se hace de una sola vez (batch=N_steps) en vez de
// dentro del kernel de payoff, porque cada hilo necesitaría releer memoria global
// n_assets veces por paso (no coalescente) o guardar n_assets acumuladores locales.
static void correlate_multi_dupire(cublasHandle_t cublas, const DeviceMultiData& dm,
                                   const float* d_dW_in, float* d_dW_out,
                                   int n_steps, int n_paths) {
    const float one = 1.0f, zero = 0.0f;
    long long stride = (long long)dm.n * n_paths;
    // dm.d_L guarda L fila-principal (n_assets×n_assets)
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n_paths, dm.n, dm.n,
        &one,
        d_dW_in, n_paths, stride,
        dm.d_L, dm.n, 0,
        &zero,
        d_dW_out, n_paths, stride,
        n_steps));
}

// Fase 1 PCA con Tensor Cores (F16→F32): dW = M_pca[D×D] @ Z[D×N]
static void phase1_pca(cublasHandle_t cublas,
                       const __half* d_M_pca, const __half* d_Z_f16,
                       float* d_dW_f32, int D, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    // La fórmula es dW = Z·M_pca^T
    CUBLAS_CHECK(cublasGemmEx(cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, D, D,
        &alpha,
        d_Z_f16, CUDA_R_16F, N,
        d_M_pca, CUDA_R_16F, D,
        &beta,
        d_dW_f32, CUDA_R_32F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // Traspone la matriz
}


// -------------------------------------------- //
// run_mc_fixed_impl — constructor interno      //
// -------------------------------------------- //

static std::pair<double,double> run_mc_fixed_impl(
    const ModelVariant& model, const PayoffVariant& payoff,
    int n_steps, long long n_paths, unsigned seed,
    NoiseMode mode = NoiseMode::Raw,
    DeviceBBData* dev_bb = nullptr,
    DevicePCAData* dev_pca = nullptr,
    cublasHandle_t cublas = nullptr)
{
    n_paths = (n_paths + 1) & ~1LL; // cuRAND exige count par
    int d_noise = model_noise_dim(model);
    long long D = (long long)n_steps * d_noise;
    ModelKind mk = model_kind(model);
    bool path_dep = payoff_needs_full_path(payoff);
    int pidx = payoff_idx(payoff);
    bool is_multi = (mk == ModelKind::MultiDupire);
    int n_assets = is_multi ? std::get<MultiDupireParams>(model).n : 1;

    KernelParams kp = make_params(model, payoff, n_steps);
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    float* d_Z = nullptr;
    float* d_dW = nullptr;
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Z, D * n_paths * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW, D * n_paths * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 4 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 4 * sizeof(double)));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateNormal(gen, d_Z, D * n_paths, 0.0f, 1.0f));
    curandDestroyGenerator(gen);

    float sqrt_h = sqrtf(kp.h);
    long long total_elems = D * n_paths;
    int scale_blk = (int)((total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (mode == NoiseMode::Raw) {
        CUDA_CHECK(cudaMemcpy(d_dW, d_Z, total_elems*sizeof(float), cudaMemcpyDeviceToDevice));
        kernel_scale<<<scale_blk, BLOCK_SIZE>>>(d_dW, sqrt_h, total_elems);
    } else if (mode == NoiseMode::BrownianBridge && dev_bb) {
        float* d_W_scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&d_W_scratch, (long long)(n_steps + 1) * n_paths * sizeof(float)));
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

    // Cesta correlacionada (MultiDupire, uncorrelated==false): correlaciona los
    // incrementos por Cholesky antes del kernel de payoff.
    DeviceMultiData dm;
    float* d_dW_corr = nullptr;
    const float* d_S0_multi = nullptr;
    if (is_multi) {
        const auto& mp = std::get<MultiDupireParams>(model);
        dm = multi_upload(mp);
        d_S0_multi = dm.d_S0;
        if (dm.d_L) {
            cublasHandle_t cublas_local = cublas;
            bool own_handle = !cublas_local;
            if (own_handle) CUBLAS_CHECK(cublasCreate(&cublas_local));
            CUDA_CHECK(cudaMalloc(&d_dW_corr, D * n_paths * sizeof(float)));
            correlate_multi_dupire(cublas_local, dm, d_dW, d_dW_corr, n_steps, (int)n_paths);
            if (own_handle) cublasDestroy(cublas_local);
        }
    }
    const float* d_feed = d_dW_corr ? d_dW_corr : d_dW;

    launch_mc_kernel(mk, pidx, is_multi, d_feed, d_sums, (int)n_paths, n_steps, n_assets, d_S0_multi);
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_sums[4] = {};
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums, 4*sizeof(double), cudaMemcpyDeviceToHost));
    if (d_dW_corr) cudaFree(d_dW_corr);
    if (is_multi) multi_free(dm);
    cudaFree(d_Z); cudaFree(d_dW); cudaFree(d_sums);

    double mean = h_sums[0] / n_paths;
    double var = std::max(0.0, (h_sums[1]/n_paths - mean*mean)) / n_paths;
    return {mean, var};
}


// ================= //
// run_mc_cuda //
// ================= //

MCResult run_mc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                     double eps, int n_steps, const MCConfig& cfg) {
    auto t0 = Clock::now();

    auto [mu_pilot, var_pilot] = run_mc_fixed_impl(model, payoff, n_steps,
                                                    cfg.pilot_n, cfg.seed);
    double sample_var = var_pilot * cfg.pilot_n;
    long long N_needed = (long long)std::ceil(2.0 * sample_var / (eps * eps));
    N_needed = std::max(N_needed, (long long)cfg.pilot_n);

    double sum_Y = 0.0, sum_Y2 = 0.0;
    long long N_done = 0;
    unsigned cur_seed = cfg.seed + 1;

    while (N_done < N_needed) {
        long long batch = std::min((long long)cfg.N_batch, N_needed - N_done);
        auto [bm, bv] = run_mc_fixed_impl(model, payoff, n_steps, batch, cur_seed++);
        sum_Y += bm * batch;
        sum_Y2 += (bv * batch + bm * bm) * batch;
        N_done += batch;
    }

    double mean = sum_Y / N_done;
    double var = std::max(0.0, sum_Y2 / N_done - mean * mean) / N_done;
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {mean, std::sqrt(var), N_done, t_s};
}


// ================= //
// run_mlmc_cuda //
// ================= //

MCResult run_mlmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                       double eps, const MLMCConfig& cfg) {
    auto t0 = Clock::now();

    int M = cfg.M;
    int max_L = cfg.max_L;
    bool path_dep = payoff_needs_full_path(payoff);
    ModelKind mk = model_kind(model);
    int pidx = payoff_idx(payoff);
    bool is_multi = (mk == ModelKind::MultiDupire);
    int n_assets = is_multi ? std::get<MultiDupireParams>(model).n : 1;
    double T = model_T(model);

    if (mk == ModelKind::MultiDupire)
        throw std::runtime_error("run_mlmc_cuda: MultiDupire (cestas) no está soportado; "
            "no hay kernel MLMC multi-activo implementado.");

    KernelParams kp = make_params(model, payoff, 1);
    kp.M_refine = M;
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    int L = 2;
    std::vector<double> E_l(max_L+1, 0.0), V_l(max_L+1, 0.0);
    std::vector<long long> N_l(max_L+1, 0);

    // Recursos propios por nivel (stream + generador + memoria), para poder tener
    // varios niveles ejecutándose de verdad en paralelo en la GPU.
    struct LevelJob {
        cudaStream_t stream = nullptr;
        curandGenerator_t gen{};
        float* d_Z = nullptr;
        double* d_sums = nullptr;
        double h_sums[4] = {};
    };
    std::vector<LevelJob> jobs(max_L + 1);
    // NonBlocking: Así, los streams son independientes al principal.
    for (auto& j : jobs) CUDA_CHECK(cudaStreamCreateWithFlags(&j.stream, cudaStreamNonBlocking));

    // Encola en el stream del nivel l la generación + el kernel para n_paths puntos,
    // sin sincronizar (collect_level recoge el resultado después). D·batch_cap acota
    // la memoria por lote y evita que (int)n_paths desborde cuando N_opt supera 2.1e9
    // a eps pequeño; el kernel acumula en d_sums vía atomicAdd.
    auto launch_level = [&](int l, long long n_paths, unsigned seed) {
        n_paths = (n_paths + 1) & ~1LL; // cuRAND exige count par
        int N_fine = (l == 0) ? 1 : (int)std::round(std::pow(M, l));
        int N_coarse = (l == 0) ? 0 : N_fine / M;
        float h_fine = (float)(T / N_fine);
        float h_coarse = (float)(T / std::max(N_coarse, 1));
        float sqrt_hf = sqrtf(h_fine);
        int d_noise = model_noise_dim(model);
        long long D = (long long)N_fine * d_noise;

        // Lote tan grande como quepa en la GPU ahora mismo, repartido entre los
        // max_L+1 niveles.
        long long budget_floats = gpu_free_bytes() / (max_L + 1) / (long long)sizeof(float);
        long long batch_cap = std::max(2LL, budget_floats / std::max(D, 1LL));
        // Tope: con D pequeño y GPUs de mucha memoria, batch_cap podría
        // superar INT_MAX y los (int)batch de más abajo se volverían negativos por desbordamiento.
        batch_cap = std::min(batch_cap, (long long)INT_MAX / 2);
        batch_cap = (batch_cap + 1) & ~1LL;
        batch_cap = std::min(batch_cap, n_paths);

        LevelJob& j = jobs[l];
        CUDA_CHECK(cudaMalloc(&j.d_Z, D * batch_cap * sizeof(float) + 2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&j.d_sums, 4 * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(j.d_sums, 0, 4 * sizeof(double), j.stream));

        CURAND_CHECK(curandCreateGenerator(&j.gen, CURAND_RNG_PSEUDO_XORWOW)); // Creo un generador
        CURAND_CHECK(curandSetStream(j.gen, j.stream)); // Usa el generador para j.stream
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(j.gen, seed)); // Inicializa el generador (en j.stream)

        // d_Z se reutiliza entre lotes de un mismo nivel.
        // El orden dentro del stream
        // garantiza que cada generación espera a que el kernel anterior haya leído
        // d_Z antes de sobreescribirlo.
        long long done = 0;
        while (done < n_paths) {
            long long batch = std::min(batch_cap, n_paths - done);
            long long gen_count = (D * batch + 1) & ~1LL; // cuRAND exige count par
            CURAND_CHECK(curandGenerateNormal(j.gen, j.d_Z, gen_count, 0.0f, 1.0f));
            // kernel_mlmc espera el incremento dW ya escalado, no Z.
            if (mk != ModelKind::Heston) {
                int blk_s = (int)((gen_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernel_scale<<<blk_s, BLOCK_SIZE, 0, j.stream>>>(j.d_Z, sqrt_hf, gen_count);
            }
            int blocks = (int)((batch + BLOCK_SIZE - 1) / BLOCK_SIZE);
            MLMC_TABLE[(int)mk][pidx]<<<blocks, BLOCK_SIZE, 0, j.stream>>>(
                j.d_Z, j.d_sums, (int)batch, N_fine, N_coarse, M, h_fine, h_coarse, sqrt_hf);
            done += batch;
        }
    };

    // Sincroniza el stream del nivel l, recoge {sum_dY, sum_dY², sum_f, sum_f²} y
    // libera la memoria de este lanzamiento, pero el stream y el generador se conservan
    // para el siguiente launch_level de ese mismo nivel.
    auto collect_level = [&](int l) -> std::array<double,4> {
        LevelJob& j = jobs[l];
        CUDA_CHECK(cudaMemcpyAsync(j.h_sums, j.d_sums, 4*sizeof(double),
                                    cudaMemcpyDeviceToHost, j.stream));
        CUDA_CHECK(cudaStreamSynchronize(j.stream));
        curandDestroyGenerator(j.gen);
        cudaFree(j.d_Z); cudaFree(j.d_sums);
        j.d_Z = nullptr; j.d_sums = nullptr;
        return {j.h_sums[0], j.h_sums[1], j.h_sums[2], j.h_sums[3]};
    };

    // Piloto inicial: los L+1 niveles se lanzan todos a la vez.
    unsigned seed_ctr = 42u;
    for (int l = 0; l <= L; l++) launch_level(l, cfg.pilot_n, seed_ctr++);
    for (int l = 0; l <= L; l++) {
        auto s = collect_level(l);
        long long np = cfg.pilot_n;
        double em = s[0] / np;
        V_l[l] = std::max(0.0, s[1]/np - em*em);
        E_l[l] = em;
        N_l[l] = np;
    }

    // Bucle adaptativo MLMC (Giles 2008, Teorema 1)
    bool converged = false;
    int iter = 0;
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

        // Lanza todos los niveles que necesitan refinamiento antes de recoger nada.
        std::vector<long long> extra(L+1, 0);
        for (int l = 0; l <= L; l++) {
            if (N_opt[l] > N_l[l]) {
                extra[l] = N_opt[l] - N_l[l];
                launch_level(l, extra[l], seed_ctr++);
            }
        }
        for (int l = 0; l <= L; l++) {
            if (extra[l] > 0) {
                auto s = collect_level(l);
                long long N_new = N_l[l] + extra[l];
                double total_sum = E_l[l] * N_l[l] + s[0];
                double total_s2 = (V_l[l] + E_l[l]*E_l[l]) * N_l[l] + s[1];
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
            launch_level(L, cfg.pilot_n, seed_ctr++);
            auto s = collect_level(L);
            long long np = cfg.pilot_n;
            double em = s[0] / np;
            V_l[L] = std::max(0.0, s[1]/np - em*em);
            E_l[L] = em;
            N_l[L] = np;
        }
    }

    for (auto& j : jobs) cudaStreamDestroy(j.stream);

    // Estimador final: suma telescópica Σ E_l
    double price = 0.0, var_sum = 0.0;
    long long N_total = 0;
    for (int l = 0; l <= L; l++) {
        price += E_l[l];
        var_sum += (N_l[l] > 0 ? V_l[l] / N_l[l] : 0.0);
        N_total += N_l[l];
    }
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {price, std::sqrt(var_sum), N_total, t_s};
}


// ================= //
// run_qmc_cuda //
// ================= //

MCResult run_qmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                      double eps, int n_steps,
                      const QMCConfig& cfg, NoiseMode mode,
                      DeviceBBData* dev_bb, DevicePCAData* dev_pca) {
    auto t0 = Clock::now();
    int R = cfg.R;
    int d_noise = model_noise_dim(model);
    long long D = (long long)n_steps * d_noise;
    ModelKind mk = model_kind(model);
    int pidx = payoff_idx(payoff);
    bool is_multi = (mk == ModelKind::MultiDupire);
    int n_assets = is_multi ? std::get<MultiDupireParams>(model).n : 1;

    cublasHandle_t cublas = nullptr;
    if (mode == NoiseMode::PCA || is_multi) { CUBLAS_CHECK(cublasCreate(&cublas)); }

    // Cesta correlacionada: sube S0/L una sola vez para toda la función (no cambian
    // entre réplicas ni duplicados).
    DeviceMultiData dm;
    if (is_multi) dm = multi_upload(std::get<MultiDupireParams>(model));

    double var_of_means = 1e30, grand_mean = 0.0;
    long long total_N = 0;
    std::vector<double> replica_means(R);

    // Lote de puntos por réplica: tan grande como quepa en la GPU ahora mismo, sin
    // limitar el numero total de puntos.
    bool use_sobol = (D <= D_MAX_SOBOL);
    // por lote conviven d_Z_r, d_dW_r y (en modo BrownianBridge) d_W_scratch.
    long long budget_floats = gpu_free_bytes() / 3 / (long long)sizeof(float);
    long long chunk_cap = std::max((long long)BLOCK_SIZE, budget_floats / std::max(D, 1LL));
    // Tope: hay que acotar es D*chunk para evitar desbordamientos.
    chunk_cap = std::min(chunk_cap, (long long)INT_MAX / 2 / std::max(D, 1LL));
    chunk_cap = (chunk_cap + 1) & ~1LL;

    // Primera réplica ya arranca con tantos puntos como quepan en un lote (en vez de
    // un N pequeño fijo), para aprovechar la GPU desde el primer duplicado. Con Sobol,
    // redondeado a potencia de 2 para garantizar las propiedades de las redes-(t,m,s).
    long long N_per_replica = std::max(64LL, chunk_cap);
    if (use_sobol) N_per_replica = pow2_floor(N_per_replica);

    // Cada réplica acumula en su propio d_sums a través de los duplicados: al doblar
    // N_per_replica solo se generan y suman los puntos nuevos (rango [N_done[r],
    // N_per_replica)), en vez de tirar el trabajo ya hecho y regenerar todo desde
    // cero. Con Sobol, cada réplica tiene su propio scramble independiente
    // (gen_scrambled_sobol_normal_replica).
    std::vector<double*> d_sums_r(R, nullptr);
    std::vector<long long> N_done(R, 0);
    for (int r = 0; r < R; r++) {
        CUDA_CHECK(cudaMalloc(&d_sums_r[r], 4*sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sums_r[r], 0, 4*sizeof(double)));
    }

    // Genera y acumula en d_sums el rango [from, to) de la réplica r. Se usa tanto
    // para el duplicado normal como para la especulación de un duplicado extra.
    auto accumulate_range = [&](int r, long long from, long long to, double* d_sums) {
        long long done = from;
        while (done < to) {
            long long chunk = std::min(chunk_cap, to - done);
            long long gen_count = (D * chunk + 1) & ~1LL; // cuRAND exige count par

            float* d_Z_r = nullptr;
            CUDA_CHECK(cudaMalloc(&d_Z_r, gen_count * sizeof(float) + 2*sizeof(float)));

            if (use_sobol) {
                // Scramble propio de la réplica r (API de dispositivo): el offset es
                // simplemente "done" dentro de la secuencia YA independiente de esta
                // réplica, sin carril.
                gen_scrambled_sobol_normal_replica(d_Z_r, D, chunk,
                    (unsigned long long)done,
                    /*replica_salt=*/42u + (unsigned)r);
            } else {
                curandGenerator_t gen;
                CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
                CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42u + r));
                CURAND_CHECK(curandSetGeneratorOffset(gen, (unsigned long long)done * D));
                CURAND_CHECK(curandGenerateNormal(gen, d_Z_r, gen_count, 0.0f, 1.0f));
                curandDestroyGenerator(gen);
            }

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

            // Cesta correlacionada: aplica Cholesky antes del kernel de payoff.
            float* d_dW_corr = nullptr;
            const float* d_feed = d_dW_r;
            if (is_multi && dm.d_L) {
                CUDA_CHECK(cudaMalloc(&d_dW_corr, D * chunk * sizeof(float)));
                correlate_multi_dupire(cublas, dm, d_dW_r, d_dW_corr, n_steps, (int)chunk);
                d_feed = d_dW_corr;
            }
            launch_mc_kernel(mk, pidx, is_multi, d_feed, d_sums, (int)chunk, n_steps, n_assets, dm.d_S0);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (d_dW_corr) cudaFree(d_dW_corr);

            cudaFree(d_dW_r);
            cudaFree(d_Z_r);
            done += chunk;
        }
    };

    for (int doublings = 0; doublings < cfg.max_doublings; doublings++) {
        // Porque si el número de muestras es excesivamente grande, hay desbordamiento de memoria 
        if (use_sobol && (unsigned long long)N_per_replica > (unsigned long long)UINT_MAX)
            throw std::runtime_error("run_qmc_cuda: N_per_replica=" + std::to_string(N_per_replica)
                + " excede el límite de 32 bits del offset de Sobol.");

        // Si el duplicado anterior
        // ya generó de más (ver más abajo), N_done[r] == N_per_replica y no hace nada.
        for (int r = 0; r < R; r++)
            accumulate_range(r, N_done[r], N_per_replica, d_sums_r[r]);
        for (int r = 0; r < R; r++) N_done[r] = N_per_replica;

        for (int r = 0; r < R; r++) {
            double hs[4] = {};
            CUDA_CHECK(cudaMemcpy(hs, d_sums_r[r], 4*sizeof(double), cudaMemcpyDeviceToHost));
            replica_means[r] = hs[0] / N_per_replica;
        }

        double m = 0.0;
        for (double v : replica_means) m += v;
        m /= R;
        double v = 0.0;
        for (double rv : replica_means) v += (rv - m) * (rv - m);
        v = (R > 1) ? v / (R - 1) : 0.0;
        var_of_means = v / R;
        grand_mean = m;
        total_N = (long long)R * N_per_replica;

        if (var_of_means < eps * eps / 2.0) break;

        long long N_next = N_per_replica * 2;
        bool offset_ok = !use_sobol || (unsigned long long)N_next <= (unsigned long long)UINT_MAX;
        bool fits = (D * std::min(chunk_cap, N_next) * (long long)sizeof(float) * 3) <= gpu_free_bytes();
        // Si aún cabe otro duplicado en la GPU y este todavía no ha convergido (así
        // que el siguiente hace falta seguro), lo genera ya en vez de esperar a la
        // próxima vuelta del bucle.
        if (doublings + 1 < cfg.max_doublings && offset_ok && fits) {
            for (int r = 0; r < R; r++)
                accumulate_range(r, N_per_replica, N_next, d_sums_r[r]);
            for (int r = 0; r < R; r++) N_done[r] = N_next;
        }
        N_per_replica = N_next;
    }
    for (int r = 0; r < R; r++) cudaFree(d_sums_r[r]);

    if (is_multi) multi_free(dm);
    if (cublas) cublasDestroy(cublas);
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {grand_mean, std::sqrt(var_of_means), total_N, t_s};
}


// ================== //
// run_mlqmc_cuda //
// ================== //

MCResult run_mlqmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                        double eps,
                        const MLMCConfig& ml_cfg, const QMCConfig& qmc_cfg,
                        NoiseMode mode,
                        std::vector<DeviceBBData*> bb_list,
                        std::vector<DevicePCAData*> pca_list) {
    auto t0 = Clock::now();

    int M = ml_cfg.M;
    int max_L = ml_cfg.max_L;
    double T = model_T(model);
    ModelKind mk = model_kind(model);
    int pidx = payoff_idx(payoff);

    // Mismo motivo que en run_mlmc_cuda: no hay kernel MLMC multi-activo.
    if (mk == ModelKind::MultiDupire)
        throw std::runtime_error("run_mlqmc_cuda: MultiDupire (cestas) no está soportado; "
            "no hay kernel MLMC multi-activo implementado.");

    // Brownian Bridge/PCA asumen ruido 1D (kernel_bb_transform espera D==N_fine);
    // Heston tiene 2 factores de ruido y no está soportado en estos modos, igual
    // que en run_qmc_cuda.
    if (mode != NoiseMode::Raw && model_noise_dim(model) != 1)
        throw std::runtime_error("run_mlqmc_cuda: Brownian Bridge/PCA solo soportados "
            "para modelos de un factor de ruido (no Heston).");

    // kernel_mlmc (reutilizado aquí para las diferencias de nivel QMC) recibe
    // h_fine/h_coarse/M, no vía c_p pues kp no depende del nivel, se fija una
    // sola vez (misma razón que en run_mlmc_cuda: es lo que permite lanzar varios
    // niveles a la vez en streams distintos sin pisarse la variable __constant__).
    KernelParams kp = make_params(model, payoff, 1);
    kp.M_refine = M;
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    int L = 2;
    int R_reps = qmc_cfg.R;

    std::vector<double> E_l(max_L+1, 0.0), sig2_l(max_L+1, 0.0);
    std::vector<long long> N_l(max_L+1, 0);

    // Número de muestras generadas por (nivel, réplica).
    std::vector<std::vector<long long>> next_off(max_L+1, std::vector<long long>(R_reps, 0));

    // Recursos propios por nivel: un stream, un generador reutilizado
    // entre réplicas/lotes de ese nivel, y un acumulador d_sums por réplica. Así los
    // niveles pueden ejecutarse de verdad en paralelo en la GPU (streams distintos),
    // mientras que las R réplicas de un mismo nivel se serializan dentro de su
    // stream.
    struct LevelJob {
        cudaStream_t stream = nullptr;
        float* d_Z = nullptr;
        long long d_Z_cap = 0; // capacidad reservada de d_Z, en floats
        // Solo se reservan si mode!=Raw: d_dW es la salida de BB/PCA (el incremento
        // ya escalado que se pasa a kernel_mlmc); d_W_scratch/d_Z_f16 son el espacio
        // de trabajo de BB/PCA respectivamente.
        float* d_dW = nullptr;
        long long d_dW_cap = 0;
        float* d_W_scratch = nullptr;
        long long d_W_scratch_cap = 0;
        __half* d_Z_f16 = nullptr;
        long long d_Z_f16_cap = 0;
        cublasHandle_t cublas = nullptr;
        std::vector<double*> d_sums;
        std::vector<std::array<double,4>> h_sums;
    };
    std::vector<LevelJob> jobs(max_L + 1);
    for (auto& j : jobs) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&j.stream, cudaStreamNonBlocking));
        j.d_sums.assign(R_reps, nullptr);
        j.h_sums.assign(R_reps, {});
        for (int r = 0; r < R_reps; r++) CUDA_CHECK(cudaMalloc(&j.d_sums[r], 4*sizeof(double)));
        if (mode == NoiseMode::PCA) {
            CUBLAS_CHECK(cublasCreate(&j.cublas));
            CUBLAS_CHECK(cublasSetStream(j.cublas, j.stream));
        }
    }

    // Encola en el stream del nivel l la generación + kernel de las R réplicas para
    // N_per_rep puntos cada una, sin sincronizar; collect_level recoge el resultado.
    auto launch_level = [&](int l, long long N_per_rep) {
        int N_fine = (l == 0) ? 1 : (int)std::round(std::pow(M, l));
        int N_coarse = (l == 0) ? 0 : N_fine / M;
        float h_fine = (float)(T / N_fine);
        float h_coarse = (float)(T / std::max(N_coarse, 1));
        float sqrt_hf = sqrtf(h_fine);
        int d_noise = model_noise_dim(model);
        long long D = (long long)N_fine * d_noise;
        bool use_sobol = (D <= D_MAX_SOBOL);

        // bb_list[l]/pca_list[l] deben existir y estar precalculados exactamente
        // para N_fine pasos de este nivel; si no, kernel_bb_transform/phase1_pca
        // leerían índices fuera de rango (memoria basura o crash).
        if (mode == NoiseMode::BrownianBridge) {
            if (l >= (int)bb_list.size() || !bb_list[l])
                throw std::runtime_error("run_mlqmc_cuda: falta bb_list[" + std::to_string(l)
                    + "] (nivel " + std::to_string(l) + " no precalculado).");
            if (bb_list[l]->N != N_fine)
                throw std::runtime_error("run_mlqmc_cuda: bb_list[" + std::to_string(l)
                    + "] tiene N=" + std::to_string(bb_list[l]->N) + ", se esperaba N_fine="
                    + std::to_string(N_fine) + " para este nivel.");
        }
        if (mode == NoiseMode::PCA) {
            if (l >= (int)pca_list.size() || !pca_list[l])
                throw std::runtime_error("run_mlqmc_cuda: falta pca_list[" + std::to_string(l)
                    + "] (nivel " + std::to_string(l) + " no precalculado).");
            if (pca_list[l]->m != N_fine)
                throw std::runtime_error("run_mlqmc_cuda: pca_list[" + std::to_string(l)
                    + "] tiene m=" + std::to_string(pca_list[l]->m) + ", se esperaba N_fine="
                    + std::to_string(N_fine) + " para este nivel.");
        }

        // Lote tan grande como quepa en la GPU ahora mismo, repartido entre los
        // max_L+1 niveles.
        // En BB/PCA cada nivel reserva además d_dW y el scratch (d_W_scratch o
        // d_Z_f16) junto a d_Z.
        long long buffers_por_nivel = (mode == NoiseMode::Raw) ? 1 : 3;
        long long budget_floats = gpu_free_bytes() / (max_L + 1) / buffers_por_nivel
                                 / (long long)sizeof(float);
        long long batch_cap = std::max(2LL, budget_floats / std::max(D, 1LL));
        // Tope: con D pequeño y GPUs de mucha memoria, batch_cap podría
        // superar INT_MAX y los (int)batch de más abajo se volverían negativos por 
        // desbordamiento de memoria.
        batch_cap = std::min(batch_cap, (long long)INT_MAX / 2);
        batch_cap = (batch_cap + 1) & ~1LL;
        batch_cap = std::min(batch_cap, N_per_rep);

        LevelJob& j = jobs[l];
        long long need = D * batch_cap + 2; // + 2 de margen. Arregla casos límite
        if (need > j.d_Z_cap) {
            if (j.d_Z) cudaFree(j.d_Z);
            CUDA_CHECK(cudaMalloc(&j.d_Z, need * sizeof(float)));
            j.d_Z_cap = need;
        }
        // D == N_fine aquí (BB/PCA exigen d_noise==1, comprobado más arriba).
        if (mode == NoiseMode::BrownianBridge) {
            if (D * batch_cap > j.d_dW_cap) {
                if (j.d_dW) cudaFree(j.d_dW);
                CUDA_CHECK(cudaMalloc(&j.d_dW, D * batch_cap * sizeof(float)));
                j.d_dW_cap = D * batch_cap;
            }
            long long scratch_need = (N_fine + 1) * batch_cap;
            if (scratch_need > j.d_W_scratch_cap) {
                if (j.d_W_scratch) cudaFree(j.d_W_scratch);
                CUDA_CHECK(cudaMalloc(&j.d_W_scratch, scratch_need * sizeof(float)));
                j.d_W_scratch_cap = scratch_need;
            }
        } else if (mode == NoiseMode::PCA) {
            if (D * batch_cap > j.d_dW_cap) {
                if (j.d_dW) cudaFree(j.d_dW);
                CUDA_CHECK(cudaMalloc(&j.d_dW, D * batch_cap * sizeof(float)));
                j.d_dW_cap = D * batch_cap;
            }
            if (D * batch_cap > j.d_Z_f16_cap) {
                if (j.d_Z_f16) cudaFree(j.d_Z_f16);
                CUDA_CHECK(cudaMalloc(&j.d_Z_f16, D * batch_cap * sizeof(__half)));
                j.d_Z_f16_cap = D * batch_cap;
            }
        }

        for (int r = 0; r < R_reps; r++) {
            // salt_idx identifica (nivel, réplica) de forma única para el scramble
            // de Hong-Hickernell: niveles y réplicas distintos son estadísticamente
            // independientes entre sí sin necesitar además carriles de offset
            // disjuntos (dos (l,r) distintos pueden compartir el mismo offset sin
            // colisionar, porque hh_scramble los decorrelaciona).
            unsigned long long salt_idx = (unsigned long long)l * (unsigned long long)R_reps + (unsigned long long)r;

            if (use_sobol && (unsigned long long)(next_off[l][r] + N_per_rep) > (unsigned long long)UINT_MAX)
                throw std::runtime_error("run_mlqmc_cuda: nivel " + std::to_string(l)
                    + ", réplica " + std::to_string(r) + ": offset "
                    + std::to_string(next_off[l][r] + N_per_rep)
                    + " excede el límite de 32 bits del offset de Sobol.");

            CUDA_CHECK(cudaMemsetAsync(j.d_sums[r], 0, 4*sizeof(double), j.stream));

            // d_Z se reutiliza entre réplicas y lotes de este mismo nivel: como todo
            // va al mismo stream j.stream, el orden dentro del stream garantiza que
            // cada generación espera a que el kernel anterior haya leído d_Z.
            long long done = 0;
            while (done < N_per_rep) {
                long long batch = std::min(batch_cap, N_per_rep - done);
                long long gen_count = (D * batch + 1) & ~1LL; // cuRAND exige count par

                // Misma lógica que en MLMC:
                // curandSetStream antes de fijar offset/seed: el kernel de
                // inicialización del generador se lanza ahí.
                if (use_sobol) {
                    // Scramble propio de (nivel, réplica).
                    gen_scrambled_sobol_normal_replica(j.d_Z, D, batch,
                        (unsigned long long)(next_off[l][r] + done),
                        /*replica_salt=*/42u + (unsigned)salt_idx, j.stream);
                } else {
                    curandGenerator_t gen;
                    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
                    CURAND_CHECK(curandSetStream(gen, j.stream));
                    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42u + (unsigned)salt_idx));
                    CURAND_CHECK(curandSetGeneratorOffset(gen,
                        (unsigned long long)(next_off[l][r] + done) * D));
                    CURAND_CHECK(curandGenerateNormal(gen, j.d_Z, gen_count, 0.0f, 1.0f));
                    curandDestroyGenerator(gen);
                }

                // kernel_mlmc espera el incremento ya escalado: en
                // Raw se escala aquí mismo (igual que run_mlmc_cuda); en BB/PCA la
                // transformada ya devuelve el incremento real, sin escalado extra
                // (igual que en run_qmc_cuda).
                // Heston siempre recibe Z (escala internamente vía Cholesky,
                // ver kernel_mlmc): escalar aquí también en modo Raw sería un doble
                // escalado. BB/PCA ya están excluidos para Heston por el guard de
                // model_noise_dim al principio de la función.
                float* d_feed = j.d_Z;
                if (mode == NoiseMode::Raw && mk != ModelKind::Heston) {
                    int blk_s = (int)((gen_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
                    kernel_scale<<<blk_s, BLOCK_SIZE, 0, j.stream>>>(j.d_Z, sqrt_hf, gen_count);
                } else if (mode == NoiseMode::BrownianBridge) {
                    DeviceBBData* dev_bb = bb_list[l];
                    CUDA_CHECK(cudaMemsetAsync(j.d_W_scratch, 0,
                        (N_fine+1)*batch*sizeof(float), j.stream));
                    int blk = (int)(batch + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    kernel_bb_transform<<<blk, BLOCK_SIZE, 0, j.stream>>>(
                        j.d_Z, j.d_dW, j.d_W_scratch,
                        dev_bb->d_map_idx, dev_bb->d_left_idx, dev_bb->d_right_idx,
                        dev_bb->d_wl, dev_bb->d_wr, dev_bb->d_std_dev,
                        N_fine, (int)batch);
                    d_feed = j.d_dW;
                } else if (mode == NoiseMode::PCA) {
                    DevicePCAData* dev_pca = pca_list[l];
                    int blk2 = (int)((D*batch + BLOCK_SIZE - 1) / BLOCK_SIZE);
                    kernel_cast_f32_to_f16<<<blk2, BLOCK_SIZE, 0, j.stream>>>(
                        j.d_Z, j.d_Z_f16, (int)(D*batch));
                    phase1_pca(j.cublas, dev_pca->d_M_pca_f16, j.d_Z_f16, j.d_dW,
                               (int)D, (int)batch);
                    d_feed = j.d_dW;
                }

                int blocks = (int)((batch + BLOCK_SIZE - 1) / BLOCK_SIZE);
                MLMC_TABLE[(int)mk][pidx]<<<blocks, BLOCK_SIZE, 0, j.stream>>>(
                    d_feed, j.d_sums[r], (int)batch, N_fine, N_coarse, M, h_fine, h_coarse, sqrt_hf);

                done += batch;
            }
            next_off[l][r] += N_per_rep;
        }
    };

    // Sincroniza el stream del nivel l y devuelve {media, varianza} de las R medias
    // de réplica, ya combinadas.
    auto collect_level = [&](int l, long long N_per_rep) -> std::pair<double,double> {
        LevelJob& j = jobs[l];
        for (int r = 0; r < R_reps; r++)
            CUDA_CHECK(cudaMemcpyAsync(j.h_sums[r].data(), j.d_sums[r], 4*sizeof(double),
                                        cudaMemcpyDeviceToHost, j.stream));
        CUDA_CHECK(cudaStreamSynchronize(j.stream));

        std::vector<double> rmeans(R_reps);
        for (int r = 0; r < R_reps; r++) rmeans[r] = j.h_sums[r][0] / N_per_rep;

        double m = 0.0;
        for (double v : rmeans) m += v;
        m /= R_reps;
        double v = 0.0;
        for (double rv : rmeans) v += (rv - m) * (rv - m);
        v = (R_reps > 1) ? v / (R_reps - 1) : 0.0;
        return {m, v};
    };

    long long N_pilot = ml_cfg.pilot_n;
    for (int l = 0; l <= L; l++) launch_level(l, N_pilot);
    for (int l = 0; l <= L; l++) {
        auto [em, vv] = collect_level(l, N_pilot);
        E_l[l] = em; sig2_l[l] = vv * (double)N_pilot; N_l[l] = N_pilot;
    }

    bool converged = false;
    int iter = 0;
    while (!converged && L <= max_L && iter++ < 50) {
        double sum_term = 0.0;
        for (int l = 0; l <= L; l++)
            sum_term += std::sqrt(sig2_l[l] * std::pow((double)M, l));

        // Lanza todos los niveles que necesitan refinamiento antes de recoger nada,
        // así se solapan en la GPU en vez de esperar uno a uno.
        std::vector<long long> extra(L+1, 0);
        for (int l = 0; l <= L; l++) {
            double C_l = std::pow((double)M, l);
            // Factor 1/R_reps: el estimador de nivel promedia además sobre las R
            // réplicas (Var(E_l) = sig2_l/(N_l·R_reps)), así que el objetivo
            // Σ sig2_l/(N_l·R_reps) ≤ eps²/2 desplaza el 2/eps² clásico de Giles a
            // 2/(R_reps·eps²) en la asignación óptima de N_l.
            long long N_opt = (long long)std::ceil(
                2.0/((double)R_reps*eps*eps) * std::sqrt(sig2_l[l]/C_l) * sum_term);
            N_opt = std::max(N_opt, 100LL);
            if (N_opt > N_l[l]) {
                extra[l] = N_opt - N_l[l];
                launch_level(l, extra[l]);
            }
        }
        for (int l = 0; l <= L; l++) {
            if (extra[l] > 0) {
                auto [em, vv] = collect_level(l, extra[l]);
                double wold = (double)N_l[l], wnew = (double)extra[l];
                double sig2_new = vv * wnew;
                E_l[l] = (E_l[l]*wold + em*wnew) / (wold + wnew);
                sig2_l[l] = (sig2_l[l]*wold + sig2_new*wnew) / (wold + wnew);
                N_l[l] = N_l[l] + extra[l];
            }
        }

        double bias_est = std::abs(E_l[L]) / std::max(M - 1, 1);
        if (L >= 1) bias_est = std::max(bias_est, std::abs(E_l[L-1]) * M / std::max(M*(M-1), 1));
        converged = (bias_est < eps / std::sqrt(2.0));

        if (!converged && L < max_L) {
            L++;
            launch_level(L, N_pilot);
            auto [em, vv] = collect_level(L, N_pilot);
            E_l[L] = em; sig2_l[L] = vv * (double)N_pilot; N_l[L] = N_pilot;
        }
    }

    for (auto& j : jobs) {
        if (j.d_Z) cudaFree(j.d_Z);
        if (j.d_dW) cudaFree(j.d_dW);
        if (j.d_W_scratch) cudaFree(j.d_W_scratch);
        if (j.d_Z_f16) cudaFree(j.d_Z_f16);
        if (j.cublas) cublasDestroy(j.cublas);
        for (auto* p : j.d_sums) cudaFree(p);
        cudaStreamDestroy(j.stream);
    }

    double price = 0.0, var_sum = 0.0;
    long long N_total = 0;
    for (int l = 0; l <= L; l++) {
        price += E_l[l];
        var_sum += sig2_l[l] / ((double)N_l[l] * (double)R_reps);
        N_total += N_l[l] * R_reps;
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
    kp.E_ctrl = (float)E_ctrl;
    kp.beta_cv = 0.0f; // beta=0 → Y_cv = Y_main, pero el kernel también acumula Y_ctrl
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    float* d_dW = nullptr;
    double* d_sums = nullptr;
    long long D = (long long)n_steps;
    CUDA_CHECK(cudaMalloc(&d_dW, D * N_pilot * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 7 * sizeof(double))); // 7 acumuladores
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

    double N = (double)N_pilot;
    double Ea = hs[0]/N, Ea2 = hs[1]/N;
    double Eg = hs[4]/N, Eg2 = hs[5]/N, Eag = hs[6]/N;
    double var_main = std::max(0.0, Ea2 - Ea*Ea);
    double var_ctrl = std::max(1e-30, Eg2 - Eg*Eg);
    double cov = Eag - Ea*Eg;
    // beta óptimo: beta* = Cov(Y_main, Y_ctrl) / Var(Y_ctrl)
    double beta = std::clamp(cov / var_ctrl, 0.0, 5.0);
    double var_cv = std::max(0.0, var_main - cov*cov/var_ctrl);
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
    kp.E_ctrl = (float)E_ctrl;
    if (mk_main == ModelKind::Dupire)
        kp.sigma = (float)std::get<DupireLocalParams>(main_model).sigma0;
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    long long D = (long long)n_steps;
    float* d_dW = nullptr;
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dW, D * cfg.pilot_n * sizeof(float)));
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

    double mu_cv = hs[2] / cfg.pilot_n;
    double var_cv = std::max(0.0, hs[3]/cfg.pilot_n - mu_cv*mu_cv);
    long long N_needed = (long long)std::ceil(2.0 * var_cv / (eps*eps));
    N_needed = std::max(N_needed, (long long)cfg.pilot_n);
    N_needed = (N_needed + 1) & ~1LL; // cuRAND exige count par

    // Run principal por lotes: acotamos la memoria a
    // D·N_batch en vez de D·N_needed.
    // El kernel acumula en d_sums vía atomicAdd.
    long long batch_cap = std::min(N_needed, (long long)cfg.N_batch);
    CUDA_CHECK(cudaMalloc(&d_dW, D * batch_cap * sizeof(float) + 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 7 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 7*sizeof(double)));

    curandGenerator_t gen2;
    CURAND_CHECK(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen2, cfg.seed + 100));

    long long N_done = 0;
    while (N_done < N_needed) {
        long long batch = std::min(batch_cap, N_needed - N_done);
        long long gen_count = (D * batch + 1) & ~1LL; // cuRAND exige count par
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

    double mean_cv = hs[2] / N_done;
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
    kp.E_ctrl = (float)E_ctrl;
    if (mk_main == ModelKind::Dupire)
        kp.sigma = (float)std::get<DupireLocalParams>(main_model).sigma0;
    // kp es constante para toda la función (no depende de réplica ni de lote), así
    // que se fija una sola vez aquí en vez de en cada iteración.
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    int R = cfg.R;
    // Arranca con tantos puntos como quepan en la GPU (con un mínimo de 512, el
    // suficiente (argumentamos) para estimar varianza inter-réplica), redondeado a potencia de 2
    // (calidad de red-(t,m,s) completa; los duplicados sucesivos lo mantienen así).
    long long budget_floats = gpu_free_bytes() / (long long)sizeof(float);
    long long N_per_rep = std::max(512LL, budget_floats / std::max((long long)n_steps, 1LL));
    N_per_rep = pow2_floor(N_per_rep);
    double var_of_means = 1e30, grand_mean = 0.0;
    long long total_N = 0;
    std::vector<double> rmeans(R);

    // Igual que en run_qmc_cuda: acumular por réplica entre duplicados generando solo
    // los puntos nuevos (rango [N_done[r], N_per_rep)) en vez de tirar el trabajo ya
    // hecho y regenerar todo desde cero. Cada réplica tiene su propio scramble
    // independiente, así que el offset dentro de su propia secuencia es "done",
    // sin carril.
    std::vector<double*> d_sums_r(R, nullptr);
    std::vector<long long> N_done(R, 0);
    for (int r = 0; r < R; r++) {
        CUDA_CHECK(cudaMalloc(&d_sums_r[r], 7*sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sums_r[r], 0, 7*sizeof(double)));
    }

    for (int doublings = 0; doublings < cfg.max_doublings; doublings++) {
        if ((unsigned long long)N_per_rep > (unsigned long long)UINT_MAX)
            throw std::runtime_error("run_qmc_cv_cuda: N_per_rep=" + std::to_string(N_per_rep)
                + " excede el límite de 32 bits del offset de Sobol.");

        // Lote tan grande como quepa en la GPU ahora mismo, topado a INT_MAX/2 
        // puntos para que (int)batch y D*batch no
        // desborden en los kernels.
        long long D = (long long)n_steps;
        long long budget_floats_it = gpu_free_bytes() / (long long)sizeof(float);
        long long batch_cap = std::max(2LL, budget_floats_it / std::max(D, 1LL));
        batch_cap = std::min(batch_cap, (long long)INT_MAX / std::max(D, 1LL) / 2);
        batch_cap = (batch_cap + 1) & ~1LL;

        for (int r = 0; r < R; r++) {
            long long done = N_done[r];
            long long extra = N_per_rep - done;
            double* d_sums = d_sums_r[r];

            long long done_local = 0;
            while (done_local < extra) {
                long long batch = std::min(batch_cap, extra - done_local);
                // +2 de margen y gen_count par: cuRAND exige count par, y D*batch
                // puede ser impar.
                long long gen_count = (D * batch + 1) & ~1LL;
                float* d_dW = nullptr;
                CUDA_CHECK(cudaMalloc(&d_dW, gen_count * sizeof(float) + 2*sizeof(float)));

                // Scramble propio de la réplica r (API de dispositivo): el offset es
                // el progreso dentro de la secuencia YA independiente de esta réplica.
                gen_scrambled_sobol_normal_replica(d_dW, D, batch,
                    (unsigned long long)(done + done_local),
                    /*replica_salt=*/42u + (unsigned)r);

                kernel_scale<<<((int)(D*batch)+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
                    d_dW, sqrtf(kp.h), D*batch);

                int blk = ((int)batch+BLOCK_SIZE-1)/BLOCK_SIZE;
                if (mk_main == ModelKind::GBM)
                    kernel_gbm_asian_cv<<<blk,BLOCK_SIZE>>>(d_dW, d_sums, (int)batch, n_steps);
                else
                    kernel_dupire_gbm_cv<<<blk,BLOCK_SIZE>>>(d_dW, d_sums, (int)batch, n_steps);
                CUDA_CHECK(cudaDeviceSynchronize());

                cudaFree(d_dW);
                done_local += batch;
            }
            if (extra > 0) N_done[r] = N_per_rep;

            double hs[7] = {};
            CUDA_CHECK(cudaMemcpy(hs, d_sums, 7*sizeof(double), cudaMemcpyDeviceToHost));
            rmeans[r] = hs[2] / N_per_rep;
        }

        double m = 0.0;
        for (double v : rmeans) m += v;
        m /= R;
        double v = 0.0;
        for (double rv : rmeans) v += (rv - m)*(rv - m);
        v = (R > 1) ? v / (R - 1) : 0.0;
        var_of_means = v / R;
        grand_mean = m;
        total_N = (long long)R * N_per_rep;

        if (var_of_means < eps*eps / 2.0) break;
        N_per_rep *= 2;
    }
    for (int r = 0; r < R; r++) cudaFree(d_sums_r[r]);

    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {grand_mean, std::sqrt(var_of_means), total_N, t_s};
}


// ==================== //
// Importance Sampling //
// ==================== //

MCResult run_is_cuda(const GBMParams& model, const European& payoff,
                     double z_star, double eps, const MCConfig& cfg) {
    auto t0 = Clock::now();

    KernelParams kp{};
    kp.S0 = (float)model.S0; kp.mu = (float)model.mu; kp.sigma = (float)model.sigma;
    kp.T = (float)model.T;
    kp.K = (float)payoff.K; kp.r = (float)payoff.r;
    kp.discount = (float)std::exp(-payoff.r * payoff.T);
    kp.payoff_kind = PayoffKind::European;

    int n_steps = std::max(4, (int)std::ceil(model.T / eps));
    kp.z_star = (float)(z_star / std::sqrt((double)n_steps));
    kp.h = kp.T / n_steps;
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &kp, sizeof(kp)));

    long long D = (long long)n_steps;
    float* d_Z = nullptr;
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Z, D * cfg.pilot_n * sizeof(float)));
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

    double mu_is = hs[0] / cfg.pilot_n;
    double var_is = std::max(0.0, hs[1]/cfg.pilot_n - mu_is*mu_is);
    // Mismo criterio que el resto de métodos (var < eps²/2, ver run_qmc_cuda)
    long long N_needed = (long long)std::ceil(2.0 * var_is / (eps * eps));
    N_needed = std::max(N_needed, (long long)cfg.pilot_n);
    N_needed = (N_needed + 1) & ~1LL; // cuRAND exige count par

    // Sampling por lotes: a diferencia del resto de métodos, N_needed aquí se calcula
    // de una sola vez, pero puede ser igual de grande a eps pequeño
    // (n_steps también crece con eps) — sin batching, D*N_needed puede no caber en
    // la GPU o desbordar el cast a (int) del lanzamiento del kernel.
    long long batch_cap = std::max(2LL, gpu_free_bytes() / (long long)sizeof(float) / std::max(D, 1LL));
    batch_cap = std::min(batch_cap, (long long)INT_MAX / 2);
    batch_cap = (batch_cap + 1) & ~1LL;
    batch_cap = std::min(batch_cap, N_needed);

    CUDA_CHECK(cudaMalloc(&d_Z, D * batch_cap * sizeof(float) + 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, 4*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 4*sizeof(double)));

    curandGenerator_t gen2;
    CURAND_CHECK(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen2, cfg.seed + 1));

    long long done = 0;
    while (done < N_needed) {
        long long batch = std::min(batch_cap, N_needed - done);
        long long gen_count = (D * batch + 1) & ~1LL; // cuRAND exige count par
        CURAND_CHECK(curandGenerateNormal(gen2, d_Z, gen_count, 0.0f, 1.0f));
        int blk = ((int)batch + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_is_gbm<<<blk, BLOCK_SIZE>>>(d_Z, d_sums, (int)batch, n_steps);
        done += batch;
    }
    curandDestroyGenerator(gen2);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hs, d_sums, 4*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_Z); cudaFree(d_sums);

    double mean = hs[0] / N_needed;
    double var_f = std::max(0.0, hs[1]/N_needed - mean*mean) / N_needed;
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return {mean, std::sqrt(var_f), N_needed, t_s};
}


// Método público de run_mc_fixed_impl (utilizado desde los ejemplos vía SimFn)
std::pair<double, double> run_mc_fixed(const ModelVariant& model,
                                       const PayoffVariant& payoff,
                                       int n_steps, long long n_paths,
                                       unsigned seed) {
    return run_mc_fixed_impl(model, payoff, n_steps, n_paths, seed);
}
