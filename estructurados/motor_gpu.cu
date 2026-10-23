// Motor GPU de productos estructurados.
//
// Cada hilo CUDA simula una trayectoria completa llamando al mismo núcleo que el
// motor CPU (simular_trayectoria, simulacion_hd.hpp). El ruido se genera dentro
// del propio hilo, sin buffers intermedios en memoria global:
//   - MC : Philox4x32-10 contrarreloj (contador = trayectoria y dimensión)
//   - QMC: Sobol (vectores de Joe-Kuo de cuRAND) con el scrambling de
//          Hong-Hickernell de la réplica, igual que run_qmc_cuda
// Así la memoria no limita el número de pasos ni de trayectorias, y la GPU solo
// guarda el producto (unos KB) y los acumuladores.
//
// Reducción: cada warp suma sus resultados con __shfl_down_sync y el hilo 0 del
// warp hace un atomicAdd en doble precisión (32 veces menos atómicas que una por
// hilo). Los contadores por fecha (autocancelaciones, cupones cobrados) se
// acumulan en memoria compartida y cada bloque los vuelca una sola vez.

#include "plan.hpp"
#include "valoracion.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace est {

namespace {

#define EST_CK(x)                                                                                 \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " en " #x); \
    } while (0)

constexpr int NACC = 10;   // pv, pv², cupones, anticipada, final, vida, pérdida, ki, ko, nº pérdidas
constexpr int HILOS = 256;
constexpr long long LOTE = 1LL << 22; // trayectorias por lanzamiento

// Contadores por fecha en memoria compartida (los métodos son HD para poder
// instanciar simular_trayectoria; en host no se usan nunca).
struct RegistroGPU {
    unsigned int* ac;
    unsigned int* cup;
    HD_INLINE void autocancelacion(int j) {
#if defined(__CUDA_ARCH__)
        atomicAdd(&ac[j], 1u);
#else
        (void)j;
#endif
    }
    HD_INLINE void cupon(int j) {
#if defined(__CUDA_ARCH__)
        atomicAdd(&cup[j], 1u);
#else
        (void)j;
#endif
    }
};

template<typename Real, bool QMC>
__global__ void __launch_bounds__(HILOS)
kernel_estructurado(SimProducto<Real> sp, long long desde, long long n, uint32_t semilla, uint32_t replica,
                    const uint32_t* __restrict__ dv, uint32_t ndv, double* __restrict__ acc,
                    unsigned long long* __restrict__ cnt_ac, unsigned long long* __restrict__ cnt_cup) {
    extern __shared__ unsigned int s_cnt[]; // [0, nobs): autocancelaciones; [nobs, 2·nobs): cupones
    const int nobs = sp.n_obs;
    for (int i = threadIdx.x; i < 2 * nobs; i += blockDim.x) s_cnt[i] = 0u;
    __syncthreads();

    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double v[NACC];
#pragma unroll
    for (int k = 0; k < NACC; ++k) v[k] = 0.0;

    if (g < n) {
        const long long idx = desde + g;
        ResTray<Real> r{};
        RegistroGPU reg{s_cnt, s_cnt + nobs};
        if constexpr (QMC) {
            RuidoSobol<Real> rz;
            rz.iniciar(dv, ndv, (uint32_t)idx, semilla + replica, semilla, replica);
            simular_trayectoria(sp, rz, reg, r);
        } else {
            RuidoPhilox<Real> rz;
            rz.iniciar(semilla, (uint64_t)idx, 0u);
            simular_trayectoria(sp, rz, reg, r);
        }
        const double pv = (double)r.pv;
        v[0] = pv;
        v[1] = pv * pv;
        v[2] = (double)r.pv_cup;
        v[3] = (double)r.pv_ant;
        v[4] = (double)r.pv_fin;
        v[5] = (double)r.vida;
        v[6] = (double)r.perdida;
        v[7] = (double)r.ki;
        v[8] = (double)r.ko;
        v[9] = (double)r.hubo_perdida;
    }

    // Reducción por warp + una atómica por warp (todos los hilos llegan aquí)
#pragma unroll
    for (int k = 0; k < NACC; ++k) {
        double x = v[k];
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
        if ((threadIdx.x & 31) == 0 && x != 0.0) atomicAdd(&acc[k], x);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nobs; i += blockDim.x) {
        if (s_cnt[i]) atomicAdd(&cnt_ac[i], (unsigned long long)s_cnt[i]);
        if (s_cnt[nobs + i]) atomicAdd(&cnt_cup[i], (unsigned long long)s_cnt[nobs + i]);
    }
}

// Copia de los arreglos del plan a la memoria del dispositivo
template<typename Real>
class PlanDispositivo {
public:
    explicit PlanDispositivo(const PlanSim<Real>& h) {
        vista = h.vista; // escalares y arreglos fijos
        vista.act = subir(h.act);
        vista.chol = subir(h.chol);
        vista.pesos = subir(h.pesos);
        vista.dt = subir(h.dt);
        vista.sqdt = subir(h.sqdt);
        vista.rdt = subir(h.rdt);
        vista.t_ini = subir(h.t_ini);
        vista.rcum_ini = subir(h.rcum_ini);
        vista.df_fin = subir(h.df_fin);
        vista.obs_paso = subir(h.obs_paso);
        vista.t_obs = subir(h.t_obs);
        vista.df_obs = subir(h.df_obs);
        vista.cup_imp = subir(h.cup_imp);
        vista.cup_bar = subir(h.cup_bar);
        vista.ac_activa = subir(h.ac_activa);
        vista.ac_bar = subir(h.ac_bar);
        vista.ac_imp = subir(h.ac_imp);
        vista.bloques = subir(h.bloques);
    }
    ~PlanDispositivo() {
        for (void* p : buffers_) cudaFree(p);
    }
    PlanDispositivo(const PlanDispositivo&) = delete;
    PlanDispositivo& operator=(const PlanDispositivo&) = delete;

    SimProducto<Real> vista{};

private:
    std::vector<void*> buffers_;
    template<typename T>
    const T* subir(const std::vector<T>& v) {
        if (v.empty()) return nullptr;
        T* d = nullptr;
        EST_CK(cudaMalloc(&d, v.size() * sizeof(T)));
        buffers_.push_back(d);
        EST_CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        return d;
    }
};

// Vectores directores de Sobol en el dispositivo (se suben una sola vez)
const uint32_t* direcciones_dispositivo(uint32_t* ndims) {
    static uint32_t* d = nullptr;
    static uint32_t n = 0;
    if (!d) {
        const uint32_t* h = nullptr;
        if (!direcciones_sobol(&h, &n))
            throw std::runtime_error("no se pudieron obtener los vectores directores de Sobol de cuRAND");
        EST_CK(cudaMalloc(&d, (size_t)n * 32 * sizeof(uint32_t)));
        EST_CK(cudaMemcpy(d, h, (size_t)n * 32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    *ndims = n;
    return d;
}

template<typename Real>
class MotorGPU final : public Motor {
public:
    MotorGPU(const Producto& p, uint32_t semilla) : semilla_(semilla) {
        plan_h_ = construir_plan<Real>(p);
        plan_d_ = std::make_unique<PlanDispositivo<Real>>(*plan_h_);
        nobs_ = plan_h_->vista.n_obs;
        EST_CK(cudaMalloc(&d_acc_, NACC * sizeof(double)));
        EST_CK(cudaMalloc(&d_ac_, (size_t)nobs_ * sizeof(unsigned long long)));
        EST_CK(cudaMalloc(&d_cup_, (size_t)nobs_ * sizeof(unsigned long long)));
        cudaDeviceProp prop{};
        EST_CK(cudaGetDeviceProperties(&prop, 0));
        nombre_ = prop.name;
        qmc_ok_ = direcciones_sobol(&dv_h_, &ndv_);
    }
    ~MotorGPU() override {
        cudaFree(d_acc_);
        cudaFree(d_ac_);
        cudaFree(d_cup_);
    }

    std::string descripcion() const override {
        return "GPU " + nombre_ + (sizeof(Real) == 4 ? " (precisión simple)" : " (doble precisión)");
    }
    bool admite_qmc() const override { return qmc_ok_; }
    bool es_gpu() const override { return true; }

    void mc(long long desde, long long hasta, Acumulado& acc) override { lanzar(false, 0, desde, hasta, acc); }

    void qmc(int replica, long long desde, long long hasta, Acumulado& acc) override {
        if (hasta > 4294967296LL) throw std::runtime_error("QMC: más de 2^32 puntos por réplica");
        lanzar(true, replica, desde, hasta, acc);
    }

private:
    void lanzar(bool qmc, int replica, long long desde, long long hasta, Acumulado& acc) {
        if (hasta <= desde) return;
        uint32_t ndv = 0;
        const uint32_t* dv = qmc ? direcciones_dispositivo(&ndv) : nullptr;
        EST_CK(cudaMemset(d_acc_, 0, NACC * sizeof(double)));
        EST_CK(cudaMemset(d_ac_, 0, (size_t)nobs_ * sizeof(unsigned long long)));
        EST_CK(cudaMemset(d_cup_, 0, (size_t)nobs_ * sizeof(unsigned long long)));
        const size_t shm = 2 * (size_t)nobs_ * sizeof(unsigned int);
        for (long long a = desde; a < hasta; a += LOTE) {
            const long long n = std::min(LOTE, hasta - a);
            const int bloques = (int)((n + HILOS - 1) / HILOS);
            if (qmc)
                kernel_estructurado<Real, true><<<bloques, HILOS, shm>>>(
                    plan_d_->vista, a, n, semilla_, (uint32_t)replica, dv, ndv, d_acc_, d_ac_, d_cup_);
            else
                kernel_estructurado<Real, false><<<bloques, HILOS, shm>>>(
                    plan_d_->vista, a, n, semilla_, 0u, nullptr, 0u, d_acc_, d_ac_, d_cup_);
            EST_CK(cudaGetLastError());
        }
        EST_CK(cudaDeviceSynchronize());

        double h[NACC];
        std::vector<unsigned long long> hac((size_t)nobs_), hcup((size_t)nobs_);
        EST_CK(cudaMemcpy(h, d_acc_, NACC * sizeof(double), cudaMemcpyDeviceToHost));
        EST_CK(cudaMemcpy(hac.data(), d_ac_, (size_t)nobs_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        EST_CK(cudaMemcpy(hcup.data(), d_cup_, (size_t)nobs_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        if (acc.n_ac.size() != (size_t)nobs_) {
            Acumulado vacio;
            vacio.iniciar(nobs_);
            vacio.combinar(acc);
            acc = vacio;
        }
        acc.n += (double)(hasta - desde);
        acc.s_pv += h[0];
        acc.s_pv2 += h[1];
        acc.s_cup += h[2];
        acc.s_ant += h[3];
        acc.s_fin += h[4];
        acc.s_vida += h[5];
        acc.s_perd += h[6];
        acc.n_ki += h[7];
        acc.n_ko += h[8];
        acc.n_perd += h[9];
        for (int j = 0; j < nobs_; ++j) {
            acc.n_ac[(size_t)j] += (double)hac[(size_t)j];
            acc.n_cup[(size_t)j] += (double)hcup[(size_t)j];
        }
    }

    std::unique_ptr<PlanSim<Real>> plan_h_;
    std::unique_ptr<PlanDispositivo<Real>> plan_d_;
    uint32_t semilla_;
    int nobs_ = 0;
    double* d_acc_ = nullptr;
    unsigned long long* d_ac_ = nullptr;
    unsigned long long* d_cup_ = nullptr;
    std::string nombre_;
    bool qmc_ok_ = false;
    const uint32_t* dv_h_ = nullptr;
    uint32_t ndv_ = 0;
};

} // namespace

bool gpu_disponible(std::string* descripcion) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) {
        if (descripcion) *descripcion = (e != cudaSuccess) ? cudaGetErrorString(e) : "no se ha encontrado ninguna GPU";
        cudaGetLastError(); // limpiar el error pegajoso
        return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && descripcion) *descripcion = prop.name;
    return true;
}

std::unique_ptr<Motor> crear_motor_gpu(const Producto& p, uint32_t semilla, bool doble_precision) {
    if (doble_precision) return std::make_unique<MotorGPU<double>>(p, semilla);
    return std::make_unique<MotorGPU<float>>(p, semilla);
}

} // namespace est
