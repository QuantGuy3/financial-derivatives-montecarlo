// Motor de simulación de productos estructurados en CPU (multihilo, doble
// precisión). Usa exactamente el mismo núcleo que la GPU (simulacion_hd.hpp):
// sirve de referencia, para validar y para trabajar sin GPU.

#include "plan.hpp"
#include "valoracion.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace est {

namespace {

struct RegistroCPU {
    double* ac;
    double* cup;
    void autocancelacion(int j) { ac[j] += 1.0; }
    void cupon(int j) { cup[j] += 1.0; }
};

class MotorCPU final : public Motor {
public:
    MotorCPU(const Producto& p, uint32_t semilla, int hilos)
        : plan_(construir_plan<double>(p)), semilla_(semilla) {
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        hilos_ = hilos > 0 ? hilos : (int)hw;
        qmc_ok_ = direcciones_sobol(&dv_, &ndv_);
    }

    std::string descripcion() const override {
        return "CPU (" + std::to_string(hilos_) + " hilos, doble precisión)";
    }
    bool admite_qmc() const override { return qmc_ok_; }
    bool es_gpu() const override { return false; }

    void mc(long long desde, long long hasta, Acumulado& acc) override {
        ejecutar(desde, hasta, acc, [this](long long idx, RegistroCPU& reg, ResTray<double>& r) {
            RuidoPhilox<double> rz;
            rz.iniciar(semilla_, (uint64_t)idx, 0u);
            simular_trayectoria(plan_->vista, rz, reg, r);
        });
    }

    void qmc(int replica, long long desde, long long hasta, Acumulado& acc) override {
        if (!qmc_ok_) throw std::runtime_error("QMC no disponible en CPU (hace falta cuRAND para las direcciones de Sobol)");
        if (hasta > 4294967296LL) throw std::runtime_error("QMC: más de 2^32 puntos por réplica");
        const uint32_t sal = semilla_ + (uint32_t)replica;
        ejecutar(desde, hasta, acc, [this, replica, sal](long long idx, RegistroCPU& reg, ResTray<double>& r) {
            RuidoSobol<double> rz;
            rz.iniciar(dv_, ndv_, (uint32_t)idx, sal, semilla_, (uint32_t)replica);
            simular_trayectoria(plan_->vista, rz, reg, r);
        });
    }

private:
    template<class F>
    void ejecutar(long long desde, long long hasta, Acumulado& acc, F&& simular) {
        const long long total = hasta - desde;
        if (total <= 0) return;
        const int nobs = plan_->vista.n_obs;
        int nh = (int)std::min<long long>(hilos_, std::max<long long>(1, total / 64));
        std::vector<Acumulado> parciales((size_t)nh);
        auto trabajo = [&](int h) {
            Acumulado& a = parciales[(size_t)h];
            a.iniciar(nobs);
            RegistroCPU reg{a.n_ac.data(), a.n_cup.data()};
            long long ini = desde + total * h / nh, fin = desde + total * (h + 1) / nh;
            ResTray<double> r{};
            for (long long idx = ini; idx < fin; ++idx) {
                simular(idx, reg, r);
                a.sumar(r);
            }
        };
        if (nh == 1) {
            trabajo(0);
        } else {
            std::vector<std::thread> th;
            for (int h = 0; h < nh; ++h) th.emplace_back(trabajo, h);
            for (auto& t : th) t.join();
        }
        if (acc.n_ac.size() != (size_t)nobs) {
            Acumulado vacio;
            vacio.iniciar(nobs);
            vacio.combinar(acc);
            acc = vacio;
        }
        for (const auto& a : parciales) acc.combinar(a);
    }

    std::unique_ptr<PlanSim<double>> plan_;
    uint32_t semilla_;
    int hilos_ = 1;
    bool qmc_ok_ = false;
    const uint32_t* dv_ = nullptr;
    uint32_t ndv_ = 0;
};

} // namespace

std::unique_ptr<Motor> crear_motor_cpu(const Producto& p, uint32_t semilla, int hilos) {
    return std::make_unique<MotorCPU>(p, semilla, hilos);
}

} // namespace est
