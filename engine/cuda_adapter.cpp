#include "cuda_adapter.hpp"

#include "../cpu/vr_cpu.hpp"   // is_n_steps (misma fórmula que run_is_cuda / run_qmc_is_cuda)

#include <cmath>
#include <map>
#include <mutex>

namespace mc::detail {

namespace {

std::mutex g_mutex;
using Key = std::pair<int, long long>;   // (N, round(T*1e6))
std::map<Key, DeviceBBData*>  g_bb;
std::map<Key, DevicePCAData*> g_pca;

Key key_of(int n, double T) { return {n, (long long)std::llround(T * 1e6)}; }

DeviceBBData* bb_for(int n, double T) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_bb.find(key_of(n, T));
    if (it == g_bb.end()) it = g_bb.emplace(key_of(n, T), bb_upload(bb_precompute(n, T))).first;
    return it->second;
}

DevicePCAData* pca_for(int n, double T) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pca.find(key_of(n, T));
    if (it == g_pca.end()) it = g_pca.emplace(key_of(n, T), pca_upload(pca_compute(n, T))).first;
    return it->second;
}

struct Handles {
    DeviceBBData* bb = nullptr;
    DevicePCAData* pca = nullptr;
};

Handles single_handles(NoiseMode mode, int n_steps, double T) {
    Handles h;
    if (mode == NoiseMode::BrownianBridge) h.bb = bb_for(n_steps, T);
    else if (mode == NoiseMode::PCA) h.pca = pca_for(n_steps, T);
    return h;
}

struct LevelHandles {
    std::vector<DeviceBBData*> bb;
    std::vector<DevicePCAData*> pca;
};

// Un manejador por nivel con N_fine = M^l pasos (la GPU exige que coincida con el nivel).
LevelHandles level_handles(NoiseMode mode, const MLMCConfig& ml, double T) {
    LevelHandles h;
    if (mode == NoiseMode::Raw) return h;
    for (int l = 0; l <= ml.max_L; l++) {
        const int n = (l == 0) ? 1 : (int)std::llround(std::pow((double)ml.M, l));
        if (mode == NoiseMode::BrownianBridge) h.bb.push_back(bb_for(n, T));
        else h.pca.push_back(pca_for(n, T));
    }
    return h;
}

} // namespace

bool cuda_compiled_in() {
#ifdef MC_HAS_CUDA
    return true;
#else
    return false;
#endif
}

bool cuda_present() { return cuda_compiled_in() && ::cuda_device_available(); }

void cuda_release_cache() {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (auto& [k, v] : g_bb) bb_free(v);
    for (auto& [k, v] : g_pca) pca_free(v);
    g_bb.clear();
    g_pca.clear();
}

MCResult cuda_mc(const ModelVariant& m, const PayoffVariant& p, double eps, int n_steps, const MCConfig& c) {
    return ::run_mc_cuda(m, p, eps, n_steps, c);
}

MCResult cuda_qmc(const ModelVariant& m, const PayoffVariant& p, double eps, int n_steps,
                  const QMCConfig& c, NoiseMode mode) {
    Handles h = single_handles(mode, n_steps, model_T(m));
    return ::run_qmc_cuda(m, p, eps, n_steps, c, mode, h.bb, h.pca);
}

MCResult cuda_mlmc(const ModelVariant& m, const PayoffVariant& p, double eps, const MLMCConfig& c) {
    return ::run_mlmc_cuda(m, p, eps, c);
}

MCResult cuda_mlqmc(const ModelVariant& m, const PayoffVariant& p, double eps, const MLMCConfig& ml,
                    const QMCConfig& q, NoiseMode mode) {
    LevelHandles h = level_handles(mode, ml, model_T(m));
    return ::run_mlqmc_cuda(m, p, eps, ml, q, mode, h.bb, h.pca);
}

std::pair<double, double> cuda_mc_fixed(const ModelVariant& m, const PayoffVariant& p, int n_steps,
                                        long long n_paths, unsigned seed) {
    return ::run_mc_fixed(m, p, n_steps, n_paths, seed);
}

CVPilot cuda_cv_pilot(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                      const PayoffVariant& cp, double E_ctrl, int n_steps, int N_pilot, unsigned seed) {
    return ::cv_pilot(mm, cm, mp, cp, E_ctrl, n_steps, N_pilot, seed);
}

MCResult cuda_mc_cv(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                    const PayoffVariant& cp, double E_ctrl, double beta, double eps, int n_steps,
                    const MCConfig& c) {
    return ::run_mc_cv_cuda(mm, cm, mp, cp, E_ctrl, beta, eps, n_steps, c);
}

MCResult cuda_qmc_cv(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                     const PayoffVariant& cp, double E_ctrl, double beta, double eps, int n_steps,
                     const QMCConfig& c, NoiseMode mode) {
    Handles h = single_handles(mode, n_steps, model_T(mm));
    return ::run_qmc_cv_cuda(mm, cm, mp, cp, E_ctrl, beta, eps, n_steps, c, mode, h.bb, h.pca);
}

MCResult cuda_mlmc_cv(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                      const PayoffVariant& cp, double E_ctrl, double beta, double eps, const MLMCConfig& c) {
    return ::run_mlmc_cv_cuda(mm, cm, mp, cp, E_ctrl, beta, eps, c);
}

MCResult cuda_mlqmc_cv(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                       const PayoffVariant& cp, double E_ctrl, double beta, double eps,
                       const MLMCConfig& ml, const QMCConfig& q, NoiseMode mode) {
    LevelHandles h = level_handles(mode, ml, model_T(mm));
    return ::run_mlqmc_cv_cuda(mm, cm, mp, cp, E_ctrl, beta, eps, ml, q, mode, h.bb, h.pca);
}

MCResult cuda_is(const GBMParams& m, const European& p, double z_star, double eps, const MCConfig& c) {
    return ::run_is_cuda(m, p, z_star, eps, c);
}

MCResult cuda_qmc_is(const GBMParams& m, const European& p, double z_star, double eps,
                     const QMCConfig& c, NoiseMode mode) {
    Handles h = single_handles(mode, cpu::is_n_steps(m.T, eps), m.T);
    return ::run_qmc_is_cuda(m, p, z_star, eps, c, mode, h.bb, h.pca);
}

MCResult cuda_mlmc_is(const GBMParams& m, const European& p, double z_star, double eps, const MLMCConfig& c) {
    return ::run_mlmc_is_cuda(m, p, z_star, eps, c);
}

MCResult cuda_mlqmc_is(const GBMParams& m, const European& p, double z_star, double eps,
                       const MLMCConfig& ml, const QMCConfig& q, NoiseMode mode) {
    LevelHandles h = level_handles(mode, ml, m.T);
    return ::run_mlqmc_is_cuda(m, p, z_star, eps, ml, q, mode, h.bb, h.pca);
}

} // namespace mc::detail
