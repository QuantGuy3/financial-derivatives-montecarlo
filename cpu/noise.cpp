#include "noise.hpp"

#include <vector>

namespace mc::cpu {

void RngNoise::fill(uint64_t first_path, int n, double* Z, int ld) const {
    thread_local std::vector<double> tmp;
    if ((int)tmp.size() < D_) tmp.resize(D_);
    for (int p = 0; p < n; p++) {
        Xoshiro256pp g = Xoshiro256pp::for_path(seed_, stream_, level_, first_path + (uint64_t)p);
        fill_normals(method_, g, tmp.data(), D_);
        for (int d = 0; d < D_; d++) Z[(size_t)d * ld + p] = tmp[d] * scale_;
    }
}

} // namespace mc::cpu
