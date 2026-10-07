#include "noise.hpp"

namespace mc::cpu {

namespace {

class RngStream final : public NoiseStream {
public:
    RngStream(uint64_t seed, Stream stream, uint64_t level, int D, double scale,
              NormalMethod method, uint64_t first)
        : seed_(seed), stream_(stream), level_(level), D_(D), scale_(scale), method_(method),
          next_(first), tmp_((size_t)D) {}

    void fill(int n, double* Z, int ld) override {
        for (int p = 0; p < n; p++) {
            Xoshiro256pp g = Xoshiro256pp::for_path(seed_, stream_, level_, next_++);
            fill_normals(method_, g, tmp_.data(), D_);
            for (int d = 0; d < D_; d++) Z[(size_t)d * ld + p] = tmp_[d] * scale_;
        }
    }

private:
    uint64_t seed_;
    Stream stream_;
    uint64_t level_;
    int D_;
    double scale_;
    NormalMethod method_;
    uint64_t next_;
    std::vector<double> tmp_;
};

} // namespace

std::unique_ptr<NoiseStream> RngNoise::open(uint64_t first_path, uint64_t /*count*/) const {
    return std::make_unique<RngStream>(seed_, stream_, level_, D_, scale_, method_, first_path);
}

} // namespace mc::cpu
