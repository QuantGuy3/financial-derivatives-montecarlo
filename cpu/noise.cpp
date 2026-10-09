#include "noise.hpp"

#include <algorithm>

namespace mc::cpu {

namespace {

class RngStream final : public NoiseStream {
public:
    RngStream(uint64_t seed, Stream stream, uint64_t level, int D, double scale,
              NormalMethod method, uint64_t first)
        : base_(Xoshiro256pp::path_hash_base(seed, stream, level)), D_(D), scale_(scale), method_(method),
          next_(first) {}

    // Por bloques de kLanes caminos: un generador por camino, avanzando intercalados (ver LaneRng).
    void fill(int n, double* Z, int ld) override {
        for (int p0 = 0; p0 < n; p0 += kLanes) {
            const int m = std::min(kLanes, n - p0);
            LaneRng g;
            g.seed(base_, next_);
            next_ += (uint64_t)m;
            fill_normals_lanes(method_, g, m, Z + p0, ld, D_, scale_);
        }
    }

private:
    uint64_t base_;      // hash de (semilla, flujo, nivel)
    int D_;
    double scale_;
    NormalMethod method_;
    uint64_t next_;
};

} // namespace

std::unique_ptr<NoiseStream> RngNoise::open(uint64_t first_path, uint64_t /*count*/) const {
    return std::make_unique<RngStream>(seed_, stream_, level_, D_, scale_, method_, first_path);
}

} // namespace mc::cpu
