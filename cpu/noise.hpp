#pragma once
// Fuentes de ruido por bloque del motor CPU.
//
// Una NoiseSource entrega, para un bloque de n caminos consecutivos, las D normales (o los D
// incrementos brownianos) de cada camino en el layout de la GPU Z[d*ld + p]. Las fuentes
// aleatorias derivan el estado de cada camino de su índice (ver rng.hpp), así que el
// resultado no depende del reparto de trabajo.

#include "normal.hpp"
#include "rng.hpp"

#include <cstdint>

namespace mc::cpu {

class NoiseSource {
public:
    virtual ~NoiseSource() = default;
    virtual int dim() const = 0;   // D por camino
    // Escribe Z[d*ld + p] para d in [0,D), p in [0,n) del bloque de caminos first..first+n-1.
    // (Los carriles p >= n no se tocan: los rellena el llamante.)
    virtual void fill(uint64_t first_path, int n, double* Z, int ld) const = 0;
};

// Normales pseudoaleatorias i.i.d. multiplicadas por `scale` (sqrt(h) para obtener dW).
class RngNoise final : public NoiseSource {
public:
    RngNoise(uint64_t seed, Stream stream, uint64_t level, int D, double scale,
             NormalMethod method = NormalMethod::BoxMuller)
        : seed_(seed), stream_(stream), level_(level), D_(D), scale_(scale), method_(method) {}
    int dim() const override { return D_; }
    void fill(uint64_t first_path, int n, double* Z, int ld) const override;

private:
    uint64_t seed_;
    Stream stream_;
    uint64_t level_;
    int D_;
    double scale_;
    NormalMethod method_;
};

} // namespace mc::cpu
