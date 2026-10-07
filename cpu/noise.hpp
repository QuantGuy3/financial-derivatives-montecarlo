#pragma once
// Fuentes de ruido del motor CPU.
//
// Una NoiseSource describe un generador de "vectores de ruido por camino" (D valores por
// camino): normales pseudoaleatorias, normales de Sobol scrambleado, o esas mismas
// transformadas por Brownian Bridge / PCA. `open(first_path, count)` crea un NoiseStream: un cursor
// con estado propio (un hilo, un chunk) que entrega los caminos first_path, first_path+1, ...
// en bloques. Así el Sobol puede avanzar por Gray-code incremental (1 XOR por dimensión y
// punto) en vez de reconstruir el estado en cada bloque.
//
// Layout de salida de fill: Z[d*ld + p] (d = dimensión, p = camino dentro del bloque), el mismo
// que usa la GPU ([k*N_paths + p]).

#include "normal.hpp"
#include "rng.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace mc::cpu {

class NoiseStream {
public:
    virtual ~NoiseStream() = default;
    // Escribe Z[d*ld + p] para d in [0,D), p in [0,n) con los siguientes n caminos del cursor.
    // Los carriles p >= n no se tocan (los rellena el llamante).
    virtual void fill(int n, double* Z, int ld) = 0;
};

class NoiseSource {
public:
    virtual ~NoiseSource() = default;
    virtual int dim() const = 0;   // D por camino
    // `count` = nº de caminos que se pedirán al cursor como máximo (permite no generar de más).
    virtual std::unique_ptr<NoiseStream> open(uint64_t first_path, uint64_t count) const = 0;
};

// Normales pseudoaleatorias i.i.d. multiplicadas por `scale` (sqrt(h) para obtener dW).
// El camino de índice i usa el estado Xoshiro256pp::for_path(seed, stream, level, i).
class RngNoise final : public NoiseSource {
public:
    RngNoise(uint64_t seed, Stream stream, uint64_t level, int D, double scale,
             NormalMethod method = NormalMethod::BoxMuller)
        : seed_(seed), stream_(stream), level_(level), D_(D), scale_(scale), method_(method) {}
    int dim() const override { return D_; }
    std::unique_ptr<NoiseStream> open(uint64_t first_path, uint64_t count) const override;

private:
    uint64_t seed_;
    Stream stream_;
    uint64_t level_;
    int D_;
    double scale_;
    NormalMethod method_;
};

} // namespace mc::cpu
