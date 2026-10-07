#pragma once
// Fuentes de ruido cuasi-aleatorias: Sobol scrambleado (por réplica) y las transformaciones
// Brownian Bridge y PCA que se aplican sobre normales "crudas" antes del esquema de Euler.

#include "noise.hpp"
#include "sobol.hpp"
#include "../utils.hpp"

namespace mc::cpu {

// Normales de Sobol scrambleado por Hong-Hickernell (una réplica) multiplicadas por `scale`.
// El camino de índice i es el punto i-ésimo de la secuencia scrambleada de la réplica.
class SobolNoise final : public NoiseSource {
public:
    // `sob` debe sobrevivir al objeto; sob.dims() == D.
    SobolNoise(const ScrambledSobol& sob, double scale) : sob_(sob), scale_(scale) {}
    int dim() const override { return sob_.dims(); }
    std::unique_ptr<NoiseStream> open(uint64_t first_path, uint64_t count) const override;

private:
    const ScrambledSobol& sob_;
    double scale_;
};

// Brownian Bridge: Z (normales, el 1er valor fija W_T, los siguientes los puntos medios) ->
// incrementos brownianos dW en orden temporal. Solo ruido de dimensión 1 (D = n_pasos, potencia de 2).
class BrownianBridgeNoise final : public NoiseSource {
public:
    // `inner` debe entregar normales SIN escalar (scale = 1).
    BrownianBridgeNoise(const NoiseSource& inner, const BBData& bb);
    int dim() const override { return bb_.N; }
    std::unique_ptr<NoiseStream> open(uint64_t first_path, uint64_t count) const override;

private:
    const NoiseSource& inner_;
    const BBData& bb_;
};

// PCA del movimiento browniano: dW = M_pca · Z (M_pca column-major, ver utils.hpp). El producto
// se hace por bloques de caminos (GEMM en doble precisión).
class PcaNoise final : public NoiseSource {
public:
    PcaNoise(const NoiseSource& inner, const PCAData& pca);
    int dim() const override { return pca_.m; }
    std::unique_ptr<NoiseStream> open(uint64_t first_path, uint64_t count) const override;

private:
    const NoiseSource& inner_;
    const PCAData& pca_;
};

// Caminos que las transformaciones procesan de una vez (amortizan la lectura de M_pca).
inline constexpr int kTransformBlock = 64;

} // namespace mc::cpu
