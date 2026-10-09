#include "qmc_noise.hpp"

#include "../mc_types.hpp"
#include "simd.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace mc::cpu {

// ---- Sobol ----------------------------------------------------------------------------------

namespace {

class SobolStream final : public NoiseStream {
public:
    SobolStream(const ScrambledSobol& sob, double scale, uint64_t first)
        : sob_(sob), scale_(scale), D_(sob.dims()), next_(first), x_((size_t)D_), tmp_((size_t)D_) {
        sob_.state_at(first, x_.data());   // lanza SobolLimitReached si first > 2^32-1
    }

    void fill(int n, double* Z, int ld) override {
        for (int p = 0; p < n; p++) {
            if (next_ > (uint64_t)UINT_MAX) throw SobolLimitReached{};
            if (started_) sob_.advance(next_, x_.data());
            started_ = true;
            sob_.normals_from_state(x_.data(), tmp_.data());
            for (int d = 0; d < D_; d++) Z[(size_t)d * ld + p] = tmp_[d] * scale_;
            ++next_;
        }
    }

private:
    const ScrambledSobol& sob_;
    double scale_;
    int D_;
    uint64_t next_;
    bool started_ = false;
    std::vector<uint32_t> x_;
    std::vector<double> tmp_;
};

// ---- núcleos de las transformaciones (se compilan también para AVX2, ver simd.hpp) ------------------
//
// En los dos el bucle interno recorre los caminos del bloque (p) y cada camino tiene su propio
// acumulador, así que vectorizar a 2 o a 4 carriles no cambia el orden de ninguna suma.

// Brownian Bridge: W en los nodos del puente y después incrementos. Z, W y out son [fila][PB].
MC_ALWAYS_INLINE void bb_transform_body(const BBData& bb, const double* Z, double* W, double* out, int m) {
    constexpr int PB = kTransformBlock;
    const int N = bb.N;
    std::memset(W, 0, sizeof(double) * PB);                      // W[0] = 0
    {
        double* wt = W + (size_t)bb.map_idx[0] * PB;
        const double sd = bb.std_dev[0];
        for (int p = 0; p < m; p++) wt[p] = sd * Z[p];
    }
    for (int step = 1; step < N; step++) {
        double* wm = W + (size_t)bb.map_idx[step] * PB;
        const double* wl = W + (size_t)bb.left_idx[step] * PB;
        const double* wr = W + (size_t)bb.right_idx[step] * PB;
        const double a = bb.weight_left[step], b = bb.weight_right[step], sd = bb.std_dev[step];
        const double* z = Z + (size_t)step * PB;
        for (int p = 0; p < m; p++) wm[p] = a * wl[p] + b * wr[p] + sd * z[p];
    }
    for (int k = 0; k < N; k++) {
        const double* w1 = W + (size_t)(k + 1) * PB;
        const double* w0 = W + (size_t)k * PB;
        double* o = out + (size_t)k * PB;
        for (int p = 0; p < m; p++) o[p] = w1[p] - w0[p];
    }
}

// PCA: dW[i][p] = sum_k M(i,k) * Z[k][p]; M(i,k) = M[i + k*D] (column-major). GEMM por bloques de
// IB filas para que el acumulador quepa en L1.
MC_ALWAYS_INLINE void pca_transform_body(const double* M, int D, const double* Z, double* out, int m_cols) {
    constexpr int PB = kTransformBlock;
    constexpr int IB = 32;
    double acc[IB][PB];
    for (int i0 = 0; i0 < D; i0 += IB) {
        const int ib = std::min(IB, D - i0);
        for (int i = 0; i < ib; i++) std::memset(acc[i], 0, sizeof(double) * PB);
        for (int k = 0; k < D; k++) {
            const double* col = M + (size_t)k * D + i0;
            const double* zk = Z + (size_t)k * PB;
            for (int i = 0; i < ib; i++) {
                const double mik = col[i];
                for (int p = 0; p < m_cols; p++) acc[i][p] += mik * zk[p];
            }
        }
        for (int i = 0; i < ib; i++)
            std::memcpy(out + (size_t)(i0 + i) * PB, acc[i], sizeof(double) * PB);
    }
}

void bb_transform_generic(const BBData& bb, const double* Z, double* W, double* out, int m) { bb_transform_body(bb, Z, W, out, m); }
void pca_transform_generic(const double* M, int D, const double* Z, double* out, int m) { pca_transform_body(M, D, Z, out, m); }
#ifdef MC_HAVE_AVX2_CLONES
MC_TARGET_AVX2 void bb_transform_avx2(const BBData& bb, const double* Z, double* W, double* out, int m) { bb_transform_body(bb, Z, W, out, m); }
MC_TARGET_AVX2 void pca_transform_avx2(const double* M, int D, const double* Z, double* out, int m) { pca_transform_body(M, D, Z, out, m); }
#endif

inline bool use_avx2_clones() {
#if defined(MC_HAVE_AVX2_CLONES) && defined(MC_X86_64)
    return simd_level() == SimdLevel::Avx2;
#else
    return false;
#endif
}

// Base de las transformaciones por bloques: produce kTransformBlock caminos de una vez en un
// buffer intermedio [D][PB] y entrega trozos de n carriles.
class StagedStream : public NoiseStream {
public:
    StagedStream(std::unique_ptr<NoiseStream> inner, int D, uint64_t count)
        : inner_(std::move(inner)), D_(D), stage_((size_t)D * kTransformBlock),
          zin_((size_t)D * kTransformBlock), remaining_(count) {}

    void fill(int n, double* Z, int ld) override {
        for (int p = 0; p < n; p++) {
            if (pos_ == avail_) refill();
            for (int d = 0; d < D_; d++) Z[(size_t)d * ld + p] = stage_[(size_t)d * kTransformBlock + pos_];
            ++pos_;
        }
    }

protected:
    // Transforma zin_[D][PB] (primeros m carriles válidos) en stage_[D][PB].
    virtual void transform(int m) = 0;

    std::unique_ptr<NoiseStream> inner_;
    int D_;
    std::vector<double> stage_, zin_;

private:
    void refill() {
        // No se genera más ruido del que el consumidor va a pedir (remaining_).
        const int m = (int)std::min<uint64_t>(kTransformBlock, remaining_);
        if (m <= 0) throw std::logic_error("StagedStream: se pidieron más caminos de los anunciados");
        inner_->fill(m, zin_.data(), kTransformBlock);
        transform(m);
        remaining_ -= (uint64_t)m;
        avail_ = m;
        pos_ = 0;
    }

    uint64_t remaining_;
    int pos_ = 0, avail_ = 0;
};

class BbStream final : public StagedStream {
public:
    BbStream(std::unique_ptr<NoiseStream> inner, const BBData& bb, uint64_t count)
        : StagedStream(std::move(inner), bb.N, count), bb_(bb), W_((size_t)(bb.N + 1) * kTransformBlock) {}

protected:
    void transform(int m) override {
#ifdef MC_HAVE_AVX2_CLONES
        if (use_avx2_clones()) { bb_transform_avx2(bb_, zin_.data(), W_.data(), stage_.data(), m); return; }
#endif
        bb_transform_generic(bb_, zin_.data(), W_.data(), stage_.data(), m);
    }

private:
    const BBData& bb_;
    std::vector<double> W_;
};

class PcaStream final : public StagedStream {
public:
    PcaStream(std::unique_ptr<NoiseStream> inner, const PCAData& pca, uint64_t count)
        : StagedStream(std::move(inner), pca.m, count), pca_(pca) {}

protected:
    void transform(int m_cols) override {
#ifdef MC_HAVE_AVX2_CLONES
        if (use_avx2_clones()) { pca_transform_avx2(pca_.M_pca.data(), pca_.m, zin_.data(), stage_.data(), m_cols); return; }
#endif
        pca_transform_generic(pca_.M_pca.data(), pca_.m, zin_.data(), stage_.data(), m_cols);
    }

private:
    const PCAData& pca_;
};

} // namespace

std::unique_ptr<NoiseStream> SobolNoise::open(uint64_t first_path, uint64_t /*count*/) const {
    return std::make_unique<SobolStream>(sob_, scale_, first_path);
}

BrownianBridgeNoise::BrownianBridgeNoise(const NoiseSource& inner, const BBData& bb)
    : inner_(inner), bb_(bb) {
    if (inner.dim() != bb.N)
        throw std::invalid_argument("BrownianBridgeNoise: la dimensión del ruido no coincide con N del puente");
}

std::unique_ptr<NoiseStream> BrownianBridgeNoise::open(uint64_t first_path, uint64_t count) const {
    return std::make_unique<BbStream>(inner_.open(first_path, count), bb_, count);
}

PcaNoise::PcaNoise(const NoiseSource& inner, const PCAData& pca) : inner_(inner), pca_(pca) {
    if (inner.dim() != pca.m)
        throw std::invalid_argument("PcaNoise: la dimensión del ruido no coincide con m de la PCA");
}

std::unique_ptr<NoiseStream> PcaNoise::open(uint64_t first_path, uint64_t count) const {
    return std::make_unique<PcaStream>(inner_.open(first_path, count), pca_, count);
}

} // namespace mc::cpu
