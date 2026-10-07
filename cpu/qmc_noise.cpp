#include "qmc_noise.hpp"

#include "../mc_types.hpp"

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
        constexpr int PB = kTransformBlock;
        const int N = bb_.N;
        double* W = W_.data();
        const double* Z = zin_.data();
        std::memset(W, 0, sizeof(double) * PB);                      // W[0] = 0
        {
            double* wt = W + (size_t)bb_.map_idx[0] * PB;
            const double sd = bb_.std_dev[0];
            for (int p = 0; p < m; p++) wt[p] = sd * Z[p];
        }
        for (int step = 1; step < N; step++) {
            double* wm = W + (size_t)bb_.map_idx[step] * PB;
            const double* wl = W + (size_t)bb_.left_idx[step] * PB;
            const double* wr = W + (size_t)bb_.right_idx[step] * PB;
            const double a = bb_.weight_left[step], b = bb_.weight_right[step], sd = bb_.std_dev[step];
            const double* z = Z + (size_t)step * PB;
            for (int p = 0; p < m; p++) wm[p] = a * wl[p] + b * wr[p] + sd * z[p];
        }
        for (int k = 0; k < N; k++) {
            const double* w1 = W + (size_t)(k + 1) * PB;
            const double* w0 = W + (size_t)k * PB;
            double* out = stage_.data() + (size_t)k * PB;
            for (int p = 0; p < m; p++) out[p] = w1[p] - w0[p];
        }
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
    // dW[i][p] = sum_k M(i,k) * Z[k][p]; M(i,k) = M_pca[i + k*m] (column-major).
    void transform(int m_cols) override {
        constexpr int PB = kTransformBlock;
        constexpr int IB = 32;                       // filas de dW por bloque (acumulador en L1)
        const int D = pca_.m;
        const double* M = pca_.M_pca.data();
        const double* Z = zin_.data();
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
                std::memcpy(stage_.data() + (size_t)(i0 + i) * PB, acc[i], sizeof(double) * PB);
        }
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
