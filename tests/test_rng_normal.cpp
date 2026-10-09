// RNG por camino y normales: momentos, Kolmogorov-Smirnov, inversa de la CDF.
#include "doctest.h"
#include "cpu/noise.hpp"
#include "cpu/normal.hpp"
#include "cpu/rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace mc::cpu;

namespace {

struct Mom { double mean, var, skew, exkurt; };

Mom moments(const std::vector<double>& x) {
    const double n = (double)x.size();
    double m = 0; for (double v : x) m += v; m /= n;
    double m2 = 0, m3 = 0, m4 = 0;
    for (double v : x) { double d = v - m; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; }
    m2 /= n; m3 /= n; m4 /= n;
    return {m, m2, m3 / std::pow(m2, 1.5), m4 / (m2 * m2) - 3.0};
}

// Estadistico de Kolmogorov-Smirnov frente a la CDF normal
double ks_statistic(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    const double n = (double)x.size();
    double D = 0;
    for (size_t i = 0; i < x.size(); i++) {
        double F = norm_cdf(x[i]);
        D = std::max({D, std::abs((double)(i + 1) / n - F), std::abs(F - (double)i / n)});
    }
    return D;
}

// Generador secuencial clásico (un Xoshiro256pp por camino, normales una detrás de otra).
std::vector<double> gen_normals_seq(NormalMethod m, size_t n_paths, int per_path, uint64_t seed) {
    std::vector<double> out(n_paths * per_path);
    for (size_t p = 0; p < n_paths; p++) {
        auto g = Xoshiro256pp::for_path(seed, Stream::Test, 0, p);
        fill_normals(m, g, out.data() + p * per_path, per_path);
    }
    return out;
}

// Generador del motor (RngNoise, por bloques de kLanes caminos). out[p*per_path + d].
std::vector<double> gen_normals(NormalMethod m, size_t n_paths, int per_path, uint64_t seed) {
    std::vector<double> out(n_paths * per_path), Z((size_t)per_path * kLanes);
    RngNoise noise(seed, Stream::Test, 0, per_path, 1.0, m);
    auto st = noise.open(0, n_paths);
    for (size_t p0 = 0; p0 < n_paths; p0 += kLanes) {
        const int n = (int)std::min<size_t>(kLanes, n_paths - p0);
        st->fill(n, Z.data(), kLanes);
        for (int l = 0; l < n; l++)
            for (int d = 0; d < per_path; d++) out[(p0 + l) * per_path + d] = Z[(size_t)d * kLanes + l];
    }
    return out;
}

} // namespace

TEST_SUITE("fast") {

TEST_CASE("norm_inv_cdf: valores conocidos") {
    CHECK(norm_inv_cdf(0.5) == doctest::Approx(0.0).epsilon(1e-15));
    CHECK(norm_inv_cdf(0.975) == doctest::Approx(1.959963984540054).epsilon(1e-14));
    CHECK(norm_inv_cdf(0.99) == doctest::Approx(2.3263478740408408).epsilon(1e-14));
    CHECK(norm_inv_cdf(0.9) == doctest::Approx(1.2815515655446004).epsilon(1e-14));
    CHECK(norm_inv_cdf(0.001) == doctest::Approx(-3.090232306167813).epsilon(1e-14));
    CHECK(norm_inv_cdf(1e-10) == doctest::Approx(-6.361340902404056).epsilon(1e-13));
    CHECK(norm_inv_cdf(0.0) == -INFINITY);
    CHECK(norm_inv_cdf(1.0) == INFINITY);
    CHECK(std::isnan(norm_inv_cdf(-0.1)));
}

// Valores de referencia calculados con mpmath (40 dígitos): cubren la zona central, los dos lados de
// la frontera con la cola (|p - 0.5| = 0.425) y la cola hasta el uniforme más pequeño que puede dar
// el Sobol de 32 bits, 0.5/2^32. La cola usa fast_log, así que esto también valida ese logaritmo.
TEST_CASE("norm_inv_cdf: valores de referencia de alta precision (centro, frontera y colas)") {
    const double ref[][2] = {
        {1.16415321826934814453125e-10, -6.3379577545537892525},
        {1e-8, -5.6120012441747887315},
        {1e-5, -4.2648907939228246285},
        {0.001, -3.0902323061678135415},
        {0.02, -2.0537489106318230529},
        {0.07, -1.4757910281791707352},
        {0.0749, -1.4402382675279636232},
        {0.0751, -1.4388253927525399831},
        {0.08, -1.405071560309632556},
        {0.25, -0.6744897501960817432},
        {0.4, -0.2533471031357997988},
        {0.6, 0.2533471031357997988},
        {0.9249, 1.4388253927525399831},
        {0.9251, 1.4402382675279636232},
        {0.93, 1.4757910281791707352},
        {0.999, 3.0902323061678135415},
        {0.99999, 4.2648907939228246285},
    };
    for (const auto& r : ref) {
        // en la cola superior 1 - p pierde dígitos del p decimal: de ahí la tolerancia mayor para p > 0.99
        const double tol = (r[0] > 0.99) ? 1e-11 : 2e-15;
        CHECK_MESSAGE(norm_inv_cdf(r[0]) == doctest::Approx(r[1]).epsilon(tol), "p=" << r[0]);
    }
}

// La versión por bloque (AVX2 de 4 en 4) debe dar los mismos bits que norm_inv_cdf valor a valor.
TEST_CASE("norm_inv_cdf_n: identico bit a bit a norm_inv_cdf, con y sin AVX2") {
    Xoshiro256pp g = Xoshiro256pp::for_path(21, Stream::Test, 0, 0);
    const int N = 200000;
    std::vector<double> p((size_t)N), ref((size_t)N), out((size_t)N);
    for (int kind = 0; kind < 4; kind++) {
        for (int i = 0; i < N; i++) {
            const double u = g.uniform_open();
            double v;
            if (kind == 0) v = u;                                                       // uniforme: 15 % en las colas
            else if (kind == 1) v = ((double)(g.next() >> 32) + 0.5) * (1.0 / 4294967296.0);   // como el Sobol de 32 bits
            else if (kind == 2) v = (i % 2) ? std::exp(-40.0 * u) : 1.0 - std::exp(-36.0 * u); // colas, incluida la lejana (p < 1.4e-11)
            else {                                                                      // valores especiales mezclados
                switch (i % 9) {
                case 0: v = 0.0; break;
                case 1: v = 1.0; break;
                case 2: v = -0.25; break;
                case 3: v = 1.5; break;
                case 4: v = std::numeric_limits<double>::quiet_NaN(); break;
                case 5: v = 5e-324; break;
                case 6: v = 0.075; break;     // justo en la frontera centro/cola
                case 7: v = 0.925; break;
                default: v = u;
                }
            }
            p[(size_t)i] = v;
            ref[(size_t)i] = norm_inv_cdf(v);
        }
        for (SimdLevel lv : {SimdLevel::Scalar, SimdLevel::Avx2}) {
            if (lv > simd_level_available()) continue;
            set_simd_level(lv);
            for (int n : {N, 1, 3, 4, 5, 7, 8, 63}) {
                std::fill(out.begin(), out.begin() + n, -777.0);
                norm_inv_cdf_n(p.data(), out.data(), n);
                CHECK_MESSAGE(std::memcmp(out.data(), ref.data(), sizeof(double) * (size_t)n) == 0,
                              "simd=" << (int)lv << " tipo=" << kind << " n=" << n);
            }
            std::vector<double> inplace(p.begin(), p.begin() + 101);     // en sitio
            norm_inv_cdf_n(inplace.data(), inplace.data(), 101);
            CHECK(std::memcmp(inplace.data(), ref.data(), sizeof(double) * 101) == 0);
        }
    }
    set_simd_level(simd_level_available());
}

TEST_CASE("norm_inv_cdf: ida y vuelta con norm_cdf y simetria") {
    double worst = 0.0;
    for (double lp = -12.0; lp <= -0.31; lp += 0.07) {
        double p = std::pow(10.0, lp);
        double back = norm_cdf(norm_inv_cdf(p));
        worst = std::max(worst, std::abs(back - p) / p);
        // simetria (1-p no es representable con precision para p muy pequeño)
        if (p >= 1e-5)
            CHECK(norm_inv_cdf(1.0 - p) == doctest::Approx(-norm_inv_cdf(p)).epsilon(1e-9));
    }
    CHECK(worst < 1e-12);
    for (double p : {0.3, 0.45, 0.5, 0.6, 0.77}) {   // zona central
        CHECK(norm_cdf(norm_inv_cdf(p)) == doctest::Approx(p).epsilon(1e-14));
    }
}

TEST_CASE("Xoshiro256pp::for_path: determinista y separado por camino/flujo/nivel/semilla") {
    auto a = Xoshiro256pp::for_path(42, Stream::Main, 3, 1000);
    auto b = Xoshiro256pp::for_path(42, Stream::Main, 3, 1000);
    for (int i = 0; i < 16; i++) CHECK(a.next() == b.next());
    uint64_t base = Xoshiro256pp::for_path(42, Stream::Main, 3, 1000).next();
    CHECK(base != Xoshiro256pp::for_path(42, Stream::Main, 3, 1001).next());
    CHECK(base != Xoshiro256pp::for_path(42, Stream::Pilot, 3, 1000).next());
    CHECK(base != Xoshiro256pp::for_path(42, Stream::Main, 4, 1000).next());
    CHECK(base != Xoshiro256pp::for_path(43, Stream::Main, 3, 1000).next());
}

TEST_CASE("uniform_open nunca es 0 ni 1") {
    auto g = Xoshiro256pp::for_path(1, Stream::Test, 0, 0);
    for (int i = 0; i < 200000; i++) {
        double u = g.uniform_open();
        REQUIRE(u > 0.0);
        REQUIRE(u < 1.0);
    }
}

TEST_CASE("normales por camino: caminos consecutivos no estan correlacionados") {
    const size_t n = 200000;
    std::vector<double> a(n), b(n);
    for (size_t p = 0; p < n; p++) {
        auto g1 = Xoshiro256pp::for_path(7, Stream::Test, 0, p);
        auto g2 = Xoshiro256pp::for_path(7, Stream::Test, 0, p + 1);
        fill_normals(NormalMethod::BoxMuller, g1, &a[p], 1);
        fill_normals(NormalMethod::BoxMuller, g2, &b[p], 1);
    }
    double sab = 0, sa = 0, sb = 0, saa = 0, sbb = 0;
    for (size_t p = 0; p < n; p++) { sab += a[p]*b[p]; sa += a[p]; sb += b[p]; saa += a[p]*a[p]; sbb += b[p]*b[p]; }
    double cov = sab / n - (sa / n) * (sb / n);
    double r = cov / std::sqrt((saa / n - (sa/n)*(sa/n)) * (sbb / n - (sb/n)*(sb/n)));
    CHECK(std::abs(r) < 5.0 / std::sqrt((double)n));
}

} // TEST_SUITE

TEST_SUITE("stat") {

TEST_CASE("normales: momentos y KS (Box-Muller, CDF inversa y Ziggurat; generador del motor y secuencial)") {
    for (bool engine : {true, false})
    for (auto m : {NormalMethod::BoxMuller, NormalMethod::InverseCdf, NormalMethod::Ziggurat}) {
        auto gen = engine ? gen_normals : gen_normals_seq;
        for (int per_path : {1, 2, 7, 64}) {      // incluye n impar
            const size_t n_paths = 2000000 / per_path;
            auto x = gen(m, n_paths, per_path, 12345);
            const double N = (double)x.size();
            auto mo = moments(x);
            CHECK(std::abs(mo.mean) < 5.0 / std::sqrt(N));
            CHECK(std::abs(mo.var - 1.0) < 5.0 * std::sqrt(2.0 / N));
            CHECK(std::abs(mo.skew) < 5.0 * std::sqrt(6.0 / N));
            CHECK(std::abs(mo.exkurt) < 5.0 * std::sqrt(24.0 / N));
        }
        auto y = gen(m, 20000, 5, 999);   // 100000 normales
        CHECK(ks_statistic(y) < 1.95 / std::sqrt((double)y.size()));   // p > 1e-3
    }
}

TEST_CASE("Ziggurat: colas y cuña correctas (fracciones de |z| > c frente a la normal)") {
    const size_t N = 8000000;
    for (bool engine : {true, false}) {
        // per_path = 64 en el motor: la cuña y la cola usan el generador auxiliar sembrado con (camino, d)
        auto x = engine ? gen_normals(NormalMethod::Ziggurat, N / 64, 64, 31337)
                        : gen_normals_seq(NormalMethod::Ziggurat, N / 4, 4, 31337);
        // 3.654 y 3.852 = inicio de la cola con 256 y 512 capas
        for (double c : {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.6541528853610088, 3.8520461503683912, 4.2}) {
            size_t cnt = 0; for (double v : x) if (std::abs(v) > c) ++cnt;
            const double p = 2.0 * (1.0 - norm_cdf(c));
            const double se = std::sqrt(p * (1 - p) / (double)N);
            CHECK_MESSAGE(std::abs((double)cnt / N - p) < 5.0 * se,
                          "motor=" << engine << " c=" << c << " frac=" << (double)cnt / N << " esperado=" << p);
        }
        // simetría: la cola usa el signo de la palabra original
        size_t pos = 0, neg = 0;
        for (double v : x) { if (v > 3.0) ++pos; if (v < -3.0) ++neg; }
        CHECK(std::abs((double)pos - (double)neg) < 5.0 * std::sqrt((double)(pos + neg)));
        double mx = 0; for (double v : x) mx = std::max(mx, std::abs(v));
        CHECK(mx > 4.5);              // la cola se alcanza
        CHECK(mx < 7.0);
    }
}

// Las excepciones del Ziggurat del motor (cuña y cola) se resuelven con un generador auxiliar: se
// comprueba que la distribución DENTRO de las capas (donde se decide aceptar o rechazar) es la correcta
// con un contraste chi-cuadrado sobre 400 intervalos equiprobables.
TEST_CASE("Ziggurat del motor: chi-cuadrado sobre 400 intervalos equiprobables") {
    const size_t N = 16000000;
    const int B = 400;
    auto x = gen_normals(NormalMethod::Ziggurat, N / 16, 16, 777);
    std::vector<double> cnt((size_t)B, 0.0);
    for (double v : x) {
        int b = (int)(norm_cdf(v) * B);
        cnt[(size_t)std::clamp(b, 0, B - 1)] += 1.0;
    }
    const double expct = (double)N / B;
    double chi2 = 0;
    for (double c : cnt) chi2 += (c - expct) * (c - expct) / expct;
    // chi2 ~ chi-cuadrado con 399 g.l.: media 399, desviación sqrt(798) = 28.2; se admite hasta +-5 sigma
    CHECK_MESSAGE(chi2 < 399.0 + 5.0 * 28.25, "chi2=" << chi2);
    CHECK_MESSAGE(chi2 > 399.0 - 5.0 * 28.25, "chi2=" << chi2);
}

TEST_CASE("normales: los elementos de un mismo camino no estan correlacionados") {
    const size_t n_paths = 400000;
    for (auto m : {NormalMethod::BoxMuller, NormalMethod::Ziggurat}) {
    auto x = gen_normals(m, n_paths, 4, 2024);
    auto corr = [&](int i, int j) {
        double sij = 0, si = 0, sj = 0, sii = 0, sjj = 0;
        for (size_t p = 0; p < n_paths; p++) {
            double a = x[p*4+i], b = x[p*4+j];
            sij += a*b; si += a; sj += b; sii += a*a; sjj += b*b;
        }
        double n = (double)n_paths;
        return (sij/n - si/n*sj/n) / std::sqrt((sii/n - si/n*si/n) * (sjj/n - sj/n*sj/n));
    };
    const double lim = 5.0 / std::sqrt((double)n_paths);
    CHECK(std::abs(corr(0, 1)) < lim);
    CHECK(std::abs(corr(1, 2)) < lim);
    CHECK(std::abs(corr(0, 2)) < lim);
    }
}

} // TEST_SUITE

// ---- generador por bloque (el que usa el motor) --------------------------------------------------------

TEST_SUITE("fast") {

// Box-Muller y CDF inversa por bloque dan EXACTAMENTE los números de fill_normals camino a camino.
TEST_CASE("fill_normals_lanes: Box-Muller y CDF inversa identicos bit a bit a fill_normals") {
    const uint64_t seed = 99, level = 3, first = 123456789;
    const double scale = 0.125 * std::sqrt(0.7);   // un escalado que no sea potencia de 2
    for (NormalMethod m : {NormalMethod::BoxMuller, NormalMethod::InverseCdf}) {
        for (int D : {1, 2, 5, 64, 65}) {
            for (int n : {1, 3, kLanes}) {
                if (n > kLanes) continue;
                const int ld = kLanes + 3;
                std::vector<double> Z((size_t)D * ld, -777.0), ref((size_t)D);
                LaneRng g;
                g.seed(Xoshiro256pp::path_hash_base(seed, Stream::Test, level), first);
                fill_normals_lanes(m, g, n, Z.data(), ld, D, scale);
                bool same = true, untouched = true;
                for (int l = 0; l < ld; l++) {
                    if (l < n) {
                        auto gp = Xoshiro256pp::for_path(seed, Stream::Test, level, first + (uint64_t)l);
                        fill_normals(m, gp, ref.data(), D);
                        for (int d = 0; d < D; d++) {
                            const double want = ref[(size_t)d] * scale;
                            if (std::memcmp(&Z[(size_t)d * ld + l], &want, sizeof(double)) != 0) same = false;
                        }
                    } else {
                        for (int d = 0; d < D; d++) if (Z[(size_t)d * ld + l] != -777.0) untouched = false;
                    }
                }
                CHECK_MESSAGE(same, "metodo=" << (int)m << " D=" << D << " n=" << n);
                CHECK_MESSAGE(untouched, "carriles >= n modificados: metodo=" << (int)m << " D=" << D << " n=" << n);
            }
        }
    }
}

// El Ziggurat por bloque tiene dos implementaciones (escalar y AVX2): el resultado no puede
// depender del procesador, así que deben dar los mismos bits para cualquier nº de carriles.
TEST_CASE("fill_normals_lanes: Ziggurat escalar y AVX2 identicos bit a bit") {
    const uint64_t base = Xoshiro256pp::path_hash_base(99, Stream::Test, 3), first = 123456789;
    const double scale = 0.125 * std::sqrt(0.7);
    for (int D : {1, 2, 5, 63, 64, 65, 129, 4097}) {      // 64 = tamaño del tramo entre correcciones
        for (int n : {1, 2, 3, 4, 5, 6, 7, kLanes}) {
            if (n > kLanes) continue;
            const int ld = kLanes + 3;
            std::vector<double> ref((size_t)D * ld, -777.0), Z((size_t)D * ld);
            set_simd_level(SimdLevel::Scalar);
            LaneRng g0; g0.seed(base, first);
            fill_normals_lanes(NormalMethod::Ziggurat, g0, n, ref.data(), ld, D, scale);
            bool untouched = true;
            for (int l = n; l < ld; l++)
                for (int d = 0; d < D; d++) if (ref[(size_t)d * ld + l] != -777.0) untouched = false;
            CHECK_MESSAGE(untouched, "carriles >= n modificados: D=" << D << " n=" << n);
            for (SimdLevel lv : {SimdLevel::Avx2}) {
                if (lv > simd_level_available()) continue;
                set_simd_level(lv);
                std::fill(Z.begin(), Z.end(), -777.0);
                LaneRng g; g.seed(base, first);
                fill_normals_lanes(NormalMethod::Ziggurat, g, n, Z.data(), ld, D, scale);
                CHECK_MESSAGE(std::memcmp(Z.data(), ref.data(), sizeof(double) * Z.size()) == 0,
                              "simd=" << (int)lv << " D=" << D << " n=" << n);
                // el estado final de los carriles generados también coincide
                bool st = (g.pos == g0.pos);
                for (int l = 0; l < n; l++)
                    st = st && g.s0[l] == g0.s0[l] && g.s1[l] == g0.s1[l] && g.s2[l] == g0.s2[l] && g.s3[l] == g0.s3[l];
                CHECK_MESSAGE(st, "estado final distinto: simd=" << (int)lv << " D=" << D << " n=" << n);
            }
        }
    }
    set_simd_level(simd_level_available());
}

// La normal nº d de un camino depende solo de (semilla, flujo, nivel, camino, d): no cambia si se
// piden más dimensiones, si se generan en dos llamadas, o si el camino cae en otro carril.
TEST_CASE("fill_normals_lanes: Ziggurat independiente de D, del troceado en llamadas y del carril") {
    const uint64_t base = Xoshiro256pp::path_hash_base(5, Stream::Main, 1);
    const int D = 300, ld = kLanes;
    for (SimdLevel lv : {SimdLevel::Scalar, SimdLevel::Avx2}) {
        if (lv > simd_level_available()) continue;
        set_simd_level(lv);
        std::vector<double> full((size_t)D * ld), part((size_t)D * ld), shifted((size_t)D * ld);
        LaneRng g; g.seed(base, 1000);
        fill_normals_lanes(NormalMethod::Ziggurat, g, kLanes, full.data(), ld, D, 1.0);
        // dos llamadas: 77 + 223 dimensiones
        LaneRng g2; g2.seed(base, 1000);
        fill_normals_lanes(NormalMethod::Ziggurat, g2, kLanes, part.data(), ld, 77, 1.0);
        fill_normals_lanes(NormalMethod::Ziggurat, g2, kLanes, part.data() + (size_t)77 * ld, ld, D - 77, 1.0);
        CHECK_MESSAGE(std::memcmp(full.data(), part.data(), sizeof(double) * full.size()) == 0, "simd=" << (int)lv);
        // los mismos caminos desplazados un carril (el camino 1001 pasa del carril 1 al 0)
        if (kLanes >= 2) {
            LaneRng g3; g3.seed(base, 1001);
            fill_normals_lanes(NormalMethod::Ziggurat, g3, kLanes, shifted.data(), ld, D, 1.0);
            bool same = true;
            for (int d = 0; d < D; d++)
                for (int l = 0; l + 1 < kLanes; l++)
                    if (shifted[(size_t)d * ld + l] != full[(size_t)d * ld + l + 1]) same = false;
            CHECK_MESSAGE(same, "simd=" << (int)lv);
        }
    }
    set_simd_level(simd_level_available());
}

// La cola del Ziggurat solo sale 1 de cada ~8000 normales: se comprueba aparte, con muchas más
// muestras, que la versión vectorial la resuelve igual que la escalar.
TEST_CASE("fill_normals_lanes: escalar y AVX2 coinciden tambien en la cola (2^23 normales)") {
    const int D = 1 << 15, ld = kLanes;
    const int n_blocks = (1 << 23) / (D * kLanes) + 1;
    std::vector<double> ref((size_t)D * ld), Z((size_t)D * ld);
    long long tails = 0;
    for (int b = 0; b < n_blocks; b++) {
        const uint64_t base = Xoshiro256pp::path_hash_base(2024, Stream::Test, 0);
        set_simd_level(SimdLevel::Scalar);
        LaneRng g0; g0.seed(base, (uint64_t)b * kLanes);
        fill_normals_lanes(NormalMethod::Ziggurat, g0, kLanes, ref.data(), ld, D, 1.0);
        for (double v : ref) if (std::abs(v) > 4.0388498461095045) ++tails;   // más allá de R para 256, 512 y 1024 capas
        for (SimdLevel lv : {SimdLevel::Avx2}) {
            if (lv > simd_level_available()) continue;
            set_simd_level(lv);
            LaneRng g; g.seed(base, (uint64_t)b * kLanes);
            fill_normals_lanes(NormalMethod::Ziggurat, g, kLanes, Z.data(), ld, D, 1.0);
            CHECK_MESSAGE(std::memcmp(Z.data(), ref.data(), sizeof(double) * Z.size()) == 0, "simd=" << (int)lv << " bloque=" << b);
        }
    }
    set_simd_level(simd_level_available());
    CHECK(tails > 250);     // se esperan ~450: la cola se ha ejercitado de verdad
}

TEST_CASE("RngNoise: un cursor entrega los mismos caminos sea cual sea el troceado en bloques") {
    const int D = 37;
    const double scale = 0.3;
    for (NormalMethod m : {NormalMethod::Ziggurat, NormalMethod::BoxMuller, NormalMethod::InverseCdf}) {
        RngNoise noise(5, Stream::Main, 2, D, scale, m);
        const int N = 5 * kLanes + 3;
        // referencia: camino a camino (bloques de 1)
        std::vector<double> ref((size_t)N * D);
        {
            auto st = noise.open(1000, (uint64_t)N);
            for (int p = 0; p < N; p++) st->fill(1, &ref[(size_t)p * D], 1);
        }
        for (int block : {2, kLanes, kLanes + 1, 64}) {
            auto st = noise.open(1000, (uint64_t)N);
            std::vector<double> Z((size_t)D * block);
            bool same = true;
            for (int p0 = 0; p0 < N; p0 += block) {
                const int n = std::min(block, N - p0);
                st->fill(n, Z.data(), block);
                for (int l = 0; l < n; l++)
                    for (int d = 0; d < D; d++)
                        if (Z[(size_t)d * block + l] != ref[(size_t)(p0 + l) * D + d]) same = false;
            }
            CHECK_MESSAGE(same, "metodo=" << (int)m << " bloque=" << block);
        }
        // y un cursor abierto en mitad del rango entrega los mismos caminos (propiedad de prefijo)
        auto st = noise.open(1000 + 11, 4);
        std::vector<double> one((size_t)D);
        st->fill(1, one.data(), 1);
        bool same = true;
        for (int d = 0; d < D; d++) if (one[(size_t)d] != ref[(size_t)11 * D + d]) same = false;
        CHECK_MESSAGE(same, "cursor desplazado, metodo=" << (int)m);
    }
}

} // TEST_SUITE
