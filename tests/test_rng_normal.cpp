// RNG por camino y normales: momentos, Kolmogorov-Smirnov, inversa de la CDF.
#include "doctest.h"
#include "cpu/normal.hpp"
#include "cpu/rng.hpp"

#include <algorithm>
#include <cmath>
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

std::vector<double> gen_normals(NormalMethod m, size_t n_paths, int per_path, uint64_t seed) {
    std::vector<double> out(n_paths * per_path);
    for (size_t p = 0; p < n_paths; p++) {
        auto g = Xoshiro256pp::for_path(seed, Stream::Test, 0, p);
        fill_normals(m, g, out.data() + p * per_path, per_path);
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

TEST_CASE("fill_normals: momentos y KS (Box-Muller y CDF inversa)") {
    for (auto m : {NormalMethod::BoxMuller, NormalMethod::InverseCdf}) {
        for (int per_path : {1, 2, 7, 64}) {      // incluye n impar
            const size_t n_paths = 2000000 / per_path;
            auto x = gen_normals(m, n_paths, per_path, 12345);
            const double N = (double)x.size();
            auto mo = moments(x);
            CHECK(std::abs(mo.mean) < 5.0 / std::sqrt(N));
            CHECK(std::abs(mo.var - 1.0) < 5.0 * std::sqrt(2.0 / N));
            CHECK(std::abs(mo.skew) < 5.0 * std::sqrt(6.0 / N));
            CHECK(std::abs(mo.exkurt) < 5.0 * std::sqrt(24.0 / N));
        }
        auto y = gen_normals(m, 20000, 5, 999);   // 100000 normales
        CHECK(ks_statistic(y) < 1.95 / std::sqrt((double)y.size()));   // p > 1e-3
    }
}

TEST_CASE("fill_normals: los elementos de un mismo camino no estan correlacionados") {
    const size_t n_paths = 400000;
    auto x = gen_normals(NormalMethod::BoxMuller, n_paths, 4, 2024);
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

} // TEST_SUITE
