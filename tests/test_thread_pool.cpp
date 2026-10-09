// Pool de hilos: reparto, excepciones, cancelación, anidado y reducción determinista.
#include "doctest.h"
#include "cpu/normal.hpp"
#include "cpu/reduce.hpp"
#include "cpu/rng.hpp"
#include "cpu/thread_pool.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace mc::cpu;

namespace {

// Resultado de una "simulacion" por chunk: momentos de 1000 caminos por chunk
Moments run_chunks(ThreadPool& pool, size_t n_chunks) {
    std::vector<Padded<Moments>> slots(n_chunks);
    pool.parallel_for(n_chunks, [&](size_t c, int) {
        ShiftedAcc acc;
        for (int i = 0; i < 1000; i++) {
            auto g = Xoshiro256pp::for_path(77, Stream::Test, 0, c * 1000 + i);
            double z;
            fill_normals(NormalMethod::BoxMuller, g, &z, 1);
            acc.add(std::exp(0.2 * z));
        }
        slots[c].v = acc.to_moments();
    });
    Moments total;
    for (auto& s : slots) total.merge(s.v);
    return total;
}

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

} // namespace

TEST_SUITE("fast") {

TEST_CASE("parallel_for ejecuta cada tarea exactamente una vez") {
    for (int threads : {1, 2, 4, 8}) {
        ThreadPool pool(threads);
        for (size_t n : {size_t(0), size_t(1), size_t(7), size_t(1000)}) {
            std::vector<std::atomic<int>> hits(n);
            for (auto& h : hits) h = 0;
            bool done = pool.parallel_for(n, [&](size_t t, int w) {
                CHECK(w >= 0);
                CHECK(w < threads);
                hits[t].fetch_add(1);
            });
            CHECK(done);
            for (size_t i = 0; i < n; i++) CHECK(hits[i].load() == 1);
        }
    }
}

TEST_CASE("parallel_for relanza la excepción y el pool sigue siendo utilizable") {
    ThreadPool pool(4);
    CHECK_THROWS_AS(pool.parallel_for(100, [&](size_t t, int) {
        if (t == 5) throw std::runtime_error("boom");
    }), std::runtime_error);
    std::atomic<int> count{0};
    pool.parallel_for(50, [&](size_t, int) { count++; });
    CHECK(count.load() == 50);
}

TEST_CASE("parallel_for: cancelación cooperativa") {
    ThreadPool pool(4);
    std::atomic<bool> cancel{false};
    std::atomic<int> executed{0};
    bool done = pool.parallel_for(100000, [&](size_t, int) {
        if (++executed >= 100) cancel = true;
    }, &cancel);
    CHECK_FALSE(done);
    CHECK(executed.load() < 100000);
}

TEST_CASE("parallel_for anidado se ejecuta en serie sin bloquearse") {
    ThreadPool pool(4);
    std::atomic<int> inner{0};
    pool.parallel_for(8, [&](size_t, int) {
        pool.parallel_for(5, [&](size_t, int) { inner++; });
    });
    CHECK(inner.load() == 40);
}

TEST_CASE("set_num_threads / num_threads") {
    set_num_threads(3);
    CHECK(num_threads() == 3);
    set_num_threads(1);
    CHECK(num_threads() == 1);
    set_num_threads(0);
    CHECK(num_threads() >= 1);
}

TEST_CASE("Moments::merge reproduce el cálculo directo") {
    Moments a, b, all;
    for (int i = 0; i < 1000; i++) {
        double x = std::sin(i) * 3 + 1e6;
        (i < 400 ? a : b).add(x);
        all.add(x);
    }
    a.merge(b);
    CHECK(a.n == all.n);
    CHECK(a.mean == doctest::Approx(all.mean).epsilon(1e-14));
    CHECK(a.M2 == doctest::Approx(all.M2).epsilon(1e-9));
}

TEST_CASE("ShiftedAcc es estable con media >> desviación") {
    // Referencia en long double (Welford en double pierde ~1e-5 aqui, el desplazado no).
    ShiftedAcc s;
    long double sum = 0, sum2 = 0;
    const int n = 100000;
    for (int i = 0; i < n; i++) {
        double c = std::cos(i * 0.37);
        s.add(1e9 + c);
        sum += c; sum2 += (long double)c * c;
    }
    const double ref_mean = 1e9 + (double)(sum / n);
    const double ref_M2 = (double)(sum2 - sum * sum / n);
    CHECK(s.to_moments().mean == doctest::Approx(ref_mean).epsilon(1e-15));
    CHECK(s.to_moments().M2 == doctest::Approx(ref_M2).epsilon(1e-6));
}

} // TEST_SUITE

TEST_SUITE("determinism") {

TEST_CASE("reducción por chunks idéntica bit a bit con 1, 2, 3, 5, 8 y 16 hilos") {
    const size_t n_chunks = 257;
    ThreadPool ref_pool(1);
    Moments ref = run_chunks(ref_pool, n_chunks);
    for (int threads : {2, 3, 5, 8, 16}) {
        ThreadPool pool(threads);
        for (int rep = 0; rep < 3; rep++) {
            Moments m = run_chunks(pool, n_chunks);
            CHECK(m.n == ref.n);
            CHECK(bits_equal(m.mean, ref.mean));
            CHECK(bits_equal(m.M2, ref.M2));
        }
    }
}

} // TEST_SUITE
