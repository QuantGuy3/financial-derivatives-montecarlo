#pragma once
// Acumuladores estadísticos y reducción determinista.
//
// Cada tarea (chunk) acumula con ShiftedAcc (sumas desplazadas: estables aunque la
// varianza sea mucho menor que la media^2) y entrega un Moments. Los Moments de los
// chunks se funden EN ORDEN DE ÍNDICE con la fórmula de Chan et al., por lo que el
// resultado es idéntico bit a bit para cualquier nº de hilos.

#include <cmath>
#include <cstddef>
#include <vector>

namespace mc::cpu {

// Momentos de una muestra: n, media y M2 = sum (x - media)^2.
struct Moments {
    long long n = 0;
    double mean = 0.0;
    double M2 = 0.0;

    double variance() const { return n > 1 ? M2 / (double)(n - 1) : 0.0; }       // insesgada
    double variance_biased() const { return n > 0 ? M2 / (double)n : 0.0; }       // E[Y^2]-E[Y]^2
    double std_error() const { return n > 1 ? std::sqrt(variance() / (double)n) : 0.0; }

    // Fusión de Chan, Golub & LeVeque (a primero, b después: el orden importa para el
    // determinismo en coma flotante).
    void merge(const Moments& b) {
        if (b.n == 0) return;
        if (n == 0) { *this = b; return; }
        const double na = (double)n, nb = (double)b.n, nt = na + nb;
        const double delta = b.mean - mean;
        mean += delta * (nb / nt);
        M2 += b.M2 + delta * delta * (na * nb / nt);
        n += b.n;
    }

    void add(double x) {   // Welford
        ++n;
        const double d = x - mean;
        mean += d / (double)n;
        M2 += d * (x - mean);
    }
};

// Acumulador rápido por chunk: sumas desplazadas por el primer valor.
struct ShiftedAcc {
    double c = 0.0, s1 = 0.0, s2 = 0.0;
    long long n = 0;

    inline void add(double x) {
        if (n == 0) c = x;
        const double d = x - c;
        s1 += d;
        s2 += d * d;
        ++n;
    }

    Moments to_moments() const {
        Moments m;
        if (n == 0) return m;
        m.n = n;
        m.mean = c + s1 / (double)n;
        m.M2 = s2 - s1 * s1 / (double)n;
        if (m.M2 < 0.0) m.M2 = 0.0;
        return m;
    }
};

// Covarianza muestral de dos variables: n, medias, M2 de cada una y co-momento.
struct Cov2 {
    long long n = 0;
    double mx = 0, my = 0, M2x = 0, M2y = 0, Cxy = 0;

    double var_x() const { return n > 0 ? M2x / (double)n : 0.0; }      // sesgada (E[x²]-E[x]²)
    double var_y() const { return n > 0 ? M2y / (double)n : 0.0; }
    double cov_xy() const { return n > 0 ? Cxy / (double)n : 0.0; }

    void merge(const Cov2& b) {   // Chan et al., generalizado al co-momento
        if (b.n == 0) return;
        if (n == 0) { *this = b; return; }
        const double na = (double)n, nb = (double)b.n, nt = na + nb;
        const double dx = b.mx - mx, dy = b.my - my;
        const double w = na * nb / nt;
        M2x += b.M2x + dx * dx * w;
        M2y += b.M2y + dy * dy * w;
        Cxy += b.Cxy + dx * dy * w;
        mx += dx * (nb / nt);
        my += dy * (nb / nt);
        n += b.n;
    }
};

// Acumulador por chunk de un par (x,y) con sumas desplazadas.
struct ShiftedAcc2 {
    ShiftedAcc a, b;
    double sab = 0.0;

    inline void add(double x, double y) {
        a.add(x);
        b.add(y);
        sab += (x - a.c) * (y - b.c);
    }

    Cov2 to_cov2() const {
        Cov2 r;
        if (a.n == 0) return r;
        const Moments ma = a.to_moments(), mb = b.to_moments();
        r.n = a.n; r.mx = ma.mean; r.my = mb.mean; r.M2x = ma.M2; r.M2y = mb.M2;
        r.Cxy = sab - a.s1 * b.s1 / (double)a.n;
        return r;
    }
};

// Evita falso compartir: cada ranura de salida ocupa su propia línea de caché.
template <class T>
struct alignas(64) Padded {
    T v{};
};

// Funde en orden de índice un rango de ranuras.
template <class T, class Get>
Moments merge_in_order(const std::vector<Padded<T>>& slots, Get&& get) {
    Moments total;
    for (const auto& s : slots) total.merge(get(s.v));
    return total;
}

} // namespace mc::cpu
