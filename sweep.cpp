// Barrido de precisión con corte por método. Movido SIN cambios de lógica desde el
// final de methods_cuda.cu (era C++ de host puro, sin CUDA) para que lo compartan los
// backends CPU y CUDA. El formato de salida es byte a byte el de la versión con
// std::format (lo parsea gen_informe.py); ver tests/test_sweep_format.cpp.

#include "sweep.hpp"
#include "mc_format.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

std::vector<double> eps_scale_125(double eps_finest) {
    // Escala redonda 1-2-5 descendente, empezando en 0.05.
    static const double BASE[] = {
        0.05, 0.02, 0.01,
        0.005, 0.002, 0.001,
        0.0005, 0.0002, 0.0001,
        0.00005, 0.00002, 0.00001
    };
    std::vector<double> out;
    for (double e : BASE) {
        if (e < eps_finest - 1e-15) break; // ya pasamos el limite pedido
        out.push_back(e);
    }
    if (out.empty() || (out.back() - eps_finest) > 1e-15 * std::max(1.0, eps_finest)) {
        // El usuario pidio un eps_finest mas fino que el ultimo de la escala
        // predefinida (o esta vacia): anadimos ese punto final explicitamente
        // para no dejar de intentar la precision solicitada.
        out.push_back(eps_finest);
    }
    return out;
}

void run_precision_sweep(const std::string& example_name,
                         std::vector<SweepMethod>& methods,
                         double price_ref,
                         const std::vector<double>& eps_list,
                         double T_BUDGET_S,
                         int R_MAX,
                         int R_MIN,
                         double SE_REL) {

    MC_PRINTLN("\n%s", std::string(78, '#').c_str());
    MC_PRINTLN("### %s", example_name.c_str());
    MC_PRINTLN("%s", std::string(78, '#').c_str());

    std::vector<bool> cut(methods.size(), false);
    // Coste de una corrida individual en el ultimo eps donde el metodo se ejecuto,
    // para proyectar el del siguiente eps (todos los metodos son >= O(eps^-2), asi
    // que eps^-2 es cota superior de la extrapolacion) y no pagar una corrida de
    // sondeo carisima solo para cortarla despues.
    std::vector<double> last_run_s(methods.size(), 0.0);
    std::vector<double> last_run_eps(methods.size(), 0.0);

    for (size_t lvl = 0; lvl < eps_list.size(); ++lvl) {
        double eps = eps_list[lvl];
        auto t_lvl0 = Clock::now();

        std::vector<TableRow>   rows;
        std::vector<std::string> cut_names;
        std::vector<std::string> na_names;

        std::vector<std::string> proj_names;
        for (size_t mi = 0; mi < methods.size(); ++mi) {
            if (cut[mi]) continue;
            SweepMethod& mth = methods[mi];

            // Proyeccion: si la corrida de sondeo ya se preve por encima del
            // presupuesto, no se ejecuta; el metodo sale -- aqui y en los eps mas finos.
            if (last_run_s[mi] > 0.0) {
                double ratio = last_run_eps[mi] / eps;
                double proj  = last_run_s[mi] * ratio * ratio;
                if (proj > T_BUDGET_S) {
                    cut[mi] = true;
                    proj_names.push_back(mth.name);
                    continue;
                }
            }

            // Presupuesto: se mide SOLO esta primera corrida individual.
            MCResult r0;
            try {
                r0 = mth.run_once(0u, eps);
            } catch (const SobolLimitReached&) {
                // El metodo necesitaria mas de 2^32 puntos Sobol por replica a
                // este eps: no aplicable. Se marca con -- y se corta para los
                // niveles mas finos (que solo empeoran).
                cut[mi] = true;
                na_names.push_back(mth.name);
                continue;
            }
            if (r0.time_s > T_BUDGET_S) {
                cut[mi] = true;
                cut_names.push_back(mth.name);
                continue;
            }

            // Dentro de presupuesto: repetir con semilla distinta hasta R_MAX
            // veces, parando en cuanto el tiempo acumulado supera T_BUDGET_S.
            RunningStats price_stats;
            price_stats.update(r0.price);
            double    time_total = r0.time_s;
            long long n_last     = r0.n_samples;
            double    se_last    = r0.std_error;
            int       R_done     = 1;

            for (int rep = 1; rep < R_MAX; ++rep) {
                if (time_total > T_BUDGET_S) break; // presupuesto = tope de tiempo total del metodo en este nivel
                // Criterio adaptativo: parar cuando el error estandar de la media
                // entre repeticiones cae por debajo de SE_REL * |media|, con un
                // minimo de R_MIN repeticiones para que la stddev tenga sentido.
                if (R_done >= R_MIN &&
                    price_stats.std_error() < SE_REL * std::abs(price_stats.mean)) break;
                MCResult r;
                try { r = mth.run_once((unsigned)rep * 977u, eps); }
                catch (const SobolLimitReached&) { break; }
                price_stats.update(r.price);
                time_total += r.time_s;
                n_last  = r.n_samples;
                se_last = r.std_error;
                ++R_done;
            }

            double price = price_stats.mean;
            double se    = (R_done > 1) ? price_stats.std_error() : se_last;
            rows.push_back({mth.name, price, se, n_last, time_total, R_done});
            last_run_s[mi] = r0.time_s; last_run_eps[mi] = eps;
            MC_PRINTLN("    [nivel %lld] %s listo: %.1fs totales, %d reps",
                       (long long)lvl, mth.name.c_str(), time_total, R_done);
        }

        double wall = std::chrono::duration<double>(Clock::now() - t_lvl0).count();

        MC_PRINTLN("--- nivel %lld eps=%s wall=%.3fs ---", (long long)lvl, mc_shortest(eps).c_str(), wall);
        MC_PRINTLN("  Referencia: %.6f", price_ref);
        MC_PRINTLN("  %-22s%10s%10s%12s%9s%10s%6s%5s",
                   "Metodo", "Precio", "StdErr", "N", "T(s)", "|Error|", "OK?", "R");
        MC_PRINTLN("  %s", std::string(84, '-').c_str());
        for (const auto& r : rows) {
            double err = std::abs(r.price - price_ref);
            bool   ok  = (err < 2.0 * eps);
            MC_PRINTLN("  %-22s%10.4f%10.4f%12lld%9.3f%10.4f%6s%5d",
                       r.method.c_str(), r.price, r.std_error, r.n_samples,
                       r.time_s, err, ok ? "SI" : "NO", r.n_reps);
        }
        for (const auto& nm : cut_names) {
            MC_PRINTLN("  %-22s--  (una corrida supero el presupuesto de %.0fs; cortado aqui y en los eps mas finos)",
                       nm.c_str(), T_BUDGET_S);
        }
        for (const auto& nm : na_names) {
            MC_PRINTLN("  %-22s--  (necesitaria > 2^32 puntos Sobol por replica a este eps)", nm.c_str());
        }
        for (const auto& nm : proj_names) {
            MC_PRINTLN("  %-22s--  (coste proyectado > %.0fs; no ejecutado, cortado aqui y en los eps mas finos)",
                       nm.c_str(), T_BUDGET_S);
        }
        MC_PRINTLN("%s", "");

        bool all_cut = true;
        for (bool b : cut) if (!b) { all_cut = false; break; }
        if (all_cut) break;
    }
}
