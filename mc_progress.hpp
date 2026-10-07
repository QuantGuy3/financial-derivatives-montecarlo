#pragma once
// Progreso y cancelación de las ejecuciones (independiente del backend).
//
// Los motores llaman a ProgressSink::on_snapshot SOLO desde el hilo que coordina la
// ejecución (nunca desde los trabajadores) y consultan should_cancel() entre rondas de
// trabajo. La GUI usa esto para dibujar las curvas de convergencia en vivo.

#include <cstdint>

enum class Stage : int {
    Plan = 0,       // planificación (c1 de Richardson, pasos, piloto de CV...)
    Pilot,          // piloto (MC: varianza; MLMC: V_l por nivel)
    Main,           // corrida principal de MC: n_done / mean / std_error crecen
    Level,          // iteración de MLMC/MLQMC: se rellenan 'levels'
    Doubling,       // iteración de QMC: se duplican los puntos por réplica
    Done,
};

// Estadísticos de un nivel de MLMC/MLQMC (l = 0..L)
struct LevelStat {
    int       l = 0;
    long long N = 0;       // muestras acumuladas en el nivel
    double    E = 0.0;     // media de la corrección P_l - P_{l-1}
    double    V = 0.0;     // varianza de la corrección por muestra
    double    cost = 0.0;  // coste relativo por muestra (M^l)
};

struct Snapshot {
    Stage     stage = Stage::Main;
    uint64_t  seq = 0;               // contador creciente dentro de una ejecución
    double    elapsed_s = 0.0;
    long long n_done = 0;            // muestras (caminos) usadas hasta ahora
    double    mean = 0.0;            // estimación actual del precio
    double    std_error = 0.0;       // error estándar actual
    double    eps_target = 0.0;
    // MLMC / MLMC-QMC
    int              L = -1;
    const LevelStat* levels = nullptr;
    int              n_levels = 0;
    // QMC / MLQMC
    int           doubling = -1;
    long long     n_per_replica = 0;
    const double* replica_means = nullptr;
    int           n_replicas = 0;
    bool          is_final = false;
};

struct ProgressSink {
    virtual ~ProgressSink() = default;
    virtual void on_snapshot(const Snapshot&) {}
    virtual bool should_cancel() const { return false; }
};
