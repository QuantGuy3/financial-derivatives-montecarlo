#pragma once
// Tipos comunes a todos los backends (CPU y CUDA): configuración de los métodos,
// resultados y excepciones. Antes vivían en methods_cuda.cuh; se movieron aquí SIN
// cambios para que el build solo-CPU no dependa de CUDA.

#include <utility>
#include <vector>

// ------------------------ //
// Structs de configuración //
// ------------------------ //

struct MCConfig {
    long long N_batch = 1LL << 16; // Trayectorias por lote GPU
    int       pilot_n = 10000;
    unsigned  seed    = 123u;
};

struct QMCConfig {
    // Réplicas Sobol: cada una recibe su propia matriz triangular de
    // Hong-Hickernell (scrambling sobre F_2) y su propio desplazamiento
    // digital, calculados a mano vía la API de dispositivo de cuRAND (ver
    // gen_scrambled_sobol_normal_replica/hh_scramble en methods_cuda.cu), no
    // solo un carril de offset distinto sobre un scramble fijo compartido.
    // var_of_means entre réplicas es así el estimador insesgado descrito en
    // la observación 3.19 de la memoria (Owen, 1997).
    int R            = 32;
    int max_doublings = 20; // Máximo de duplicaciones del número de puntos
    // Puntos por réplica de la primera ronda. La GPU arranca en min(chunk_cap, 4096) (valor por
    // defecto, que ignora este campo); el motor CPU lo usa (redondeado a potencia de 2 con Sobol)
    // y la GUI puede bajarlo (p. ej. 256) para dibujar curvas de convergencia más suaves.
    int n0            = 4096;
    // Semilla base para el generador pseudoaleatorio / los scrambles Sobol.
    // Se usa como cfg.seed (Raw) o como sal de scramble por réplica (Sobol),
    // reemplazando la constante 42u que antes iba fija en el código: así los
    // ejemplos pueden repetir un método con semillas distintas (ver TAREA 2).
    unsigned seed = 42u;
};

struct MLMCConfig {
    int M      = 2;   // Factor de refinamiento entre niveles
    int max_L  = 10;  // Número máximo de niveles
    int pilot_n = 400; // Trayectorias piloto por nivel
    // Semilla base de la cadena de semillas por nivel/iteración (antes un 42u
    // fijo dentro de run_mlmc_cuda/run_mlqmc_cuda). Permite repetir MLMC con
    // distinta semilla sin tocar el motor (ver TAREA 2).
    unsigned seed = 42u;
};

// Modo de transformación del ruido antes de aplicar el esquema de Euler
enum class NoiseMode { Raw, BrownianBridge, PCA };


// -------------------- //
// Struct de resultados //
// -------------------- //

// Se lanza cuando un metodo Sobol/QMC necesitaria mas de 2^32 puntos por
// replica y choca con el limite de 32 bits del offset de la API de
// dispositivo de cuRAND. El barrido lo captura y marca el metodo como
// no aplicable a ese eps (--) en vez de abortar el ejecutable.
struct SobolLimitReached {};

struct MCResult {
    double    price     = 0.0;
    double    std_error = 0.0;
    long long n_samples = 0;
    double    time_s    = 0.0;
};

// Resultado del piloto de variable de control
struct CVPilot { double beta, var_plain, var_cv; };
