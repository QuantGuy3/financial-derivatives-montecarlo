# Arquitectura

```
 ejemplos 01–11        GUI web (mc_gui)         tests / bench
        │                   │                        │
        └─────────┬─────────┴────────────────────────┘
                  ▼
        engine/api.hpp  ── fachada: mc::run_*  (capa 1)  y  mc::run(RunSpec)  (capa 2: planifica)
                  │
        ┌─────────┴───────────┐
        ▼                     ▼
   cpu/ (multihilo, double)   engine/cuda_adapter.cpp ──► methods_cuda.cu (GPU, float)
                                  └─ sin nvcc: engine/cuda_stub.cpp (mismas firmas, lanza excepción)
```

Los ficheros raíz originales (`models.hpp`, `payoffs.hpp`, `utils.*`, `methods_cuda.*`) siguen en su sitio.
Cambios en ellos (mínimos): configuración/resultados/barrido movidos a `mc_types.hpp` y `sweep.*`; `utils.*` sin
Eigen y con caché por referencia; `methods_cuda.cu` solo pierde la cola de host puro (el barrido) y gana
`cuda_device_available()`.

## Motor CPU (`cpu/`)

### Determinismo (contrato)

> **El mismo binario produce resultados idénticos bit a bit con cualquier número de hilos.**

Se sostiene en cuatro decisiones (todas con tests en la etiqueta `determinism`):

1. **Semilla por camino.** El estado de `xoshiro256++` del camino *i* se deriva por hash de
   `(semilla, flujo, nivel, i)` (`cpu/rng.hpp`). Un camino vale lo mismo esté en el bloque, el chunk o el hilo
   que esté, y los primeros *n* caminos son los mismos sea cual sea *N* (propiedad de prefijo: ampliar la
   precisión no repite ni descarta muestras).
2. **Chunks de tamaño fijo** que dependen solo del coste por camino (≈0,5–2 ms de trabajo), nunca del número de
   hilos (`cpu/path_sim.hpp::chunk_paths_for`).
3. **Reducción en orden de índice.** Cada tarea escribe en su propia ranura (`Padded<T>` evita falso compartir) y
   el hilo coordinador funde las ranuras una a una con la fórmula de Chan (`cpu/reduce.hpp`). Los chunks se
   funden individualmente, por lo que ni siquiera la partición en "rondas" altera los bits.
4. **Sin contracción FMA ni fast-math** en el motor (`-ffp-contract=off`), para que el compilador no escoja
   contracciones distintas en funciones distintas.

La garantía es **por binario**: otro compilador, otra libm u otra bandera `-march` cambian los últimos bits. Los
tests entre plataformas son estadísticos.

### Piezas

| Fichero | Papel |
|---|---|
| `cpu/thread_pool.*` | Pool propio (sin OpenMP): contador atómico de tareas, excepciones, cancelación, llamadas anidadas en serie. |
| `cpu/rng.hpp`, `cpu/normal.*` | `xoshiro256++`, inversa de la CDF (AS 241), Box–Muller, **Ziggurat** (por defecto). |
| `cpu/sobol.*`, `cpu/joe_kuo_data.cpp` | Sobol hasta 21 201 dimensiones (parámetros Joe-Kuo generados por `tools/gen_joe_kuo.py`) con el scrambling de Hong-Hickernell **idéntico al de la GPU**. |
| `cpu/noise.*`, `cpu/qmc_noise.*` | `NoiseSource`/`NoiseStream`: cursores por chunk que entregan `Z[d*ld+p]` (mismo layout que la GPU). Pseudoaleatorio, Sobol, y las transformaciones Brownian Bridge y PCA por bloques. |
| `cpu/kernels.*` | Núcleos de simulación: `kLanes` (8) caminos en paso sincronizado, plantillas Modelo × Payoff, cesta con Cholesky, par fino/grueso de MLMC, evaluadores CV/IS. |
| `cpu/path_sim.*` | Simulador por chunks (`PathSim`, `CoupledSim`) y rondas con reducción determinista. |
| `cpu/mc_cpu.*`, `cpu/qmc_cpu.*`, `cpu/mlmc_cpu.*`, `cpu/vr_cpu.*` | Los cuatro métodos, y variables de control / importance sampling. |
| `cpu/path_sampler.*` | Trayectorias completas para la GUI con **los mismos pasos de Euler** que el motor. |

### Algoritmos

* **MC**: piloto (`pilot_n`) → `N = 2·Var/ε²` → corrida principal (igual que `run_mc_cuda`).
* **QMC**: `R` réplicas Sobol, cada una con su scrambling; se duplican los puntos por réplica hasta
  `var_of_means < ε²/2`. La unidad paralela es *(réplica, rango de puntos)*. Los puntos nuevos de cada
  duplicación continúan la secuencia con el estado de Gray-code (`1 XOR` por dimensión y punto).
* **MLMC / MLQMC**: un **único bucle de Giles** (piloto → `N_l` óptimo → refinamiento → test de sesgo → nivel
  nuevo) que sirve a los dos: MLMC es una réplica con ruido pseudoaleatorio, MLQMC usa `R` réplicas Sobol por
  *(nivel, réplica)*. Todos los niveles pendientes de una iteración van en una sola región paralela.
* **CV / IS**: las dos parejas que implementa la GPU (GBM+Asian con control geométrica, Dupire+europea con control
  GBM; GBM+call con desplazamiento Girsanov) en MC, QMC, MLMC (solo GBM) y MLQMC.

### Progreso y cancelación

`ProgressSink` (`mc_progress.hpp`) recibe `Snapshot`s **solo desde el hilo coordinador** y se consulta
`should_cancel()` entre rondas (≤ 256 chunks, así la latencia de cancelación es de décimas de segundo).
`max_seconds` devuelve la estimación parcial marcada `truncated`.

## Diferencias deliberadas con la GPU

| Tema | GPU | CPU |
|---|---|---|
| Precisión | `float` + `--use_fast_math` | `double`, sin fast-math |
| Aleatorios (MC) | cuRAND XORWOW, semillas por lote | `xoshiro256++` por camino (la CPU no reproduce los números de la GPU en MC; en QMC **sí**: mismos puntos Sobol+HH) |
| Dupire | sin guarda: `S ≤ 0` da NaN | `S` absorbente en 0; los caminos no finitos se excluyen y se cuentan (`n_nonfinite`) |
| MLMC + Heston + payoff dependiente del camino | el acumulador no se actualiza (resultado incorrecto) | se actualiza correctamente |
| Matriz PCA | fp16 en Tensor Cores | `double` |
| Cesta Dupire | normaliza con `S0[0]` | usa `S0[i]` de cada activo (idéntico si son iguales, como en los ejemplos) |
| `MCConfig::N_batch` | tamaño de lote en GPU | ignorado (reparto automático) |
| `QMCConfig::n0` | ignorado (arranca en `min(chunk, 4096)`) | puntos iniciales por réplica |

Contraste con las cifras guardadas de la A100 (`resultados/`): el MLQMC de la europea (`ejemplo01`, ε = 0,02)
coincide en precio (10,4462), error estándar y nº de muestras (76 800); es el test `paridad con la A100`.

## Referencias analíticas (`reference_prices.*`)

Black-Scholes con deriva `μ`, Asian geométrica discreta y **Heston por inversión de Fourier** (formulación "little
trap", contrastada con el caso 5,785155450 de la literatura y con `scipy.integrate.quad`). Los motores simulan con
Euler, así que el resultado difiere de la referencia en un sesgo O(h) (visible con pocos pasos).

## Construcción

`cmake --preset mingw-release` (o `linux-release`, `msvc-release`, `mingw-native`). Opciones: `MC_ENABLE_CUDA`
(`AUTO|ON|OFF`), `MC_ARCH=native` (opt-in), `MC_LANES` (1–16), `MC_BUILD_{TESTS,EXAMPLES,BENCH,GUI}`.
El núcleo no necesita Eigen ni LAPACK; las dependencias de cabecera están vendorizadas en `third_party/`.
