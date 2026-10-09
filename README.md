# Financial Derivatives Monte Carlo (CPU multihilo + GPU / CUDA)

Motor de valoración de derivados financieros en C++ con **dos backends**: CPU multihilo (determinista, `double`, sin
dependencias) y GPU NVIDIA (CUDA, `float`, `cuRAND`/`cuBLAS`/`CUB`). Implementa cuatro familias de estimadores — Monte
Carlo estándar (MC), Quasi-Monte Carlo (QMC), Multilevel Monte Carlo (MLMC) y Multilevel Quasi-Monte Carlo (MLQMC) —
sobre varios modelos (GBM, Heston, Dupire local, cestas multi-activo) y payoffs (europeas, asiáticas, lookback, barrera,
basket), con reducción de varianza (variables de control, importance sampling) y reducción de dimensión efectiva
(Brownian Bridge, PCA).

## Inicio rápido

```bash
cmake --preset mingw-release          # o linux-release / msvc-release (sin CUDA: solo CPU)
cmake --build --preset mingw-release
ctest --preset mingw-release          # ~130 pruebas (unitarias, estadísticas, determinismo, API de la GUI)

./build/mingw-release/gui/mc_gui                      # interfaz gráfica (ver docs/GUI.md)
./build/mingw-release/ejemplo01 0.01 --backend=cpu --threads=8   # ejemplo por línea de comandos
```

* **CPU o GPU**: todos los ejemplos aceptan `--backend=cpu|cuda --threads=N` (por defecto CUDA si el binario la incluye
  y hay GPU; si no, la CPU con todos los hilos). `colab_build.sh` compila con CUDA.
* **GUI** (`mc_gui`): ventana local con convergencia y trayectorias animadas, 11 ejemplos predefinidos, ES/EN y tema
  claro/oscuro. Servidor interno solo en `127.0.0.1`; todo embebido en el ejecutable. → [`docs/GUI.md`](docs/GUI.md)
* **Reproducibilidad**: el motor CPU da resultados **idénticos bit a bit con cualquier número de hilos**, y su QMC genera
  los mismos puntos Sobol (con el mismo scrambling de Hong-Hickernell) que la GPU. → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
* **Rendimiento y optimizaciones** medidas: [`docs/perf/OPTIMIZACIONES.md`](docs/perf/OPTIMIZACIONES.md). Ideas para la GPU,
  sin aplicar: [`docs/GPU_OPTIMIZATION_IDEAS.md`](docs/GPU_OPTIMIZATION_IDEAS.md).

> **Estado de la validación.** El backend CPU, la GUI y los tests se han desarrollado y ejecutado en un equipo **sin GPU**:
> el código CUDA sigue siendo el original (solo se movió código de host y se añadió `cuda_device_available()`), el
> adaptador de la fachada se compila y enlaza contra un *stub* con las mismas firmas, y la paridad CPU↔GPU se valida con
> `tools/colab_validate.sh` (A100 en Colab). Hasta ejecutarlo en una GPU, la ruta CUDA debe considerarse **sin reverificar**.

## Problemas conocidos

* **Windows, Smart App Control** (modo "activado"): puede negarse a ejecutar un `.exe` recién enlazado sin firmar
  (`Permission denied` en Git Bash, "An Application Control policy has blocked this file" en PowerShell; evento 3077 de
  Code Integrity). El veredicto es por *hash* del binario y es intermitente: basta **reenlazar** (`touch` de un fuente del
  destino y volver a compilar) y reintentar. Desactivar Smart App Control es una decisión del usuario (irreversible sin
  reinstalar Windows); el proyecto no lo toca.
* **Mediciones de rendimiento**: usa el equipo enchufado; en batería la CPU baja de frecuencia y los tiempos absolutos no
  son comparables (ver `docs/perf/OPTIMIZACIONES.md`).

## Los cuatro métodos

**Monte Carlo (MC).** Cada trayectoria se simula con un esquema de Euler (Euler–Milstein en la varianza para Heston) a partir de incrementos brownianos generados con `cuRAND` (generador pseudoaleatorio XORWOW). Un piloto inicial estima la varianza de la muestra y, a partir de ella, el número de trayectorias necesario para alcanzar la tolerancia `eps` objetivo (regla `N = 2·Var/eps²`); el resto de trayectorias se generan y evalúan en lotes (`N_batch`) dimensionados dinámicamente según la memoria libre de la GPU.

**Quasi-Monte Carlo (QMC).** Sustituye el ruido pseudoaleatorio por una secuencia de Sobol, que cubre el hipercubo `[0,1]^D` con mejor discrepancia que puntos i.i.d. y por tanto converge más rápido en integrandos suaves. El error no se puede estimar con la varianza muestral clásica (los puntos de una única secuencia Sobol no son independientes), así que se usan `R` réplicas de un *randomized QMC*: cada réplica genera la misma secuencia Sobol (mediante vectores directores de Joe-Kuo, vía `curandStateSobol32_t`) pero se le aplica su propio scrambling independiente, siguiendo la construcción de Hong–Hickernell (una matriz triangular inferior aleatoria e invertible sobre `F₂`, con `1` en la diagonal, más un desplazamiento digital aditivo — teorema 2.20 / algoritmo 823 de ACM). Además, la matriz y el desplazamiento de cada (réplica, dimensión) no se generan ni se almacenan de antemano sino se derivan en el momento con ayuda de la función hash `splitmix32` a partir de una semilla (`replica_salt`) propia de la réplica, y que el producto fila·vector sobre `F₂` se reduce a la operación `popcount` por fila (`__popc`). La media de las `R` réplicas es el estimador del precio, y la varianza entre réplicas (`var_of_means`) es ahora el estimador insesgado estándar de un RQMC con scrambling propio por réplica. El bucle duplica progresivamente el número de puntos por réplica hasta que `var_of_means < eps²/2` o se alcanza `max_doublings`. Las normales Sobol pueden además pasar por dos transformaciones antes del esquema de Euler: **Brownian Bridge** (construye la trayectoria de "grueso a fino", concentrando la varianza explicada en las primeras coordenadas Sobol, que son las de menor discrepancia) o **PCA** (proyección sobre las componentes principales de la matriz de covarianzas del movimiento browniano, aplicada como un único producto matricial por lote vía `cuBLAS` con Tensor Cores en `fp16→fp32`).

**Multilevel Monte Carlo (MLMC).** Sigue el esquema de Giles (2008): en vez de simular todo con paso fino `h`, se estima el precio como una suma telescópica `E[Y] = Σ_l E[P_l − P_{l−1}]` sobre niveles `l = 0..L` con pasos `h_l = T/M^l` (`M` = factor de refinamiento). Cada nivel `l>0` simula **una misma trayectoria** con dos discretizaciones acopladas (fina y gruesa, compartiendo el mismo ruido browniano) para que `Var[P_l − P_{l−1}] → 0` cuando `h_l → 0`; el nivel grueso reconstruye su incremento sumando los pasos finos entre dos puntos gruesos, así que ambos ven exactamente el mismo camino browniano. Un piloto por nivel estima `V_l`, y la asignación óptima de trayectorias por nivel (`N_l ∝ sqrt(V_l/coste_l) · Σ sqrt(V_l·coste_l)`) minimiza la varianza total a coste fijo. El número de niveles `L` crece adaptativamente hasta que el sesgo estimado (extrapolado de `E_L`, `E_{L-1}`) cae por debajo de `eps/√2`.

**Multilevel Quasi-Monte Carlo (MLQMC).** Combina ambos: en cada nivel, en vez de trayectorias MC acopladas, se generan `R` réplicas Sobol con el mismo scrambling de Hong-Hickernell de QMC, pero ahora la semilla del scrambling (`replica_salt`) se deriva de `(nivel, réplica)` en vez de solo de la réplica. Como en QMC, el scrambling por punto es lo que decorrelaciona: dos pares `(nivel, réplica)` distintos pueden compartir el mismo offset dentro de la secuencia Sobol subyacente sin colisionar, porque cada uno aplica una matriz de scrambling distinta. El estimador de cada diferencia de nivel usa la media de las réplicas, y la telescópica final es la misma suma `Σ_l E[P_l−P_{l−1}]` de MLMC, pero con varianza por nivel reducida gracias a la baja discrepancia de Sobol.

## Streams de CUDA y paralelismo entre niveles

Un *stream* de CUDA es una cola ordenada de trabajo (kernels, copias, llamadas a `cuRAND`/`cuBLAS`) que la GPU ejecuta de forma asíncrona respecto a la CPU, en el orden en que se encola. `run_mlmc_cuda` y `run_mlqmc_cuda` crean un stream *non-blocking* por nivel (`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`), cada uno con su propio generador de `cuRAND` (o, en MLQMC, su propio `cublasHandle_t` si se usa PCA) y su propio entorno de memoria. Al lanzar todos los niveles antes de sincronizar ninguno (`launch_level(l, ...)` para cada `l` seguido de `collect_level(l)` para cada `l`), los kernels de niveles distintos pueden solaparse de verdad en la GPU en vez de ejecutarse uno tras otro — el nivel `l=0` (barato, muchas trayectorias) puede seguir corriendo mientras el nivel `l=L` (caro, pocas trayectorias) ya ha terminado, aprovechando mejor los *streaming multiprocessors* de la tarjeta. Dentro de un mismo nivel, sucesivas llamadas a `launch_level` (piloto, refinamientos, extensión de `L`) reutilizan el mismo stream y el mismo generador, así que el orden interno queda garantizado por la propia cola del stream sin necesitar sincronizaciones explícitas adicionales.

## Kernels, acumuladores y reducción

Cada trayectoria se asigna a un hilo CUDA (`p = blockIdx.x*BLOCK_SIZE + threadIdx.x`, `BLOCK_SIZE=256`); los hilos sobrantes del último bloque quedan protegidos por un `if (p < N_paths)` que los deja aportar exactamente `0` a la suma en vez de contaminarla. El acumulador *running* del payoff (suma para asiáticas, mínimo para lookback, máximo para barrera) se actualiza en un registro local del hilo tras cada paso de Euler mediante `d_running_update<PayoffKind>`, especializado en tiempo de compilación con `if constexpr` para no pagar el coste en tiempo de ejecución. Al terminar la trayectoria, cada hilo reduce su valor (`Y`, `Y²`, y en MLMC también `ΔY`, `ΔY²`) dentro del bloque con `cub::BlockReduce` (reducción en memoria compartida, sin comunicación entre bloques), y solo el hilo `0` de cada bloque hace un `atomicAdd` sobre el acumulador global en memoria del dispositivo (`d_sums`). Esto reduce el número de operaciones atómicas de "una por hilo" a "una por bloque" (256×), evitando que la contención en `atomicAdd` sea el cuello de botella, y evita cualquier condición de carrera entre bloques porque cada uno solo escribe una vez, de forma atómica, sobre una posición ya inicializada a cero (`cudaMemset`/`cudaMemsetAsync` antes del lanzamiento).

## Evitación de condiciones de carrera

El código evita carreras en varios niveles: (1) los parámetros del kernel (`KernelParams c_p`) viven en memoria `__constant__`, de solo lectura durante el kernel, y se fijan una única vez por llamada con `cudaMemcpyToSymbol` antes de lanzar ningún nivel — en MLMC/MLQMC `c_p` no depende del nivel `l` precisamente para poder lanzar todos los niveles en paralelo sin que uno sobrescriba los parámetros que otro está leyendo; (2) cada nivel en MLMC/MLQMC tiene su propio entorno de memoria para el ruido (`d_Z`), su propio generador de `cuRAND` y su propio acumulador `d_sums`, así que streams distintos nunca comparten memoria de escritura; (3) dentro de un mismo stream, el orden FIFO garantiza que la generación de un nuevo lote de ruido en `d_Z` espera a que el kernel anterior haya terminado de leerlo antes de sobreescribirlo, sin necesitar el uso de un lock; (4) el producto de Cholesky para cestas correlacionadas y la proyección PCA se hacen como una única llamada por lotes a `cuBLAS` (`cublasSgemmStridedBatched` / `cublasGemmEx`) en vez de dentro del kernel de payoff, evitando que cada hilo tenga que releer memoria global de forma no coalescente o mantener `n_assets` acumuladores locales.

## Estructura del repositorio

- `models.hpp` / `payoffs.hpp`: parámetros de los modelos (GBM, Heston, Dupire local, cesta Dupire) y payoffs (europea, asiática aritmética/geométrica, lookback, barrera, basket), como `std::variant`.
- `mc_types.hpp`, `sweep.*`, `mc_progress.hpp`, `mc_format.hpp`: configuración y resultados comunes a ambos backends, barrido de precisión (`run_precision_sweep`), progreso/cancelación y formato de tablas.
- `utils.*`: fórmulas analíticas de referencia, precómputo de Brownian Bridge y PCA (forma cerrada, sin Eigen, caché por referencia), Richardson, `RunningStats`. `reference_prices.*`: Black-Scholes con deriva, Asian geométrica y **Heston de Fourier**.
- `methods_cuda.cuh` / `methods_cuda.cu`: backend GPU de los cuatro métodos, variables de control e importance sampling (código original).
- `cpu/`: **motor CPU multihilo** (pool de hilos, RNG por camino, Sobol+Hong-Hickernell, núcleos, MC/QMC/MLMC/MLQMC, CV/IS, muestreador de trayectorias). → `docs/ARCHITECTURE.md`
- `engine/`: fachada `mc::run_*` / `mc::run(RunSpec)` que despacha a CPU o GPU; `cuda_stub.cpp` para builds sin CUDA.
- `gui/`: servidor local + interfaz web (`gui/web`), embebida en `mc_gui`. → `docs/GUI.md`
- `examples/`: 11 ejemplos (`ejemplo01`…`ejemplo11`), con `--backend` y `--threads`.
- `tests/`: suite de doctest (etiquetas `fast`, `stat`, `determinism`, `cuda`). `bench/`: `bench_cpu` (cargas W1–W8, escalado por hilos) y `bench_micro`. `tools/`: generadores de tablas y valores dorados, informe de benchmarks, scripts de Colab.
- `docs/`: arquitectura, GUI, rendimiento (`docs/perf/`) e ideas para la GPU. `third_party/`: doctest, cpp-httplib, nlohmann/json, ECharts (con sus licencias).
- `colab_build.sh`, `informes_finales/`, `resultados/`, `gen_informe*.py`: compilación en Colab e informes numéricos de la A100 (originales).
