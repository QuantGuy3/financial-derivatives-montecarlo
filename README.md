# Financial Derivatives Monte Carlo (GPU / CUDA)

Motor de valoración de derivados financieros en C++/CUDA, pensado para GPUs NVIDIA (objetivo `sm_80`, A100) con `cuRAND`, `cuBLAS` y `CUB`. Implementa cuatro familias de estimadores — Monte Carlo estándar (MC), Quasi-Monte Carlo (QMC), Multilevel Monte Carlo (MLMC) y Multilevel Quasi-Monte Carlo (MLQMC) — sobre ocho modelos de precio (GBM, Heston, Dupire local, cestas multi-activo, Merton, Bates, SABR y NIG) y varios payoffs (europeas, asiáticas, lookback, barrera, basket), con soporte de reducción de varianza (variables de control, importance sampling) y de técnicas de reducción de dimensión efectiva (Brownian Bridge, PCA con Tensor Cores).

Incluye además un **módulo de productos estructurados** (autocallables Phoenix y Athena, reverse convertibles, notas con capital protegido, certificados bonus, twin-win, shark fin, range accrual, cliquets y cualquier combinación de sus piezas) sobre cestas de hasta 16 subyacentes, cada uno con su propio modelo. Los productos se definen con un asistente interactivo o con plantillas, se guardan en ficheros JSON legibles y se valoran en GPU (MC o QMC) o en CPU; ver [Productos estructurados](#productos-estructurados).

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

## Modelos del motor GPU

| Modelo (`models.hpp`) | Dinámica | Esquema | Ruido por paso | Ejemplo |
|---|---|---|---|---|
| `GBMParams` | Black-Scholes | Euler | 1 | 01–04, 09, 11 |
| `HestonParams` | Heston | Euler en S, media exacta en v | 2 | 05 |
| `DupireLocalParams` | vol local σ₀·e^(-αt)·(S/S₀)^(β-1) | Euler | 1 | 06, 10 |
| `MultiDupireParams` | cesta de n activos con vol local | Euler + Cholesky (cuBLAS) | n | 07, 08 |
| `MertonParams` | Black-Scholes + saltos lognormales de Poisson | logarítmico exacto | 3 | 12 |
| `BatesParams` | Heston + saltos de Merton | log-Euler, truncamiento completo | 4 | 13 |
| `SABRParams` | dF = α F^β dW₁, dα = ν α dW₂ | log-Euler, α exacto | 2 | 14 |
| `NIGParams` | Normal Inverse Gaussian (Lévy) | exacto (subordinador IG) | 3 | 15 |

Los cuatro modelos nuevos funcionan con los cuatro métodos (MC, QMC, MLMC, MLQMC) y con todos los payoffs. En los saltos y en la NIG todo el ruido sigue siendo normal (el número de saltos y la uniforme del muestreo IG se obtienen como Φ(z)), así que QMC y MLQMC los tratan como una dimensión más; Brownian Bridge y PCA solo se aplican a modelos con un único factor de ruido. En MLMC el paso grueso reutiliza la suma exacta de los incrementos brownianos, de los saltos y de los incrementos del subordinador de los pasos finos, de modo que ambos niveles ven exactamente el mismo camino. Las referencias de los ejemplos 12–15 son exactas o semianalíticas (`referencias.hpp`): serie de Merton, fórmula de Lewis con las funciones características de Bates y NIG, y para SABR un MC de alta resolución (la fórmula de Hagan se imprime como aproximación).

## Productos estructurados

```bash
./build/estructurados nuevo                        # asistente: pregunta paso a paso y guarda el JSON
./build/estructurados plantilla phoenix mi.json    # plantilla lista para editar (ver: estructurados plantillas)
./build/estructurados mostrar mi.json              # valida y muestra la ficha con el perfil de pago
./build/estructurados valorar mi.json --griegas    # valora (GPU si hay; si no, CPU) y calcula sensibilidades
```

**Definición.** Un producto es un fichero JSON con las secciones de una ficha de producto: mercado (tipo plano o curva), subyacentes (spot, nivel inicial, dividendos y modelo), correlación, cesta (peor, mejor, media o ponderada), calendario de observación, cupones (fijos, condicionales con o sin memoria, *range accrual*), autocancelación (con *step-down* y primas acumuladas), barrera de capital y barrera desactivante (observadas a vencimiento, en fechas, diariamente o de forma continua con corrección BGK) y la amortización final como suma de componentes (fijo, lineal, call, put, put vendida, digitales y cliquet, condicionados o no a la barrera). Los porcentajes se escriben como `"60%"`, los plazos como `"18M"` o `"3A"`, y se admiten fechas reales para productos ya vivos. La referencia completa está en [`estructurados/FORMATO.md`](estructurados/FORMATO.md) y en `productos/` hay un ejemplo de cada familia.

```json
{
  "nombre": "Phoenix worst-of SX5E / SPX",
  "nominal": 1000, "vencimiento": "3A",
  "mercado": {"tipo_interes": "3%"},
  "subyacentes": [
    {"nombre": "SX5E", "spot": 4800, "dividendos": "3%",
     "modelo": {"tipo": "heston", "vol_inicial": "18%", "kappa": 1.5, "vol_largo_plazo": "20%", "xi": 0.6, "rho": -0.7}},
    {"nombre": "SPX", "spot": 5600, "dividendos": "1.5%", "modelo": {"tipo": "black_scholes", "volatilidad": "18%"}}
  ],
  "correlacion": 0.65, "cesta": "peor",
  "observaciones": {"frecuencia": "trimestral"},
  "cupones": {"tipo": "condicional", "importe": "2%", "barrera": "70%", "memoria": true},
  "autocancelacion": {"barrera": "100%", "desde": 2},
  "barrera_capital": {"nivel": "60%", "observacion": "vencimiento"},
  "amortizacion_final": [{"tipo": "fijo", "importe": "100%"},
                         {"tipo": "put_vendida", "strike": "100%", "condicion": "barrera_tocada"}]
}
```

**Facilidad de uso.** El asistente (`estructurados nuevo`) ofrece las plantillas habituales (o la construcción pieza a pieza), propone valores por defecto, explica cada dato con `?`, valida cada respuesta al momento (rangos, matriz de correlación válida, parámetros compatibles) y, antes de guardar, muestra la ficha: calendario con barreras y cupones por fecha, y el perfil de pago a vencimiento con y sin barrera tocada. Al leer un fichero, los errores indican la clave y la línea, y las erratas en las claves se avisan con una sugerencia («¿quería decir "observacion"?»).

**Modelos por subyacente.** `black_scholes`, `heston`, `merton`, `bates`, `vol_local` (paramétrica tipo Dupire/CEV), `sabr` y `nig`; cada subyacente puede llevar uno distinto y la correlación se aplica a sus brownianos principales (`estructurados modelos` lista los parámetros).

**Valoración.** El núcleo que simula una trayectoria y evalúa el producto (`estructurados/simulacion_hd.hpp`) es código `__host__ __device__`: el mismo lo ejecutan el kernel CUDA (un hilo por trayectoria, ruido generado en el propio hilo con Philox o Sobol + scrambling de Hong-Hickernell, sin buffers de ruido en memoria global) y el motor CPU multihilo de doble precisión. Con la misma semilla ambos recorren las mismas trayectorias. El número de trayectorias se adapta al error estándar objetivo (MC: piloto + N = Var/error²; QMC: duplicación de puntos con 16 réplicas). El informe incluye la descomposición del valor (cupones, amortización anticipada y final), probabilidades de autocancelación por fecha, vida media, probabilidad de tocar la barrera y de pérdida, y con `--griegas` delta, gamma, vega y rho por diferencias finitas con números aleatorios comunes.

**Validación.** `tests/test_estructurados.cpp` contrasta el motor con precios cerrados o semianalíticos: casos deterministas exactos (cupón cero, bono con curva, memoria, range accrual, cliquet, autocancelación), Black-Scholes (call, put, digital, reverse convertible, put down-and-in continua frente a Reiner-Rubinstein, shark fin), los siete modelos (Fourier, serie de Merton, Hagan, martingala), cestas worst-of/best-of frente a la normal bivariante, QMC y la equivalencia precisión simple/doble. Si hay GPU, compara además GPU y CPU trayectoria a trayectoria.

## Compilación

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release        # -DCMAKE_CUDA_ARCHITECTURES=80 por defecto
make -j
ctest                                       # tests del módulo de productos estructurados
```

Sin `nvcc` (o con `-DMC_USAR_CUDA=OFF`) se compila igualmente la parte CPU: el programa `estructurados` (motor CPU, Monte Carlo) y los tests. Con CUDA se compilan además los 15 ejemplos y el motor GPU de estructurados. `colab_build.sh` hace todo esto en Google Colab. Eigen se busca en `/usr/include/eigen3` (`-DEIGEN3_INCLUDE_DIR=...` para cambiarlo).

## Estructura del repositorio

- `models.hpp` / `payoffs.hpp`: parámetros de los modelos de precio (GBM, Heston, Dupire local, cesta Dupire multi-activo, Merton, Bates, SABR, NIG) y de los payoffs (europea, asiática aritmética/geométrica, lookback, barrera, basket), como `std::variant`.
- `utils.hpp` / `utils.cpp`: fórmulas analíticas de referencia, precómputo de Brownian Bridge y PCA en CPU (Eigen), estimación del sesgo por Richardson, acumuladores *running* (`RunningStats`, media/varianza en un solo paso) y tablas de resultados.
- `methods_cuda.cuh` / `methods_cuda.cu`: implementación GPU de los cuatro métodos (`run_mc_cuda`, `run_qmc_cuda`, `run_mlmc_cuda`, `run_mlqmc_cuda`), variables de control e importance sampling.
- `hd_comun.hpp` / `modelos_hd.hpp`: código compartido CPU/GPU — Philox4x32-10, Sobol con scrambling de Hong-Hickernell, inversa de la normal y los pasos de los siete modelos de subyacente.
- `referencias.hpp` / `referencias.cpp`: precios de referencia (Black-Scholes con dividendos, digitales y barreras, Merton, Fourier para Heston/Bates/NIG, Hagan para SABR, normal bivariante).
- `estructurados/`: módulo de productos estructurados — lectura/escritura JSON (`json.*`), definición y validación (`producto.*`), ficha (`ficha.cpp`), plantillas y asistente (`plantillas.*`, `asistente.*`), núcleo de simulación (`simulacion_hd.hpp`, `plan.hpp`), motores CPU y GPU (`motor_cpu.cpp`, `motor_gpu.cu`), valoración y griegas (`valoracion.*`), programa `estructurados` (`main.cpp`) y la referencia del formato (`FORMATO.md`).
- `productos/`: productos de ejemplo en JSON (Phoenix, Athena, reverse convertible, capital protegido, bonus, twin-win, shark fin, range accrual, cliquet, opción europea y un autocallable vivo con fechas reales).
- `examples/`: 15 ejemplos ejecutables (`ejemplo01`…`ejemplo15`) que cubren europea, asiática, lookback, barrera, Heston, Dupire, cestas correlacionadas/no correlacionadas, variables de control, importance sampling, Merton, Bates, SABR y NIG.
- `tests/`: tests del módulo de productos estructurados.
- `colab_build.sh`: script de compilación pensado para un entorno Google Colab con GPU.
- `informes_finales/`: informes de resultados numéricos generados sobre los ejemplos anteriores.
- `gen_informe.py` / `gen_informe_completo.py`: generación automática de esos informes a partir de las salidas de los ejemplos.
