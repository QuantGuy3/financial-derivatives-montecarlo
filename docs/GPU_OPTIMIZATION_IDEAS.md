# Ideas de optimización para el backend CUDA (sin aplicar)

> **Estado.** Las optimizaciones de este documento **no se han aplicado ni medido**: el equipo de desarrollo
> no tenía GPU NVIDIA ni `nvcc`, y no se quiere tocar `methods_cuda.cu` sin poder compilarlo y medirlo. Cada
> punto indica *qué* se observó leyendo el código, *por qué* debería ayudar y *cómo medirlo* (ver
> `tools/colab_validate.sh` y `tools/colab_bench.sh`). Se ordenan por relación esperada beneficio/riesgo.
> Antes de aplicar cualquiera: línea base con `ejemplo01 --backend=cuda` y perfil con Nsight Systems
> (el repositorio ya compila con `-lineinfo`).

El motor CPU (carpeta `cpu/`) ya incorpora dos de estas ideas y las mide (`docs/perf/OPTIMIZACIONES.md`):
Sobol por Gray-code con vectores directores pre-scrambleados (`V' = L·V`, ×265 frente a la versión de la GPU)
y generador de normales sin funciones transcendentes. Los números de la CPU **no** son extrapolables a la GPU.

## 1. Quick wins en los controladores (host)

| # | Observación en el código | Propuesta | Medir |
|---|---|---|---|
| 1 | `run_qmc_cuda::accumulate_range` hace `cudaMalloc/cudaFree` de `d_Z_r`, `d_dW_r`, `d_W_scratch`/`d_Z_f16` **por réplica y por lote** (~`R × duplicaciones × 4` pares). `cudaMalloc` serializa el dispositivo. | Reservar una vez los búferes (tamaño del lote máximo) y reutilizarlos; un pool por stream. | Tiempo de `QMC BB/PCA` en `ejemplo01` a eps pequeño; nº de llamadas `cudaMalloc` en Nsight. |
| 2 | `cudaDeviceSynchronize()` tras cada lote en `accumulate_range`. | Quitarlo: el orden FIFO del stream ya garantiza dependencias; sincronizar solo antes de leer `d_sums`. | Solapamiento de kernels en la línea de tiempo. |
| 3 | `make_params(...)` + `cudaMemcpyToSymbol(c_p, ...)` dentro del bucle de lotes (mismos parámetros cada vez). | Hoistear fuera del bucle (como ya hace `run_mlmc_cuda`). | Nº de `cudaMemcpyToSymbol`. |
| 4 | `run_mlmc_cuda` y `run_mlqmc_cuda` **crean y destruyen** el generador cuRAND y la memoria en cada `launch_level`/`collect_level`. | Crear el generador y `d_Z` una vez por nivel (el stream y los búferes ya se conservan). | Tiempo de MLMC en eps fino (muchas iteraciones). |
| 5 | Réplicas QMC se ejecutan en serie en el stream por defecto. | Un stream por réplica (o por grupo de réplicas) y `cudaMemcpyAsync` de sumas al final. | Utilización de SM en Nsight. |

## 2. Núcleos

| # | Observación | Propuesta | Riesgo |
|---|---|---|---|
| 6 | **Escalado √h**: en modo Raw se hace `cudaMemcpy` D2D de `d_Z` a `d_dW` y después `kernel_scale` (2 pasadas sobre `D·N` floats). | Fusionar el factor `sqrt(h)` en `kernel_gen_hh_scrambled_sobol_normal` (escribir ya escalado) y usar `d_Z` directamente. | Bajo. |
| 7 | **Scrambling HH**: cada hilo carga `dv[32]` de memoria global y hace 32 `splitmix32` + 32 `__popc` por palabra. | Precalcular **una vez por (réplica, dimensión)** los vectores directores scrambleados `V'_j = L·V_j` (32 palabras; `hh_scramble` es lineal sobre F₂ salvo el desplazamiento) y generar el punto como XOR de `V'_j` sobre los bits de `gray(idx)` + desplazamiento. En CPU esto cuesta ~0,8 ns/palabra frente a ~200 ns de la versión ingenua (`bench_micro`). Verificar bit a bit contra la versión actual. | Medio (reescritura del kernel de generación). |
| 8 | **`kernel_bb_transform`** usa un búfer global `d_W_scratch` de `(N+1)×N_paths` floats con lecturas/escrituras no necesarias en memoria global y luego `d_dW_out` otra vez. | Fusionar BB con el kernel de payoff (el camino se reconstruye en registros/memoria compartida para N pequeño) y evitar `d_dW`. | Medio. |
| 9 | `kernel_mc`/`kernel_mlmc` leen `d_dW[k*N_paths+p]` en cada paso: son **limitados por ancho de banda de memoria** (≈10 flops por 4 B leídos). | Generar las normales dentro del kernel con la API de dispositivo de cuRAND (Philox) y no materializar `D·N` floats. Los lotes dejan de depender de la memoria libre. | Medio-alto (cambia la reproducibilidad entre versiones). |
| 10 | Niveles pequeños de MLMC (nivel 0: un paso, muchísimos caminos) están dominados por latencia de lanzamiento. | Agrupar lanzamientos con **CUDA Graphs** por iteración de Giles. | Bajo-medio. |
| 11 | **PCA**: la matriz va en **fp16** a Tensor Cores (`cublasGemmEx` F16→F32). Introduce un sesgo del orden de 1e-3 relativo en la covarianza de `dW`. | Cuantificarlo (ver abajo) y valorar `CUBLAS_COMPUTE_32F_FAST_TF32` o BF16 según el sesgo aceptable. | Bajo (solo medir). |

### Cómo cuantificar el sesgo de fp16 en PCA (sin GPU)

El motor CPU trabaja en `double`, así que sirve de referencia: emular el redondeo `binary16` de `M_pca`
(`float → half → float`) y comparar `M Mᵀ` frente a `h·I` (la covarianza exacta de los incrementos). Un
experimento de 20 líneas sobre `pca_compute(m, T)` da el error de covarianza por `m`; si supera `eps/√2` del
objetivo de precisión, el modo PCA de la GPU sesga el precio de forma medible.

## 3. Cómo se validaría una optimización

1. `tools/colab_validate.sh` (compila con CUDA, ejecuta `ctest -L cuda`: paridad CPU↔GPU y volcado de Sobol).
2. `tools/colab_bench.sh` (mide `ejemplo01` y `ejemplo09` con `--backend=cuda` y `--backend=cpu`).
3. Mantener un cambio solo si gana ≥ 5 % en su carga objetivo, no pierde > 2 % en las demás y la paridad
   estadística con la CPU se conserva (los resultados GPU y CPU deben coincidir dentro de sus errores
   estándar; el QMC coincide a 4 decimales porque ambas generan los mismos puntos Sobol+HH).
