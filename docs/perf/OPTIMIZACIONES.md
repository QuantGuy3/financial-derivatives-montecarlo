# Optimizaciones del motor CPU: mediciones y decisiones

Criterio de aceptación (fijado antes de medir): una optimización se **conserva** si gana ≥ 5 % en su carga
objetivo, no pierde > 2 % en las demás, los tests pasan y se mantiene el determinismo (las declaradas bit-exactas se
verifican bit a bit). Si no, se **descarta** y se deja constancia aquí.

## Cómo se mide

* `bench/bench_cpu` — cargas de extremo a extremo W1–W8 (MC GBM europea, asiática, Dupire, Heston; QMC BB/PCA; MLMC; MLQMC),
  2 repeticiones de calentamiento y ≥ 5 medidas; mediana, mínimo, IQR y CV %, Mcaminos/s y ns por paso-camino; escalado
  por hilos. Salida JSON → `tools/bench_report.py` genera las tablas.
* `bench/bench_micro` — piezas aisladas (un hilo, mejor de N repeticiones) con la variante antigua y la nueva en el mismo
  binario; el experimento de extremo a extremo **intercala** A y B para que un cambio de frecuencia de la CPU afecte a
  ambos por igual.
* Variantes de compilación (`MC_LANES`, `MC_ARCH=native`): `tools/run_variant_bench.sh`.

**Máquina**: AMD Ryzen 7 7730U (8 núcleos / 16 hilos, 15 W), g++ 15.2 (MSYS2 UCRT64), `-O3`, sin `-march=native`,
`-ffp-contract=off`.

> **Aviso de fiabilidad.** Es un portátil de 15 W cuyos relojes dependen del estado de energía. La línea base
> (`baseline-naive-cpu`) se midió con la máquina "normal" (≈16 ns por paso en W1). Las mediciones posteriores se hicieron
> **con el equipo funcionando con batería (6–7 %)**, lo que reduce la frecuencia a la mitad (la misma carga tarda ≈2× más).
> Por eso las comparaciones de esta sesión son **razones dentro de la misma ejecución** (A/B intercalado), que sí son
> robustas; los tiempos absolutos de las tablas del apartado 2 no son comparables con los de la línea base. Conviene repetir
> `bench_cpu` enchufado y con el plan de alto rendimiento antes de citar cifras absolutas.

## 1. Línea base (`git tag baseline-naive-cpu`)

Motor ingenuo: Box–Muller, bloques de 8 caminos, sin más trucos. Mediana (s), 2^22 caminos × 64 pasos (W1); 2^20 × 256 (W2);
2^19 × 256 (W3); 2^20 × 128 pasos con 2 factores (W4). Detalle completo en `baseline-naive-cpu.json/.txt`.

| carga | 1 hilo | 2 | 4 | 8 | 16 | speedup 8 hilos (eficiencia) | speedup 16 (eficiencia) |
|---|---|---|---|---|---|---|---|
| W1 GBM europea | 4.35 | 2.79 | 1.48 | 0.80 | 0.58 | ×5.4 (68 %) | ×7.5 (47 %) |
| W2 GBM asiática | 4.29 | 2.74 | 1.47 | 0.80 | 0.56 | ×5.4 (67 %) | ×7.7 (48 %) |
| W3 Dupire | 4.81 | 3.09 | 1.74 | 0.96 | 0.69 | ×5.0 (62 %) | ×7.0 (44 %) |
| W4 Heston | 4.79 | 2.95 | 1.53 | 0.82 | 0.56 | ×5.9 (73 %) | ×8.5 (53 %) |

El escalado a 8 hilos es de ×5–6 (eficiencia 62–73 %): el 7730U comparte presupuesto de 15 W, así que los relojes bajan al
cargar más núcleos; con 16 hilos (SMT) la ganancia adicional es de ×1.3–1.4. La reducción es en orden de índice, sin
sincronización por camino, por lo que la pérdida no viene del reparto de trabajo.

## 2. Optimizaciones conservadas

| # | Cambio | Carga objetivo | Antes → después | Razón | Verificación |
|---|---|---|---|---|---|
| 1 | **Generador Ziggurat** (256 capas) en lugar de Box–Muller | MC GBM europea, 1 hilo, **extremo a extremo** | 30.5 → 7.86 ns/paso | **×3.9** | momentos, KS (p > 10⁻³) y fracciones de cola (hasta |z| > 4) en `stat` |
| | | solo la normal (`bench_micro`) | 27.7 → 5.7 ns/normal | ×4.9 | idem |
| 2 | **Sobol por Gray-code con `V' = L·V`** (1 XOR por punto y dimensión) frente a los 32 hashes `splitmix32` + 32 popcounts de la versión de la GPU | generar palabras de 32 bits | 203.6 → 0.77 ns/punto·dim | **×265** | idéntico bit a bit a la versión ingenua (`test_sobol`), valores de scipy (21 201 dims × 64 puntos) |
| | | con la inversa de la normal incluida | 257 → 23.0 ns/punto·dim | ×11.2 | idem; el cuello de botella pasa a ser AS 241 (≈20 ns) |
| 3 | **PCA: GEMM por bloques** de 64 caminos (acumulador en L1) frente al producto por camino | m = 64 | 7 949 → 3 364 ns/camino | ×2.4 | `M·Mᵀ = h·I` (`test_utils`, `test_qmc_cpu`) |
| | | m = 256 | 318 279 → 32 114 | ×9.9 | |
| | | m = 1024 | 12 508 180 → 391 557 | ×31.9 | |
| 4 | **Brownian Bridge**: `bb_apply` reutiliza el vector `W` (antes 1 `std::vector` por simulación) | N = 64 / N = 1024 | 256 → 173 / 3 418 → 3 060 ns/camino | ×1.48 / ×1.12 | `B·Bᵀ = h·I` |
| 5 | **Caché de BB/PCA por referencia** (antes `return it->second` copiaba `2·m²` valores por llamada) | `pca_compute` desde la caché, m = 2048 | 12.6 ms → ~0 | — | caché estable bajo concurrencia (`test_utils`) |
| | | m = 512 | 0.70 ms → ~0 | — | |
| 6 | **Matriz PCA con tabla de senos** (4m+2 `sin()` en lugar de m²) | construir la matriz, m = 512 / 2048 | 3.48 → 0.46 ms / 59.5 → 7.2 ms | ×7.5 / ×8.3 | comparada con la fórmula directa (`test_utils`) |
| 7 | **Reducción determinista en orden de índice** con ranuras de 64 B y pool propio | escalado | ver §1 | — | `determinism` (1/3/5/8/16 hilos, bit a bit) |

## 3. Probado y descartado

| Cambio | Resultado | Decisión |
|---|---|---|
| Inversa de la normal **por bloques** (rama central sin ramas, para solapar las cadenas de Horner) | 21.8 → 20.3 ns (×1.08) en U(0,1); ×1.01 en el generador Sobol | **Descartado**: < 5 % en la carga objetivo (el coste está en la cola, ≈66 ns por valor, y en la rama central sin FMA) |
| `-march=native` (AVX2) para la inversa de la normal | 21.8 → 19.5 ns | Se deja como opción (`MC_ARCH=native`); con `-ffp-contract=off` no hay FMA y el compilador no vectoriza la cadena |
| `exp(b·log x)` en lugar de `pow(x, b)` en Dupire | 28.1 → 25.3 ns/llamada (×1.11), ≈ 6 % de la carga W3 estimada | **Descartado**: en el umbral, cambia los bits y no elimina las dos transcendentes |
| Box–Muller / CDF inversa como generador por defecto | superados por Ziggurat | siguen disponibles (`NormalMethod`) |

## 4. No medido todavía (honestidad)

* **Anchura de bloque `MC_LANES`** (1/2/4/8/16) y **tamaño de chunk** (`kChunkWork`): la infraestructura está
  (`tools/run_variant_bench.sh`), pero no se pudieron obtener mediciones fiables con el equipo en batería.
* **Escalado por hilos tras el Ziggurat** (la tabla del §1 es la de la línea base con Box–Muller).
* **Vía O(m log m) para PCA** (base seno cerrada + FFT): se evaluó sobre el papel; con m ≤ 2048 el coste de la FFT de
  longitud 2m+1 (Bluestein) es del mismo orden que el GEMM por bloques, así que no se implementó.
* **`float` en CPU**: no se probó; la GPU ya usa `float` y la ventaja de la CPU en `double` es la reproducibilidad.

## 5. Cuellos de botella actuales (para la siguiente ronda)

1. **QMC**: la inversa de la normal (AS 241) es ≈ 20 de los ≈ 23 ns por punto·dimensión; una aproximación por tramos
   con tablas o vectorización AVX2/FMA sería la palanca (aceptando romper el bit a bit entre binarios).
2. **PCA**: O(m²) por camino; a m = 1024 cuesta ≈ 390 µs/camino frente a ≈ 10 µs del resto del camino.
3. **MC**: tras el Ziggurat, el generador y el paso de Euler pesan parecido; el siguiente salto sería generar y consumir las
   normales en el mismo bucle (sin el búfer transpuesto `Z[d*ld+p]`) y con SIMD explícito.
