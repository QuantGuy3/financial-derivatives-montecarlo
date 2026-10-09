# Optimizaciones del motor CPU: mediciones y decisiones

Criterio de aceptación (fijado antes de medir): una optimización se **conserva** si gana ≥ 5 % en su carga
objetivo, no pierde > 2 % en las demás, los tests pasan y se mantiene el determinismo (las declaradas bit-exactas se
verifican bit a bit). Si no, se **descarta** y se deja constancia aquí.

## Cómo se mide

* `bench/bench_cpu` — cargas de extremo a extremo W1–W8 (MC GBM europea, asiática, Dupire, Heston; QMC BB/PCA; MLMC;
  MLQMC), 2 repeticiones de calentamiento y ≥ 5 medidas; mediana, mínimo, IQR y CV %, Mcaminos/s y ns por paso-camino;
  escalado por hilos. Salida JSON → `tools/bench_report.py` genera las tablas.
* `bench/bench_micro` — piezas aisladas (un hilo, mejor de 7 repeticiones) con la variante antigua y la nueva en el mismo
  binario; el experimento de extremo a extremo **intercala** A y B para que un cambio de frecuencia de la CPU afecte a
  ambos por igual.
* Variantes de compilación (`MC_LANES`, `MC_ARCH=native`): `tools/run_variant_bench.sh` (ver §4: no se llegaron a medir).

**Máquina**: AMD Ryzen 7 7730U (8 núcleos / 16 hilos, 15 W), g++ 15.2 (MSYS2 UCRT64), `-O3`, sin `-march=native`,
`-ffp-contract=off`.

**Ficheros de datos** (`docs/perf/`):

| fichero | qué es | condiciones |
|---|---|---|
| `baseline-naive-cpu.{json,txt}` | motor ingenuo (Box–Muller) | equipo "normal" |
| `opt-ziggurat-ac.{json,txt,md}` | motor actual, W1–W8, 1–16 hilos, 5 repeticiones | **enchufado a la corriente** |
| `micro-ac.{json,txt}` | `bench_micro` (A/B por pieza) | **enchufado a la corriente** |

> **Nota de fiabilidad.** Es un portátil de 15 W cuyos relojes dependen del estado de energía. Una primera tanda se midió
> con la batería al 6–7 % (la misma carga tardaba ≈ 2×) y **se descartó**: todas las cifras de este documento salen de las
> ejecuciones en corriente (`*-ac.*`). Aun así, la línea base y `opt-ziggurat-ac` son tandas distintas del mismo equipo,
> no un A/B intercalado; la razón W1b (Box–Muller) / W1 (Ziggurat) **dentro de la misma ejecución** es la comparación más
> robusta. El CV % de algunas celdas llega al 10–25 % (ver `.txt`): léanse los factores con ± 10 %.

## 1. Línea base frente al estado actual

Mediana (s): 2^22 caminos × 64 pasos (W1); 2^20 × 256 (W2); 2^19 × 256 (W3); 2^20 × 128 pasos con 2 factores (W4).

| carga | hilos | línea base | actual | aceleración |
|---|---|---|---|---|
| W1 GBM europea | 1 | 4.348 | 0.854 | ×5.1 |
| | 8 | 0.801 | 0.226 | ×3.6 |
| | 16 | 0.579 | 0.156 | ×3.7 |
| W2 GBM asiática | 1 | 4.288 | 0.833 | ×5.1 |
| | 8 | 0.797 | 0.186 | ×4.3 |
| | 16 | 0.560 | 0.165 | ×3.4 |
| W3 Dupire | 1 | 4.810 | 2.659 | ×1.8 |
| | 8 | 0.964 | 0.653 | ×1.5 |
| | 16 | 0.687 | 0.477 | ×1.4 |
| W4 Heston | 1 | 4.794 | 0.978 | ×4.9 |
| | 8 | 0.817 | 0.201 | ×4.1 |
| | 16 | 0.563 | 0.196 | ×2.9 |

Tabla completa (1/2/4/8/16 hilos): `opt-ziggurat-ac.md`.

**Lectura.**

* La ganancia a 1 hilo (×5) viene casi entera del generador (Ziggurat). W3 (Dupire) mejora menos (×1.8) porque su coste
  dominante son dos funciones transcendentes por paso (`pow` y `exp`), no la normal (ver §3).
* **El escalado por hilos empeora tras el Ziggurat** — es esperable: al ser el paso mucho más barato, pesan más la parte
  serie (reducción ordenada, arranque del pool, relleno del búfer de ruido) y el reloj compartido de 15 W. Con el
  Box–Muller (W1b) la eficiencia a 8 hilos es del 64 %; con el Ziggurat (W1) del 47 %. En términos absolutos el Ziggurat
  sigue ganando en todos los puntos (W1b 16 hilos: 0.600 s; W1 16 hilos: 0.156 s, ×3.8).
* Eficiencia a 16 hilos (SMT sobre 8 núcleos físicos): 31–35 % en MC y 21–42 % en QMC (aceleración ×5–×6.7 sobre un hilo
  en 16 hilos; el SMT aporta poco más allá de los 8 núcleos y el límite de potencia comparte el reloj). W6 (PCA, m = 256)
  no mejora de 8 a 16 hilos (el GEMM por bloques ya ocupa las unidades de coma flotante de cada núcleo).
* MLMC/MLQMC escalan mejor que el MC puro a 4–8 hilos (76–84 % a 4 hilos): hay más trabajo por chunk. W8 (MLQMC) resuelve
  eps = 5·10⁻³ en 19 ms con 16 hilos.

## 2. Optimizaciones conservadas

Cifras de `micro-ac.txt` (un hilo, mejor de 7).

| # | Cambio | Carga objetivo | Antes → después | Razón | Verificación |
|---|---|---|---|---|---|
| 1 | **Generador Ziggurat** (256 capas) en lugar de Box–Muller | MC GBM europea, 1 hilo, **extremo a extremo** (A/B intercalado) | 11.67 → 3.08 ns/paso | **×3.8** | momentos, KS (p > 10⁻³) y fracciones de cola (hasta \|z\| > 4) en `stat` |
| | | solo la normal | 11.06 → 2.34 ns/normal | ×4.7 | idem |
| | | MLMC W7 (Box–Muller W7b → Ziggurat), 1 hilo | 1.587 → 0.898 s | ×1.8 | idem |
| 2 | **Sobol por Gray-code con `V' = L·V`** (1 XOR por punto y dimensión) frente a los 32 hashes `splitmix32` + 32 popcounts de la versión de la GPU | generar palabras de 32 bits | 83.6 → 0.307 ns/punto·dim | **×273** | idéntico bit a bit a la versión ingenua (`test_sobol`), valores de scipy (21 201 dims × 64 puntos) |
| | | con la inversa de la normal incluida | 102.8 → 9.49 ns/punto·dim | ×10.8 | idem; el cuello de botella pasa a ser AS 241 (≈ 9 ns) |
| 3 | **PCA: GEMM por bloques** de 64 caminos (acumulador en L1) frente al producto por camino | m = 64 | 2 461 → 808 ns/camino | ×3.1 | `M·Mᵀ = h·I` (`test_utils`, `test_qmc_cpu`) |
| | | m = 256 | 126 177 → 10 320 | ×12.2 | |
| | | m = 1024 | 5 523 225 → 156 355 | ×35.3 | |
| 4 | **Brownian Bridge**: `bb_apply` reutiliza el vector `W` (antes 1 `std::vector` por simulación) | N = 64 / N = 1024 | 107.5 → 75.5 / 1 745 → 1 607 ns/camino | ×1.42 / ×1.09 | `B·Bᵀ = h·I` |
| 5 | **Caché de BB/PCA por referencia** (antes `return it->second` copiaba `2·m²` valores por llamada) | `pca_compute` desde la caché, m = 2048 | 4.97 ms → ~0 | — | caché estable bajo concurrencia (`test_utils`) |
| | | m = 512 | 0.31 ms → ~0 | — | |
| 6 | **Matriz PCA con tabla de senos** (4m+2 `sin()` en lugar de m²) | construir la matriz, m = 512 / 2048 | 1.44 → 0.19 ms / 22.8 → 2.85 ms | ×7.5 / ×8.0 | comparada con la fórmula directa (`test_utils`) |
| 7 | **Reducción determinista en orden de índice** con ranuras de 64 B y pool propio | escalado | ver §1 | — | `determinism` (1/3/5/8/16 hilos, bit a bit) |

## 3. Probado y descartado

| Cambio | Resultado | Decisión |
|---|---|---|
| Inversa de la normal **por bloques** (rama central sin ramas, para solapar las cadenas de Horner) | ×1.08 en U(0,1); ×1.01 en el generador Sobol | **Descartado**: < 5 % en la carga objetivo (el coste está en la cola y en la rama central sin FMA) |
| `-march=native` (AVX2) para la inversa de la normal | ≈ 10 % en la normal sola | Se deja como opción (`MC_ARCH=native`); con `-ffp-contract=off` no hay FMA y el compilador no vectoriza la cadena. Cambia los bits entre binarios |
| `exp(b·log x)` en lugar de `pow(x, b)` en Dupire | 11.53 → 10.64 ns/llamada (×1.08); ≈ 6 % de la carga W3 | **Descartado**: en el umbral, cambia los bits y no elimina las dos transcendentes |
| Box–Muller / CDF inversa como generador por defecto | superados por Ziggurat (×4.7 / ×4.0 en la normal) | siguen disponibles (`NormalMethod`) y se usan como referencia en W1b/W7b |
| BB "por bloques" integrado con el generador de normales | el flujo completo (incl. normales) cuesta 2.2–3.5× el `bb_apply` aislado | no es una optimización, es el coste real del camino QMC-BB; se anota como cuello de botella (§5) |

## 4. No medido todavía (honestidad)

* **Anchura de bloque `MC_LANES`** (1/2/4/8/16) y **tamaño de chunk** (`kChunkWork`): la infraestructura está
  (`tools/run_variant_bench.sh`), pero **no se ejecutaron los barridos** a tiempo. El valor por defecto (8) viene del
  diseño (ILP sobre la cadena de dependencia del paso de Euler), no de una medición; es el primer candidato para la
  siguiente ronda.
* **`-march=native` extremo a extremo** (solo se midió sobre la normal aislada).
* **Vía O(m log m) para PCA** (base seno cerrada + FFT): se evaluó sobre el papel; con m ≤ 2048 el coste de la FFT de
  longitud 2m+1 (Bluestein) es del mismo orden que el GEMM por bloques, así que no se implementó.
* **`float` en CPU**: no se probó; la GPU ya usa `float` y la ventaja de la CPU en `double` es la reproducibilidad.
* **GPU**: ninguna de las optimizaciones de `docs/GPU_OPTIMIZATION_IDEAS.md` está aplicada ni medida (esta máquina no tiene
  CUDA); los scripts `tools/colab_validate.sh` y `tools/colab_bench.sh` están para medirlas en Colab.

## 5. Cuellos de botella actuales (para la siguiente ronda)

1. **Escalado por hilos del MC con Ziggurat** (47 % a 8 hilos en W1): medir la fracción serie (reducción, arranque de
   chunks) y probar chunks mayores; hoy el chunk depende solo del coste por camino (contrato de determinismo).
2. **QMC**: la inversa de la normal (AS 241) es ≈ 9 de los ≈ 9.5 ns por punto·dimensión; una aproximación por tramos con
   tablas o vectorización AVX2/FMA sería la palanca (aceptando romper el bit a bit entre binarios).
3. **PCA**: O(m²) por camino; a m = 1024 cuesta ≈ 156 µs/camino frente a ≈ 10 µs del resto del camino (Sobol + AS 241 a ≈ 9.5 ns × 1024).
4. **Dupire**: dos transcendentes por paso (≈ 20 ns/paso a 1 hilo frente a ≈ 3 ns de GBM).
5. **MC**: tras el Ziggurat, el generador y el paso de Euler pesan parecido; el siguiente salto sería generar y consumir las
   normales en el mismo bucle (sin el búfer transpuesto `Z[d*ld+p]`) y con SIMD explícito.
