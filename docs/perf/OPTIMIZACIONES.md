# Optimizaciones del motor CPU: mediciones y decisiones

Criterio de aceptación (fijado antes de medir): una optimización se **conserva** si gana ≥ 5 % en su carga
objetivo, no pierde > 2 % en las demás, los tests pasan y se mantiene el determinismo. Si no, se **descarta** y se deja
constancia aquí.

Se han hecho dos rondas. La primera (generador Ziggurat, Sobol por Gray-code, PCA por bloques…) está en el §4. La
segunda, que empezó por la pregunta "¿por qué escala tan poco con los hilos?", está en los §1–§3.

## Cómo se mide

* `bench/bench_cpu` — cargas de extremo a extremo W1–W8 (MC GBM europea, asiática, Dupire, Heston; QMC BB/PCA; MLMC;
  MLQMC), 2 repeticiones de calentamiento y ≥ 3 medidas; mediana, mínimo, IQR y CV %.
* `tools/ab_bench.py` — **A/B entre dos binarios, intercalando ejecuciones** (orden ABBA, varias rondas). Da la razón de
  las medianas, el rango por ronda y la razón de los mínimos. Es lo que decide si algo se conserva.
* `bench/bench_micro` — piezas aisladas (un hilo, mejor de 7) con la variante antigua y la nueva en el mismo binario,
  y el desglose de un bloque de MC (ruido / núcleo / total).
* `bench/bench_scaling` — la misma carga con hilos propios y reparto estático (techo del hardware), con el pool en una
  sola ronda y con el motor tal cual.

**Máquina**: AMD Ryzen 7 7730U (Zen 3, 8 núcleos / 16 hilos, 15 W), g++ 15.2 (MSYS2 UCRT64), `-O3`, x86-64 genérico
(sin `-march=native`), `-ffp-contract=off`. Enchufado a la corriente.

> **Fiabilidad.** Es un portátil de 15 W. Entre dos tandas separadas el mismo binario puede variar un ± 15 % (temperatura,
> estado de energía); por eso ninguna decisión sale de comparar dos tandas, sino del A/B intercalado. Aun así, léanse
> los factores con ± 5–10 %. Donde el rango por ronda es ancho se da también la razón de los mínimos.

**Ficheros de datos** (`docs/perf/`): `sesion2-antes-despues.md` (A/B de la segunda ronda), `sesion2-escalado.txt`,
`sesion2-micro.{txt,json}`, `sesion2-cargas.{txt,json}` (todas las cargas, 1–16 hilos); de la primera ronda,
`baseline-naive-cpu.*`, `opt-ziggurat-ac.*` y `micro-ac.*`. (El campo `commit` de los JSON es el de la última
configuración de CMake, no el del binario: las cargas de `sesion2-*` se midieron con el commit `0e12f30`.)

## 0. Resumen

Segundos (mediana). "Ronda 1" es el estado al abrir el PR; "ahora", tras la segunda ronda. Las dos últimas columnas son
A/B intercalado; la columna de la línea base es de otra tanda y solo orienta.

| carga | hilos | línea base | ronda 1 | ahora | ronda 1 → ahora |
|---|---|---|---|---|---|
| W1 MC GBM europea, 64 pasos | 1 | 4.35 | 0.868 | 0.352 | **×2.5** |
| | 16 | 0.58 | 0.158 | 0.059 | **×2.7** |
| W2 MC GBM asiática, 256 pasos | 1 | 4.29 | 0.820 | 0.337 | **×2.4** |
| | 16 | 0.56 | 0.163 | 0.060 | **×2.7** |
| W3 MC Dupire, 256 pasos | 1 | 4.81 | 2.494 | 0.979 | **×2.5** |
| | 16 | 0.69 | 0.463 | 0.146 | **×3.2** |
| W4 MC Heston, 128 pasos | 1 | 4.79 | 0.963 | 0.588 | **×1.6** |
| | 16 | 0.56 | 0.182 | 0.097 | **×1.9** |
| W5 QMC Brownian Bridge, 64 pasos | 1 | — | 0.398 | 0.204 | **×1.9** |
| | 16 | — | 0.061 | 0.039 | **×1.6** |
| W6 QMC PCA, 256 pasos | 1 | — | 0.216 | 0.135 | **×1.6** |
| | 16 | — | 0.066 | 0.043 | **×1.5** |
| W7 MLMC ε = 5·10⁻³ | 1 | — | 0.896 | 0.416 | **×2.2** |
| | 16 | — | 0.145 | 0.069 | **×2.1** |
| W8 MLQMC ε = 5·10⁻³ | 1 | — | 0.101 | 0.063 | **×1.6** |
| | 16 | — | 0.016 | 0.010 | **×1.5** |

Tabla completa con 8 hilos, rango por ronda y mínimos: `sesion2-antes-despues.md`.

## 1. Escalado por hilos: de dónde viene la pérdida

La tabla de la primera ronda daba un 47 % de eficiencia a 8 hilos en W1. Antes de tocar nada se midió qué parte es del
programa y qué parte es de la máquina, con cuatro variantes de la misma carga (`bench_scaling`):

* **enteros**: aritmética entera pura, sin memoria ni coma flotante, en hilos propios. Techo de frecuencia.
* **sin pool**: el código de MC en hilos propios con reparto estático. Techo de lo que puede dar el motor.
* **una ronda**: el pool del motor con todos los chunks en una sola ronda.
* **motor**: `simulate_range` tal cual.

Eficiencia (aceleración sobre 1 hilo / nº de hilos), 2²³ caminos × 64 pasos, mediana de 9 repeticiones intercaladas
(`sesion2-escalado.txt`). Con 1 hilo las tres variantes de MC tardan lo mismo (0.66 s).

| hilos | enteros | sin pool | una ronda | motor | motor (s) |
|---|---|---|---|---|---|
| 1 | 100 % | 100 % | 100 % | 100 % | 0.662 |
| 2 | 96 % | 85 % | 75 % | 74 % | 0.445 |
| 4 | 80 % | 64 % | 76 % | 73 % | 0.227 |
| 8 | 64 % | 50 % | 60 % | 58 % | 0.142 |
| 16 | 42 % | 34 % | 36 % | 35 % | 0.119 |

(Las filas de 2 y 4 hilos son las más ruidosas: por mínimos, el motor da 83 % y 76 %.)

Conclusiones:

1. **Casi toda la pérdida es del procesador, no del programa.** Un bucle de enteros puro, sin memoria compartida ni
   sincronización, se queda en el 64 % a 8 hilos y en el 42 % a 16: es un chip de 15 W, con más núcleos activos baja la
   frecuencia, y de 8 a 16 hilos solo queda el SMT. El motor va 6–7 puntos por debajo de ese techo y a 1–3 de "una
   ronda". El reparto estático ("sin pool") sale incluso peor que el dinámico a 4–8 hilos: basta un hilo lento para
   retrasar el final. El dato del 47 % de la primera tanda era además una medición mala (tanda única en un equipo que
   varía ± 15 %); medido bien, el motor ya estaba en torno al 60 %.
2. **Lo que sí era del programa eran las fronteras de ronda**: los hilos esperan al último chunk de cada ronda. Con el
   reparto fino (pensado para la curva de convergencia de la GUI) el motor iba un 6 % (8 hilos) y un 13 % (16 hilos) por
   detrás de "una ronda". Arreglado (§2, nº 4): ahora va a menos de un 2 %.
3. El reparto de tareas del pool cuesta 10–50 µs por ronda (medido con `bench_scaling --dispatch`): no es un cuello de
   botella y no se ha tocado.

Como el margen por el lado de los hilos era pequeño, el resto de la ronda se dedicó a abaratar cada camino.

## 2. Segunda ronda: optimizaciones conservadas

Cada fila es un commit con su A/B intercalado frente al commit anterior.

| # | Cambio | Carga objetivo | Ganancia medida | Otras cargas | ¿Mismos bits que antes? |
|---|---|---|---|---|---|
| 1 | **Ziggurat por bloque vectorizado** (AVX2 elegido en tiempo de ejecución) y siembra por camino más barata | W1, W2, W4, W7 | W1 ×2.2–2.4, W2 ×2.0–2.2, W4 ×1.7–2.0, W7 ×2.1–2.3 | W3 ×1.1 | No: otra secuencia, misma distribución |
| 2 | **`fast_pow`** para la volatilidad local de Dupire (tablas + polinomios, AVX2 de 4 en 4) y tabla de `exp(−αt)` | W3 | ×2.2 | W1 sin cambio | No: ≤ 4 ulp en la potencia |
| 3 | **Inversa de la normal en árbol y por bloque** (AVX2 de 4 en 4) | W5, W8 | W5 ×2.0, W8 ×1.5–1.6 | W6 ×1.17–1.34 | No: último dígito |
| 4 | **Rondas gruesas** cuando nadie escucha el progreso; una sola tarea no despierta al pool | 8 y 16 hilos | coste de las rondas +6 % / +13 % → +2 % / +2 % | — | Sí |
| 5 | **Paso de Euler factorizado**, `S·(1 + μh + σΔW)` | W1, W2 | W1 ×1.24–1.26, W2 ×1.16–1.23 | W3, W4, W7 ×1.01–1.04 | No: último bit |
| 6 | **PCA y Brownian Bridge compilados también para AVX2** | W6 | ×1.40–1.45 | W5, W8 sin cambio | Sí |

Donde pone "No", lo que cambia es el redondeo o la secuencia de números aleatorios respecto al commit anterior; el
contrato (mismo binario ⇒ mismos bits con cualquier nº de hilos) se mantiene, y además **las versiones AVX2 dan los
mismos bits que sus gemelas escalares**, así que el resultado tampoco depende del procesador. Todos los tests
estadísticos pasan sin tocar umbrales.

### 2.1 El generador de normales

El desglose de un bloque de MC (`bench_micro --filter desglose`) decía dónde estaba el tiempo: de 2.75 ns por
paso-camino, **2.35 eran generar la normal** y 0.36 el paso de Euler.

La primera idea, intercalar 8 generadores para solapar sus cadenas de dependencias, **no ganó nada** (2.36 → 2.35 ns):
el generador escalar no está limitado por la latencia sino por el número de micro-operaciones (~40 por normal). Lo que
funciona es vectorizar, y para eso hubo que cambiar la definición del generador:

* Cada normal consume **exactamente una palabra** del `xoshiro256++` del camino. En el Ziggurat clásico la cuña y la
  cola (1.5 % de los casos) siguen consumiendo el mismo generador, lo que obliga a parar el bucle vectorial, volcar el
  estado y volver; medido, eso costaba más que todo el camino rápido. Ahora esos sorteos salen de un SplitMix64
  auxiliar sembrado con *(camino, d)*, las excepciones se apuntan en una lista sin saltos y se corrigen después.
* El uniforme se forma con 52 bits como mantisa de un `double` en [1, 2) menos 1.5: tres operaciones vectoriales
  exactas, sin conversión entero → `double` (que AVX2 no tiene).

| Generador (ns por normal, D entre 64 y 1024, incluye la siembra) | escalar | AVX2 |
|---|---|---|
| Ziggurat secuencial clásico, 256 capas (ronda 1) | 2.35 | — |
| por bloque, 256 capas | 1.78–1.90 | 1.10–1.18 |
| por bloque, **512 capas** (elegido) | 1.59–1.71 | **0.97–1.04** |
| por bloque, 1024 capas | 1.55–1.66 | 1.21–1.28 |

Con más capas hay menos excepciones (1.49 % → 0.80 % → 0.43 %), pero con 1024 la tabla (16 KB) ya compite por la caché
L1 con el bloque de ruido.

Validación: momentos, Kolmogorov–Smirnov, fracciones de cola en 9 umbrales, simetría y un χ² sobre 400 intervalos
equiprobables con 1.6·10⁷ normales (`test_rng_normal`); identidad bit a bit escalar/AVX2, incluida la cola con 2²³
normales.

### 2.2 Potencia de Dupire e inversa de la normal: el mismo problema

`std::pow` (11.7 ns) y la inversa de la normal AS 241 (9.4 ns) comparten diagnóstico: sin FMA, una evaluación de
polinomio por Horner es una cadena de ~42 ciclos, y la función entera pasa de 100. Lo que limita es esa latencia. Dos
medidas: evaluar los polinomios en árbol (Estrin) y llevar 4 valores por cadena con AVX2.

| ns por valor | biblioteca / antes | escalar nuevo | AVX2 |
|---|---|---|---|
| `x^b` (rango de Dupire) | 11.7 (`std::pow`) | 7.4 | **3.2** |
| punto Sobol + inversa de la normal | 9.5 | — | **3.9** |

`fast_pow` **no es un sustituto general de `std::pow`**: su error relativo es < 4·10⁻¹⁶·(2 + |b·ln x|), hasta 4 ulp en
el rango de Dupire, frente a ~0.5 ulp de la biblioteca. Para una volatilidad local es irrelevante. A cambio, con las
tablas generadas (`tools/gen_fastmath_tables.py`) el resultado ya no depende de la libm de cada plataforma.

## 3. Segunda ronda: probado y descartado

| Cambio | Resultado | Decisión |
|---|---|---|
| Intercalar 8 generadores **escalares** (sin vectorizar) | 2.36 → 2.35 ns por normal | **Descartado** como optimización del generador; se conserva la estructura porque abarata la siembra (MLMC ×1.9) |
| Ziggurat **SSE2** (2 carriles por vector) | 2.04–2.13 ns frente a 1.84–1.99 del escalar | **Descartado**: más lento que el escalar en Zen 3 |
| Ziggurat AVX2 con el camino lento **en línea** (volcar estado, corregir, recargar) | 1.57 ns | **Descartado** por la lista de excepciones (1.0 ns) |
| Excepciones con salto en vez de sin salto | 14.8 frente a 13.3 ciclos por vuelta | **Descartado** |
| Guardar las máscaras y revisarlas después (sin `vmovmskpd` en el bucle) | 15.4 frente a 13.0 ciclos | **Descartado** |
| Índices de la tabla por memoria en vez de extraerlos del registro | 13.7 frente a 13.3 ciclos | **Descartado** |
| Salida `xoshiro256+` (sin rotación), mantisa en los bits bajos | 12.1 frente a 13.3 ciclos (+9 %) | **Descartado**: no compensa pasar a un mezclador con bits bajos débiles |
| Anchura de bloque `MC_LANES` = 2 | W1 ×0.55, W3 ×0.39 | **Descartado** |
| `MC_LANES` = 4 | W1 ×0.93, W3 ×0.61, W7 ×0.88 | **Descartado** |
| `MC_LANES` = 16 | W3 ×1.11–1.14, pero W1 ×0.95, W2 ×0.91, W4 ×0.85–0.91 (8 rondas) | **Descartado**: se queda en 8 |
| `-march=native` por defecto | W6 ×1.74 y W7 ×1.18; W1 ×1.07, resto ×1.01–1.04 | Sigue siendo opción (`MC_ARCH=native`). Lo de W6 se ha llevado al binario genérico (§2, nº 6); lo de W7 queda pendiente (§5) |
| Rehacer el reparto del pool (que los hilos tardíos no se esperen, espera activa) | el reparto cuesta 10–50 µs por ronda | **No se hizo**: no es el cuello de botella |

## 4. Primera ronda (estado al abrir el PR)

Cifras de `micro-ac.txt` (un hilo, mejor de 7, enchufado).

| # | Cambio | Carga objetivo | Antes → después | Razón | Verificación |
|---|---|---|---|---|---|
| 1 | **Generador Ziggurat** (256 capas) en lugar de Box–Muller | MC GBM europea, 1 hilo, extremo a extremo | 11.67 → 3.08 ns/paso | **×3.8** | momentos, KS y fracciones de cola en `stat` |
| 2 | **Sobol por Gray-code con `V' = L·V`** (1 XOR por punto y dimensión) frente a 32 hashes + 32 popcounts | generar palabras de 32 bits | 83.6 → 0.307 ns/punto·dim | **×273** | idéntico bit a bit a la versión ingenua; valores de scipy |
| 3 | **PCA: GEMM por bloques** de 64 caminos | m = 64 / 256 / 1024 | — | ×3.1 / ×12.2 / ×35.3 | `M·Mᵀ = h·I` |
| 4 | **Brownian Bridge**: `bb_apply` reutiliza el vector `W` | N = 64 / 1024 | — | ×1.42 / ×1.09 | `B·Bᵀ = h·I` |
| 5 | **Caché de BB/PCA por referencia** (antes se copiaban `2·m²` valores por llamada) | m = 2048 | 4.97 ms → ~0 | — | caché estable bajo concurrencia |
| 6 | **Matriz PCA con tabla de senos** | m = 512 / 2048 | — | ×7.5 / ×8.0 | comparada con la fórmula directa |
| 7 | **Reducción determinista en orden de índice** con ranuras de 64 B y pool propio | escalado | — | — | `determinism` |

Descartado en la primera ronda: la inversa de la normal "por bloques" sin cambiar la evaluación (×1.08; lo que hacía
falta era el árbol y AVX2, §2 nº 3) y `exp(b·log x)` con las funciones de la biblioteca en lugar de `pow` (×1.08).

## 5. No medido o pendiente

* **Núcleos de simulación en AVX2.** `-march=native` da todavía ×1.18 en MLMC y ×1.07 en W1 sobre el binario genérico,
  casi seguro por los bucles de los núcleos. Llevarlo al binario genérico exige compilar las plantillas de
  `kernels.hpp` dos veces sin que el enlazador mezcle las copias; no se ha hecho.
* **Camino lento del Ziggurat**: 0.15 de los ~1.0 ns por normal (18 ns por excepción, casi todo `exp`). Una prueba de
  aceptación por cotas lineales evitaría la mayoría de las llamadas.
* **Tamaño de chunk** (`kChunkWork`): no se ha barrido.
* **Compiladores y procesadores**: todo está medido con g++ 15 en un Zen 3. Con MSVC las rutinas con intrínsecos
  (generador, potencia, inversa) usan AVX2 igual, pero los clones del §2 nº 6 no. En procesadores Intel o Zen 4 el
  reparto de puertos es distinto y los descartes del §3 podrían salir de otra forma.
* **`float` en CPU** y la vía O(m log m) para PCA: sin probar.
* **GPU**: nada de `docs/GPU_OPTIMIZATION_IDEAS.md` está aplicado ni medido (esta máquina no tiene CUDA).

## 6. Cuellos de botella actuales

1. **MC**: el generador es ahora ~3/4 del coste por paso (≈ 1.0 de ≈ 1.3 ns; el núcleo de Euler, 0.13). El bucle AVX2
   tarda ~13 ciclos por vuelta de 4 normales con ~55 instrucciones; la hipótesis, sin confirmar con contadores, es que
   en Zen 3 lo limitan los desplazamientos vectoriales y las transferencias registro vectorial → entero.
2. **Dupire**: la potencia, 3.2 ns de ≈ 7 ns por paso.
3. **Heston**: la que menos ha mejorado (×1.6–1.9): dos normales por paso y una raíz cuadrada.
4. **QMC**: Sobol + inversa de la normal son 3.9 de los ≈ 6 ns por punto y dimensión de W5; en PCA, el producto O(m²).
5. **Hilos**: lo que queda es del procesador (§1).
