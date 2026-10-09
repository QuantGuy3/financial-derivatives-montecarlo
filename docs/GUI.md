# Monte Carlo Studio (GUI)

`mc_gui` es un único ejecutable: lleva dentro un **servidor HTTP que solo escucha en `127.0.0.1`** (no usa
internet, no instala nada, desaparece al cerrar la ventana) y la interfaz web embebida (ECharts incluido, sin CDN).
Al arrancar abre Edge/Chrome en **modo aplicación** (ventana sin barra de direcciones).

```bash
cmake --preset mingw-release && cmake --build --preset mingw-release
./build/mingw-release/gui/mc_gui          # abre la ventana
./build/mingw-release/gui/mc_gui --no-browser --port 8765 --token abc   # solo servidor (desarrollo)
./build/mingw-release/gui/mc_gui --web-dir gui/web                      # sirve la interfaz desde disco (sin recompilar)
```

La ventana se apaga sola unos 20 s después de cerrarla (el navegador hace *ping* cada 4 s); el botón ⏻ la cierra
al instante. Atajos: **Ctrl+Intro** simula, **Esc** cancela.

## Qué hay en pantalla

1. **Escenario** — ejemplo predefinido (los 11 del repositorio), modelo (GBM, Heston, Dupire, cesta), derivado y
   sus parámetros con deslizador y campo numérico.
2. **Método** — MC, QMC, MLMC, MLQMC; construcción del ruido (directo, Brownian Bridge, PCA); reducción de
   varianza (variable de control, importance sampling). Las combinaciones no soportadas se deshabilitan con una
   explicación. *Opciones avanzadas*: pasos, réplicas, puntos iniciales, factor de refinamiento, niveles.
3. **Precisión y ejecución** — ε (escala logarítmica), dispositivo (CPU/GPU), hilos, semilla, tiempo máximo.
4. **Convergencia** (animada en vivo): precio con banda ±2σ y referencia analítica con su zona ±ε; error estándar y
   error real en log-log con las pendientes guía N^(−1/2) y N^(−1) y el objetivo ε/√2.
5. **Trayectorias** (5 por defecto, de 1 a 200): reproducción animada con velocidad, *nueva tirada*, ruido
   pseudoaleatorio o Sobol × Raw/BB/PCA, percentiles 5–95 y 25–75, media teórica `E[S_t]`, estadístico corriente
   (media, mínimo o máximo), barrera con marcas ✕ de *knock-out*, varianza `v(t)` de Heston, **par acoplado de MLMC**
   (camino fino y grueso con el mismo ruido) y distribución de `S_T`.
6. **Resultados** — tabla acumulativa persistente con copia a Markdown/CSV y descarga.

Tema claro/oscuro, español/inglés, respeta `prefers-reduced-motion`.

## API local

Todas las rutas `/api/*` exigen el token (cabecera `X-MC-Token` o `?token=`), que viaja en el **fragmento** de la
URL (`#token=…`: no llega al servidor ni a otros sitios). Además se comprueba la cabecera `Host`
(anti DNS-rebinding). Un solo trabajo se ejecuta a la vez; los demás esperan en cola.

| Ruta | Descripción |
|---|---|
| `GET /api/capabilities` | backends, hilos, modelos y payoffs con rangos, presets |
| `POST /api/run` | `{backend, threads, seed, max_seconds, model:{type,…}, payoff:{type,…}, method:{family,noise,variance,eps,n_steps,R,n0,M,max_L}}` → `202 {job_id}`; `400 {error}` si es inválida |
| `GET /api/jobs/{id}/events` | **SSE** (`started`, `progress`, `result`, `cancelled`, `error`; latido cada 15 s; `Last-Event-ID`) |
| `GET /api/jobs/{id}/poll?after=N&wait=ms` | lo mismo por sondeo largo: `{events:[…], next, finished}` |
| `POST /api/jobs/{id}/cancel` | cancela (la CPU responde entre rondas) |
| `POST /api/paths` | `{model, payoff, n_steps, n_paths, seed, sobol, construction, fan, coupled}` → trayectorias, abanico, histograma, par acoplado |
| `POST /api/shutdown` | apaga el servidor |

Eventos `progress`: `{seq, stage, elapsed, n_done, mean, se, eps, [n_steps], [L, levels], [doubling, n_per_replica,
replica_means]}`, regulados a unos 400 puntos por ejecución (casi log-espaciados).
`result`: `{price, std_error, n_samples, time_s, ci95, n_steps, beta, E_ctrl, z_star, levels, reference:{value,kind},
abs_error, z_score, within, mpaths_per_s, truncated, …}`.

## Banco de ideas (fuera de la v1)

* **v1.1**: diagnóstico MLMC por niveles (log₂V_l, log₂|E_l|, N_l y tasas α/β); nube 2-D de Sobol frente a
  aleatorio y animación de la construcción del Brownian Bridge; barrido de precisión (carrera coste-vs-ε entre
  métodos); pestaña de benchmarks (escalado por hilos, Mpaths/s, antes/después); histograma del payoff terminal y
  Greeks (bump y pathwise con números aleatorios comunes); permalinks, comparación de ejecuciones guardadas y
  exportar informe HTML/PDF.
* **Más adelante**: superficie de volatilidad local de Dupire, scatter de la variable de control con su recta de
  regresión, histograma de pesos de importance sampling, importar resultados de una GPU remota, compilar el motor a
  WASM para una versión sin servidor.
