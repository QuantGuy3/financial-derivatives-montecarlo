# Calibración de modelos

Dos formas de obtener los parámetros del modelo de un subyacente:

- **A precios de opciones** (`estructurados calibrar`): los parámetros que mejor reproducen la superficie de volatilidades implícitas del mercado. Es la calibración que corresponde a valorar (medida riesgo neutro).
- **Con la serie histórica de precios** (`estructurados historico`): volatilidad, saltos y rugosidad estimados con los rendimientos pasados (medida real).

En los dos casos el modelo calibrado puede escribirse directamente en el subyacente de un producto (`--producto`), listo para `estructurados valorar`.

```bash
./build/estructurados calibrar productos/mercado_ejemplo_sx5e.json --modelo heston
./build/estructurados calibrar productos/mercado_ejemplo_sx5e.json --modelo bates --optimizador de
./build/estructurados calibrar productos/cadena_ejemplo_spx.csv --modelo sabr --volumen-min 10
./build/estructurados calibrar productos/mercado_ejemplo_sx5e.json --modelo svi          # superficie SVI
./build/estructurados calibrar mercado.json --modelo heston --producto productos/phoenix_worst_of.json --subyacente SX5E
./build/estructurados historico precios.csv --modelo auto --producto mi_producto.json
```

## Datos de mercado

`calibrar` lee un fichero **JSON de mercado** o una **cadena de opciones en CSV** (el mismo formato que `replicar --cadena`; ver [FORMATO.md](FORMATO.md#réplica-estática-con-opciones), y `herramientas/descargar_cadena.py` para bajarla de Yahoo Finance).

El JSON admite una superficie de volatilidades (lo más cómodo para escribir datos a mano), una lista de cotizaciones o una referencia a una cadena CSV:

```json
{
  "subyacente": "SX5E", "fecha": "2026-10-23", "spot": 4800,
  "tipo_interes": "3%", "dividendos": "2.5%",
  "superficie": {
    "vencimientos": ["1M", "3M", "6M", "1A", "2A"],
    "strikes": ["80%", "90%", "100%", "110%", "120%"],
    "volatilidades": [["28.1%", "23.9%", "17.6%", "14.6%", "17.2%"],
                      ["26.7%", "22.6%", "18.0%", "14.3%", "14.6%"],
                      ...]
  },
  "cotizaciones": [{"vencimiento": "6M", "strike": 5200, "tipo": "call", "bid": 61.0, "ask": 63.2},
                   {"vencimiento": "1A", "strike": "90%", "vol": "21.3%", "peso": 2}],
  "cadena": "opciones_sx5e.csv"
}
```

- Los strikes con `%` son relativos al spot; un número es un strike absoluto. Un `null` en la superficie es un hueco.
- Los vencimientos son plazos (`"3M"`, `"1A"`, años) o fechas (con `"fecha"` en el fichero).
- Cada cotización puede traer volatilidad, precio, o bid y ask (se usa el medio y, con la horquilla, las volatilidades bid y ask).

**Forward y descuento.** Para cada vencimiento, si hay calls y puts del mismo strike cerca del spot, el forward F y el descuento DF salen de la **paridad put-call** (regresión de C − P = DF·(F − K) en los strikes): así no hace falta conocer los dividendos ni el tipo exacto. Si no, se usan `tipo_interes` y `dividendos`. `--forward dado` obliga a usar los tipos dados y `--forward paridad` avisa si algún vencimiento no la admite.

**Filtros.** De cada strike se usa la opción **fuera del dinero** (put si K < F, call si K ≥ F), la más líquida y la que mejor fija la volatilidad (`--con-dentro-del-dinero` admite también las otras). `--vencimientos 1M:3A`, `--strikes 70%:130%` (K/F) y `--volumen-min 10` acotan los datos.

## El problema de optimización

Se minimiza la suma de cuadrados de los errores en **volatilidad implícita**

  Σᵢ wᵢ · (σ_modelo(Tᵢ, Kᵢ; θ) − σ_mercado,ᵢ)²

que pesa por igual opciones de precios muy distintos (una put muy fuera del dinero vale céntimos, pero su volatilidad informa del *skew*). Pesos (`--pesos`): `iguales`, `vega` (más peso cerca del dinero) u `horquilla` (inverso del cuadrado de la horquilla en volatilidad: las opciones con precio fiable mandan); la columna `peso` del fichero multiplica.

Cada parámetro se busca en un rango razonable (p. ej. κ ∈ [0.01, 20], ξ ∈ [0.01, 5], ρ ∈ (−1, 1), volatilidades entre 0.5% y 200%) mediante una transformación logística, de modo que los optimizadores trabajan sin restricciones. Las restricciones entre parámetros (la de martingala de NIG) se imponen descartando los puntos inválidos.

| Optimizador | Cuándo |
|---|---|
| `lm` (por defecto) | Levenberg-Marquardt con jacobiano por diferencias, escalado de Marquardt y la regla de Nielsen para la amortiguación, desde `--arranques` (4) puntos de partida: el de los datos y perturbaciones reproducibles. Rápido; los arranques evitan mínimos locales (en Heston son frecuentes). |
| `de` | Evolución diferencial (DE/rand/1/bin, semilla fija) en todo el rango y refinado final con LM: búsqueda global, más lenta. |
| `nm` | Nelder-Mead (parámetros adaptativos de Gao y Han) y refinado con LM: sin derivadas, útil con objetivos ruidosos. |

`--fijar k=v` deja parámetros fijos (p. ej. `--fijar kappa=2`), `--inicial k=v` cambia el punto de partida y `--liberar beta` calibra la β de SABR, que por defecto se mantiene fija (con datos de un solo día β y ρ apenas se distinguen). Con `--producto`, el modelo actual del subyacente es el punto de partida si es del mismo tipo.

## Precios del modelo

Cada evaluación valora toda la superficie con el mismo servicio de precios que la réplica estática (`vanilla.hpp`), en forma forward (con el F y el DF de cada vencimiento):

| Modelo | Método |
|---|---|
| `black_scholes` | fórmula de Black |
| `merton` | serie de Poisson de precios de Black |
| `heston`, `bates`, `nig` | Fourier (fórmula de Lewis) con la función característica del modelo evaluada **una sola vez por vencimiento** y reutilizada en todos los strikes, con Black-Scholes como variable de control (precisión en las alas) |
| `sabr` | aproximación de Hagan et al. (2002) |
| `vol_local` y los modelos sin fórmula | Monte Carlo con el paso del motor de estructurados, antitéticas, ajuste del forward y **números aleatorios comunes** (la misma semilla en cada evaluación): el objetivo es suave en los parámetros y LM funciona con pasos de jacobiano mayores. `--trayectorias` fija la precisión |

Con la superficie de ejemplo, una calibración de Heston tarda unos segundos; la de volatilidad local por Monte Carlo, del orden de diez.

## Resultado

- **Parámetros** y su **error estándar** aproximado, a partir de JᵀJ en la solución (Cov = s²·(JᵀJ)⁻¹ en el espacio de búsqueda, trasladado a cada parámetro). Un error grande indica que los datos apenas fijan ese parámetro.
- **Avisos**: parámetros en el borde de su rango, pares con correlación > 0.95 (los datos fijan una combinación, p. ej. κ y θ en Heston), condición de Feller (2κθ ≥ ξ²; si no se cumple la varianza puede tocar 0, cosa que el motor admite) y el ruido de muestreo en los modelos por Monte Carlo.
- **Ajuste** por vencimiento y strike: volatilidad de mercado, del modelo, diferencia en puntos básicos y, si hay horquilla, la fracción de volatilidades del modelo que caen dentro de [bid, ask].
- `--salida calibracion.json` guarda parámetros, errores y notas; `--csv ajuste.csv` guarda cada cotización con su precio y volatilidad de mercado y del modelo (para dibujar la sonrisa).
- `--producto p.json --subyacente SX5E` escribe el modelo calibrado en ese subyacente y guarda `p_calibrado.json` (o el fichero de `--guardar`). El spot, el nivel inicial y los dividendos del producto no se tocan (se avisa si el spot difiere del de los datos).

## SVI

`--modelo svi` ajusta en cada vencimiento la parametrización SVI de Gatheral de la varianza total implícita w(k) = σ²T:

  w(k) = a + b·(ρ(k − m) + √((k − m)² + σ²)),  k = ln(K/F)

con varios puntos de partida por vencimiento, y comprueba la ausencia de arbitraje:

- **mariposa**: la función g(k) de Gatheral y Jacquier (2014), proporcional a la densidad implícita, debe ser ≥ 0;
- **calendario**: la varianza total no puede bajar con el plazo para un mismo k.

SVI no es un modelo dinámico (no se puede escribir en un producto): sirve para suavizar e interpolar la superficie, detectar cotizaciones inconsistentes y medir lo bien que podría ajustar cualquier modelo.

## Estimación histórica

`estructurados historico precios.csv` lee una serie de precios (columnas de fecha y de cierre; reconoce `Date`/`fecha` y `Adj Close`/`Close`/`cierre`/`precio`, como las descargas de Yahoo Finance) y estima:

- **Black-Scholes**: volatilidad y tendencia anuales con su error estándar; asimetría y exceso de curtosis de los rendimientos.
- **Merton** por máxima verosimilitud (cada rendimiento es una mezcla de Poisson de normales), con errores estándar por la hessiana numérica y el **contraste de razón de verosimilitudes** frente a Black-Scholes (χ² con 3 grados de libertad). `--modelo auto` propone Merton solo si los saltos son significativos al 1%.
- **Exponentes de Hurst** por el variograma (E|X(t+Δ) − X(t)|² ∝ Δ^{2H}): del log-precio (0.5 para un browniano) y de la log-volatilidad realizada por bloques (la volatilidad rugosa da H ≈ 0.1; ver los modelos fraccionarios). El error se estima con la dispersión entre subperíodos.

La volatilidad histórica no es la implícita: la diferencia es la prima de riesgo de volatilidad. Para valorar, la tendencia estimada no se usa (bajo la medida riesgo neutro la deriva es r − q).
