# Formato de los productos estructurados (JSON)

Cada producto se guarda en **un fichero JSON** que se lee como una ficha de producto (*term sheet*): datos generales, mercado, subyacentes, calendario, cupones, autocancelación, barreras y amortización final. El programa `estructurados` lo valida, lo muestra (`mostrar`) y lo valora (`valorar`). Los ficheros de `productos/` son ejemplos completos de cada familia.

La forma más cómoda de crear uno es el diseñador visual (`herramientas/disenador_productos.html`, se abre en el navegador), el asistente de terminal (`estructurados nuevo`) o una plantilla (`estructurados plantilla phoenix mi_producto.json`) que luego se edita con cualquier editor de texto. Este documento describe todas las claves por si se quiere escribir o retocar a mano.

## Convenciones

| Qué | Cómo se escribe | Ejemplos |
|---|---|---|
| Porcentajes (barreras, cupones, volatilidades, tipos, dividendos, correlaciones) | texto con `%` **o** número en tanto por uno | `"60%"` = `0.6`; `"2.5%"` = `0.025`; `"25pb"` = `0.0025` |
| Plazos | número de años **o** texto con unidades `A`/`Y` (años), `M` (meses), `S`/`W` (semanas), `D` (días) | `3`, `"3A"`, `"18M"`, `"2A6M"`, `"26S"` |
| Fechas | `"AAAA-MM-DD"` o `"DD/MM/AAAA"`; exigen `fecha_valoracion` y se convierten a años con ACT/365 | `"2027-04-26"`, `"26/04/2027"` |
| Sí / no | `true` / `false` (también `"si"` / `"no"`) | `"memoria": true` |

- **P** es el *rendimiento de la cesta* en una fecha: cada subyacente aporta `P_i = S_i / nivel_inicial_i` y la cesta los combina (peor, mejor, media o media ponderada). Todas las barreras, *strikes* y rangos se expresan en ese mismo nivel relativo (100% = nivel inicial).
- Los importes (cupones, primas, componentes de la amortización) están en **% del nominal**.
- El lector admite comentarios `// ...` y `/* ... */` y comas finales, para que sea fácil anotar el fichero a mano (al guardar se escribe JSON estándar).
- Las claves que empiezan por `_` se ignoran sin aviso (el diseñador visual guarda ahí el estado de su formulario, en `_disenador`).
- Las claves desconocidas no son un error, pero se avisa de ellas sugiriendo la clave más parecida (`"obsrevacion"` → «¿quería decir "observacion"?»). Los errores indican la clave y la línea/columna del fichero.

## Esquema general

```json
{
  "formato": "producto-estructurado/1",
  "nombre": "...", "descripcion": "...",
  "nominal": 1000, "moneda": "EUR",
  "fecha_valoracion": "2026-10-23",          // solo si se usan fechas
  "vencimiento": "3A",
  "mercado": {"tipo_interes": "3%"},
  "subyacentes": [ { "nombre": "SX5E", "spot": 4800, "dividendos": "3%", "modelo": {...} } ],
  "correlacion": 0.6,
  "cesta": "peor",
  "observaciones": {"frecuencia": "trimestral"},
  "cupones": {...},
  "autocancelacion": {...},
  "barrera_capital": {...},
  "barrera_desactivante": {...},
  "amortizacion_final": [ ... ],
  "estado": {...},
  "simulacion": {...}
}
```

Solo son obligatorios `vencimiento` (o una lista de fechas de observación), el tipo de interés, los subyacentes con su modelo, la correlación si hay más de un subyacente y la `amortizacion_final`.

## Datos generales

| Clave | Por defecto | Descripción |
|---|---|---|
| `nombre`, `descripcion` | — | Textos libres |
| `nominal` | 100 | Importe de referencia; los precios se dan en moneda y en % del nominal |
| `moneda` | `"EUR"` | Solo informativa |
| `fecha_valoracion` | — | Fecha de hoy; necesaria si alguna fecha se escribe como `AAAA-MM-DD` |
| `vencimiento` | — | Plazo o fecha de vencimiento. Debe coincidir con la última fecha de observación |
| `formato`, `plantilla` | — | Informativas (versión del formato y plantilla de origen) |

## Mercado

```json
"mercado": {"tipo_interes": "3%"}
"mercado": {"curva": [["6M", "2.2%"], ["1A", "2.3%"], ["5A", "2.9%"]]}
```

Tipos cero en capitalización continua. Con una curva se interpola linealmente en el tipo cero y se extrapola plano; el descuento de cada pago y la deriva de los subyacentes (`r(t) - q`) salen de ella.

## Subyacentes

| Clave | Por defecto | Descripción |
|---|---|---|
| `nombre` | `S1`, `S2`... | Identificador (se usa en la correlación por parejas) |
| `spot` | 100 | Precio de hoy |
| `nivel_inicial` | = spot | Fijación de referencia de los rendimientos. Para un producto ya emitido, el nivel del día de emisión |
| `dividendos` | 0 | Rentabilidad por dividendo continua (en divisas: el tipo de interés extranjero) |
| `modelo` | — | `{"tipo": ..., parámetros}`; ver la tabla de modelos |

Se admiten hasta 16 subyacentes. Si el nivel inicial es igual al spot, el precio no depende del valor absoluto del spot.

## Modelos

Cada subyacente tiene su propio modelo (se pueden mezclar). Todos se simulan bajo la medida riesgo neutro en logaritmo del precio. `estructurados modelos` muestra esta misma lista.

| `tipo` | Parámetros | Dinámica | Paso |
|---|---|---|---|
| `black_scholes` (`bs`, `gbm`) | `volatilidad` | dS/S = (r-q)dt + σ dW | exacto |
| `heston` | `vol_inicial` (√v0), `kappa`, `vol_largo_plazo` (√θ), `xi`, `rho` | dv = κ(θ-v)dt + ξ√v dW₂ | Euler con truncamiento completo |
| `merton` | `volatilidad`, `lambda`, `media_salto`, `vol_salto` | BS + saltos lognormales de Poisson | exacto |
| `bates` | los de Heston + los de saltos | Heston + saltos de Merton | truncamiento completo |
| `vol_local` (`dupire`, `cev`) | `sigma0`, `alpha`, `beta`, `vol_max` | σ(S,t) = σ0·e^(-αt)·(S/S0)^(β-1) | log-Euler |
| `sabr` | `alpha`, `beta`, `nu`, `rho`, `vol_max` | dF = α F^β dW₁, dα = να dW₂ (sobre el forward) | log-Euler (α exacto) |
| `nig` | `sigma`, `theta`, `kappa` | browniano con deriva θ y vol σ subordinado por un IG de varianza κ | exacto |

Notas:

- En `heston` y `bates` también se aceptan `v0` y `theta` como varianzas (se convierten a volatilidades). Si no se cumple la condición de Feller (2κθ ≥ ξ², lo habitual en calibraciones reales) la varianza puede tocar 0: el esquema de truncamiento completo lo admite.
- `media_salto` es la media del logaritmo del salto (-0.10 ≈ caída media del 9.5%) y `vol_salto` su desviación típica. La deriva incluye el compensador λ·(e^(μ_J+δ²/2) - 1).
- En `vol_local`, `beta = 1` y `alpha = 0` es Black-Scholes; `beta < 1` da *skew* negativo (tipo CEV). En `sabr`, `alpha` es la volatilidad en el forward inicial (la α de Hagan es `alpha·F0^(1-beta)`). `vol_max` (300% por defecto) acota la volatilidad instantánea cuando el precio cae mucho.
- En `nig` hace falta 1 - σ²κ - 2θκ > 0 (si no, el precio descontado no sería una martingala); `theta < 0` da asimetría bajista y `kappa` controla el grosor de las colas.
- La correlación entre subyacentes se aplica a sus brownianos principales; la correlación precio-volatilidad (`rho`) es la de cada modelo; los saltos son independientes entre subyacentes.

## Correlación y cesta

```json
"correlacion": 0.6                                     // la misma para todas las parejas
"correlacion": [[1, 0.7, 0.5], [0.7, 1, 0.5], [0.5, 0.5, 1]]
"correlacion": {"comun": "55%", "SX5E/SPX": "70%"}     // común + excepciones por parejas
```

La matriz debe ser simétrica, con unos en la diagonal y semidefinida positiva (si no, se indica qué pivote de Cholesky falla).

```json
"cesta": "peor"          // worst-of: P = min P_i (por defecto)
"cesta": "mejor"         // best-of:  P = max P_i
"cesta": "media"         // media equiponderada
"cesta": {"tipo": "ponderada", "pesos": [0.5, 0.3, 0.2]}
```

## Calendario de observación

```json
"observaciones": {"frecuencia": "trimestral"}     // mensual, bimestral, trimestral, cuatrimestral, semestral, anual, "3M"...
"observaciones": {"fechas": ["6M", "1A", "18M", "2A"]}
"observaciones": {"fechas": ["2027-01-25", "2027-04-26"]}   // con fecha_valoracion
```

Con una frecuencia, las fechas se cuentan hacia atrás desde el vencimiento (si el plazo no es múltiplo del periodo, el primer periodo es más corto). Sin `observaciones`, la única fecha es el vencimiento. Las fechas deben ser crecientes y la última debe ser el vencimiento.

## Cupones

Un objeto o una lista de hasta 4 patas que se suman:

| Clave | Descripción |
|---|---|
| `tipo` | `"fijo"` (se paga siempre), `"condicional"` (se paga si P ≥ barrera; *Phoenix*), `"rango"` (*range accrual*) |
| `importe` | Cupón por periodo (escalar o lista con un valor por fecha) |
| `tasa_anual` | Alternativa a `importe`: tasa anual × duración del periodo |
| `barrera` | Solo condicional: nivel de P (escalar o lista por fecha) |
| `memoria` | Solo condicional: si un cupón no se paga, se acumula y se paga cuando P vuelva a superar la barrera |
| `rango` | Solo rango: `[inferior, superior]`; el cupón del periodo se multiplica por la fracción de días con P dentro del rango |
| `desde`, `hasta` | Primera y última fecha (número de observación, desde 1) en que se observa el cupón |

```json
"cupones": {"tipo": "condicional", "importe": "2%", "barrera": "70%", "memoria": true}
"cupones": [{"tipo": "fijo", "tasa_anual": "1%"}, {"tipo": "condicional", "importe": "1.5%", "barrera": "80%"}]
"cupones": {"tipo": "rango", "tasa_anual": "6%", "rango": ["90%", "110%"]}
```

## Autocancelación

| Clave | Por defecto | Descripción |
|---|---|---|
| `barrera` | — | Nivel de P que provoca la amortización anticipada (escalar o lista por fecha) |
| `escalon` | 0 | Variación de la barrera en cada fecha (*step-down*: `"-5%"`) a partir de `desde` |
| `desde`, `hasta` | 1, última | Fechas en que se puede autocancelar |
| `en_vencimiento` | `true` | Si la fecha final también es fecha de autocancelación |
| `importe` | 100% | Amortización al autocancelar |
| `prima` | 0 | Prima añadida al importe (escalar o lista por fecha) |
| `acumulativa` | `false` | Si es `true`, la prima se multiplica por el número de la observación (*Athena/Express*: 1·p, 2·p, 3·p...) |
| `prima_anual` | — | Alternativa: prima anual × plazo de la fecha |

Si en la fecha final P supera la barrera de autocancelación, el producto se amortiza con ese importe (más la prima) y no se aplica la amortización final.

## Barreras

```json
"barrera_capital": {"nivel": "60%", "observacion": "vencimiento"}
"barrera_desactivante": {"nivel": "130%", "observacion": "continua", "importe": "105%", "pago": "vencimiento"}
```

- **Barrera de capital** (*knock-in*, hacia abajo): se considera tocada si P ≤ nivel. Su único efecto es activar los componentes de la amortización final con `"condicion": "barrera_tocada"` (o desactivar los de `"barrera_no_tocada"`).
- **Barrera desactivante** (*knock-out*, hacia arriba): si P ≥ nivel, el producto termina y paga `importe` (`"pago": "vencimiento"` o `"inmediato"`).

Formas de observación (`observacion`):

| Valor | Significado |
|---|---|
| `vencimiento` | solo en la fecha final (barrera europea; no vale para la desactivante) |
| `fechas` | en cada fecha de observación |
| `diaria` | en cada paso de la simulación (≥ 252 por año): cierres diarios |
| `continua` | en todo momento (americana); se aproxima con pasos diarios y la corrección de continuidad de Broadie-Glasserman-Kou (`"correccion_bgk": false` la desactiva) |

## Amortización final

Lo que se cobra a vencimiento si el producto sigue vivo, como suma de componentes en función de P (el rendimiento final de la cesta, o la media de las observaciones con `"rendimiento": "medio"`):

```json
"amortizacion_final": [ {...}, {...} ]
"amortizacion_final": {"rendimiento": "medio", "componentes": [ {...}, {...} ]}
```

| `tipo` | Pago (en % del nominal) | Claves |
|---|---|---|
| `fijo` | importe | `importe` (100%) |
| `lineal` | part · P (acotado a [suelo, techo]) | `participacion`, `suelo`, `techo` |
| `call` | part · max(min(P, techo) - strike, 0) | `strike`, `participacion`, `techo` |
| `put` | part · max(strike - max(P, suelo), 0) | `strike`, `participacion`, `suelo` |
| `put_vendida` | -part · max(strike - P, 0) / strike | `strike`, `participacion` (la pérdida de una *reverse convertible*: con barrera tocada se recibe P/strike) |
| `digital` | importe si P ≥ strike | `strike`, `importe` |
| `digital_put` | importe si P < strike | `strike`, `importe` |
| `cliquet` | part · clamp(Σⱼ clamp(Pⱼ/Pⱼ₋₁ - 1, suelo_local, techo_local), suelo_global, techo_global) | `suelo_local`, `techo_local`, `suelo_global` (0), `techo_global`, `participacion` |

Cada componente admite `"condicion"`: `"siempre"` (por defecto), `"barrera_tocada"` o `"barrera_no_tocada"`. Ejemplos:

```json
// Reverse convertible / Phoenix: devuelve el nominal salvo que se toque la barrera
[{"tipo": "fijo", "importe": "100%"}, {"tipo": "put_vendida", "strike": "100%", "condicion": "barrera_tocada"}]
// Capital protegido con participación del 80% y tope del 150%
[{"tipo": "fijo", "importe": "100%"}, {"tipo": "call", "strike": "100%", "participacion": "80%", "techo": "150%"}]
// Bonus 120% con barrera
[{"tipo": "fijo", "importe": "120%", "condicion": "barrera_no_tocada"},
 {"tipo": "call", "strike": "120%", "condicion": "barrera_no_tocada"},
 {"tipo": "lineal", "condicion": "barrera_tocada"}]
```

`estructurados mostrar` imprime el **perfil de pago** a vencimiento para P entre 0% y 150% (con y sin barrera tocada), que es la forma más rápida de comprobar que los componentes dicen lo que se quería.

## Productos ya vivos

```json
"estado": {"barrera_capital_tocada": false, "cupones_en_memoria": "2.25%"}
```

Para valorar un producto emitido en el pasado: use `fecha_valoracion` y fechas reales (las fechas ya pasadas no se incluyen), el `nivel_inicial` de cada subyacente del día de emisión, y el estado: si la barrera de capital ya se tocó y los cupones impagados acumulados en memoria (un valor, o uno por pata de cupón).

## Simulación

| Clave | Por defecto | Descripción |
|---|---|---|
| `metodo` | `auto` | `mc` (Monte Carlo, Philox), `qmc` (Sobol con *scrambling* de Hong-Hickernell, réplicas independientes) o `auto` (QMC si la dimensión del problema es ≤ 1024 y el motor lo admite) |
| `dispositivo` | `auto` | `gpu`, `cpu` o `auto` (GPU si hay) |
| `error_objetivo` | 0.05% | Error estándar buscado, en % del nominal (`error_objetivo_importe` para darlo en moneda) |
| `pasos_por_anio` | automático | Pasos de la rejilla temporal. Automático: 252 si hay vigilancia diaria/continua, *range accrual* o algún modelo con error de discretización; solo las fechas de observación si todos los modelos son exactos |
| `semilla` | 2024 | Semilla (misma semilla = mismas trayectorias en CPU y GPU) |
| `trayectorias` | adaptativo | Número fijo de trayectorias (sin adaptación) |
| `trayectorias_max` | 2·10⁸ | Tope de la adaptación |
| `replicas_qmc` | 16 | Réplicas de QMC (el error se estima con la dispersión entre réplicas) |
| `precision_gpu` | `simple` | `simple` (float, más rápido) o `doble` |
| `hilos` | todos | Hilos del motor CPU |

Las opciones de la línea de comandos (`--metodo`, `--error`, `--pasos`, `--gpu`/`--cpu`, `--doble`...) prevalecen sobre este bloque.

## Orden de los eventos en cada fecha de observación

En cada paso de la simulación se avanzan todos los subyacentes; si hay vigilancia diaria o continua se comprueban las barreras y se cuentan los días dentro del rango. En cada fecha de observación, en este orden:

1. Barrera de capital (si se observa en fechas o a vencimiento) y barrera desactivante (si se toca, el producto termina aquí).
2. Cupones: fijos, condicionales (pagando la memoria si P ≥ barrera) y *range accrual*.
3. Acumuladores del rendimiento medio y del cliquet.
4. Autocancelación; si no la hay y es la última fecha, amortización final.

Todos los pagos se descuentan desde su fecha con la curva de tipos.

## Resultados de la valoración

- **Precio** en moneda y en % del nominal, error estándar e intervalo de confianza del 95%.
- **Descomposición** del valor: cupones, amortización anticipada (autocancelación o desactivación) y amortización a vencimiento.
- **Escenarios**: probabilidad de autocancelación (total y por fecha), vida media esperada, probabilidad de tocar la barrera de capital, de desactivación y de amortizar por debajo del 100% (con la pérdida media en ese caso), y probabilidad de cobrar cupón en cada fecha.
- **Griegas** (`--griegas`): delta y gamma por subyacente (spot ±1%), vega por subyacente (+1 punto de volatilidad en los parámetros de nivel de vol del modelo) y rho (+1% en toda la curva), por diferencias finitas con las mismas trayectorias (números aleatorios comunes). Al mover el spot se mantienen fijos el nivel inicial y la superficie de volatilidad local/SABR.
- `--salida resultado.json` guarda todo en JSON.
