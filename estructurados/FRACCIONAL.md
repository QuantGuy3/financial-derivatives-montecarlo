# Modelos fraccionarios

El cálculo fraccionario entra en la valoración de opciones por cuatro caminos, y el programa tiene un modelo de cada uno con sus métodos numéricos:

| Dónde está lo fraccionario | Modelo | En el motor | Métodos |
|---|---|---|---|
| Núcleo de la varianza, K(t) = t^{H-½}/Γ(H+½) (volatilidad rugosa) | `rough_heston` | sí | Fourier con la ecuación de Riccati fraccionaria · Monte Carlo (varianza forward + QE) |
| | `rough_bergomi` | sí | esquema híbrido de Bennedsen-Lunde-Pakkanen (FFT) · híbrido-markoviano |
| Derivada fraccionaria en espacio (saltos α-estables) | `fmls` | sí | Fourier · EDP fraccionaria (WSGD + Crank-Nicolson) · Monte Carlo exacto (CMS) |
| Ruido fraccionario en el precio (fBm) | `fbs`, `fbs_mixto` | no | fórmula cerrada · Monte Carlo con fBm (Davies-Harte o Cholesky) |
| Derivada fraccionaria en tiempo (subdifusión) | `subdifusivo` | no | cuadratura de Kanter · EDP de Caputo (esquema L1) · Monte Carlo exacto |

"En el motor" quiere decir que el modelo se puede asignar a un subyacente de un producto estructurado (`"modelo": {"tipo": "rough_heston", ...}`), que lo usan la réplica estática y la calibración, y que se simula igual en CPU y en GPU. Los otros dos viven en la biblioteca `fraccional/` y en el comando `estructurados fraccional`.

```bash
./build/estructurados fraccional                                  # catálogo de modelos y métodos
./build/estructurados fraccional precio fmls alpha=1.5 volatilidad=25%
./build/estructurados fraccional precio rough_heston hurst=0.1 --vencimiento 3M --strike 95
./build/estructurados fraccional precio fmls alpha=1.4 --barrera 85    # down-and-out por EDP
./build/estructurados fraccional precio subdifusivo alpha=0.7 --put
./build/estructurados fraccional precio fbs_mixto hurst=0.8 --asiatica 12
./build/estructurados fraccional sonrisa rough_bergomi hurst=0.07 eta=1.9 rho=-0.9
./build/estructurados fraccional fbm --hurst 0.3 --csv fbm.csv
./build/estructurados fraccional nucleo --hurst 0.1 --factores 17
./build/estructurados calibrar mercado.json --modelo rough_heston
```

`precio` valora la misma opción por todos los métodos del modelo y muestra precio, error estándar, volatilidad implícita y tiempo de cada uno; `sonrisa` da la superficie de volatilidades implícitas y la pendiente del *skew* ATM en log-log (con volatilidad rugosa, ψ(T) ~ T^{H-½} cuando T → 0). Opciones comunes: `--spot`, `--strike`, `--vencimiento`, `--tipo`, `--dividendos`, `--put`, `--trayectorias`, `--pasos` (por año), y los parámetros del modelo como `clave=valor` (o `--param k=v,...`).

## Volatilidad rugosa

Las series de volatilidad realizada se comportan como un proceso con exponente de Hurst H ≈ 0.1 (Gatheral, Jaisson y Rosenbaum 2018; `estructurados historico` lo estima con el variograma de la log-volatilidad), y el *skew* ATM de las opciones sobre índices decae como T^{-0.4}, algo que Heston no reproduce (su *skew* tiende a una constante en plazos cortos). Los modelos rugosos sustituyen el browniano de la varianza por una integral de Volterra con el núcleo fraccionario

  K(t) = t^{α-1}/Γ(α),  α = H + ½ ∈ (½, 1]

que es singular en t = 0 si H < ½. Con H = ½ ambos modelos se reducen a los clásicos.

### Rough Heston

```json
"modelo": {"tipo": "rough_heston", "vol_inicial": "20%", "kappa": 0.3, "vol_largo_plazo": "20%",
           "xi": 0.3, "rho": -0.7, "hurst": 0.1}
```

  V_t = V0 + ∫₀ᵗ K(t-s)·κ(θ - V_s) ds + ∫₀ᵗ K(t-s)·ξ√V_s dW_s,  dS/S = (r-q)dt + √V dB,  d⟨W,B⟩ = ρ dt

(El Euch y Rosenbaum 2019). `kappa` es λ, `xi` es ν (también se aceptan `lambda` y `nu`). `factores` (17 por defecto, opcional) solo afecta a la simulación.

**Fourier.** La función característica del log-precio es φ(u) = exp(κθ·∫₀ᵀh + V0·∫₀ᵀF(h)), donde h resuelve la ecuación de Riccati fraccionaria

  D^α h = F(h) = -½(u² + iu) + (iρξu - κ)·h + ½ξ²·h²,  h(0) = 0

(derivada de Caputo de orden α). Se integra con la **regla trapezoidal fraccionaria** (el corrector de Adams-Bashforth-Moulton de Diethelm, de orden 1+α) **en forma implícita**: el término nuevo entra en la ecuación del paso, que es cuadrática en h y se resuelve de forma exacta eligiendo la raíz que tiende a la del paso explícito. El predictor-corrector explícito clásico se vuelve inestable para |u| grande y vencimientos largos (la ecuación es rígida, con escala de tiempo (ξ|u|)^{-1/α}); la versión implícita es estable para cualquier u. Con N = 128 pasos (192 a más de un año):

- con H = ½ coincide con la fórmula cerrada de Heston (|Δφ| ≈ 1e-6);
- |φ(N=128) - φ(N=1024)| ≈ 4e-6 con H = 0.1;
- la integral de Lewis (con Black-Scholes como variable de control) usa Gauss-Legendre de 16 nodos por tramo y reparte la evaluación de φ entre los núcleos: una superficie de 6 vencimientos y 12 strikes cuesta unos 30 ms con un error máximo de 0.2 pb en volatilidad frente a una referencia de alta precisión. La función característica de rough Heston decae despacio en u en los plazos cortos (colas gruesas a corto plazo), así que el corte de la integral se hace con cuidado: un corte prematuro llega a costar decenas de puntos básicos en las alas.

**Simulación (motor).** La simulación de procesos de Volterra con núcleo singular es delicada: el esquema de Euler pierde varianza en el tramo cercano y, con H = 0.1, en un paso diario la varianza cambia tanto como su nivel, de modo que truncarla en 0 introduce un sesgo grande. El motor usa:

1. **Factores markovianos.** K(t) ≈ Σ_j c_j e^{-x_j t} a partir de la representación de Bernstein K(t) = ∫ e^{-xt} x^{-α} dx / (Γ(α)Γ(1-α)): un nodo de masa en [0, x_min] y Gauss-Legendre de 2 puntos en paneles de ln x hasta x_max = 30/Δt. Con 17 factores el error relativo del núcleo es ≈ 5e-4 en [Δt, T] (`fraccional nucleo` lo muestra).
2. **Varianza forward media de cada paso.** En lugar de la varianza instantánea (que es extremadamente rugosa) se simula ξ̄_k = (1/Δ)∫ E[V_s | F_{t_k}] ds sobre el paso, una cantidad mucho más regular que los factores dan de forma exacta.
3. **Esquema QE de Andersen** para ξ̄_{k+1} dado el pasado: dos momentos exactos (media condicionada y varianza A1²ξ²ξ̄Δ, con A1 el peso del paso en la media del siguiente) y positividad garantizada, en lugar de una normal truncada.
4. **Respuesta dentro del paso**: la varianza integrada del paso responde al ruido del propio paso con el peso exacto del núcleo, I = Δ·(ξ̄ + ξ·A0·√Δ·√ξ̄·Z), A0 = Δ^{α-1}/Γ(α+2), que es lo que genera el *skew* a escalas menores que el paso.
5. **Corrección de martingala exacta**: E[exp(tt·Z)] se calcula en forma cerrada en las dos ramas del QE, así que E[S_T] = F_T sin ajustes posteriores.

Contrastado con el precio de Fourier (H = 0.1, ξ = 0.3, ρ = -0.7, 1 000 000 de trayectorias): sesgo ≈ 0.1 puntos de volatilidad a 3 meses con 252 pasos por año y ≈ 0.04 con 1 008, y dentro del error de Monte Carlo a un año. Como referencia, el esquema implícito-explícito habitual del Heston "levantado" (Abi Jaber 2019) da ≈ 0.3 puntos de sesgo a un año con 252 pasos, y un tramo cercano gaussiano truncado, ≈ 1 punto. Si el producto es muy sensible a la forma de la sonrisa de corto plazo, conviene subir `pasos_por_anio`.

### Rough Bergomi

```json
"modelo": {"tipo": "rough_bergomi", "vol_inicial": "20%", "hurst": 0.1, "eta": 1.9, "rho": -0.9}
```

  V_t = ξ0·exp(η·Y_t - ½η²t^{2H}),  Y_t = √(2H)∫₀ᵗ (t-s)^{H-½} dW_s

(Bayer, Friz y Gatheral 2016), con curva de varianza forward plana ξ0 = `vol_inicial`². No tiene fórmula de precios: todo es simulación.

- **Esquema híbrido** de Bennedsen, Lunde y Pakkanen (2017) con κ = 1 (`frac::rbergomi_hibrido`, comando `fraccional precio`): el tramo del último paso se simula de forma exacta (un par gaussiano correlacionado con ΔW) y el resto de la integral con el núcleo evaluado en los puntos óptimos b_k; la suma de convolución se hace por FFT en O(n log n).
- **Esquema híbrido-markoviano** (motor): el mismo tramo cercano exacto y el resto del núcleo con los factores markovianos, con estado O(factores) por trayectoria, lo que permite simularlo paso a paso en la GPU junto con el resto del producto.

Con 1 000 000 de trayectorias ambos esquemas dan el mismo precio en todos los strikes dentro del error estándar.

## FMLS: saltos α-estables

```json
"modelo": {"tipo": "fmls", "volatilidad": "20%", "alpha": 1.5}
```

El modelo log-estable de momentos finitos de Carr y Wu (2003): ln S_T = ln S_0 + (r - q + μ)T + L_T, con L un proceso α-estable (α ∈ (1, 2]) con asimetría máxima β = -1, es decir, saltos solo a la baja; así E[S_T] es finito. La escala estable es `volatilidad`/√2, de modo que α = 2 es exactamente Black-Scholes con esa volatilidad; cuanto menor es α, más gruesa es la cola izquierda y más pronunciado el *skew*, que además no se aplana con el plazo.

- **Fourier**: φ(u) = exp(iuμT - T·(iu·s)^α·sec(πα/2)), μ = s^α·sec(πα/2).
- **EDP fraccionaria en espacio** (`frac::fmls_edp`): en x = ln S el generador de L es κ_α·D^α con D^α la derivada de Weyl (Riemann-Liouville desde -∞) y κ_α = -s^α sec(πα/2) > 0, así que

    V_τ = (r - q - κ_α)·V_x + κ_α·D^α V - rV.

  D^α se discretiza con el **operador de Grünwald desplazado y ponderado** (WSGD, Tian, Zhou y Deng 2015), de segundo orden; en el tiempo, **Crank-Nicolson** con dos semipasos implícitos de Rannacher al principio (el pago tiene un codo). La matriz es Hessenberg inferior (densa por debajo de la diagonal: el operador es no local), con una factorización LU de coste O(n²) en la que U es bidiagonal; los dos tipos de paso comparten matriz. El pago se promedia en cada celda. Converge con orden 2 (errores 4.5e-4 → 1.1e-4 → 2.8e-5 al doblar la malla) y con 1 200 × 200 nodos queda a ≈ 1e-4 del precio de Fourier (S = 100) en medio segundo. Las puts europeas salen por paridad; las **barreras desactivantes inferiores** se tratan de forma natural poniendo el borde de la malla en la barrera (el valor es 0 por debajo, que es justo lo que ve el operador no local cuando un salto la cruza): `fraccional precio fmls --barrera 85`.
- **Monte Carlo exacto** por Chambers-Mallows-Stuck con β = -1. En el motor, el ángulo de CMS sale de la normal principal del subyacente (que es creciente en el ángulo), de modo que la correlación entre subyacentes actúa como una cópula gaussiana; la exponencial, de una normal propia.

## Movimiento browniano fraccionario

`frac::GeneradorFBM` genera trayectorias de B^H en una rejilla uniforme:

- **Davies-Harte** (inmersión circulante, Davies y Harte 1987; Wood y Chan 1994): la covarianza del ruido gaussiano fraccionario se sumerge en una matriz circulante de tamaño 2N, cuyos autovalores (no negativos para el fGn) da una FFT; cada trayectoria cuesta una FFT, O(n log n) (≈ 40 µs para 252 pasos).
- **Cholesky** de la covarianza: exacto, O(n²) por trayectoria tras una factorización O(n³); útil como referencia.

`fraccional fbm` comprueba las varianzas, la covarianza, la correlación de incrementos y estima H por el variograma de las trayectorias simuladas; `--csv` las guarda para dibujarlas.

## Black-Scholes fraccionario y mixto

  S_t = S_0·exp((r - q)t - ½(σ²t + σ_H²t^{2H}) + σW_t + σ_H B^H_t)

Con σ = 0 es el modelo de Necula (2002) / Hu y Øksendal (2003); con σ > 0, el mixto (Cheridito 2001). Las opciones europeas tienen fórmula cerrada con varianza total σ²T + σ_H²T^{2H}, y las **asiáticas geométricas** también (ln G es gaussiano con la covarianza del fBm); las **aritméticas** se valoran por Monte Carlo con trayectorias del fBm y la geométrica como variable de control (el error estándar baja de ≈ 0.03 a ≈ 0.001).

Advertencia: con un fBm puro (H ≠ ½) los precios admiten arbitraje en el sentido clásico (Rogers 1997), y la noción de autofinanciación de Wick en la que se basa la fórmula fraccionaria es discutida (Björk y Hult 2005); en el modelo mixto con H > ¾ la medida es equivalente a la de un browniano (Cheridito 2001), así que el precio sin arbitraje sería el de Black-Scholes con σ. Se incluye como modelo de referencia de la literatura y como banco de pruebas del fBm.

## Black-Scholes subdifusivo

  S_t = F_t·exp(σB(E_t) - ½σ²E_t),  E_t = inf{τ : U_α(τ) > t}

(Magdziarz 2009). E_t es la inversa de un subordinador α-estable U_α (α ∈ (0, 1]): un "tiempo operacional" que se detiene a ratos, de modo que el precio tiene periodos sin movimiento (mercados poco líquidos). Con α = 1 es Black-Scholes. E[E_T] = T^α/Γ(1+α) y E[e^{-sE_t}] = E_α(-s·t^α) (Mittag-Leffler, `frac::mittag_leffler`).

- **Cuadratura de la subordinación**: E_T = T^α·(W/A(πV))^{1-α} con V ~ U(0,1) y W ~ Exp(1) (representación de Kanter 1975 del subordinador estable), así que el precio es E[Black(σ²E_T)]: Gauss-Legendre en V (con V = 1 - s² para suavizar el extremo) y la regla trapezoidal en ln W. Unos milisegundos, con error ≈ 1e-6.
- **EDP de Caputo**: c(y, t) en y = ln(F/K) cumple ∂_t^α c = ½σ²(c_yy - c_y). La derivada de Caputo se discretiza con el **esquema L1** en una malla graduada t_n = T(n/N)^γ, γ = 2(2-α)/α (la solución cerca del dinero se comporta como t^{α/2} y la malla uniforme perdería orden), con diferencias centradas implícitas en y; la historia completa entra en cada paso (O(N²) en total), y la **extrapolación de Richardson** en t (orden 2-α) y en y (orden 2) lleva el error a ≈ 2e-5 (S = 100) con 800 × 300 nodos en unos 60 ms.
- **Monte Carlo exacto** de E_T con la misma representación de Kanter.

## Calibración

`estructurados calibrar ... --modelo rough_heston | rough_bergomi | fmls` funciona como con los demás modelos (ver [CALIBRACION.md](CALIBRACION.md)): rough Heston y FMLS por Fourier, rough Bergomi por Monte Carlo con números aleatorios comunes. `hurst` se calibra en [0.02, 0.5] y `factores` queda fijo. Con una superficie sintética de rough Heston de 72 cotizaciones, la calibración recupera los seis parámetros (H incluido) en unos 30 s; con κ y θ fijos, en unos 3 s. Es habitual que ξ y H salgan muy correlacionados (los dos controlan cómo decae el *skew* con el plazo).

## Biblioteca `fraccional/`

| Fichero | Contenido |
|---|---|
| `fraccional.hpp` | interfaz de todo el módulo (namespace `frac`) |
| `nucleo.cpp` | Mittag-Leffler E_α(-x), Gauss-Kronrod adaptativo, núcleo fraccionario y sus factores markovianos, FFT |
| `fbm.cpp` | fBm por Davies-Harte y Cholesky; Black-Scholes fraccionario y mixto, asiáticas |
| `rugosos.cpp` | Riccati fraccionaria de rough Heston; esquema híbrido BLP de rough Bergomi |
| `fmls.cpp` | FMLS: función característica, Chambers-Mallows-Stuck, EDP WSGD + Crank-Nicolson |
| `subdifusivo.cpp` | Black-Scholes subdifusivo: cuadratura de Kanter, EDP L1, Monte Carlo |

En el motor (`modelos_hd.hpp`, compartido CPU/GPU), los modelos rugosos guardan sus factores markovianos en el estado de cada trayectoria (hasta `MAX_RUGOSOS_HD` = 4 subyacentes rugosos por producto) y leen de una tabla por pasos que se prepara en el host (`preparar_rugoso` en `plan.hpp`), porque los pesos dependen de la longitud de cada paso de la rejilla.

## Referencias

- Abi Jaber, E. (2019). Lifting the Heston model. *Quantitative Finance*.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance*.
- Bayer, C., Friz, P. y Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*.
- Bennedsen, M., Lunde, A. y Pakkanen, M. (2017). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics*.
- Carr, P. y Wu, L. (2003). The finite moment log stable process and option pricing. *Journal of Finance*.
- Chambers, J., Mallows, C. y Stuck, B. (1976). A method for simulating stable random variables. *JASA*.
- Cheridito, P. (2001). Mixed fractional Brownian motion. *Bernoulli*.
- Davies, R. y Harte, D. (1987). Tests for Hurst effect. *Biometrika*.
- Diethelm, K., Ford, N. y Freed, A. (2002). A predictor-corrector approach for the numerical solution of fractional differential equations. *Nonlinear Dynamics*.
- El Euch, O. y Rosenbaum, M. (2019). The characteristic function of rough Heston models. *Mathematical Finance*.
- Gatheral, J., Jaisson, T. y Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*.
- Kanter, M. (1975). Stable densities under change of scale and total variation inequalities. *Annals of Probability*.
- Magdziarz, M. (2009). Black-Scholes formula in subdiffusive regime. *Journal of Statistical Physics*.
- Necula, C. (2002). Option pricing in a fractional Brownian motion environment.
- Stynes, M., O'Riordan, E. y Gracia, J. L. (2017). Error analysis of a finite difference method on graded meshes for a time-fractional diffusion equation. *SIAM J. Numer. Anal.*
- Tian, W., Zhou, H. y Deng, W. (2015). A class of second order difference approximations for solving space fractional diffusion equations. *Mathematics of Computation*.
