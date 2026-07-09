========================================================================
PonyTail VII - Informes finales (curados)
========================================================================

Esta carpeta contiene solo los informes VÁLIDOS. Cada uno indica su alcance
(qué ejemplos y qué métodos) y con qué versión del código se generó.

Versión del código de referencia (la última):
  - CV: guarda de hilos de relleno + batching (sin crashes de memoria).
  - MLMC: batching (sin overflow de int a eps fino).
  - QMC: arreglo del deslizamiento de dimensiones Sobol + batching SIN techo
    de memoria (honra epsilons finos; el limite pasa a ser el tiempo).
  - Metrica MSE_est = |Error|^2 + StdErr^2 (sesgo^2 + varianza).

------------------------------------------------------------------------
ACTUALIZADOS (ultima version del codigo) - ejemplos 1-4
------------------------------------------------------------------------
informe_completo_sin_mc_v2.txt
    ej1-4. Metodos: MLMC, QMC (Raw/BB/PCA), MLQMC (Raw/BB/PCA). Sin MC.
    QMC ya SIN techo de memoria (converge a eps finos; QMC Raw es lento).

informe_completo_sin_mc_mlmc_v2.txt
    ej1-4. Solo QMC (Raw/BB/PCA) y MLQMC (Raw/BB/PCA). Sin MC ni MLMC.

informe_completo_sin_mc_qr.txt
    ej1-4. Sin MC ni QMC Raw. Metodos: MLMC, QMC BB, QMC PCA, MLQMC x3.
    (Confirmó que el cuello a eps fino es QMC PCA, no MLMC.)

informe_completo_mlmc_mlqmc.txt
    ej1-4. Solo MLMC y MLQMC (Raw/BB/PCA). Muestra que MLQMC logra la misma
    precision que MLMC con ~10^4-10^5 veces menos muestras y tiempo.

------------------------------------------------------------------------
BASE DE 11 EJEMPLOS (correcta con una salvedad)
------------------------------------------------------------------------
informe_completo.txt
    Los 11 ejemplos, todos los metodos. Detalle completo con MSE_est.
    SALVEDAD: se genero antes del batching, asi que las COLAS a eps muy fino
    de ej04 (barrier), ej05 (heston), ej09/10 (CV) estan cortadas por crashes
    de memoria (marcados '***' o ausentes) que despues se arreglaron. Y su
    QMC usa el techo viejo. Para eps moderado y los otros 7 ejemplos es valido.

informe_final.txt
    Misma data que informe_completo.txt pero como matriz compacta de
    convergencia (metodo x eps -> |Error|). Misma salvedad.

------------------------------------------------------------------------
NOTA
------------------------------------------------------------------------
No existe un informe de LOS 11 ejemplos con la ULTIMA version del codigo:
solo se re-corrieron los ejemplos 1-4 con el QMC sin techo. Si en el futuro
quieres los 11 al dia, hay que re-lanzar el barrido completo tras reconstruir.

Conclusion de fondo: MLQMC (cualquier variante) es el metodo mas eficiente;
los limites a eps fino son estructurales (sesgo de discretizacion Euler,
coste del matmul de PCA en alta dimension, incertidumbre de la referencia),
no limites artificiales.
