# Valoración de Derivados Financieros con Monte Carlo (C++/CUDA)

Librería en C++ y CUDA para valorar derivados financieros mediante simulación de Monte Carlo, con soporte para múltiples modelos estocásticos, técnicas de reducción de varianza y aceleración en GPU.

## Derivados implementados

Diez ejemplos que cubren una amplia gama de productos:

| Ejemplo | Derivado | Técnica |
|---------|----------|---------|
| 01 | Opción europea (Call/Put) | MC estándar |
| 02 | Opción asiática (media aritmética) | MC estándar |
| 03 | Opción lookback | MC estándar |
| 04 | Opción barrera | MC estándar |
| 05 | Opción sobre Heston | Modelo de volatilidad estocástica |
| 06 | Opción sobre Dupire | Volatilidad local |
| 07 | Cesta no correlacionada | MC multidimensional |
| 08 | Cesta correlacionada | Cholesky + MC |
| 09 | Asiática con variable de control | Reducción de varianza |
| 10 | Dupire con variable de control | Reducción de varianza |
| 11 | OTM con muestreo por importancia | Reducción de varianza |

## Técnicas de reducción de varianza y estimación

- **MLMC** (Multilevel Monte Carlo) con refinamiento adaptativo de niveles
- **MLQMC** (Multilevel Quasi-Monte Carlo) con secuencias de Sobol y scrambling
- **Variables de control** y **muestreo por importancia**
- **Brownian Bridge** y **PCA** como modos de transformación del ruido

## Arquitectura

- Simulación en GPU con CUDA (`methods_cuda.cu`) mediante lotes de trayectorias configurables
- Generación de informes automáticos (`gen_informe.py`)
- Barridos de precisión y comparativas entre métodos

## Tecnologías

- C++17 / CUDA
- CMake
- Python (análisis de resultados)
