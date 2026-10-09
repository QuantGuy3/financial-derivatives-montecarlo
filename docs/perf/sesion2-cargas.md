Máquina: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD · 16 hilos · gcc 15.2.0 · commit a145ec5+dirty

### Comparativa (mediana en segundos; entre paréntesis, aceleración frente a la primera columna con datos)

| carga | hilos | sesion2-cargas |
|---|---|---|
| W1_gbm_euro_64 | 1 | 0.336 |
| W1_gbm_euro_64 | 2 | 0.176 |
| W1_gbm_euro_64 | 4 | 0.105 |
| W1_gbm_euro_64 | 8 | 0.066 |
| W1_gbm_euro_64 | 16 | 0.057 |
| W2_gbm_asian_256 | 1 | 0.317 |
| W2_gbm_asian_256 | 2 | 0.165 |
| W2_gbm_asian_256 | 4 | 0.101 |
| W2_gbm_asian_256 | 8 | 0.067 |
| W2_gbm_asian_256 | 16 | 0.060 |
| W3_dupire_euro_256 | 1 | 0.941 |
| W3_dupire_euro_256 | 2 | 0.484 |
| W3_dupire_euro_256 | 4 | 0.278 |
| W3_dupire_euro_256 | 8 | 0.193 |
| W3_dupire_euro_256 | 16 | 0.141 |
| W1b_gbm_euro_64_boxmuller | 1 | 3.551 |
| W1b_gbm_euro_64_boxmuller | 2 | 1.919 |
| W1b_gbm_euro_64_boxmuller | 4 | 1.094 |
| W1b_gbm_euro_64_boxmuller | 8 | 0.759 |
| W1b_gbm_euro_64_boxmuller | 16 | 0.517 |
| W4_heston_euro_128 | 1 | 0.554 |
| W4_heston_euro_128 | 2 | 0.283 |
| W4_heston_euro_128 | 4 | 0.174 |
| W4_heston_euro_128 | 8 | 0.110 |
| W4_heston_euro_128 | 16 | 0.096 |
| W5_qmc_bb_64 | 1 | 0.193 |
| W5_qmc_bb_64 | 2 | 0.097 |
| W5_qmc_bb_64 | 4 | 0.057 |
| W5_qmc_bb_64 | 8 | 0.039 |
| W5_qmc_bb_64 | 16 | 0.034 |
| W6_qmc_pca_256 | 1 | 0.132 |
| W6_qmc_pca_256 | 2 | 0.072 |
| W6_qmc_pca_256 | 4 | 0.044 |
| W6_qmc_pca_256 | 8 | 0.042 |
| W6_qmc_pca_256 | 16 | 0.044 |
| W7_mlmc_euro_5e-3 | 1 | 0.396 |
| W7_mlmc_euro_5e-3 | 2 | 0.202 |
| W7_mlmc_euro_5e-3 | 4 | 0.120 |
| W7_mlmc_euro_5e-3 | 8 | 0.077 |
| W7_mlmc_euro_5e-3 | 16 | 0.068 |
| W7b_mlmc_euro_5e-3_boxmuller | 1 | 1.223 |
| W7b_mlmc_euro_5e-3_boxmuller | 2 | 0.630 |
| W7b_mlmc_euro_5e-3_boxmuller | 4 | 0.357 |
| W7b_mlmc_euro_5e-3_boxmuller | 8 | 0.230 |
| W7b_mlmc_euro_5e-3_boxmuller | 16 | 0.178 |
| W8_mlqmc_euro_5e-3 | 1 | 0.061 |
| W8_mlqmc_euro_5e-3 | 2 | 0.034 |
| W8_mlqmc_euro_5e-3 | 4 | 0.019 |
| W8_mlqmc_euro_5e-3 | 8 | 0.013 |
| W8_mlqmc_euro_5e-3 | 16 | 0.012 |

### Escalado por hilos (sesion2-cargas)

| carga | hilos | mediana (s) | Mcaminos/s | aceleración vs 1 hilo | eficiencia |
|---|---|---|---|---|---|
| W1_gbm_euro_64 | 1 | 0.336 | 12.496 | ×1.00 | 100 % |
| W1_gbm_euro_64 | 2 | 0.176 | 23.811 | ×1.91 | 95 % |
| W1_gbm_euro_64 | 4 | 0.105 | 39.913 | ×3.19 | 80 % |
| W1_gbm_euro_64 | 8 | 0.066 | 63.437 | ×5.08 | 63 % |
| W1_gbm_euro_64 | 16 | 0.057 | 73.183 | ×5.86 | 37 % |
| W1b_gbm_euro_64_boxmuller | 1 | 3.551 | 1.181 | ×1.00 | 100 % |
| W1b_gbm_euro_64_boxmuller | 2 | 1.919 | 2.186 | ×1.85 | 93 % |
| W1b_gbm_euro_64_boxmuller | 4 | 1.094 | 3.834 | ×3.25 | 81 % |
| W1b_gbm_euro_64_boxmuller | 8 | 0.759 | 5.528 | ×4.68 | 59 % |
| W1b_gbm_euro_64_boxmuller | 16 | 0.517 | 8.116 | ×6.87 | 43 % |
| W2_gbm_asian_256 | 1 | 0.317 | 3.308 | ×1.00 | 100 % |
| W2_gbm_asian_256 | 2 | 0.165 | 6.354 | ×1.92 | 96 % |
| W2_gbm_asian_256 | 4 | 0.101 | 10.427 | ×3.15 | 79 % |
| W2_gbm_asian_256 | 8 | 0.067 | 15.537 | ×4.70 | 59 % |
| W2_gbm_asian_256 | 16 | 0.060 | 17.551 | ×5.31 | 33 % |
| W3_dupire_euro_256 | 1 | 0.941 | 0.557 | ×1.00 | 100 % |
| W3_dupire_euro_256 | 2 | 0.484 | 1.083 | ×1.95 | 97 % |
| W3_dupire_euro_256 | 4 | 0.278 | 1.887 | ×3.39 | 85 % |
| W3_dupire_euro_256 | 8 | 0.193 | 2.717 | ×4.88 | 61 % |
| W3_dupire_euro_256 | 16 | 0.141 | 3.720 | ×6.68 | 42 % |
| W4_heston_euro_128 | 1 | 0.554 | 1.892 | ×1.00 | 100 % |
| W4_heston_euro_128 | 2 | 0.283 | 3.705 | ×1.96 | 98 % |
| W4_heston_euro_128 | 4 | 0.174 | 6.013 | ×3.18 | 79 % |
| W4_heston_euro_128 | 8 | 0.110 | 9.530 | ×5.04 | 63 % |
| W4_heston_euro_128 | 16 | 0.096 | 10.951 | ×5.79 | 36 % |
| W5_qmc_bb_64 | 1 | 0.193 | 2.720 | ×1.00 | 100 % |
| W5_qmc_bb_64 | 2 | 0.097 | 5.419 | ×1.99 | 100 % |
| W5_qmc_bb_64 | 4 | 0.057 | 9.221 | ×3.39 | 85 % |
| W5_qmc_bb_64 | 8 | 0.039 | 13.444 | ×4.94 | 62 % |
| W5_qmc_bb_64 | 16 | 0.034 | 15.388 | ×5.66 | 35 % |
| W6_qmc_pca_256 | 1 | 0.132 | 0.124 | ×1.00 | 100 % |
| W6_qmc_pca_256 | 2 | 0.072 | 0.227 | ×1.83 | 91 % |
| W6_qmc_pca_256 | 4 | 0.044 | 0.374 | ×3.01 | 75 % |
| W6_qmc_pca_256 | 8 | 0.042 | 0.387 | ×3.11 | 39 % |
| W6_qmc_pca_256 | 16 | 0.044 | 0.374 | ×3.01 | 19 % |
| W7_mlmc_euro_5e-3 | 1 | 0.396 | 80.268 | ×1.00 | 100 % |
| W7_mlmc_euro_5e-3 | 2 | 0.202 | 157.123 | ×1.96 | 98 % |
| W7_mlmc_euro_5e-3 | 4 | 0.120 | 264.728 | ×3.30 | 82 % |
| W7_mlmc_euro_5e-3 | 8 | 0.077 | 413.163 | ×5.15 | 64 % |
| W7_mlmc_euro_5e-3 | 16 | 0.068 | 471.047 | ×5.87 | 37 % |
| W7b_mlmc_euro_5e-3_boxmuller | 1 | 1.223 | 25.889 | ×1.00 | 100 % |
| W7b_mlmc_euro_5e-3_boxmuller | 2 | 0.630 | 50.219 | ×1.94 | 97 % |
| W7b_mlmc_euro_5e-3_boxmuller | 4 | 0.357 | 88.780 | ×3.43 | 86 % |
| W7b_mlmc_euro_5e-3_boxmuller | 8 | 0.230 | 137.550 | ×5.31 | 66 % |
| W7b_mlmc_euro_5e-3_boxmuller | 16 | 0.178 | 178.001 | ×6.88 | 43 % |
| W8_mlqmc_euro_5e-3 | 1 | 0.061 | 14.582 | ×1.00 | 100 % |
| W8_mlqmc_euro_5e-3 | 2 | 0.034 | 26.139 | ×1.79 | 90 % |
| W8_mlqmc_euro_5e-3 | 4 | 0.019 | 46.763 | ×3.21 | 80 % |
| W8_mlqmc_euro_5e-3 | 8 | 0.013 | 68.189 | ×4.68 | 58 % |
| W8_mlqmc_euro_5e-3 | 16 | 0.012 | 77.411 | ×5.31 | 33 % |
