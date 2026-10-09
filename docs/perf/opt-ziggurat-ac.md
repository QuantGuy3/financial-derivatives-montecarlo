Máquina: AMD Ryzen 7 7730U (8C/16T) · 16 hilos · gcc 15.2.0 · commit 0dc165d+dirty

### Comparativa (mediana en segundos; entre paréntesis, aceleración frente a la primera columna con datos)

| carga | hilos | baseline-naive-cpu | opt-ziggurat-ac |
|---|---|---|---|
| W1_gbm_euro_64 | 1 | 4.348 | 0.854 (×5.09) |
| W1_gbm_euro_64 | 2 | 2.789 | 0.689 (×4.05) |
| W1_gbm_euro_64 | 4 | 1.479 | 0.368 (×4.02) |
| W1_gbm_euro_64 | 8 | 0.801 | 0.226 (×3.55) |
| W1_gbm_euro_64 | 16 | 0.579 | 0.156 (×3.70) |
| W2_gbm_asian_256 | 1 | 4.288 | 0.833 (×5.14) |
| W2_gbm_asian_256 | 2 | 2.744 | 0.425 (×6.45) |
| W2_gbm_asian_256 | 4 | 1.471 | 0.259 (×5.68) |
| W2_gbm_asian_256 | 8 | 0.797 | 0.186 (×4.28) |
| W2_gbm_asian_256 | 16 | 0.560 | 0.165 (×3.38) |
| W3_dupire_euro_256 | 1 | 4.810 | 2.659 (×1.81) |
| W3_dupire_euro_256 | 2 | 3.086 | 1.870 (×1.65) |
| W3_dupire_euro_256 | 4 | 1.740 | 1.010 (×1.72) |
| W3_dupire_euro_256 | 8 | 0.964 | 0.653 (×1.48) |
| W3_dupire_euro_256 | 16 | 0.687 | 0.477 (×1.44) |
| W4_heston_euro_128 | 1 | 4.794 | 0.978 (×4.90) |
| W4_heston_euro_128 | 2 | 2.945 | 0.543 (×5.42) |
| W4_heston_euro_128 | 4 | 1.526 | 0.308 (×4.95) |
| W4_heston_euro_128 | 8 | 0.817 | 0.201 (×4.07) |
| W4_heston_euro_128 | 16 | 0.563 | 0.196 (×2.87) |
| W1b_gbm_euro_64_boxmuller | 1 | — | 3.628 |
| W1b_gbm_euro_64_boxmuller | 2 | — | 2.406 |
| W1b_gbm_euro_64_boxmuller | 4 | — | 1.295 |
| W1b_gbm_euro_64_boxmuller | 8 | — | 0.711 |
| W1b_gbm_euro_64_boxmuller | 16 | — | 0.600 |
| W5_qmc_bb_64 | 1 | — | 0.442 |
| W5_qmc_bb_64 | 2 | — | 0.295 |
| W5_qmc_bb_64 | 4 | — | 0.159 |
| W5_qmc_bb_64 | 8 | — | 0.098 |
| W5_qmc_bb_64 | 16 | — | 0.066 |
| W6_qmc_pca_256 | 1 | — | 0.233 |
| W6_qmc_pca_256 | 2 | — | 0.207 |
| W6_qmc_pca_256 | 4 | — | 0.118 |
| W6_qmc_pca_256 | 8 | — | 0.068 |
| W6_qmc_pca_256 | 16 | — | 0.070 |
| W7_mlmc_euro_5e-3 | 1 | — | 0.898 |
| W7_mlmc_euro_5e-3 | 2 | — | 0.464 |
| W7_mlmc_euro_5e-3 | 4 | — | 0.294 |
| W7_mlmc_euro_5e-3 | 8 | — | 0.192 |
| W7_mlmc_euro_5e-3 | 16 | — | 0.156 |
| W7b_mlmc_euro_5e-3_boxmuller | 1 | — | 1.587 |
| W7b_mlmc_euro_5e-3_boxmuller | 2 | — | 0.835 |
| W7b_mlmc_euro_5e-3_boxmuller | 4 | — | 0.471 |
| W7b_mlmc_euro_5e-3_boxmuller | 8 | — | 0.370 |
| W7b_mlmc_euro_5e-3_boxmuller | 16 | — | 0.241 |
| W8_mlqmc_euro_5e-3 | 1 | — | 0.108 |
| W8_mlqmc_euro_5e-3 | 2 | — | 0.057 |
| W8_mlqmc_euro_5e-3 | 4 | — | 0.033 |
| W8_mlqmc_euro_5e-3 | 8 | — | 0.022 |
| W8_mlqmc_euro_5e-3 | 16 | — | 0.019 |

### Escalado por hilos (opt-ziggurat-ac)

| carga | hilos | mediana (s) | Mcaminos/s | aceleración vs 1 hilo | eficiencia |
|---|---|---|---|---|---|
| W1_gbm_euro_64 | 1 | 0.854 | 4.911 | ×1.00 | 100 % |
| W1_gbm_euro_64 | 2 | 0.689 | 6.084 | ×1.24 | 62 % |
| W1_gbm_euro_64 | 4 | 0.368 | 11.396 | ×2.32 | 58 % |
| W1_gbm_euro_64 | 8 | 0.226 | 18.597 | ×3.79 | 47 % |
| W1_gbm_euro_64 | 16 | 0.156 | 26.828 | ×5.46 | 34 % |
| W1b_gbm_euro_64_boxmuller | 1 | 3.628 | 1.156 | ×1.00 | 100 % |
| W1b_gbm_euro_64_boxmuller | 2 | 2.406 | 1.743 | ×1.51 | 75 % |
| W1b_gbm_euro_64_boxmuller | 4 | 1.295 | 3.240 | ×2.80 | 70 % |
| W1b_gbm_euro_64_boxmuller | 8 | 0.711 | 5.901 | ×5.11 | 64 % |
| W1b_gbm_euro_64_boxmuller | 16 | 0.600 | 6.988 | ×6.04 | 38 % |
| W2_gbm_asian_256 | 1 | 0.833 | 1.258 | ×1.00 | 100 % |
| W2_gbm_asian_256 | 2 | 0.425 | 2.467 | ×1.96 | 98 % |
| W2_gbm_asian_256 | 4 | 0.259 | 4.045 | ×3.21 | 80 % |
| W2_gbm_asian_256 | 8 | 0.186 | 5.634 | ×4.48 | 56 % |
| W2_gbm_asian_256 | 16 | 0.165 | 6.336 | ×5.04 | 31 % |
| W3_dupire_euro_256 | 1 | 2.659 | 0.197 | ×1.00 | 100 % |
| W3_dupire_euro_256 | 2 | 1.870 | 0.280 | ×1.42 | 71 % |
| W3_dupire_euro_256 | 4 | 1.010 | 0.519 | ×2.63 | 66 % |
| W3_dupire_euro_256 | 8 | 0.653 | 0.803 | ×4.07 | 51 % |
| W3_dupire_euro_256 | 16 | 0.477 | 1.099 | ×5.57 | 35 % |
| W4_heston_euro_128 | 1 | 0.978 | 1.072 | ×1.00 | 100 % |
| W4_heston_euro_128 | 2 | 0.543 | 1.930 | ×1.80 | 90 % |
| W4_heston_euro_128 | 4 | 0.308 | 3.403 | ×3.17 | 79 % |
| W4_heston_euro_128 | 8 | 0.201 | 5.221 | ×4.87 | 61 % |
| W4_heston_euro_128 | 16 | 0.196 | 5.351 | ×4.99 | 31 % |
| W5_qmc_bb_64 | 1 | 0.442 | 1.187 | ×1.00 | 100 % |
| W5_qmc_bb_64 | 2 | 0.295 | 1.775 | ×1.50 | 75 % |
| W5_qmc_bb_64 | 4 | 0.159 | 3.304 | ×2.78 | 70 % |
| W5_qmc_bb_64 | 8 | 0.098 | 5.324 | ×4.49 | 56 % |
| W5_qmc_bb_64 | 16 | 0.066 | 7.932 | ×6.68 | 42 % |
| W6_qmc_pca_256 | 1 | 0.233 | 0.070 | ×1.00 | 100 % |
| W6_qmc_pca_256 | 2 | 0.207 | 0.079 | ×1.13 | 56 % |
| W6_qmc_pca_256 | 4 | 0.118 | 0.139 | ×1.97 | 49 % |
| W6_qmc_pca_256 | 8 | 0.068 | 0.241 | ×3.43 | 43 % |
| W6_qmc_pca_256 | 16 | 0.070 | 0.234 | ×3.33 | 21 % |
| W7_mlmc_euro_5e-3 | 1 | 0.898 | 38.518 | ×1.00 | 100 % |
| W7_mlmc_euro_5e-3 | 2 | 0.464 | 74.635 | ×1.94 | 97 % |
| W7_mlmc_euro_5e-3 | 4 | 0.294 | 117.739 | ×3.06 | 76 % |
| W7_mlmc_euro_5e-3 | 8 | 0.192 | 180.490 | ×4.69 | 59 % |
| W7_mlmc_euro_5e-3 | 16 | 0.156 | 222.260 | ×5.77 | 36 % |
| W7b_mlmc_euro_5e-3_boxmuller | 1 | 1.587 | 19.954 | ×1.00 | 100 % |
| W7b_mlmc_euro_5e-3_boxmuller | 2 | 0.835 | 37.902 | ×1.90 | 95 % |
| W7b_mlmc_euro_5e-3_boxmuller | 4 | 0.471 | 67.265 | ×3.37 | 84 % |
| W7b_mlmc_euro_5e-3_boxmuller | 8 | 0.370 | 85.684 | ×4.29 | 54 % |
| W7b_mlmc_euro_5e-3_boxmuller | 16 | 0.241 | 131.484 | ×6.59 | 41 % |
| W8_mlqmc_euro_5e-3 | 1 | 0.108 | 8.311 | ×1.00 | 100 % |
| W8_mlqmc_euro_5e-3 | 2 | 0.057 | 15.598 | ×1.88 | 94 % |
| W8_mlqmc_euro_5e-3 | 4 | 0.033 | 26.942 | ×3.24 | 81 % |
| W8_mlqmc_euro_5e-3 | 8 | 0.022 | 39.858 | ×4.80 | 60 % |
| W8_mlqmc_euro_5e-3 | 16 | 0.019 | 46.532 | ×5.60 | 35 % |
