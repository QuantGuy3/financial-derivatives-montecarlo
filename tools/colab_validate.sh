#!/usr/bin/env bash
# Valida el backend CUDA en Google Colab (A100) o en cualquier equipo con GPU NVIDIA y nvcc:
#   1. compila con CUDA (MC_ENABLE_CUDA=ON) y ejecuta los tests rápidos/estadísticos de la CPU,
#   2. vuelca los puntos Sobol de cuRAND y los compara con el generador de la CPU,
#   3. ejecuta la paridad CPU<->GPU (ctest -L cuda).
# Uso:  bash tools/colab_validate.sh        (desde la raíz del repositorio)
set -euo pipefail

ARCH="${CUDA_ARCH:-80}"          # 80 = A100; 75 = T4; 89 = L4
BUILD="${BUILD_DIR:-build-cuda}"

cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DMC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$BUILD" -j"$(nproc)"

echo "== Tests de la CPU =="
ctest --test-dir "$BUILD" -L "fast|determinism" --output-on-failure

echo "== Volcado de Sobol de cuRAND =="
"$BUILD/sobol_dump" "$BUILD/sobol_dump.txt"

echo "== Paridad CPU <-> GPU =="
MC_SOBOL_DUMP="$BUILD/sobol_dump.txt" ctest --test-dir "$BUILD" -L cuda --output-on-failure

echo "Validación completada."
