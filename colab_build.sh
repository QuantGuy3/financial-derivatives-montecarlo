#!/usr/bin/env bash
# Instala dependencias y compila el proyecto en Google Colab (A-100, Ubuntu 20.04+).
set -e

apt-get install -y --no-install-recommends libeigen3-dev > /dev/null

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DEigen3_DIR=/usr/share/eigen3/cmake
make -j"$(nproc)"
echo "Compilación completada."

# Tests del módulo de productos estructurados (con GPU incluyen la comparación GPU/CPU)
./test_estructurados rapido
echo "Pruebe (desde build/):  ./estructurados plantillas  |  ./estructurados valorar ../productos/phoenix_worst_of.json"
