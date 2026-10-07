#!/usr/bin/env bash
# Compila el proyecto en Google Colab (A-100, Ubuntu 20.04+). Ya no hace falta Eigen:
# el núcleo es autocontenido; solo se necesitan CMake y la toolchain de CUDA.
set -e

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DMC_ENABLE_CUDA=ON
make -j"$(nproc)"
echo "Compilación completada."
