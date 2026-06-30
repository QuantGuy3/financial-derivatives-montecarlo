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
