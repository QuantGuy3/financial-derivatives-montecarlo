#!/usr/bin/env bash
# Compara tiempos CPU (todos los hilos) y GPU en los ejemplos 01 y 09 (eps fino 0.002).
# Requiere el build de tools/colab_validate.sh (directorio build-cuda).
set -euo pipefail
BUILD="${BUILD_DIR:-build-cuda}"
for ex in ejemplo01 ejemplo09; do
  for be in cpu cuda; do
    echo "=== $ex --backend=$be ==="
    /usr/bin/time -f "%e s de pared" "$BUILD/$ex" 0.002 --backend="$be" 2>/dev/null | grep -E "^###|nivel [0-9]+ eps" | tail -4
  done
done
