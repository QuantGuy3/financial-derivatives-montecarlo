#!/usr/bin/env bash
# Compila bench_cpu con una variante de configuración y la mide en cargas de un solo hilo.
# Sirve para reproducir los experimentos de docs/perf/OPTIMIZACIONES.md (anchura de bloque, -march=native...).
#
#   tools/run_variant_bench.sh <nombre> [flags de cmake...]
#   tools/run_variant_bench.sh lanes1  -DMC_LANES=1
#   tools/run_variant_bench.sh native  -DMC_ARCH=native
#
# Deja docs/perf/variant-<nombre>.{txt,json}. Requiere g++ y ninja en el PATH.
set -euo pipefail
NAME="$1"; shift
BUILD="build/variant-$NAME"
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DMC_ENABLE_CUDA=OFF \
      -DMC_BUILD_TESTS=OFF -DMC_BUILD_GUI=OFF -DMC_BUILD_EXAMPLES=OFF "$@" >/dev/null
cmake --build "$BUILD" --target bench_cpu
THREADS="${THREADS:-1}"
REPS="${REPS:-5}"
FILTERS="${FILTERS:-W1_,W2_,W3_,W4_,W5_,W6_}"
OUT="docs/perf/variant-$NAME"
"$BUILD/bench/bench_cpu" --filter "$FILTERS" --threads "$THREADS" --reps "$REPS" --json "$OUT.json" | tee "$OUT.txt"
