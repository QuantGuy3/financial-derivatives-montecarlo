#!/usr/bin/env python3
"""Genera valores de referencia ("golden") a partir de scipy para los tests del motor CPU.

  tests/golden/sobol_scipy.inc : primeros 64 puntos de Sobol (sin scrambling, orden Gray,
                                 enteros de 32 bits) de varias dimensiones, mas un hash
                                 FNV-1a de las 21201 dimensiones x 64 puntos.

scipy.stats.qmc.Sobol usa las mismas tablas Joe-Kuo (new-joe-kuo-6.21201), por lo que sirve
de verificacion independiente del generador de tablas y de la recurrencia de vectores
directores. Requiere numpy y scipy.   Uso:  python tools/gen_golden.py
"""
import pathlib
import numpy as np
from scipy.stats import qmc

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden" / "sobol_scipy.inc"
DIMS_SAMPLE = [0, 1, 2, 3, 4, 7, 10, 31, 32, 63, 100, 255, 1000, 5000, 10000, 19999, 21200]
N = 64
D_ALL = 21201


def fnv1a64(arr_u32):
    h = 0xCBF29CE484222325
    for v in arr_u32.tolist():
        for b in (v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF):
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    sob = qmc.Sobol(d=D_ALL, scramble=False, bits=32)
    pts = sob.random(N)                                   # (N, D) en [0,1), orden Gray
    words = np.rint(pts * 4294967296.0).astype(np.uint64)   # exacto: x/2^32
    assert np.all(words < 2**32)
    words = words.astype(np.uint32)
    # primeros puntos de dim 0: 0, 0.5, 0.75, 0.25
    assert pts[1, 0] == 0.5 and pts[2, 0] == 0.75 and pts[3, 0] == 0.25, pts[:4, 0]

    h = 0xCBF29CE484222325
    flat = np.ascontiguousarray(words.T).reshape(-1)       # dim-major
    h = fnv1a64(flat)

    lines = ["// ARCHIVO GENERADO por tools/gen_golden.py (scipy.stats.qmc.Sobol, scramble=False)",
             f"static const int kGoldenSobolN = {N};",
             f"static const int kGoldenSobolNDimsSample = {len(DIMS_SAMPLE)};",
             "static const int kGoldenSobolDims[] = {" + ",".join(map(str, DIMS_SAMPLE)) + "};",
             "static const unsigned kGoldenSobolWords[] = {"]
    for d in DIMS_SAMPLE:
        lines.append("    " + ",".join(f"{int(w)}u" for w in words[:, d]) + ",   // dim " + str(d))
    lines.append("};")
    lines.append(f"static const unsigned long long kGoldenSobolFnv = {h}ULL;   // todas las dims x {N} puntos")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("escrito", OUT, "hash", h)


if __name__ == "__main__":
    main()
