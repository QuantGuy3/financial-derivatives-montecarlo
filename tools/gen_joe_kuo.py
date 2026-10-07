#!/usr/bin/env python3
"""Genera cpu/joe_kuo_data.cpp a partir del fichero oficial de numeros directores de Sobol
(Joe & Kuo, "new-joe-kuo-6.21201": hasta 21201 dimensiones).

Uso:  python tools/gen_joe_kuo.py [ruta/new-joe-kuo-6.21201]
Sin argumento descarga el fichero de https://web.maths.unsw.edu.au/~fkuo/sobol/

Se embeben los parametros (s, a, m_i) y no los vectores directores expandidos (que
ocuparian ~5x mas); cpu/sobol.cpp los expande al arrancar. Dimension 1 (van der Corput)
no aparece en el fichero (todos sus m_i = 1).

Licencia de los datos: Copyright (c) 2008, Frances Y. Kuo and Stephen Joe (licencia tipo
BSD de 3 clausulas, ver https://web.maths.unsw.edu.au/~fkuo/sobol/ ). Se conserva la
atribucion en el fichero generado.
"""
import hashlib
import pathlib
import sys
import urllib.request

URL = "https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201"
OUT = pathlib.Path(__file__).resolve().parent.parent / "cpu" / "joe_kuo_data.cpp"


def main():
    if len(sys.argv) > 1:
        raw = pathlib.Path(sys.argv[1]).read_bytes()
    else:
        raw = urllib.request.urlopen(URL, timeout=60).read()
    sha = hashlib.sha256(raw).hexdigest()
    lines = raw.decode("ascii").splitlines()
    assert lines[0].split()[0] == "d"
    S, A, M_off, M = [], [], [0], []
    expect_d = 2
    for ln in lines[1:]:
        t = ln.split()
        if not t:
            continue
        d, s, a = int(t[0]), int(t[1]), int(t[2])
        m = [int(x) for x in t[3:]]
        assert d == expect_d, (d, expect_d)
        assert len(m) == s, (d, s, len(m))
        S.append(s)
        A.append(a)
        M.extend(m)
        M_off.append(len(M))
        expect_d += 1
    n_dims = expect_d - 1          # incluye la dimension 1 (van der Corput), no listada

    def emit(name, ctype, arr):
        out = [f"extern const {ctype} {name}[] = {{"]
        for i in range(0, len(arr), 24):
            out.append("    " + ",".join(str(x) for x in arr[i:i + 24]) + ",")
        out.append("};")
        return "\n".join(out)

    body = f"""// ARCHIVO GENERADO por tools/gen_joe_kuo.py -- no editar a mano.
// Origen: {URL}
// sha256 del fichero de origen: {sha}
// Datos: Copyright (c) 2008, Frances Y. Kuo and Stephen Joe. Licencia tipo BSD de 3 clausulas
// (redistribucion permitida conservando este aviso); ver la pagina de origen.
//
// Para la dimension d = 2 .. {n_dims} (indice d-2): grado s, coeficientes a del polinomio
// primitivo y numeros iniciales m_1..m_s (kJoeKuoM[kJoeKuoMOff[d-2] .. kJoeKuoMOff[d-1])).
#include <cstdint>

namespace mc::cpu::detail {{

extern const int kJoeKuoDims = {n_dims};   // incluye la dimension 1 (van der Corput)

{emit("kJoeKuoS", "uint8_t", S)}

{emit("kJoeKuoA", "uint32_t", A)}

{emit("kJoeKuoMOff", "uint32_t", M_off)}

{emit("kJoeKuoM", "uint32_t", M)}

}} // namespace mc::cpu::detail
"""
    OUT.write_text(body, encoding="utf-8", newline="\n")
    print(f"escrito {OUT} ({len(body)/1e6:.2f} MB), dims={n_dims}, sha256={sha}")


if __name__ == "__main__":
    main()
