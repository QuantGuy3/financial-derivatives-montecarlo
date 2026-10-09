#!/usr/bin/env python3
"""Genera cpu/fastmath_tables.inc: tablas de fast_log / fast_exp (cpu/fastmath.hpp).

Los valores se calculan con mpmath (40 dígitos) y se escriben como literales hexadecimales de
coma flotante, así que no dependen de la libm de la plataforma.

  log:  z = x * 2^-k en [0.748, 1.496), 128 intervalos definidos por los bits de z (ver LOG_OFF).
        Por intervalo: invc ~ 1/c (c = centro; c = 1 exacto en el intervalo que contiene al 1) y
        logc = -log(invc) redondeado, de modo que log(z) = log1p(z*invc - 1) + logc.
  exp:  T[j] = 2^(j/128), j = 0..127.

Uso: python tools/gen_fastmath_tables.py > cpu/fastmath_tables.inc
"""
import struct
import sys

import mpmath as mp

mp.mp.dps = 40
LOG_BITS = 7
N = 1 << LOG_BITS
LOG_OFF = 0x3FE8000000000000 - (1 << (52 - LOG_BITS - 1))   # bits de 0.75 menos medio intervalo


def from_bits(u):
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def to_double(x):
    return float(mp.nstr(x, 25))   # redondeo al double más cercano


def main():
    out = []
    out.append("// GENERADO por tools/gen_fastmath_tables.py -- no editar a mano.")
    out.append(f"// LOG_OFF = 0x{LOG_OFF:016X}, {N} intervalos para log y {N} entradas para exp.")
    out.append("static constexpr double kLogTab[%d][2] = {   // {invc, logc}" % N)
    max_r = 0.0
    for i in range(N):
        lo = from_bits(LOG_OFF + (i << (52 - LOG_BITS)))
        hi = from_bits(LOG_OFF + ((i + 1) << (52 - LOG_BITS)))
        if lo < 1.0 <= hi:
            c = mp.mpf(1)
        else:
            c = (mp.mpf(lo) + mp.mpf(hi)) / 2
        invc = to_double(1 / c)
        logc = to_double(-mp.log(mp.mpf(invc)))
        if c == 1:
            assert invc == 1.0 and logc == 0.0
        max_r = max(max_r, abs(lo * invc - 1), abs(hi * invc - 1))
        out.append("    {%s, %s}," % (float.hex(invc), float.hex(logc)))
    out.append("};")
    out.append(f"// max |z*invc - 1| = {max_r:.6f}")
    out.append("static constexpr double kExp2Tab[%d] = {   // 2^(j/%d)" % (N, N))
    for j in range(N):
        out.append("    %s," % float.hex(to_double(mp.power(2, mp.mpf(j) / N))))
    out.append("};")
    # ln2 partido en alto (los 21 bits bajos a cero: k*hi es exacto para |k| < 2^21) y bajo
    ln2 = mp.log(2)
    hi_bits = struct.unpack("<Q", struct.pack("<d", to_double(ln2)))[0] & ~((1 << 21) - 1)
    ln2hi = from_bits(hi_bits)
    ln2lo = to_double(ln2 - mp.mpf(ln2hi))
    out.append("static constexpr double kLn2Hi = %s, kLn2Lo = %s;" % (float.hex(ln2hi), float.hex(ln2lo)))
    # ln2/N partido igual; N/ln2
    l = ln2 / N
    hi_bits = struct.unpack("<Q", struct.pack("<d", to_double(l)))[0] & ~((1 << 21) - 1)
    lhi = from_bits(hi_bits)
    llo = to_double(l - mp.mpf(lhi))
    out.append("static constexpr double kLn2NHi = %s, kLn2NLo = %s, kInvLn2N = %s;"
               % (float.hex(lhi), float.hex(llo), float.hex(to_double(N / ln2))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
