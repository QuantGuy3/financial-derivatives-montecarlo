# -*- coding: utf-8 -*-
# Informe COMPLETO: por ejemplo / eps / metodo -> Precio, |Error|, StdErr, N, T(s), MSE_est
# MSE_est = sesgo^2 + varianza ~= |Error|^2 + StdErr^2
#   - sesgo  ~ |precio - referencia|  (incluye el ruido de la referencia)
#   - var    ~ StdErr^2               (varianza del estimador de la media)
import re, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "barrido_precision.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "informe_completo.txt"

lines = open(SRC, encoding="utf-8").read().splitlines()

ex_re  = re.compile(r"^### (ejemplo\S+)")
lvl_re = re.compile(r"^--- nivel \d+\s+eps=([\d.eE+-]+)\s+wall=([\d.]+)s ---")
ref_re = re.compile(r"Referencia:\s*([\d.]+)")
row_re = re.compile(r"^\s{2}(.+?)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(SI|NO)\s*$")
# Fallback: StdErr y N pegados cuando N desborda la columna {:>12} de print_table.
# StdErr sale como D.DDDD (4 decimales), el resto de digitos es N.
merged_re = re.compile(r"^\s{2}(.+?)\s+([\d.]+)\s+(\d+\.\d{4})(\d+)\s+([\d.]+)\s+([\d.]+)\s+(SI|NO)\s*$")

examples = []
cur_ex = cur_levels = cur_level = None
for ln in lines:
    m = ex_re.match(ln)
    if m:
        if cur_ex: examples.append((cur_ex, cur_levels))
        cur_ex, cur_levels, cur_level = m.group(1), [], None
        continue
    m = lvl_re.match(ln)
    if m:
        cur_level = {"eps": m.group(1), "wall": float(m.group(2)), "ref": None, "rows": []}
        cur_levels.append(cur_level); continue
    m = ref_re.search(ln)
    if m and cur_level is not None: cur_level["ref"] = float(m.group(1)); continue
    m = row_re.match(ln) or merged_re.match(ln)
    if m and cur_level is not None:
        cur_level["rows"].append({
            "metodo": m.group(1).strip(), "precio": float(m.group(2)),
            "stderr": float(m.group(3)), "N": int(m.group(4)),
            "T": float(m.group(5)), "err": float(m.group(6)), "ok": m.group(7),
        })
if cur_ex: examples.append((cur_ex, cur_levels))

out = []
out.append("=" * 96)
out.append("PonyTail VII - INFORME COMPLETO (barrido de precision, A100)")
out.append("Por ejemplo / eps / metodo: Precio, |Error|, StdErr, N, T(s), MSE_est")
out.append("MSE_est = |Error|^2 + StdErr^2   (sesgo^2 + varianza del estimador)")
out.append("=" * 96)
out.append("")

HDR = f"  {'metodo':<16}{'Precio':>11}{'|Error|':>10}{'StdErr':>10}{'N':>16}{'T(s)':>9}{'MSE_est':>12}  OK"
for name, levels in examples:
    out.append("#" * 96)
    out.append(f"### {name}    (referencia = {levels[0]['ref'] if levels else '?'})")
    out.append("#" * 96)
    for lv in levels:
        out.append(f"-- eps = {lv['eps']:<10}  wall = {lv['wall']:.2f}s --")
        out.append(HDR)
        out.append("  " + "-" * (len(HDR) - 2))
        for r in lv["rows"]:
            mse = r["err"]**2 + r["stderr"]**2
            flag = "" if r["err"] <= 1.0 else "  <-- corrida rota (pre-crash)"
            out.append(
                f"  {r['metodo']:<16}{r['precio']:>11.4f}{r['err']:>10.4f}"
                f"{r['stderr']:>10.4f}{r['N']:>16d}{r['T']:>9.3f}{mse:>12.3e}  {r['ok']}{flag}"
            )
        out.append("")

out.append("=" * 96)
out.append("NOTA: la referencia es un MC fijo de 500k; su incertidumbre (~0.01-0.02 en")
out.append("varios ejemplos) domina el |Error| y el MSE_est cuando eps baja de ese umbral.")
out.append("Por eso el MSE_est se 'estanca': lo limita el sesgo frente a la referencia,")
out.append("no la varianza del metodo (que sigue cayendo, visible en StdErr).")
out.append("=" * 96)

open(OUT, "w", encoding="utf-8").write("\n".join(out))
print(f"OK -> {OUT}  ({len(examples)} ejemplos)")
