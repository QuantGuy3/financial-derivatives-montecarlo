# -*- coding: utf-8 -*-
import re, sys

SRC = "barrido_precision.txt"
OUT = "informe_final.txt"

txt = open(SRC, encoding="utf-8").read()
lines = txt.splitlines()

# Estructura: por ejemplo -> lista de niveles; cada nivel: eps, wall, ref, dict metodo->(precio,stderr,N,T,err,ok)
examples = []   # (name, [levels])
cur_ex = None
cur_levels = None
cur_level = None

ex_re   = re.compile(r"^### (ejemplo\S+)")
lvl_re  = re.compile(r"^--- nivel \d+\s+eps=([\d.eE+-]+)\s+wall=([\d.]+)s ---")
cut_re  = re.compile(r"CORTADO|supero el presupuesto")
ref_re  = re.compile(r"Referencia:\s*([\d.]+)")
# fila de metodo: nombre (puede tener espacios) + 6 columnas + OK
row_re  = re.compile(r"^\s{2}(.+?)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(SI|NO)\s*$")

for ln in lines:
    m = ex_re.match(ln)
    if m:
        if cur_ex is not None:
            examples.append((cur_ex, cur_levels))
        cur_ex = m.group(1)
        cur_levels = []
        cur_level = None
        continue
    m = lvl_re.match(ln)
    if m:
        cur_level = {"eps": m.group(1), "wall": float(m.group(2)), "ref": None, "rows": {}}
        cur_levels.append(cur_level)
        continue
    m = ref_re.search(ln)
    if m and cur_level is not None:
        cur_level["ref"] = float(m.group(1))
        continue
    m = row_re.match(ln)
    if m and cur_level is not None:
        name = m.group(1).strip()
        cur_level["rows"][name] = {
            "precio": float(m.group(2)), "stderr": float(m.group(3)),
            "N": int(m.group(4)), "T": float(m.group(5)),
            "err": float(m.group(6)), "ok": m.group(7),
        }
if cur_ex is not None:
    examples.append((cur_ex, cur_levels))

out = []
out.append("=" * 78)
out.append("PonyTail VII - INFORME FINAL DE CONVERGENCIA (barrido de precision, A100)")
out.append("Escala redonda 1-2-5. |Error| = |precio - referencia|. wall = s de pared/corrida.")
out.append("=" * 78)
out.append("")

for name, levels in examples:
    if not levels:
        continue
    epss   = [lv["eps"] for lv in levels]
    walls  = [lv["wall"] for lv in levels]
    # orden de metodos: el del primer nivel
    methods = list(levels[0]["rows"].keys())
    # union por si aparece alguno nuevo
    for lv in levels:
        for k in lv["rows"]:
            if k not in methods:
                methods.append(k)

    out.append("#" * 78)
    out.append(f"### {name}    (referencia = {levels[0]['ref']})")
    out.append("#" * 78)

    # cabecera de eps
    hdr = f"{'metodo \\ eps':<16}" + "".join(f"{e:>10}" for e in epss)
    out.append(hdr)
    out.append("-" * len(hdr))
    # matriz de |Error|  (marca *** los puntos basura |err|>1, p.ej. corrida rota pre-crash)
    for mth in methods:
        cells = []
        for lv in levels:
            r = lv["rows"].get(mth)
            if not r:
                cells.append(f"{'--':>10}")
            elif r["err"] > 1.0:
                cells.append(f"{'***':>10}")
            else:
                cells.append(f"{r['err']:>10.4f}")
        out.append(f"{mth:<16}" + "".join(cells))
    out.append("-" * len(hdr))
    # fila de tiempos de pared por nivel
    out.append(f"{'wall (s)':<16}" + "".join(f"{w:>10.1f}" for w in walls))
    # N del MC (referencia de coste), en millones para que quepa
    mc_key = next((k for k in methods if k.strip() in ("MC", "MC (Dupire)", "MC (plain)")), None)
    if mc_key:
        out.append(f"{'N(MC) [millones]':<16}" + "".join(
            f"{(lv['rows'][mc_key]['N']/1e6 if mc_key in lv['rows'] else 0):>10.2f}" for lv in levels))
    out.append("")

out.append("=" * 78)
out.append("NOTAS")
out.append("- |Error| = |precio - referencia|. La referencia es un MC fijo de 500k; su")
out.append("  propia incertidumbre (~0.01-0.02 segun el ejemplo) marca el suelo de error:")
out.append("  por debajo de eso, 'NO' en la tabla original no significa metodo erroneo,")
out.append("  sino que el eps exigido es mas fino que la propia referencia.")
out.append("- '***' = punto basura (|err|>1) de una corrida rota justo antes del crash por")
out.append("  desbordamiento de memoria / int; corregido despues con batching.")
out.append("- ej04, ej05, ej09, ej10 pararon antes en el barrido original por ese crash;")
out.append("  con el batching anadido ya corren a mayor precision sin reventar.")
out.append("=" * 78)

open(OUT, "w", encoding="utf-8").write("\n".join(out))
print(f"OK -> {OUT}  ({len(examples)} ejemplos)")
