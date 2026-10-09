#!/usr/bin/env python3
"""Genera tablas Markdown a partir de los JSON de bench_cpu.

Uso:
  python tools/bench_report.py docs/perf/baseline-naive-cpu.json docs/perf/opt1-ziggurat.json ...

  * Tabla "comparativa": mediana (s) por carga/hilos en cada fichero y aceleracion respecto al
    primero que contenga esa carga.
  * Tabla "escalado por hilos" del ultimo fichero: aceleracion frente a 1 hilo y eficiencia.
"""
import json
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # la consola de Windows no es UTF-8 por defecto


def load(path):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    res = {}
    for r in d["results"]:
        res[(r["id"], r["threads"])] = r
    return d["machine"], res


def main(paths):
    if not paths:
        print(__doc__)
        return 1
    data = [(pathlib.Path(p).stem, *load(p)) for p in paths]
    machine = data[-1][1]
    print(f"Máquina: {machine['cpu']} · {machine['hw_threads']} hilos · {machine['compiler']} · commit {machine['commit']}\n")

    keys = []
    for _, _, res in data:
        for k in res:
            if k not in keys:
                keys.append(k)

    print("### Comparativa (mediana en segundos; entre paréntesis, aceleración frente a la primera columna con datos)\n")
    head = ["carga", "hilos"] + [name for name, _, _ in data]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] * len(head)) + "|")
    for (wid, th) in keys:
        cells = []
        base = None
        for _, _, res in data:
            r = res.get((wid, th))
            if r is None:
                cells.append("—")
                continue
            t = r["median_s"]
            if base is None:
                base = t
                cells.append(f"{t:.3f}")
            else:
                cells.append(f"{t:.3f} (×{base / t:.2f})")
        print(f"| {wid} | {th} | " + " | ".join(cells) + " |")

    name, _, res = data[-1]
    print(f"\n### Escalado por hilos ({name})\n")
    print("| carga | hilos | mediana (s) | Mcaminos/s | aceleración vs 1 hilo | eficiencia |")
    print("|---|---|---|---|---|---|")
    ids = sorted({w for (w, _) in res})
    for wid in ids:
        t1 = res.get((wid, 1))
        for th in sorted(t for (w, t) in res if w == wid):
            r = res[(wid, th)]
            sp = (t1["median_s"] / r["median_s"]) if t1 else float("nan")
            eff = sp / th if t1 else float("nan")
            print(f"| {wid} | {th} | {r['median_s']:.3f} | {r['mpaths_per_s']:.3f} | ×{sp:.2f} | {eff * 100:.0f} % |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
