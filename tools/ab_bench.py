#!/usr/bin/env python3
"""A/B entre dos binarios de bench_cpu, intercalando ejecuciones.

En un portátil la frecuencia cambia con la temperatura y el estado de energía, así que dos tandas
separadas no son comparables. Este script alterna A y B (orden ABBA) varias rondas con las mismas
cargas e hilos y compara las medianas.

Uso:
  python tools/ab_bench.py --a build/ab/base_bench_cpu.exe --b build/mingw-release/bench/bench_cpu.exe \
         --filter W1_,W7_ --threads 1,8 --rounds 5 --reps 3 [--md salida.md]
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_once(exe, flt, threads, reps, tmp):
    out = pathlib.Path(tmp) / "ab.json"
    if out.exists():
        out.unlink()
    cmd = [str(pathlib.Path(exe).resolve()), "--threads", threads, "--reps", str(reps), "--json", str(out)]
    if flt:
        cmd += ["--filter", flt]
    for _ in range(4):   # Smart App Control puede negar la ejecución de forma intermitente
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            last = str(e)
            continue
        if r.returncode == 0 and out.exists():
            d = json.loads(out.read_text(encoding="utf-8"))
            return {(x["id"], x["threads"]): x["median_s"] for x in d["results"]}
        last = (r.stdout + r.stderr)[-400:]
    raise SystemExit(f"no se pudo ejecutar {exe}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A (antes)")
    ap.add_argument("--label-b", default="B (después)")
    ap.add_argument("--filter", default="")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    ta, tb = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for r in range(args.rounds):
            order = [("a", ta), ("b", tb)] if r % 2 == 0 else [("b", tb), ("a", ta)]
            for which, acc in order:
                res = run_once(getattr(args, which), args.filter, args.threads, args.reps, tmp)
                for k, v in res.items():
                    acc.setdefault(k, []).append(v)
            print(f"ronda {r + 1}/{args.rounds}", file=sys.stderr, flush=True)

    lines = [f"| carga | hilos | {args.label_a} (s) | {args.label_b} (s) | A/B | por ronda (mín–máx) |",
             "|---|---|---|---|---|---|"]
    for k in ta:
        if k not in tb:
            continue
        a, b = statistics.median(ta[k]), statistics.median(tb[k])
        per = [x / y for x, y in zip(ta[k], tb[k])]
        lines.append(f"| {k[0]} | {k[1]} | {a:.4f} | {b:.4f} | ×{a / b:.3f} | ×{min(per):.2f}–×{max(per):.2f} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        pathlib.Path(args.md).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
