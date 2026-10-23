#!/usr/bin/env python3
"""Descarga una cadena de opciones de Yahoo Finance (yfinance) y la guarda en el
CSV que leen `estructurados replicar --cadena` y la calibración.

    python herramientas/descargar_cadena.py ^SPX                      # vencimiento más próximo
    python herramientas/descargar_cadena.py ^SPX --vencimiento 2027-12-17 -o spx.csv
    python herramientas/descargar_cadena.py ^SPX --todos --volumen-min 10

Por defecto solo admite índices con opciones de ejercicio europeo (^SPX, ^XSP,
^NDX, ^NQX, ^RUT, ^DJX), igual que TooManyOptions: la réplica estática y la
calibración suponen ejercicio a vencimiento. Las opciones sobre acciones y ETF
son americanas; con --permitir-americanas se descargan igualmente (avisando).

Requisitos: pip install yfinance pandas  (y conexión a internet).

Formato de salida (CSV con cabecera; '#' son comentarios con datos generales):
    # subyacente: ^SPX
    # fecha: 2026-10-23
    # spot: 5612.3
    # fuente: Yahoo Finance (yfinance)
    vencimiento,tipo,strike,bid,ask,ultimo,volumen,interes_abierto,vol_implicita
    2026-12-18,put,5000,12.1,12.6,12.4,350,10240,0.2134
"""

import argparse
import datetime as dt
import math
import sys

INDICES_EUROPEOS = ["^SPX", "^XSP", "^NDX", "^NQX", "^RUT", "^DJX"]


def precio_actual(ticker) -> float:
    """Último precio: cotización rápida o, si no hay, el último cierre disponible."""
    try:
        p = float(ticker.fast_info.get("lastPrice"))
    except (AttributeError, KeyError, TypeError, ValueError):
        p = float("nan")
    if not math.isfinite(p) or p <= 0:
        hist = ticker.history(period="5d")
        if hist is None or hist.empty or "Close" not in hist:
            raise ValueError("no hay precio disponible para el subyacente")
        cierres = hist["Close"].dropna()
        if cierres.empty:
            raise ValueError("no hay precio disponible para el subyacente")
        p = float(cierres.iloc[-1])
    return p


def numero(x):
    """Número o cadena vacía (para los NaN de pandas)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(v) else repr(v) if v != int(v) else str(int(v))


def filas_cadena(ticker, vencimiento: str, volumen_min: float):
    cadena = ticker.option_chain(vencimiento)
    for tipo, tabla in (("call", cadena.calls), ("put", cadena.puts)):
        for _, f in tabla.iterrows():
            vol = f.get("volume")
            try:
                vol_ok = float(vol) >= volumen_min
            except (TypeError, ValueError):
                vol_ok = volumen_min <= 0
            if not vol_ok:
                continue
            yield [vencimiento, tipo, numero(f.get("strike")), numero(f.get("bid")), numero(f.get("ask")),
                   numero(f.get("lastPrice")), numero(vol), numero(f.get("openInterest")),
                   numero(f.get("impliedVolatility"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ticker", help="símbolo de Yahoo Finance, p. ej. ^SPX")
    ap.add_argument("--vencimiento", help="fecha AAAA-MM-DD (por defecto, la más próxima)")
    ap.add_argument("--todos", action="store_true", help="todos los vencimientos (para calibrar)")
    ap.add_argument("--volumen-min", type=float, default=0.0, help="solo opciones con volumen >= N contratos")
    ap.add_argument("-o", "--salida", help="fichero CSV (por defecto cadena_<ticker>_<fecha>.csv)")
    ap.add_argument("--permitir-americanas", action="store_true",
                    help="admite subyacentes con opciones americanas (acciones, ETF)")
    a = ap.parse_args(argv)

    simbolo = a.ticker.strip().upper()
    if simbolo not in INDICES_EUROPEOS and not a.permitir_americanas:
        print(f"'{simbolo}' no está entre los índices con opciones europeas ({', '.join(INDICES_EUROPEOS)}).\n"
              "Las opciones de acciones y ETF son americanas: use --permitir-americanas si aun así las quiere.",
              file=sys.stderr)
        return 2
    try:
        import yfinance as yf
    except ImportError:
        print("Falta yfinance: pip install yfinance pandas", file=sys.stderr)
        return 2

    t = yf.Ticker(simbolo)
    vencimientos = tuple(t.options)
    if not vencimientos:
        print(f"No hay vencimientos de opciones para {simbolo}", file=sys.stderr)
        return 1
    if a.todos:
        elegidos = list(vencimientos)
    elif a.vencimiento:
        if a.vencimiento not in vencimientos:
            print(f"Vencimiento {a.vencimiento} no disponible. Disponibles: {', '.join(vencimientos)}",
                  file=sys.stderr)
            return 1
        elegidos = [a.vencimiento]
    else:
        elegidos = [vencimientos[0]]

    spot = precio_actual(t)
    hoy = dt.date.today().isoformat()
    salida = a.salida or f"cadena_{simbolo.strip('^').lower()}_{hoy}.csv"
    n = 0
    with open(salida, "w", encoding="utf-8") as f:
        f.write(f"# subyacente: {simbolo}\n# fecha: {hoy}\n# spot: {spot:.6g}\n")
        f.write("# fuente: Yahoo Finance (yfinance); volumen = contratos negociados en el día\n")
        if simbolo not in INDICES_EUROPEOS:
            f.write("# aviso: opciones americanas (ejercicio anticipado no modelizado)\n")
        f.write("vencimiento,tipo,strike,bid,ask,ultimo,volumen,interes_abierto,vol_implicita\n")
        for v in elegidos:
            for fila in filas_cadena(t, v, a.volumen_min):
                f.write(",".join(fila) + "\n")
                n += 1
    print(f"{n} opciones de {simbolo} ({len(elegidos)} vencimiento{'s' if len(elegidos) != 1 else ''}, "
          f"spot {spot:.6g}) guardadas en {salida}")
    print(f"Réplica:      estructurados replicar producto.json --cadena {salida} --volumen-min 10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
