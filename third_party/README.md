# Código de terceros (vendorizado)

Todo está copiado tal cual, sin modificar, para compilar sin red. sha256 del fichero principal:

| componente | versión | licencia | fichero | sha256 |
|---|---|---|---|---|
| doctest | 2.4.11 | MIT | `doctest/doctest.h` | `44faa038e9c3f9728efbda143748d01124ea0a27f4bf78f35a15d8fab2e039fb` |
| cpp-httplib | 0.18.3 | MIT | `cpp-httplib/httplib.h` | `a0a0c13dc086663863dbe6730a19716f7d3744904b6205496e8301750814dbd5` |
| nlohmann/json | 3.11.3 | MIT | `nlohmann/json.hpp` | `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6` |
| Apache ECharts | 5.5.1 | Apache-2.0 | `echarts/echarts.min.js` | `e84270bd0cd5bdf60fefc26d00c2a391cb2e81f4d26a7a9ee16185a54773a3cf` |

## Datos generados

`cpu/joe_kuo_data.cpp` lo genera `tools/gen_joe_kuo.py` a partir del fichero de direcciones de Joe & Kuo
(`new-joe-kuo-6.21201`, 21 201 dimensiones). El sha256 del fichero de origen figura en la cabecera del `.cpp` generado.
