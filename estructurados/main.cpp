// estructurados: definición, almacenamiento y valoración de productos estructurados.
//
//   estructurados nuevo [fichero.json]       asistente interactivo
//   estructurados plantilla <tipo> [fichero]  plantilla JSON lista para editar
//   estructurados plantillas | modelos        catálogos disponibles
//   estructurados mostrar <fichero.json>      valida y muestra la ficha
//   estructurados valorar <fichero.json> [opciones]
//   estructurados replicar <fichero.json> [opciones]  réplica estática con opciones
//   estructurados calibrar <mercado.json|cadena.csv> --modelo M [opciones]
//   estructurados historico <precios.csv> [opciones]   estimación con una serie de precios
//   estructurados fraccional [modelos|precio|sonrisa|fbm|nucleo] ...  modelos fraccionarios

#include "asistente.hpp"
#include "calibracion.hpp"
#include "cotizaciones.hpp"
#include "fraccional_cli.hpp"
#include "historico.hpp"
#include "mercado.hpp"
#include "plantillas.hpp"
#include "producto.hpp"
#include "replicacion.hpp"
#include "valoracion.hpp"

#include <cmath>

#include <cstring>
#include <map>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace est;

namespace {

void uso(std::ostream& os) {
    os << "Uso:\n"
          "  estructurados nuevo [fichero.json]          Asistente interactivo para definir un producto\n"
          "  estructurados plantilla <tipo> [fichero]    Escribe una plantilla JSON lista para editar\n"
          "  estructurados plantillas                    Lista las plantillas (tipos de producto)\n"
          "  estructurados modelos                       Lista los modelos y sus parámetros\n"
          "  estructurados mostrar <fichero.json>        Valida el producto y muestra su ficha\n"
          "  estructurados valorar <fichero.json> [opciones]\n"
          "  estructurados replicar <fichero.json> [opciones]  Réplica estática con bonos, subyacente y opciones\n"
          "  estructurados calibrar <mercado.json|cadena.csv> --modelo M [opciones]   Calibra un modelo a opciones\n"
          "  estructurados historico <precios.csv> [--modelo auto|black_scholes|merton] [--producto ...]\n"
          "                                              Estima un modelo con la serie histórica de precios\n";
    uso_fraccional(os);
    os <<
          "\nOpciones de valoración (prevalecen sobre la sección \"simulacion\" del fichero):\n"
          "  --metodo auto|mc|qmc     --dispositivo auto|gpu|cpu   (--cpu / --gpu)\n"
          "  --error 0.05%            error estándar objetivo, en % del nominal\n"
          "  --trayectorias N         número fijo de trayectorias (sin adaptación)\n"
          "  --max N                  máximo de trayectorias       --replicas R   réplicas QMC\n"
          "  --pasos N|auto           pasos de simulación por año  --semilla S\n"
          "  --doble | --simple       precisión en GPU             --hilos H      hilos en CPU\n"
          "  --griegas                delta, gamma, vega y rho     --salida res.json  guarda resultados\n"
          "  --sin-ficha              no muestra la ficha del producto antes de valorar\n"
          "\nOpciones de la réplica estática:\n"
          "  --cadena opciones.csv    usa los strikes y precios de una cadena de opciones cotizada\n"
          "  --volumen-min N          (con --cadena) solo opciones con volumen >= N contratos\n"
          "  --pesos interpolacion|curvatura   pesos de las opciones cotizadas (por defecto interpolación)\n"
          "  --fecha AAAA-MM-DD       fecha de los datos de la cadena, si sus vencimientos son fechas\n"
          "  --paso 5%                rejilla de strikes para pagos con fórmulas\n"
          "  --rango 50%:200%         strikes que se usan y tramo donde se mide el error\n"
          "  --precios-mc             valora las opciones por Monte Carlo con el modelo del subyacente\n"
          "  --comparar               valora también el producto por simulación y compara\n"
          "  --csv cartera.csv        guarda la cartera      --perfil perfil.csv   pago objetivo y réplica\n"
          "  --salida replica.json    guarda la réplica en JSON\n"
          "\nOpciones de la calibración (ver estructurados/CALIBRACION.md):\n"
          "  --modelo M               black_scholes, heston, merton, bates, nig, sabr, vol_local, rough_heston,\n"
          "                           rough_bergomi, fmls o svi\n"
          "  --optimizador auto|lm|de|nm   --arranques N   --pesos iguales|vega|horquilla\n"
          "  --fijar k=v,...   --liberar k,...   --inicial k=v,...   (parámetros del modelo)\n"
          "  --vencimientos 1M:3A   --strikes 70%:130% (K/F)   --volumen-min N   --con-dentro-del-dinero\n"
          "  --forward auto|dado|paridad   --tipo 3%   --dividendos 2%   --spot S   --fecha AAAA-MM-DD\n"
          "  --trayectorias N (modelos por Monte Carlo)   --semilla S\n"
          "  --producto p.json [--subyacente NOMBRE] [--guardar nuevo.json]  pone el modelo calibrado en el producto\n"
          "  --salida calibracion.json   --csv ajuste.csv\n";
}

// Ajusta un texto a un ancho de línea (cortando por palabras, sin partir UTF-8)
std::string ajustar(const std::string& texto, size_t ancho, size_t sangria) {
    std::string out, linea, palabra;
    auto vis = [](const std::string& s) {
        size_t n = 0;
        for (unsigned char c : s) n += ((c & 0xC0) != 0x80);
        return n;
    };
    auto volcar = [&]() {
        if (linea.empty()) return;
        out += std::string(sangria, ' ') + linea + "\n";
        linea.clear();
    };
    for (size_t i = 0; i <= texto.size(); ++i) {
        if (i == texto.size() || texto[i] == ' ') {
            if (!palabra.empty()) {
                if (vis(linea) + 1 + vis(palabra) > ancho) volcar();
                linea += (linea.empty() ? "" : " ") + palabra;
                palabra.clear();
            }
        } else {
            palabra += texto[i];
        }
    }
    volcar();
    return out;
}

int cmd_plantillas() {
    std::cout << "\nPlantillas disponibles (estructurados plantilla <tipo> fichero.json):\n\n";
    for (const auto& p : plantillas()) {
        std::cout << "  " << p.id << std::string(p.id.size() < 22 ? 22 - p.id.size() : 1, ' ') << p.titulo << "\n";
        std::cout << ajustar(p.descripcion, 72, 24) << "\n";
    }
    return 0;
}

int cmd_modelos() {
    std::cout << "\nModelos disponibles para cada subyacente (\"modelo\": {\"tipo\": ..., parámetros}):\n";
    for (const auto& m : catalogo_modelos()) {
        std::cout << "\n  " << m.clave;
        if (!m.alias.empty()) {
            std::cout << "  (alias:";
            for (const auto& a : m.alias) std::cout << " " << a;
            std::cout << ")";
        }
        std::cout << "\n" << ajustar(m.nombre + ". " + m.descripcion, 90, 4);
        for (const auto& p : m.params) {
            std::string k = "\"" + p.clave + "\"";
            std::cout << "      " << k << std::string(k.size() < 20 ? 20 - k.size() : 1, ' ') << p.descripcion
                      << " (por defecto " << (p.porcentaje ? porcentaje(p.defecto) : fmt(p.defecto, 4));
            if (p.clave == "vol_max" || p.clave == "factores") std::cout << ", opcional";
            std::cout << ")\n";
        }
    }
    std::cout << "\n  Los porcentajes se pueden escribir como \"20%\" o como 0.20.\n";
    return 0;
}

int cmd_plantilla(const std::vector<std::string>& a) {
    if (a.empty()) {
        std::cerr << "Indique el tipo de plantilla. ";
        cmd_plantillas();
        return 2;
    }
    const Plantilla* p = buscar_plantilla(a[0]);
    if (!p) {
        std::cerr << "Plantilla desconocida: " << a[0] << "\n";
        cmd_plantillas();
        return 2;
    }
    json::Valor doc = construir_producto(*p, comunes_por_defecto(*p), {});
    compilar(doc); // comprobación
    if (a.size() > 1) {
        json::escribir_fichero(a[1], doc);
        std::cout << "Plantilla \"" << p->id << "\" escrita en " << a[1] << "\n"
                  << "Edítela con cualquier editor de texto y compruébela con:  estructurados mostrar " << a[1] << "\n";
    } else {
        std::cout << json::escribir(doc);
    }
    return 0;
}

int cmd_mostrar(const std::vector<std::string>& a) {
    if (a.empty()) { uso(std::cerr); return 2; }
    Producto p = cargar(a[0]);
    imprimir_ficha(std::cout, p);
    std::cout << "El producto es válido.\n";
    return 0;
}

int valorar_producto(const Producto& p, const OpcionesValoracion& o, bool griegas, const std::string& salida) {
    ResultadoValoracion r = valorar(p, o, &std::cout);
    imprimir_resultado(std::cout, p, r);
    Griegas g;
    if (griegas) {
        g = calcular_griegas(p, o, r, &std::cout);
        imprimir_griegas(std::cout, p, g);
    }
    if (!salida.empty()) {
        json::escribir_fichero(salida, resultado_a_json(p, r, griegas ? &g : nullptr));
        std::cout << "Resultados guardados en " << salida << "\n";
    }
    return 0;
}

int cmd_valorar(const std::vector<std::string>& a) {
    if (a.empty()) { uso(std::cerr); return 2; }
    OpcionesValoracion o;
    bool griegas = false, ficha = true;
    std::string salida;
    for (size_t i = 1; i < a.size(); ++i) {
        const std::string& k = a[i];
        auto sig = [&]() -> const std::string& {
            if (i + 1 >= a.size()) throw std::runtime_error("falta el valor de la opción " + k);
            return a[++i];
        };
        if (k == "--metodo") o.metodo = sig();
        else if (k == "--dispositivo") o.dispositivo = sig();
        else if (k == "--cpu") o.dispositivo = "cpu";
        else if (k == "--gpu") o.dispositivo = "gpu";
        else if (k == "--error") o.error_objetivo = texto_a_real(sig());
        else if (k == "--trayectorias") o.trayectorias = (long long)texto_a_real(sig());
        else if (k == "--max") o.trayectorias_max = (long long)texto_a_real(sig());
        else if (k == "--semilla") o.semilla = (long long)texto_a_real(sig());
        else if (k == "--replicas") o.replicas_qmc = (int)texto_a_real(sig());
        else if (k == "--hilos") o.hilos = (int)texto_a_real(sig());
        else if (k == "--pasos") { std::string v = sig(); o.pasos_por_anio = (v == "auto") ? -1.0 : texto_a_real(v); }
        else if (k == "--doble") o.doble_precision = 1;
        else if (k == "--simple") o.doble_precision = 0;
        else if (k == "--griegas") griegas = true;
        else if (k == "--salida") salida = sig();
        else if (k == "--sin-ficha") ficha = false;
        else throw std::runtime_error("opción desconocida: " + k);
    }
    if (!o.metodo.empty() && o.metodo != "auto" && o.metodo != "mc" && o.metodo != "qmc")
        throw std::runtime_error("--metodo debe ser auto, mc o qmc");
    if (!o.dispositivo.empty() && o.dispositivo != "auto" && o.dispositivo != "gpu" && o.dispositivo != "cpu")
        throw std::runtime_error("--dispositivo debe ser auto, gpu o cpu");
    Producto p = cargar(a[0]);
    if (ficha) imprimir_ficha(std::cout, p);
    else for (const auto& av : p.avisos) std::cout << " ! " << av << "\n";
    return valorar_producto(p, o, griegas, salida);
}

int cmd_replicar(const std::vector<std::string>& a) {
    if (a.empty()) { uso(std::cerr); return 2; }
    OpcionesReplica o;
    std::string ruta_cadena, csv, perfil, salida, fecha;
    bool comparar = false;
    for (size_t i = 1; i < a.size(); ++i) {
        const std::string& k = a[i];
        auto sig = [&]() -> const std::string& {
            if (i + 1 >= a.size()) throw std::runtime_error("falta el valor de la opción " + k);
            return a[++i];
        };
        if (k == "--cadena") ruta_cadena = sig();
        else if (k == "--volumen-min" || k == "--volumen") o.volumen_min = texto_a_real(sig());
        else if (k == "--pesos") {
            o.pesos = sig();
            if (o.pesos != "interpolacion" && o.pesos != "curvatura")
                throw std::runtime_error("--pesos debe ser interpolacion o curvatura");
        }
        else if (k == "--fecha") fecha = sig();
        else if (k == "--paso") o.paso = texto_a_real(sig());
        else if (k == "--rango") {
            std::string v = sig();
            size_t dp = v.find(':');
            if (dp == std::string::npos) throw std::runtime_error("--rango se escribe mínimo:máximo, p. ej. 50%:200%");
            o.p_min = texto_a_real(v.substr(0, dp));
            o.p_max = texto_a_real(v.substr(dp + 1));
            if (!(o.p_max > o.p_min) || o.p_min < 0) throw std::runtime_error("--rango no válido");
        }
        else if (k == "--precios-mc") o.precios_mc = true;
        else if (k == "--trayectorias") o.mc.trayectorias = (long long)texto_a_real(sig());
        else if (k == "--comparar") comparar = true;
        else if (k == "--csv") csv = sig();
        else if (k == "--perfil") perfil = sig();
        else if (k == "--salida") salida = sig();
        else throw std::runtime_error("opción desconocida: " + k);
    }
    json::Valor doc;
    Producto p = cargar(a[0], &doc);
    CadenaOpciones cadena;
    if (!ruta_cadena.empty()) {
        int dias = -1;
        if (!fecha.empty() && !texto_a_fecha(fecha, dias)) throw std::runtime_error("--fecha no válida (AAAA-MM-DD)");
        if (dias < 0) {
            if (const json::Valor* fv = doc.buscar("fecha_valoracion")) texto_a_fecha(fv->texto(), dias);
        }
        cadena = leer_cadena_csv(ruta_cadena, dias);
        for (const auto& av : cadena.avisos) std::cout << " ! " << av << "\n";
        if (cadena.spot > 0 && std::fabs(cadena.spot / p.activos[0].spot - 1.0) > 0.02)
            std::cout << " ! el spot de la cadena (" << fmt(cadena.spot, 2) << ") y el del producto ("
                      << fmt(p.activos[0].spot, 2) << ") difieren más de un 2%\n";
        o.cadena = &cadena;
    }
    Replica r = replicar(p, o);
    imprimir_replica(std::cout, p, r);
    if (comparar) {
        OpcionesValoracion ov;
        ov.metodo = "auto"; // QMC si el motor lo admite; si no, MC
        ResultadoValoracion rv = valorar(p, ov, nullptr);
        double vr = r.valor / p.nominal;
        std::cout << " Comparación con la valoración por simulación (" << rv.metodo << ", " << rv.motor << ")\n"
                  << "   Simulación ...... " << porcentaje(rv.precio, 4) << " ± " << porcentaje(rv.error, 4)
                  << " (error estándar)\n"
                  << "   Réplica ......... " << porcentaje(vr, 4) << "\n"
                  << "   Diferencia ...... " << porcentaje(vr - rv.precio, 4);
        if (rv.error > 0) std::cout << " (" << fmt((vr - rv.precio) / rv.error, 2) << " errores estándar)";
        std::cout << "\n";
    }
    if (!csv.empty()) { guardar_replica_csv(csv, r); std::cout << "Cartera guardada en " << csv << "\n"; }
    if (!perfil.empty()) { guardar_perfil_csv(perfil, r); std::cout << "Perfil guardado en " << perfil << "\n"; }
    if (!salida.empty()) {
        json::escribir_fichero(salida, replica_a_json(p, r));
        std::cout << "Réplica guardada en " << salida << "\n";
    }
    return 0;
}

// "k=v,k2=v2" -> mapa (valores con % o en tanto por uno)
std::map<std::string, double> pares(const std::string& t) {
    std::map<std::string, double> m;
    size_t i = 0;
    while (i < t.size()) {
        size_t c = t.find(',', i);
        std::string par = t.substr(i, c == std::string::npos ? std::string::npos : c - i);
        size_t e = par.find('=');
        if (e == std::string::npos) throw std::runtime_error("se esperaba clave=valor en \"" + par + "\"");
        bool ok = false;
        double v = texto_a_real(par.substr(e + 1), &ok);
        if (!ok) throw std::runtime_error("valor no válido en \"" + par + "\"");
        m[par.substr(0, e)] = v;
        if (c == std::string::npos) break;
        i = c + 1;
    }
    return m;
}

// Escribe el modelo en el subyacente i del producto y guarda el resultado
void escribir_modelo_en_producto(json::Valor& doc, const Producto& prod, int i_sub, const Modelo& mod,
                                 const std::string& producto, const std::string& guardar) {
    json::Valor* subs = doc.buscar("subyacentes");
    json::Valor& s = (*subs)[(size_t)i_sub];
    s.poner("modelo", modelo_a_json(mod));
    std::string destino = guardar;
    if (destino.empty()) {
        destino = producto;
        size_t pto = destino.rfind(".json");
        destino = (pto != std::string::npos ? destino.substr(0, pto) : destino) + "_calibrado.json";
    }
    compilar(doc); // comprobación
    json::escribir_fichero(destino, doc);
    std::cout << "Producto con el nuevo modelo de " << prod.activos[(size_t)i_sub].nombre << " guardado en " << destino
              << "\n  Para valorarlo:  estructurados valorar " << destino << "\n";
}

int buscar_subyacente(const Producto& prod, const std::string& subyacente, const std::string& otro_nombre) {
    int i_sub = -1;
    for (size_t i = 0; i < prod.activos.size(); ++i)
        if ((!subyacente.empty() && prod.activos[i].nombre == subyacente) ||
            (subyacente.empty() && (prod.activos.size() == 1 || prod.activos[i].nombre == otro_nombre)))
            i_sub = (int)i;
    if (i_sub < 0) throw std::runtime_error("indique con --subyacente cuál de los subyacentes del producto se usa");
    return i_sub;
}

int cmd_historico(const std::vector<std::string>& a) {
    if (a.empty()) { uso(std::cerr); return 2; }
    std::string modelo = "auto", producto, subyacente, guardar;
    for (size_t i = 1; i < a.size(); ++i) {
        const std::string& k = a[i];
        auto sig = [&]() -> const std::string& {
            if (i + 1 >= a.size()) throw std::runtime_error("falta el valor de la opción " + k);
            return a[++i];
        };
        if (k == "--modelo") modelo = sig();
        else if (k == "--producto") producto = sig();
        else if (k == "--subyacente") subyacente = sig();
        else if (k == "--guardar") guardar = sig();
        else throw std::runtime_error("opción desconocida: " + k);
    }
    if (modelo == "bs" || modelo == "gbm") modelo = "black_scholes";
    if (modelo != "auto" && modelo != "black_scholes" && modelo != "merton")
        throw std::runtime_error("--modelo debe ser auto, black_scholes o merton");
    SerieHistorica s = leer_serie_csv(a[0]);
    ResultadoHistorico r = estimar_historico(s, modelo);
    imprimir_historico(std::cout, s, r);
    if (!producto.empty()) {
        json::Valor doc;
        Producto prod = cargar(producto, &doc);
        int i_sub = buscar_subyacente(prod, subyacente, s.nombre);
        escribir_modelo_en_producto(doc, prod, i_sub, r.modelo, producto, guardar);
    }
    return 0;
}

int cmd_calibrar(const std::vector<std::string>& a) {
    if (a.empty()) { uso(std::cerr); return 2; }
    OpcionesMercado om;
    OpcionesCalibracion oc;
    std::string producto, subyacente, guardar, salida, csv, fecha;
    for (size_t i = 1; i < a.size(); ++i) {
        const std::string& k = a[i];
        auto sig = [&]() -> const std::string& {
            if (i + 1 >= a.size()) throw std::runtime_error("falta el valor de la opción " + k);
            return a[++i];
        };
        auto rango = [&](const std::string& v, double& lo, double& hi, bool plazo) {
            size_t dp = v.find(':');
            if (dp == std::string::npos) throw std::runtime_error(k + " se escribe mínimo:máximo");
            std::string x = v.substr(0, dp), y = v.substr(dp + 1);
            if (!x.empty()) lo = plazo ? texto_a_plazo(x) : texto_a_real(x);
            if (!y.empty()) hi = plazo ? texto_a_plazo(y) : texto_a_real(y);
        };
        if (k == "--modelo") oc.modelo = sig();
        else if (k == "--optimizador") oc.optimizador = sig();
        else if (k == "--arranques") oc.arranques = (int)texto_a_real(sig());
        else if (k == "--pesos") oc.pesos = sig();
        else if (k == "--fijar") for (auto& [c, v] : pares(sig())) oc.fijos[c] = v;
        else if (k == "--inicial") for (auto& [c, v] : pares(sig())) oc.iniciales[c] = v;
        else if (k == "--liberar") {
            std::string v = sig();
            size_t p0 = 0;
            while (p0 <= v.size()) {
                size_t c = v.find(',', p0);
                oc.liberar.push_back(v.substr(p0, c == std::string::npos ? std::string::npos : c - p0));
                if (c == std::string::npos) break;
                p0 = c + 1;
            }
        }
        else if (k == "--vencimientos") rango(sig(), om.T_min, om.T_max, true);
        else if (k == "--strikes") rango(sig(), om.mon_min, om.mon_max, false);
        else if (k == "--volumen-min" || k == "--volumen") om.volumen_min = texto_a_real(sig());
        else if (k == "--con-dentro-del-dinero") om.incluir_dentro_del_dinero = true;
        else if (k == "--forward") om.forward = sig();
        else if (k == "--tipo") om.tipo = texto_a_real(sig());
        else if (k == "--dividendos") om.dividendos = texto_a_real(sig());
        else if (k == "--spot") om.spot = texto_a_real(sig());
        else if (k == "--fecha") fecha = sig();
        else if (k == "--trayectorias") oc.trayectorias_mc = (long long)texto_a_real(sig());
        else if (k == "--semilla") oc.semilla = (uint32_t)texto_a_real(sig());
        else if (k == "--producto") producto = sig();
        else if (k == "--subyacente") subyacente = sig();
        else if (k == "--guardar") guardar = sig();
        else if (k == "--salida") salida = sig();
        else if (k == "--csv") csv = sig();
        else throw std::runtime_error("opción desconocida: " + k);
    }
    if (oc.modelo.empty()) throw std::runtime_error("indique el modelo con --modelo (p. ej. --modelo heston, o svi)");
    if (!fecha.empty() && !texto_a_fecha(fecha, om.dias_valoracion)) throw std::runtime_error("--fecha no válida (AAAA-MM-DD)");
    if (om.forward != "auto" && om.forward != "dado" && om.forward != "paridad")
        throw std::runtime_error("--forward debe ser auto, dado o paridad");

    DatosMercado m = cargar_mercado(a[0], om);
    for (const auto& av : m.avisos) std::cout << " ! " << av << "\n";

    // Producto en el que se escribirá el modelo (y cuyo modelo actual sirve de punto de partida)
    json::Valor doc;
    Producto prod;
    int i_sub = -1;
    if (!producto.empty()) {
        prod = cargar(producto, &doc);
        i_sub = buscar_subyacente(prod, subyacente, m.subyacente);
        oc.partida = &prod.activos[(size_t)i_sub].modelo;
        if (std::fabs(prod.activos[(size_t)i_sub].spot / m.spot - 1.0) > 0.01)
            std::cout << " ! el spot del producto (" << fmt(prod.activos[(size_t)i_sub].spot, 2)
                      << ") y el del mercado (" << fmt(m.spot, 2) << ") difieren más de un 1%\n";
    }
    oc.progreso = &std::cout;
    std::cout << "\nCalibrando " << oc.modelo << " a " << m.puntos.size() << " cotizaciones...\n";
    ResultadoCalibracion r = calibrar(m, oc);
    imprimir_calibracion(std::cout, m, r);
    if (!salida.empty()) {
        json::escribir_fichero(salida, calibracion_a_json(m, r));
        std::cout << "Calibración guardada en " << salida << "\n";
    }
    if (!csv.empty()) {
        guardar_ajuste_csv(csv, m, r);
        std::cout << "Ajuste por cotización guardado en " << csv << "\n";
    }
    if (i_sub >= 0) {
        if (!r.svi.empty()) throw std::runtime_error("SVI describe la superficie, no es un modelo del motor: no se escribe en el producto");
        escribir_modelo_en_producto(doc, prod, i_sub, r.modelo, producto, guardar);
    }
    return 0;
}

int cmd_nuevo(const std::vector<std::string>& a) {
    std::string ruta;
    json::Valor doc = asistente_nuevo_producto(std::cin, std::cout, a.empty() ? "" : a[0], &ruta);
    Producto p;
    try {
        p = compilar(doc);
    } catch (const std::exception&) {
        return ruta.empty() ? 1 : 0; // guardado para corregir a mano
    }
    if (preguntar_si_no(std::cin, std::cout, "¿Valorarlo ahora?", true)) {
        bool g = preguntar_si_no(std::cin, std::cout, "¿Calcular también las sensibilidades (griegas)?", false);
        valorar_producto(p, OpcionesValoracion{}, g, "");
    }
    if (!ruta.empty())
        std::cout << "\nPara volver a valorarlo:  estructurados valorar " << ruta << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "ayuda" || args[0] == "--help" || args[0] == "-h") {
        uso(std::cout);
        return args.empty() ? 2 : 0;
    }
    std::string cmd = args[0];
    std::vector<std::string> resto(args.begin() + 1, args.end());
    // "estructurados producto.json" = valorar
    if (cmd.size() > 5 && cmd.substr(cmd.size() - 5) == ".json") {
        resto.insert(resto.begin(), cmd);
        cmd = "valorar";
    }
    try {
        if (cmd == "nuevo" || cmd == "asistente") return cmd_nuevo(resto);
        if (cmd == "plantilla") return cmd_plantilla(resto);
        if (cmd == "plantillas") return cmd_plantillas();
        if (cmd == "modelos") return cmd_modelos();
        if (cmd == "mostrar" || cmd == "ficha") return cmd_mostrar(resto);
        if (cmd == "valorar") return cmd_valorar(resto);
        if (cmd == "replicar" || cmd == "replica") return cmd_replicar(resto);
        if (cmd == "calibrar" || cmd == "calibracion") return cmd_calibrar(resto);
        if (cmd == "historico" || cmd == "histórico") return cmd_historico(resto);
        if (cmd == "fraccional" || cmd == "fraccionarios") return cmd_fraccional(resto);
        std::cerr << "Orden desconocida: " << cmd << "\n\n";
        uso(std::cerr);
        return 2;
    } catch (const Cancelado&) {
        std::cout << "Asistente cancelado; no se ha guardado nada.\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
}
