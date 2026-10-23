#include "plantillas.hpp"
#include "producto.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace est {

namespace {

std::string minus(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
std::string recorta(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

double pct(const Respuestas& r, const std::string& k) { return texto_a_real(r.at(k)); }
json::Valor P(const Respuestas& r, const std::string& k) { return json::Valor(r.at(k)); }
bool si(const Respuestas& r, const std::string& k) { return r.at(k) == "si"; }
bool hay(const Respuestas& r, const std::string& k) { return r.count(k) && r.at(k) != "no"; }

json::Valor obj() { return json::Valor::objeto(); }

json::Valor bloque(const std::string& tipo) {
    json::Valor b = obj();
    b.poner("tipo", tipo);
    return b;
}

json::Valor capital_100() {
    json::Valor b = bloque("fijo");
    b.poner("importe", "100%");
    return b;
}

json::Valor perdida_bajo_strike(const std::string& strike, bool condicionada) {
    json::Valor b = bloque("put_vendida");
    b.poner("strike", strike);
    if (condicionada) b.poner("condicion", "barrera_tocada");
    return b;
}

json::Valor barrera_capital(const Respuestas& r, const std::string& kn, const std::string& ko) {
    json::Valor b = obj();
    b.poner("nivel", P(r, kn));
    b.poner("observacion", r.at(ko));
    return b;
}

Pregunta Q(std::string clave, std::string texto, std::string def, TipoDato t, std::string ayuda = "",
           std::vector<std::string> opciones = {}, std::string solo_si = "") {
    return Pregunta{std::move(clave), std::move(texto), std::move(def), t, std::move(ayuda), std::move(opciones),
                    std::move(solo_si)};
}

const std::string AYUDA_OBS =
    "vencimiento = solo se mira el último día; fechas = en cada fecha de observación; "
    "diaria = cada cierre; continua = en todo momento (con corrección de continuidad BGK)";

std::vector<Plantilla> construir_plantillas() {
    std::vector<Plantilla> v;
    const std::vector<SubyacenteDef> dos = {{"SX5E", 4800, "3%", "20%"}, {"SPX", 5600, "1.5%", "18%"}};
    const std::vector<SubyacenteDef> uno = {{"SX5E", 4800, "3%", "20%"}};

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "phoenix", "Autocallable Phoenix (cupón condicional con memoria)",
        "Paga un cupón en cada fecha si la cesta está por encima de la barrera de cupón (con memoria, los "
        "impagados se recuperan). Se amortiza anticipadamente si la cesta supera la barrera de autocancelación. "
        "A vencimiento devuelve el nominal salvo que se haya tocado la barrera de capital: entonces se pierde "
        "como el subyacente.",
        dos, "3A",
        {Q("frecuencia", "Frecuencia de observación", "trimestral", TipoDato::Frecuencia,
           "mensual, bimestral, trimestral, semestral, anual o un plazo como 3M"),
         Q("cupon", "Cupón por periodo (% del nominal)", "2%", TipoDato::Porcentaje,
           "importe que se cobra en cada fecha si P ≥ barrera de cupón (2% trimestral ≈ 8% anual)"),
         Q("barrera_cupon", "Barrera de cupón (% del nivel inicial)", "70%", TipoDato::Porcentaje),
         Q("memoria", "¿Cupón con memoria (recupera los impagados)?", "si", TipoDato::SiNo),
         Q("barrera_autocall", "Barrera de autocancelación (% del nivel inicial)", "100%", TipoDato::Porcentaje),
         Q("escalon", "Variación de la barrera de autocancelación en cada fecha (step-down, p. ej. -5)", "0%",
           TipoDato::Porcentaje, "0 = barrera constante; -5 = baja 5 puntos en cada observación"),
         Q("desde", "Primera fecha (nº de observación) en la que puede autocancelar", "2", TipoDato::Entero),
         Q("barrera_capital", "Barrera de capital (% del nivel inicial)", "60%", TipoDato::Porcentaje,
           "si se toca (según la observación elegida) se pierde capital como el peor subyacente"),
         Q("observacion_capital", "Observación de la barrera de capital", "vencimiento", TipoDato::Observacion,
           AYUDA_OBS)},
        [](json::Valor& doc, const Respuestas& r) {
            json::Valor obs = obj();
            obs.poner("frecuencia", r.at("frecuencia"));
            doc.poner("observaciones", obs);
            json::Valor c = obj();
            c.poner("tipo", "condicional");
            c.poner("importe", P(r, "cupon"));
            c.poner("barrera", P(r, "barrera_cupon"));
            c.poner("memoria", si(r, "memoria"));
            doc.poner("cupones", c);
            json::Valor ac = obj();
            ac.poner("barrera", P(r, "barrera_autocall"));
            if (pct(r, "escalon") != 0.0) ac.poner("escalon", P(r, "escalon"));
            ac.poner("desde", std::stoi(r.at("desde")));
            doc.poner("autocancelacion", ac);
            doc.poner("barrera_capital", barrera_capital(r, "barrera_capital", "observacion_capital"));
            json::Valor am = json::Valor::lista();
            am.anadir(capital_100());
            am.anadir(perdida_bajo_strike("100%", true));
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) {
            return "Cupón condicional del " + r.at("cupon") + " " + r.at("frecuencia") + " (barrera " +
                   r.at("barrera_cupon") + (r.at("memoria") == "si" ? ", con memoria" : ", sin memoria") +
                   "); autocancelable si P ≥ " + r.at("barrera_autocall") + " desde la observación " + r.at("desde") +
                   "; barrera de capital del " + r.at("barrera_capital") + " (" + r.at("observacion_capital") + ").";
        }});

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "athena", "Autocallable Athena / Express (prima acumulada al autocancelar)",
        "No paga cupones periódicos: si en una fecha la cesta supera la barrera de autocancelación, se amortiza "
        "pagando el nominal más una prima acumulada (prima × nº de periodos). A vencimiento, como el Phoenix.",
        dos, "3A",
        {Q("frecuencia", "Frecuencia de observación", "semestral", TipoDato::Frecuencia),
         Q("prima", "Prima por periodo transcurrido (% del nominal)", "4%", TipoDato::Porcentaje,
           "al autocancelar en la observación j se cobra nominal + j × prima"),
         Q("barrera_autocall", "Barrera de autocancelación (% del nivel inicial)", "100%", TipoDato::Porcentaje),
         Q("escalon", "Variación de la barrera de autocancelación en cada fecha (step-down)", "0%", TipoDato::Porcentaje),
         Q("desde", "Primera fecha en la que puede autocancelar", "1", TipoDato::Entero),
         Q("barrera_capital", "Barrera de capital (% del nivel inicial)", "60%", TipoDato::Porcentaje),
         Q("observacion_capital", "Observación de la barrera de capital", "vencimiento", TipoDato::Observacion,
           AYUDA_OBS)},
        [](json::Valor& doc, const Respuestas& r) {
            json::Valor obs = obj();
            obs.poner("frecuencia", r.at("frecuencia"));
            doc.poner("observaciones", obs);
            json::Valor ac = obj();
            ac.poner("barrera", P(r, "barrera_autocall"));
            if (pct(r, "escalon") != 0.0) ac.poner("escalon", P(r, "escalon"));
            ac.poner("desde", std::stoi(r.at("desde")));
            ac.poner("prima", P(r, "prima"));
            ac.poner("acumulativa", true);
            doc.poner("autocancelacion", ac);
            doc.poner("barrera_capital", barrera_capital(r, "barrera_capital", "observacion_capital"));
            json::Valor am = json::Valor::lista();
            am.anadir(capital_100());
            am.anadir(perdida_bajo_strike("100%", true));
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) {
            return "Autocancelable " + r.at("frecuencia") + " si P ≥ " + r.at("barrera_autocall") +
                   " pagando una prima acumulada del " + r.at("prima") + " por periodo; barrera de capital del " +
                   r.at("barrera_capital") + " (" + r.at("observacion_capital") + ").";
        }});

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "reverse_convertible", "Reverse convertible (cupón fijo, capital en riesgo)",
        "Paga un cupón fijo pase lo que pase. A vencimiento devuelve el nominal si el subyacente no ha caído por "
        "debajo de la barrera (o del strike, si no hay barrera); si no, el inversor recibe P/strike del nominal.",
        uno, "1A",
        {Q("frecuencia", "Frecuencia de pago del cupón", "anual", TipoDato::Frecuencia),
         Q("tasa_cupon", "Cupón fijo anual (%)", "8%", TipoDato::Porcentaje),
         Q("strike", "Strike: nivel por debajo del cual se pierde capital (%)", "100%", TipoDato::Porcentaje),
         Q("con_barrera", "¿Tiene barrera de capital (barrier reverse convertible)?", "si", TipoDato::SiNo),
         Q("barrera", "Barrera de capital (% del nivel inicial)", "70%", TipoDato::Porcentaje, "", {},
           "con_barrera=si"),
         Q("observacion", "Observación de la barrera", "continua", TipoDato::Observacion, AYUDA_OBS, {},
           "con_barrera=si")},
        [](json::Valor& doc, const Respuestas& r) {
            json::Valor obs = obj();
            obs.poner("frecuencia", r.at("frecuencia"));
            doc.poner("observaciones", obs);
            json::Valor c = obj();
            c.poner("tipo", "fijo");
            c.poner("tasa_anual", P(r, "tasa_cupon"));
            doc.poner("cupones", c);
            bool b = si(r, "con_barrera");
            if (b) doc.poner("barrera_capital", barrera_capital(r, "barrera", "observacion"));
            json::Valor am = json::Valor::lista();
            am.anadir(capital_100());
            am.anadir(perdida_bajo_strike(r.at("strike"), b));
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) {
            std::string s = "Cupón fijo del " + r.at("tasa_cupon") + " anual pagadero de forma " + r.at("frecuencia") +
                            "; ";
            if (r.at("con_barrera") == "si")
                s += "si se toca la barrera del " + r.at("barrera") + " (" + r.at("observacion") +
                     ") se recibe P/strike del nominal (strike " + r.at("strike") + ").";
            else
                s += "a vencimiento se recibe min(100%, P/strike) del nominal (strike " + r.at("strike") + ").";
            return s;
        }});

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "capital_protegido", "Nota con capital protegido (garantía + participación)",
        "Garantiza un porcentaje del nominal a vencimiento y añade una participación en la subida del "
        "subyacente, con tope opcional. El rendimiento puede ser el final o la media de varias fechas.",
        uno, "5A",
        {Q("proteccion", "Capital garantizado a vencimiento (%)", "100%", TipoDato::Porcentaje),
         Q("participacion", "Participación en la subida (%)", "80%", TipoDato::Porcentaje),
         Q("strike", "Strike de la participación (% del nivel inicial)", "100%", TipoDato::Porcentaje),
         Q("techo", "Tope para el rendimiento P (%, o 'no')", "no", TipoDato::PctOpcional,
           "p. ej. 150 limita la ganancia a participación × (150% - strike)"),
         Q("rendimiento", "Rendimiento final: 'final' o 'medio' (media asiática)", "final", TipoDato::Opcion, "",
           {"final", "medio"}),
         Q("frecuencia", "Frecuencia de las observaciones que se promedian", "anual", TipoDato::Frecuencia, "", {},
           "rendimiento=medio")},
        [](json::Valor& doc, const Respuestas& r) {
            bool medio = r.at("rendimiento") == "medio";
            if (medio) {
                json::Valor obs = obj();
                obs.poner("frecuencia", r.at("frecuencia"));
                doc.poner("observaciones", obs);
            }
            json::Valor f = bloque("fijo");
            f.poner("importe", P(r, "proteccion"));
            json::Valor c = bloque("call");
            c.poner("strike", P(r, "strike"));
            c.poner("participacion", P(r, "participacion"));
            if (hay(r, "techo")) c.poner("techo", P(r, "techo"));
            json::Valor comp = json::Valor::lista();
            comp.anadir(f);
            comp.anadir(c);
            if (medio) {
                json::Valor am = obj();
                am.poner("rendimiento", "medio");
                am.poner("componentes", comp);
                doc.poner("amortizacion_final", am);
            } else {
                doc.poner("amortizacion_final", comp);
            }
        },
        [](const Respuestas& r) {
            return "Garantía del " + r.at("proteccion") + " del nominal más el " + r.at("participacion") +
                   " de la subida por encima del " + r.at("strike") +
                   (r.at("techo") != "no" ? " (rendimiento topado en " + r.at("techo") + ")" : std::string("")) +
                   (r.at("rendimiento") == "medio" ? ", sobre la media " + r.at("frecuencia") + " del subyacente." : ".");
        }});

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "bonus", "Certificado bonus (con tope)",
        "Si el subyacente nunca toca la barrera, se cobra al menos el nivel de bonus (y la subida, hasta el "
        "tope). Si la toca, el certificado replica al subyacente (hasta el tope).",
        uno, "2A",
        {Q("bonus", "Nivel de bonus (% del nivel inicial)", "120%", TipoDato::Porcentaje),
         Q("barrera", "Barrera (% del nivel inicial)", "70%", TipoDato::Porcentaje),
         Q("observacion", "Observación de la barrera", "continua", TipoDato::Observacion, AYUDA_OBS),
         Q("tope", "Tope del pago (%, o 'no')", "150%", TipoDato::PctOpcional)},
        [](json::Valor& doc, const Respuestas& r) {
            doc.poner("barrera_capital", barrera_capital(r, "barrera", "observacion"));
            json::Valor am = json::Valor::lista();
            json::Valor f = bloque("fijo");
            f.poner("importe", P(r, "bonus"));
            f.poner("condicion", "barrera_no_tocada");
            json::Valor c = bloque("call");
            c.poner("strike", P(r, "bonus"));
            if (hay(r, "tope")) c.poner("techo", P(r, "tope"));
            c.poner("condicion", "barrera_no_tocada");
            json::Valor l = bloque("lineal");
            if (hay(r, "tope")) l.poner("techo", P(r, "tope"));
            l.poner("condicion", "barrera_tocada");
            am.anadir(f);
            am.anadir(c);
            am.anadir(l);
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) {
            return "Bonus del " + r.at("bonus") + " si no se toca la barrera del " + r.at("barrera") + " (" +
                   r.at("observacion") + ")" + (r.at("tope") != "no" ? ", tope " + r.at("tope") : std::string("")) + ".";
        }});

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "twin_win", "Twin-win (gana con subidas y con caídas moderadas)",
        "Si no se toca la barrera, a vencimiento se cobra 100% + participación × |P - 100%| (gana tanto si sube "
        "como si baja). Si se toca, replica al subyacente.",
        uno, "3A",
        {Q("part_alcista", "Participación en la subida (%)", "100%", TipoDato::Porcentaje),
         Q("part_bajista", "Participación en la caída (convertida en ganancia) (%)", "100%", TipoDato::Porcentaje),
         Q("barrera", "Barrera (% del nivel inicial)", "60%", TipoDato::Porcentaje),
         Q("observacion", "Observación de la barrera", "continua", TipoDato::Observacion, AYUDA_OBS),
         Q("tope", "Tope del rendimiento P (%, o 'no')", "no", TipoDato::PctOpcional)},
        [](json::Valor& doc, const Respuestas& r) {
            doc.poner("barrera_capital", barrera_capital(r, "barrera", "observacion"));
            json::Valor am = json::Valor::lista();
            json::Valor f = capital_100();
            f.poner("condicion", "barrera_no_tocada");
            json::Valor c = bloque("call");
            c.poner("participacion", P(r, "part_alcista"));
            if (hay(r, "tope")) c.poner("techo", P(r, "tope"));
            c.poner("condicion", "barrera_no_tocada");
            json::Valor pu = bloque("put");
            pu.poner("participacion", P(r, "part_bajista"));
            pu.poner("condicion", "barrera_no_tocada");
            json::Valor l = bloque("lineal");
            if (hay(r, "tope")) l.poner("techo", P(r, "tope"));
            l.poner("condicion", "barrera_tocada");
            am.anadir(f);
            am.anadir(c);
            am.anadir(pu);
            am.anadir(l);
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) {
            return "Twin-win: " + r.at("part_alcista") + " de la subida y " + r.at("part_bajista") +
                   " de la caída si no se toca la barrera del " + r.at("barrera") + " (" + r.at("observacion") + ").";
        }});

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "shark_fin", "Shark fin (capital protegido con barrera desactivante)",
        "Garantiza el capital y participa en la subida, pero si el subyacente toca la barrera superior la "
        "participación desaparece y se cobra un importe fijo (reembolso).",
        uno, "1A",
        {Q("proteccion", "Capital garantizado (%)", "100%", TipoDato::Porcentaje),
         Q("participacion", "Participación en la subida (%)", "100%", TipoDato::Porcentaje),
         Q("barrera_ko", "Barrera desactivante (% del nivel inicial)", "130%", TipoDato::Porcentaje),
         Q("observacion", "Observación de la barrera desactivante", "continua", TipoDato::Observacion, AYUDA_OBS),
         Q("reembolso", "Importe que se cobra a vencimiento si se desactiva (%)", "105%", TipoDato::Porcentaje)},
        [](json::Valor& doc, const Respuestas& r) {
            json::Valor ko = obj();
            ko.poner("nivel", P(r, "barrera_ko"));
            ko.poner("observacion", r.at("observacion"));
            ko.poner("importe", P(r, "reembolso"));
            ko.poner("pago", "vencimiento");
            doc.poner("barrera_desactivante", ko);
            json::Valor am = json::Valor::lista();
            json::Valor f = bloque("fijo");
            f.poner("importe", P(r, "proteccion"));
            json::Valor c = bloque("call");
            c.poner("participacion", P(r, "participacion"));
            am.anadir(f);
            am.anadir(c);
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) {
            return "Garantía del " + r.at("proteccion") + " más el " + r.at("participacion") +
                   " de la subida; si se toca el " + r.at("barrera_ko") + " (" + r.at("observacion") + ") se cobra el " +
                   r.at("reembolso") + " a vencimiento.";
        }});

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "range_accrual", "Nota range accrual (cupón por días dentro de un rango)",
        "El cupón de cada periodo se devenga en proporción a los días en que el subyacente cotiza dentro del "
        "rango [inferior, superior]. Capital protegido a vencimiento.",
        uno, "2A",
        {Q("frecuencia", "Frecuencia de pago del cupón", "trimestral", TipoDato::Frecuencia),
         Q("tasa", "Cupón anual máximo (si todos los días están en rango) (%)", "6%", TipoDato::Porcentaje),
         Q("inferior", "Límite inferior del rango (% del nivel inicial)", "90%", TipoDato::Porcentaje),
         Q("superior", "Límite superior del rango (% del nivel inicial)", "110%", TipoDato::Porcentaje),
         Q("proteccion", "Capital garantizado a vencimiento (%)", "100%", TipoDato::Porcentaje)},
        [](json::Valor& doc, const Respuestas& r) {
            json::Valor obs = obj();
            obs.poner("frecuencia", r.at("frecuencia"));
            doc.poner("observaciones", obs);
            json::Valor c = obj();
            c.poner("tipo", "rango");
            c.poner("tasa_anual", P(r, "tasa"));
            json::Valor rg = json::Valor::lista();
            rg.anadir(P(r, "inferior"));
            rg.anadir(P(r, "superior"));
            c.poner("rango", rg);
            doc.poner("cupones", c);
            json::Valor am = json::Valor::lista();
            json::Valor f = bloque("fijo");
            f.poner("importe", P(r, "proteccion"));
            am.anadir(f);
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) {
            return "Cupón de hasta el " + r.at("tasa") + " anual (" + r.at("frecuencia") + ") devengado por cada día con P en [" +
                   r.at("inferior") + ", " + r.at("superior") + "]; capital garantizado del " + r.at("proteccion") + ".";
        }});

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "cliquet", "Nota cliquet con capital garantizado",
        "Acumula los rendimientos de cada periodo, cada uno acotado entre un suelo y un techo locales; a "
        "vencimiento paga el capital garantizado más la suma acotada por un suelo global.",
        uno, "5A",
        {Q("frecuencia", "Frecuencia de los periodos (reseteo)", "anual", TipoDato::Frecuencia),
         Q("techo_local", "Techo de cada rendimiento periódico (%)", "8%", TipoDato::Porcentaje),
         Q("suelo_local", "Suelo de cada rendimiento periódico (%)", "0%", TipoDato::Porcentaje),
         Q("suelo_global", "Suelo de la suma de rendimientos (%)", "0%", TipoDato::Porcentaje),
         Q("techo_global", "Techo de la suma de rendimientos (%, o 'no')", "no", TipoDato::PctOpcional),
         Q("participacion", "Participación sobre la suma (%)", "100%", TipoDato::Porcentaje),
         Q("proteccion", "Capital garantizado a vencimiento (%)", "100%", TipoDato::Porcentaje)},
        [](json::Valor& doc, const Respuestas& r) {
            json::Valor obs = obj();
            obs.poner("frecuencia", r.at("frecuencia"));
            doc.poner("observaciones", obs);
            json::Valor am = json::Valor::lista();
            json::Valor f = bloque("fijo");
            f.poner("importe", P(r, "proteccion"));
            json::Valor c = bloque("cliquet");
            c.poner("participacion", P(r, "participacion"));
            c.poner("suelo_local", P(r, "suelo_local"));
            c.poner("techo_local", P(r, "techo_local"));
            c.poner("suelo_global", P(r, "suelo_global"));
            if (hay(r, "techo_global")) c.poner("techo_global", P(r, "techo_global"));
            am.anadir(f);
            am.anadir(c);
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) {
            return "Cliquet " + r.at("frecuencia") + " con rendimientos acotados a [" + r.at("suelo_local") + ", " +
                   r.at("techo_local") + "], suelo global " + r.at("suelo_global") + "; capital garantizado del " +
                   r.at("proteccion") + ".";
        }});

    // ------------------------------------------------------------------ //
    Plantilla op{
        "opcion", "Opción europea (call, put o digital)",
        "Opción europea sobre un subyacente. El nominal se iguala al spot, así que el precio es el de una opción "
        "sobre una unidad del subyacente.",
        uno, "1A",
        {Q("tipo_opcion", "Tipo de opción (call, put, digital_call, digital_put)", "call", TipoDato::Opcion, "",
           {"call", "put", "digital_call", "digital_put"}),
         Q("strike", "Strike (% del nivel inicial)", "100%", TipoDato::Porcentaje),
         Q("importe", "Pago de la digital (% del nominal)", "100%", TipoDato::Porcentaje, "", {},
           "tipo_opcion=digital_call|digital_put")},
        [](json::Valor& doc, const Respuestas& r) {
            const std::string& t = r.at("tipo_opcion");
            json::Valor b = bloque(t == "digital_call" ? "digital" : t);
            b.poner("strike", P(r, "strike"));
            if (t == "digital_call" || t == "digital_put") b.poner("importe", P(r, "importe"));
            json::Valor am = json::Valor::lista();
            am.anadir(b);
            doc.poner("amortizacion_final", am);
        },
        [](const Respuestas& r) { return "Opción " + r.at("tipo_opcion") + " europea con strike " + r.at("strike") + "."; },
        true};
    v.push_back(op);

    // ------------------------------------------------------------------ //
    v.push_back(Plantilla{
        "personalizado", "Personalizado (todas las piezas, para editar a mano)",
        "Ejemplo con todas las secciones del formato: varias patas de cupón, autocancelación con step-down, "
        "barrera de capital continua, componentes finales y estado de un producto vivo.",
        dos, "3A", {},
        [](json::Valor& doc, const Respuestas&) {
            json::Valor obs = obj();
            obs.poner("frecuencia", "trimestral");
            doc.poner("observaciones", obs);
            json::Valor cups = json::Valor::lista();
            json::Valor c1 = obj();
            c1.poner("tipo", "condicional");
            c1.poner("importe", "1.5%");
            c1.poner("barrera", "75%");
            c1.poner("memoria", true);
            json::Valor c2 = obj();
            c2.poner("tipo", "fijo");
            c2.poner("tasa_anual", "1%");
            c2.poner("hasta", 4);
            cups.anadir(c1);
            cups.anadir(c2);
            doc.poner("cupones", cups);
            json::Valor ac = obj();
            ac.poner("barrera", "100%");
            ac.poner("escalon", "-2.5%");
            ac.poner("desde", 4);
            ac.poner("importe", "100%");
            ac.poner("prima", "0%");
            doc.poner("autocancelacion", ac);
            json::Valor ki = obj();
            ki.poner("nivel", "55%");
            ki.poner("observacion", "continua");
            ki.poner("correccion_bgk", true);
            doc.poner("barrera_capital", ki);
            json::Valor am = obj();
            am.poner("rendimiento", "final");
            json::Valor comp = json::Valor::lista();
            comp.anadir(capital_100());
            comp.anadir(perdida_bajo_strike("100%", true));
            json::Valor dg = bloque("digital");
            dg.poner("strike", "100%");
            dg.poner("importe", "5%");
            dg.poner("condicion", "barrera_no_tocada");
            comp.anadir(dg);
            am.poner("componentes", comp);
            doc.poner("amortizacion_final", am);
            json::Valor est = obj();
            est.poner("barrera_capital_tocada", false);
            est.poner("cupones_en_memoria", "0%");
            doc.poner("estado", est);
        },
        [](const Respuestas&) {
            return std::string("Producto de ejemplo con todas las secciones del formato.");
        }});
    return v;
}

} // namespace

const std::vector<Plantilla>& plantillas() {
    static const std::vector<Plantilla> v = construir_plantillas();
    return v;
}

const Plantilla* buscar_plantilla(const std::string& id) {
    std::string b = minus(recorta(id));
    for (auto& c : b) if (c == '-' || c == ' ') c = '_';
    for (const auto& p : plantillas())
        if (p.id == b) return &p;
    // alias habituales
    if (b == "autocall" || b == "autocallable") return buscar_plantilla("phoenix");
    if (b == "express") return buscar_plantilla("athena");
    if (b == "rc" || b == "brc" || b == "reverse") return buscar_plantilla("reverse_convertible");
    if (b == "garantizado" || b == "cpn") return buscar_plantilla("capital_protegido");
    if (b == "call" || b == "put" || b == "vanilla") return buscar_plantilla("opcion");
    return nullptr;
}

json::Valor modelo_black_scholes(const std::string& vol) {
    json::Valor m = json::Valor::objeto();
    m.poner("tipo", "black_scholes");
    m.poner("volatilidad", vol);
    return m;
}

DatosComunes comunes_por_defecto(const Plantilla& p) {
    DatosComunes c;
    c.nombre = p.titulo;
    c.vencimiento = p.vencimiento_def;
    for (const auto& s : p.subyacentes_def) {
        json::Valor a = json::Valor::objeto();
        a.poner("nombre", s.nombre);
        a.poner("spot", s.spot);
        a.poner("dividendos", s.dividendos);
        a.poner("modelo", modelo_black_scholes(s.vol));
        c.subyacentes.anadir(a);
    }
    if (p.subyacentes_def.size() > 1) c.correlacion = json::Valor("60%");
    if (p.nominal_es_spot && !p.subyacentes_def.empty()) c.nominal = p.subyacentes_def[0].spot;
    return c;
}

bool normalizar_respuesta(const Pregunta& q, const std::string& texto0, std::string& can, std::string& err) {
    std::string t = recorta(texto0);
    std::string tl = minus(t);
    bool ok = false;
    switch (q.tipo) {
    case TipoDato::Texto:
        can = t;
        return true;
    case TipoDato::Real: {
        double x = texto_a_real(t, &ok);
        if (!ok) { err = "escriba un número"; return false; }
        can = fmt(x, 10);
        return true;
    }
    case TipoDato::PctOpcional:
        if (tl.empty() || tl == "no" || tl == "n" || tl == "sin" || tl == "sin_tope" || tl == "-") {
            can = "no";
            return true;
        }
        [[fallthrough]];
    case TipoDato::Porcentaje: {
        double x;
        if (!t.empty() && t.back() == '%') x = texto_a_real(t, &ok);
        else x = texto_a_real(t, &ok) / 100.0; // en el asistente los números van en puntos porcentuales
        if (!ok) { err = "escriba un porcentaje, p. ej. 70 o 70%"; return false; }
        can = porcentaje(x, 4);
        return true;
    }
    case TipoDato::Plazo: {
        double y = texto_a_plazo(t, &ok);
        if (!ok || !(y > 0)) { err = "escriba un plazo positivo: 3A, 18M, 2.5 (años)..."; return false; }
        can = t;
        return true;
    }
    case TipoDato::Entero: {
        double x = texto_a_real(t, &ok);
        if (!ok || std::fabs(x - std::round(x)) > 1e-9) { err = "escriba un número entero"; return false; }
        can = std::to_string((long long)std::llround(x));
        return true;
    }
    case TipoDato::SiNo:
        if (tl == "s" || tl == "si" || tl == "sí" || tl == "y" || tl == "yes" || tl == "1") { can = "si"; return true; }
        if (tl == "n" || tl == "no" || tl == "0") { can = "no"; return true; }
        err = "responda s (sí) o n (no)";
        return false;
    case TipoDato::Frecuencia: {
        int f = texto_a_frecuencia(tl);
        if (f <= 0) { err = "use mensual, bimestral, trimestral, cuatrimestral, semestral, anual o un plazo como 3M"; return false; }
        switch (f) {
            case 12: can = "mensual"; break;
            case 6:  can = "bimestral"; break;
            case 4:  can = "trimestral"; break;
            case 3:  can = "cuatrimestral"; break;
            case 2:  can = "semestral"; break;
            case 1:  can = "anual"; break;
            default: can = tl; break;
        }
        return true;
    }
    case TipoDato::Observacion: {
        static const std::vector<std::string> v = {"vencimiento", "fechas", "diaria", "continua"};
        for (const auto& o : v)
            if (!tl.empty() && o.rfind(tl, 0) == 0) { can = o; return true; }
        err = "use vencimiento, fechas, diaria o continua";
        return false;
    }
    case TipoDato::Opcion: {
        // número de opción o texto (o su comienzo)
        bool okn = false;
        double x = texto_a_real(tl, &okn);
        if (okn && x >= 1 && x <= (double)q.opciones.size() && std::fabs(x - std::round(x)) < 1e-9) {
            can = q.opciones[(size_t)std::llround(x) - 1];
            return true;
        }
        for (const auto& o : q.opciones)
            if (o == tl) { can = o; return true; }
        for (const auto& o : q.opciones)
            if (!tl.empty() && o.rfind(tl, 0) == 0) { can = o; return true; }
        err = "opciones válidas:";
        for (const auto& o : q.opciones) err += " " + o;
        return false;
    }
    }
    return false;
}

json::Valor a_json(TipoDato t, const std::string& c) {
    switch (t) {
        case TipoDato::Real:
        case TipoDato::Entero: return json::Valor(texto_a_real(c));
        case TipoDato::SiNo:   return json::Valor(c == "si");
        default:               return json::Valor(c);
    }
}

json::Valor construir_producto(const Plantilla& p, const DatosComunes& c, const Respuestas& r_in) {
    // Respuestas completas (valor por defecto donde falte), ya normalizadas
    Respuestas r;
    for (const auto& q : p.preguntas) {
        auto it = r_in.find(q.clave);
        std::string can, err;
        if (it != r_in.end()) {
            r[q.clave] = it->second;
        } else {
            std::string def = q.defecto;
            normalizar_respuesta(q, def, can, err);
            r[q.clave] = can.empty() ? def : can;
        }
    }

    json::Valor doc = json::Valor::objeto();
    doc.poner("formato", "producto-estructurado/1");
    doc.poner("plantilla", p.id);
    doc.poner("nombre", c.nombre.empty() ? p.titulo : c.nombre);
    std::string desc = c.descripcion;
    if (desc.empty() && p.describir) desc = p.describir(r);
    if (!desc.empty()) doc.poner("descripcion", desc);
    doc.poner("nominal", c.nominal);
    doc.poner("moneda", c.moneda);
    bool num_ok = false;
    double venc = texto_a_real(c.vencimiento, &num_ok);
    doc.poner("vencimiento", num_ok ? json::Valor(venc) : json::Valor(c.vencimiento));
    json::Valor mer = json::Valor::objeto();
    mer.poner("tipo_interes", c.tipo_interes);
    doc.poner("mercado", mer);
    doc.poner("subyacentes", c.subyacentes);
    if (c.subyacentes.tamano() > 1) {
        doc.poner("correlacion", c.correlacion.es_nulo() ? json::Valor("50%") : c.correlacion);
        doc.poner("cesta", c.cesta);
    }
    if (p.construir) p.construir(doc, r);
    if (!c.simulacion.es_nulo()) doc.poner("simulacion", c.simulacion);
    return doc;
}

} // namespace est
