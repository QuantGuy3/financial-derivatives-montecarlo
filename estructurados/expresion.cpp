// Compilador de fórmulas de pago (análisis descendente recursivo -> máquina de pila)
//
//   comparacion := suma (( < | <= | > | >= | == | != ) suma)*
//   suma        := termino (( + | - ) termino)*
//   termino     := unario (( * | / ) unario)*
//   unario      := ( - | + ) unario | potencia        (-P^2 = -(P^2))
//   potencia    := primario ( ^ unario )?             (a^b^c = a^(b^c))
//   primario    := número [% | pb] | variable | función ( args ) | ( comparacion )

#include "expresion.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace est {

namespace {

std::string minus(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

bool es_letra(unsigned char c) { return std::isalpha(c) || c == '_' || c >= 0x80; }
bool es_alnum(unsigned char c) { return std::isalnum(c) || c == '_' || c >= 0x80; }

class Compilador {
public:
    Compilador(const std::string& s, double escala) : s_(s), escala_(escala) {}

    Formula compilar() {
        blancos();
        if (i_ >= s_.size()) error("la fórmula está vacía", 0);
        comparacion();
        blancos();
        if (i_ < s_.size()) {
            if (s_[i_] == ')') error("sobra un paréntesis de cierre", i_);
            if (s_[i_] == ',') error("coma fuera de una función (los decimales se escriben con punto: 1.5)", i_);
            error("se esperaba un operador (+ - * / ^) antes de esto", i_);
        }
        Formula f;
        f.texto = s_;
        f.codigo = std::move(cod_);
        f.profundidad = prof_max_;
        f.usa_P = usa_P_;
        if (f.profundidad > MAX_PILA_FORMULA)
            error("la fórmula es demasiado larga o anidada (simplifíquela)", 0);
        return f;
    }

private:
    const std::string& s_;
    double escala_;
    size_t i_ = 0;
    std::vector<InstrFormula<double>> cod_;
    int prof_ = 0, prof_max_ = 0;
    bool usa_P_ = false;

    [[noreturn]] void error(const std::string& m, size_t pos) const {
        // Señala la posición con un acento circunflejo bajo el texto
        size_t col = 0;
        for (size_t k = 0; k < pos && k < s_.size(); ++k) col += ((unsigned char)s_[k] & 0xC0) != 0x80;
        throw ErrorFormula(m + "\n      " + s_ + "\n      " + std::string(col, ' ') + "^", pos);
    }

    void emitir(int op, double v = 0.0) {
        cod_.push_back(InstrFormula<double>{op, v});
        if (op == FX_CONST || op == FX_P) ++prof_;
        else if (op == FX_SI) prof_ -= 2;
        else if (op >= FX_SUMA) --prof_;
        prof_max_ = std::max(prof_max_, prof_);
    }

    void blancos() {
        while (i_ < s_.size() && std::isspace((unsigned char)s_[i_])) ++i_;
    }
    bool ver(char c) {
        blancos();
        return i_ < s_.size() && s_[i_] == c;
    }
    bool tomar(const char* tok) {
        blancos();
        size_t n = std::char_traits<char>::length(tok);
        if (s_.compare(i_, n, tok) == 0) { i_ += n; return true; }
        return false;
    }
    void exigir(char c, const char* que) {
        if (!ver(c)) error(std::string("se esperaba ") + que, i_);
        ++i_;
    }

    void comparacion() {
        suma();
        for (;;) {
            int op;
            if (tomar("<=")) op = FX_MENOR_IGUAL;
            else if (tomar(">=")) op = FX_MAYOR_IGUAL;
            else if (tomar("==")) op = FX_IGUAL;
            else if (tomar("!=")) op = FX_DISTINTO;
            else if (tomar("<")) op = FX_MENOR;
            else if (tomar(">")) op = FX_MAYOR;
            else return;
            suma();
            emitir(op);
        }
    }

    void suma() {
        termino();
        for (;;) {
            if (ver('+')) { ++i_; termino(); emitir(FX_SUMA); }
            else if (ver('-')) { ++i_; termino(); emitir(FX_RESTA); }
            else return;
        }
    }

    void termino() {
        unario();
        for (;;) {
            if (ver('*')) { ++i_; unario(); emitir(FX_MUL); }
            else if (ver('/')) { ++i_; unario(); emitir(FX_DIV); }
            else return;
        }
    }

    void unario() {
        if (ver('-')) { ++i_; unario(); emitir(FX_NEG); return; }
        if (ver('+')) { ++i_; unario(); return; }
        potencia();
    }

    void potencia() {
        primario();
        if (ver('^')) { ++i_; unario(); emitir(FX_POT); }
    }

    // Lista de argumentos ya dentro del paréntesis; devuelve cuántos hay
    int argumentos(const std::string& fn, int minimo, int maximo, size_t pos_fn) {
        int n = 0;
        if (ver(')')) { ++i_; }
        else {
            for (;;) {
                comparacion();
                ++n;
                if (ver(',')) { ++i_; continue; }
                exigir(')', "',' o ')' para cerrar los argumentos");
                break;
            }
        }
        if (n < minimo || n > maximo) {
            std::string esperado = (minimo == maximo) ? std::to_string(minimo)
                                 : (maximo > 100 ? "al menos " + std::to_string(minimo)
                                                 : "entre " + std::to_string(minimo) + " y " + std::to_string(maximo));
            error(fn + "() necesita " + esperado + " argumento" + (maximo == 1 && minimo == 1 ? "" : "s") +
                  " y tiene " + std::to_string(n), pos_fn);
        }
        return n;
    }

    void primario() {
        blancos();
        if (i_ >= s_.size()) error("la fórmula termina antes de tiempo (falta un valor)", i_);
        const size_t ini = i_;
        const unsigned char c = (unsigned char)s_[i_];

        if (c == '(') {
            ++i_;
            comparacion();
            exigir(')', "')'");
            return;
        }

        if (std::isdigit(c) || c == '.') {
            const char* p0 = s_.c_str() + i_;
            char* fin = nullptr;
            double v = std::strtod(p0, &fin);
            if (fin == p0) error("número mal escrito", ini);
            i_ += (size_t)(fin - p0);
            if (i_ < s_.size() && s_[i_] == '%') { v /= 100.0; ++i_; }
            else if (i_ + 1 < s_.size() && std::tolower((unsigned char)s_[i_]) == 'p' &&
                     std::tolower((unsigned char)s_[i_ + 1]) == 'b' &&
                     (i_ + 2 >= s_.size() || !es_alnum((unsigned char)s_[i_ + 2]))) {
                v /= 10000.0;
                i_ += 2;
            }
            if (i_ < s_.size() && es_letra((unsigned char)s_[i_]))
                error("falta un operador entre el número y el nombre (escriba 2*P, no 2P)", i_);
            emitir(FX_CONST, v);
            return;
        }

        if (!es_letra(c)) {
            if (c == ',') error("coma inesperada (los decimales se escriben con punto: 1.5)", ini);
            error(std::string("carácter inesperado '") + (char)c + "'", ini);
        }
        while (i_ < s_.size() && es_alnum((unsigned char)s_[i_])) ++i_;
        std::string id = minus(s_.substr(ini, i_ - ini));
        if (id == "raíz") id = "raiz";

        if (!ver('(')) {
            if (id == "p") { usa_P_ = true; emitir(FX_P); return; }
            if (id == "s") {
                if (!(escala_ > 0.0))
                    error("la variable S (precio) solo se admite con un único subyacente; use P (rendimiento, 100% = nivel inicial)", ini);
                usa_P_ = true;
                emitir(FX_P);
                emitir(FX_CONST, escala_);
                emitir(FX_MUL);
                return;
            }
            if (id == "pi") { emitir(FX_CONST, 3.14159265358979323846); return; }
            static const char* funciones[] = {"max", "min", "abs", "exp", "log", "ln", "sqrt", "raiz", "pow", "pot",
                                              "si", "if", "clamp", "acotar", "call", "put", "digital", "digital_put"};
            for (const char* f : funciones)
                if (id == f) error("a " + id + " le faltan los paréntesis: " + id + "(...)", i_);
            error("nombre desconocido \"" + s_.substr(ini, i_ - ini) + "\" (variables: P" +
                  std::string(escala_ > 0.0 ? ", S" : "") + "; funciones: " + ayuda_formulas() + ")", ini);
        }
        ++i_; // '('

        auto unaria = [&](int op) { argumentos(id, 1, 1, ini); emitir(op); };
        if (id == "max" || id == "min") {
            // max(a, b, c) = max(max(a, b), c)
            const int op = (id == "max") ? FX_MAX : FX_MIN;
            int n = 0;
            for (;;) {
                comparacion();
                if (++n >= 2) emitir(op);
                if (ver(',')) { ++i_; continue; }
                exigir(')', "',' o ')' para cerrar los argumentos");
                break;
            }
            if (n < 2) error(id + "() necesita al menos 2 argumentos", ini);
        }
        else if (id == "abs") unaria(FX_ABS);
        else if (id == "exp") unaria(FX_EXP);
        else if (id == "log" || id == "ln") unaria(FX_LOG);
        else if (id == "sqrt" || id == "raiz") unaria(FX_RAIZ);
        else if (id == "pow" || id == "pot") { argumentos(id, 2, 2, ini); emitir(FX_POT); }
        else if (id == "si" || id == "if") { argumentos(id, 3, 3, ini); emitir(FX_SI); }
        else if (id == "clamp" || id == "acotar") {
            // clamp(x, lo, hi) = min(max(x, lo), hi)
            comparacion();
            exigir(',', "',' (clamp necesita 3 argumentos: valor, mínimo, máximo)");
            comparacion();
            emitir(FX_MAX);
            exigir(',', "',' (clamp necesita 3 argumentos: valor, mínimo, máximo)");
            comparacion();
            emitir(FX_MIN);
            exigir(')', "')' (clamp necesita 3 argumentos)");
        }
        else if (id == "call") {         // max(P - k, 0)
            usa_P_ = true;
            emitir(FX_P);
            argumentos(id, 1, 1, ini);
            emitir(FX_RESTA);
            emitir(FX_CONST, 0.0);
            emitir(FX_MAX);
        }
        else if (id == "put") {          // max(k - P, 0)
            usa_P_ = true;
            argumentos(id, 1, 1, ini);
            emitir(FX_P);
            emitir(FX_RESTA);
            emitir(FX_CONST, 0.0);
            emitir(FX_MAX);
        }
        else if (id == "digital" || id == "digital_call") { // 1{P >= k}
            usa_P_ = true;
            emitir(FX_P);
            argumentos(id, 1, 1, ini);
            emitir(FX_MAYOR_IGUAL);
        }
        else if (id == "digital_put") {  // 1{P < k}
            usa_P_ = true;
            emitir(FX_P);
            argumentos(id, 1, 1, ini);
            emitir(FX_MENOR);
        }
        else error("función desconocida \"" + s_.substr(ini, i_ - 1 - ini) + "\" (funciones: " + ayuda_formulas() + ")", ini);
    }
};

} // namespace

const char* ayuda_formulas() {
    return "max, min, abs, exp, log, sqrt, pow, si(c,a,b), clamp(x,lo,hi), call(k), put(k), digital(k), digital_put(k)";
}

Formula compilar_formula(const std::string& texto, double escala_S) {
    Compilador c(texto, escala_S);
    return c.compilar();
}

void comprobar_finita(const Formula& f, double p_max) {
    const int n = 4000;
    for (int k = 0; k <= n; ++k) {
        double P = p_max * k / n;
        double v = evaluar(f, P);
        if (!std::isfinite(v)) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%.4g%%", 100.0 * P);
            throw ErrorFormula("la fórmula no da un valor finito en P = " + std::string(buf) +
                               " (¿logaritmo o raíz de un número negativo, o división por cero? Acote con max(...))",
                               0);
        }
    }
}

} // namespace est
