#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace json {

// ------------------ //
// Acceso a los datos //
// ------------------ //

const char* Valor::nombre_tipo() const {
    switch (tipo_) {
        case Tipo::Nulo:     return "nulo";
        case Tipo::Booleano: return "booleano";
        case Tipo::Numero:   return "número";
        case Tipo::Texto:    return "texto";
        case Tipo::Lista:    return "lista";
        case Tipo::Objeto:   return "objeto";
    }
    return "?";
}

std::string Valor::donde() const {
    if (linea <= 0) return "";
    return " (línea " + std::to_string(linea) + ", columna " + std::to_string(columna) + ")";
}

bool Valor::booleano() const {
    if (tipo_ != Tipo::Booleano) throw Error(std::string("se esperaba un booleano y hay un ") + nombre_tipo() + donde());
    return b_;
}
double Valor::numero() const {
    if (tipo_ != Tipo::Numero) throw Error(std::string("se esperaba un número y hay un ") + nombre_tipo() + donde());
    return n_;
}
const std::string& Valor::texto() const {
    if (tipo_ != Tipo::Texto) throw Error(std::string("se esperaba un texto y hay un ") + nombre_tipo() + donde());
    return s_;
}

size_t Valor::tamano() const {
    if (tipo_ == Tipo::Lista || tipo_ == Tipo::Objeto) return lista_.size();
    return 0;
}
const Valor& Valor::operator[](size_t i) const {
    if (tipo_ != Tipo::Lista) throw Error("acceso por índice a un valor que no es una lista" + donde());
    if (i >= lista_.size()) throw Error("índice fuera de rango" + donde());
    return lista_[i];
}
Valor& Valor::operator[](size_t i) {
    if (tipo_ != Tipo::Lista) throw Error("acceso por índice a un valor que no es una lista" + donde());
    if (i >= lista_.size()) throw Error("índice fuera de rango" + donde());
    return lista_[i];
}
void Valor::anadir(Valor v) {
    if (tipo_ == Tipo::Nulo) tipo_ = Tipo::Lista;
    if (tipo_ != Tipo::Lista) throw Error("anadir() sobre un valor que no es una lista");
    lista_.push_back(std::move(v));
}

const Valor* Valor::buscar(const std::string& clave) const {
    if (tipo_ != Tipo::Objeto) return nullptr;
    for (size_t i = 0; i < claves_.size(); ++i)
        if (claves_[i] == clave) return &lista_[i];
    return nullptr;
}
Valor* Valor::buscar(const std::string& clave) {
    if (tipo_ != Tipo::Objeto) return nullptr;
    for (size_t i = 0; i < claves_.size(); ++i)
        if (claves_[i] == clave) return &lista_[i];
    return nullptr;
}
Valor& Valor::poner(const std::string& clave, Valor v) {
    if (tipo_ == Tipo::Nulo) tipo_ = Tipo::Objeto;
    if (tipo_ != Tipo::Objeto) throw Error("poner() sobre un valor que no es un objeto");
    for (size_t i = 0; i < claves_.size(); ++i)
        if (claves_[i] == clave) { lista_[i] = std::move(v); return lista_[i]; }
    claves_.push_back(clave);
    lista_.push_back(std::move(v));
    return lista_.back();
}
bool Valor::quitar(const std::string& clave) {
    if (tipo_ != Tipo::Objeto) return false;
    for (size_t i = 0; i < claves_.size(); ++i)
        if (claves_[i] == clave) {
            claves_.erase(claves_.begin() + (long)i);
            lista_.erase(lista_.begin() + (long)i);
            return true;
        }
    return false;
}

// ------ //
// Lector //
// ------ //

namespace {

class Parser {
public:
    explicit Parser(const std::string& t) : s_(t) {}

    Valor documento() {
        saltar_espacios();
        Valor v = valor();
        saltar_espacios();
        if (pos_ < s_.size()) fallo("texto sobrante después del final del documento");
        return v;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;
    int linea_ = 1, col_ = 1;

    [[noreturn]] void fallo(const std::string& msg) const {
        throw Error("JSON inválido en línea " + std::to_string(linea_) + ", columna "
                    + std::to_string(col_) + ": " + msg);
    }
    char actual() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }
    char avanzar() {
        char c = s_[pos_++];
        if (c == '\n') { ++linea_; col_ = 1; }
        else if (((unsigned char)c & 0xC0) != 0x80) ++col_; // no contar bytes de continuación UTF-8
        return c;
    }
    void saltar_espacios() {
        for (;;) {
            while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
                                        s_[pos_] == '\n' || s_[pos_] == '\r'))
                avanzar();
            if (pos_ + 1 < s_.size() && s_[pos_] == '/' && s_[pos_ + 1] == '/') {
                while (pos_ < s_.size() && s_[pos_] != '\n') avanzar();
                continue;
            }
            if (pos_ + 1 < s_.size() && s_[pos_] == '/' && s_[pos_ + 1] == '*') {
                avanzar(); avanzar();
                while (pos_ + 1 < s_.size() && !(s_[pos_] == '*' && s_[pos_ + 1] == '/')) avanzar();
                if (pos_ + 1 >= s_.size()) fallo("comentario /* sin cerrar");
                avanzar(); avanzar();
                continue;
            }
            // BOM UTF-8 al principio del fichero
            if (pos_ == 0 && s_.size() >= 3 && (unsigned char)s_[0] == 0xEF &&
                (unsigned char)s_[1] == 0xBB && (unsigned char)s_[2] == 0xBF) {
                pos_ = 3;
                continue;
            }
            break;
        }
    }

    Valor valor() {
        saltar_espacios();
        int l = linea_, c = col_;
        Valor v;
        char ch = actual();
        if (ch == '{')      v = objeto();
        else if (ch == '[') v = lista();
        else if (ch == '"') v = Valor(cadena());
        else if (ch == 't') { literal("true");  v = Valor(true); }
        else if (ch == 'f') { literal("false"); v = Valor(false); }
        else if (ch == 'n') { literal("null");  v = Valor(); }
        else if (ch == '-' || ch == '+' || (ch >= '0' && ch <= '9') || ch == '.') v = Valor(numero());
        else if (ch == '\0') fallo("fin inesperado del texto");
        else fallo(std::string("carácter inesperado '") + ch + "'");
        v.linea = l; v.columna = c;
        return v;
    }

    void literal(const char* lit) {
        for (const char* p = lit; *p; ++p) {
            if (actual() != *p) fallo(std::string("se esperaba '") + lit + "'");
            avanzar();
        }
    }

    double numero() {
        size_t ini = pos_;
        if (actual() == '-' || actual() == '+') avanzar();
        while ((actual() >= '0' && actual() <= '9') || actual() == '.' || actual() == 'e' ||
               actual() == 'E' || actual() == '-' || actual() == '+')
            avanzar();
        std::string t = s_.substr(ini, pos_ - ini);
        char* fin = nullptr;
        double x = std::strtod(t.c_str(), &fin);
        if (fin == t.c_str() || *fin != '\0') fallo("número mal formado '" + t + "'");
        return x;
    }

    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        }
    }
    unsigned hex4() {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = actual();
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else fallo("secuencia \\u mal formada");
            avanzar();
        }
        return v;
    }

    std::string cadena() {
        avanzar(); // comilla inicial
        std::string out;
        for (;;) {
            if (pos_ >= s_.size()) fallo("texto sin comilla de cierre");
            char c = avanzar();
            if (c == '"') break;
            if (c == '\n') fallo("salto de línea dentro de un texto (falta la comilla de cierre)");
            if (c != '\\') { out += c; continue; }
            char e = avanzar();
            switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && actual() == '\\') {
                        avanzar();
                        if (actual() != 'u') fallo("par sustituto UTF-16 incompleto");
                        avanzar();
                        unsigned lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                } break;
                default: fallo(std::string("secuencia de escape desconocida \\") + e);
            }
        }
        return out;
    }

    Valor lista() {
        avanzar(); // [
        Valor v = Valor::lista();
        saltar_espacios();
        if (actual() == ']') { avanzar(); return v; }
        for (;;) {
            v.anadir(valor());
            saltar_espacios();
            if (actual() == ',') {
                avanzar();
                saltar_espacios();
                if (actual() == ']') { avanzar(); break; } // coma final tolerada
                continue;
            }
            if (actual() == ']') { avanzar(); break; }
            fallo("se esperaba ',' o ']' en la lista");
        }
        return v;
    }

    Valor objeto() {
        avanzar(); // {
        Valor v = Valor::objeto();
        saltar_espacios();
        if (actual() == '}') { avanzar(); return v; }
        for (;;) {
            saltar_espacios();
            if (actual() != '"') fallo("se esperaba el nombre de una clave entre comillas");
            std::string k = cadena();
            saltar_espacios();
            if (actual() != ':') fallo("se esperaba ':' después de la clave \"" + k + "\"");
            avanzar();
            if (v.contiene(k)) fallo("la clave \"" + k + "\" está repetida");
            v.poner(k, valor());
            saltar_espacios();
            if (actual() == ',') {
                avanzar();
                saltar_espacios();
                if (actual() == '}') { avanzar(); break; } // coma final tolerada
                continue;
            }
            if (actual() == '}') { avanzar(); break; }
            fallo("se esperaba ',' o '}' en el objeto");
        }
        return v;
    }
};

// -------- //
// Escritor //
// -------- //

void escapar(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else out += (char)c;
        }
    }
    out += '"';
}

bool es_escalar(const Valor& v) { return !v.es_lista() && !v.es_objeto(); }

// ¿Cabe la lista en una sola línea? (escalares o listas de escalares cortas)
bool lista_en_linea(const Valor& v) {
    if (!v.es_lista()) return false;
    if (v.tamano() > 16) return false;
    for (const auto& e : v.elementos()) {
        if (es_escalar(e)) continue;
        if (e.es_lista() && e.tamano() <= 16) {
            for (const auto& f : e.elementos()) if (!es_escalar(f)) return false;
            continue;
        }
        return false;
    }
    return true;
}

void escribir_rec(std::string& out, const Valor& v, int sangria, int nivel);

void escribir_en_linea(std::string& out, const Valor& v) {
    if (v.es_lista()) {
        out += '[';
        for (size_t i = 0; i < v.tamano(); ++i) {
            if (i) out += ", ";
            escribir_en_linea(out, v[i]);
        }
        out += ']';
        return;
    }
    escribir_rec(out, v, 0, 0);
}

void escribir_rec(std::string& out, const Valor& v, int sangria, int nivel) {
    std::string pad((size_t)(sangria * (nivel + 1)), ' ');
    std::string pad_fin((size_t)(sangria * nivel), ' ');
    switch (v.tipo()) {
        case Valor::Tipo::Nulo:     out += "null"; break;
        case Valor::Tipo::Booleano: out += v.booleano() ? "true" : "false"; break;
        case Valor::Tipo::Numero:   out += numero_a_texto(v.numero()); break;
        case Valor::Tipo::Texto:    escapar(out, v.texto()); break;
        case Valor::Tipo::Lista:
            if (v.tamano() == 0) { out += "[]"; break; }
            if (lista_en_linea(v)) { escribir_en_linea(out, v); break; }
            out += "[\n";
            for (size_t i = 0; i < v.tamano(); ++i) {
                out += pad;
                escribir_rec(out, v[i], sangria, nivel + 1);
                if (i + 1 < v.tamano()) out += ',';
                out += '\n';
            }
            out += pad_fin + "]";
            break;
        case Valor::Tipo::Objeto:
            if (v.tamano() == 0) { out += "{}"; break; }
            out += "{\n";
            for (size_t i = 0; i < v.tamano(); ++i) {
                out += pad;
                escapar(out, v.claves()[i]);
                out += ": ";
                escribir_rec(out, v.valores()[i], sangria, nivel + 1);
                if (i + 1 < v.tamano()) out += ',';
                out += '\n';
            }
            out += pad_fin + "}";
            break;
    }
}

} // namespace

std::string numero_a_texto(double x) {
    if (!std::isfinite(x)) return "null";
    if (x == std::floor(x) && std::fabs(x) < 1e15) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.0f", x);
        return buf;
    }
    char buf[40];
    for (int prec = 6; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, x);
        if (std::strtod(buf, nullptr) == x) break;
    }
    return buf;
}

Valor parsear(const std::string& texto) {
    Parser p(texto);
    return p.documento();
}

std::string escribir(const Valor& v, int sangria) {
    std::string out;
    escribir_rec(out, v, sangria, 0);
    out += '\n';
    return out;
}

Valor leer_fichero(const std::string& ruta) {
    std::ifstream f(ruta, std::ios::binary);
    if (!f) throw Error("no se puede abrir el fichero '" + ruta + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    try {
        return parsear(ss.str());
    } catch (const Error& e) {
        throw Error(ruta + ": " + e.what());
    }
}

void escribir_fichero(const std::string& ruta, const Valor& v) {
    std::ofstream f(ruta, std::ios::binary);
    if (!f) throw Error("no se puede escribir el fichero '" + ruta + "'");
    f << escribir(v);
    if (!f) throw Error("error al escribir el fichero '" + ruta + "'");
}

} // namespace json
