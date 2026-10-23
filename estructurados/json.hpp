#pragma once
// ---------------------------------------------------------------------------- //
// JSON mínimo sin dependencias para guardar/leer productos estructurados.      //
//                                                                              //
// - Los objetos conservan el orden de inserción de sus claves (el fichero que   //
//   se escribe se lee como una ficha de producto, en el mismo orden).          //
// - El lector es tolerante con la edición a mano: admite comentarios // y /*  */ //
//   y comas finales; el escritor produce JSON estándar.                        //
// - Cada valor recuerda la línea/columna donde se leyó, para que los errores   //
//   de validación del producto puedan señalar el sitio exacto del fichero.     //
// ---------------------------------------------------------------------------- //

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace json {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Valor {
public:
    enum class Tipo { Nulo, Booleano, Numero, Texto, Lista, Objeto };

    Valor() = default;
    Valor(std::nullptr_t) {}
    Valor(bool b) : tipo_(Tipo::Booleano), b_(b) {}
    template<typename T, std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, int> = 0>
    Valor(T n) : tipo_(Tipo::Numero), n_((double)n) {}
    Valor(const char* s) : tipo_(Tipo::Texto), s_(s) {}
    Valor(std::string s) : tipo_(Tipo::Texto), s_(std::move(s)) {}

    static Valor lista()  { Valor v; v.tipo_ = Tipo::Lista;  return v; }
    static Valor objeto() { Valor v; v.tipo_ = Tipo::Objeto; return v; }
    template<typename T>
    static Valor lista_de(const std::vector<T>& xs) {
        Valor v = lista();
        for (const auto& x : xs) v.anadir(Valor(x));
        return v;
    }

    Tipo tipo() const { return tipo_; }
    bool es_nulo()     const { return tipo_ == Tipo::Nulo; }
    bool es_booleano() const { return tipo_ == Tipo::Booleano; }
    bool es_numero()   const { return tipo_ == Tipo::Numero; }
    bool es_texto()    const { return tipo_ == Tipo::Texto; }
    bool es_lista()    const { return tipo_ == Tipo::Lista; }
    bool es_objeto()   const { return tipo_ == Tipo::Objeto; }
    const char* nombre_tipo() const;

    bool booleano() const;
    double numero() const;
    const std::string& texto() const;

    // Listas
    size_t tamano() const;
    const Valor& operator[](size_t i) const;
    Valor& operator[](size_t i);
    void anadir(Valor v);
    const std::vector<Valor>& elementos() const { return lista_; }

    // Objetos (orden de inserción conservado)
    const Valor* buscar(const std::string& clave) const;
    Valor* buscar(const std::string& clave);
    bool contiene(const std::string& clave) const { return buscar(clave) != nullptr; }
    // Inserta o sustituye la clave (al final si es nueva)
    Valor& poner(const std::string& clave, Valor v);
    bool quitar(const std::string& clave);
    const std::vector<std::string>& claves() const { return claves_; }
    const std::vector<Valor>& valores() const { return lista_; }

    // Posición en el texto de origen (1-based; 0 si el valor no se leyó de texto)
    int linea = 0, columna = 0;
    std::string donde() const; // " (línea L, columna C)" o ""

private:
    Tipo tipo_ = Tipo::Nulo;
    bool b_ = false;
    double n_ = 0.0;
    std::string s_;
    std::vector<std::string> claves_; // objetos
    std::vector<Valor> lista_;        // elementos de lista, o valores de objeto
};

// Lee un documento JSON completo. Lanza json::Error con línea y columna.
Valor parsear(const std::string& texto);
// Escribe con sangría (2 espacios). Las listas cortas de escalares van en una línea.
std::string escribir(const Valor& v, int sangria = 2);

Valor leer_fichero(const std::string& ruta);
void escribir_fichero(const std::string& ruta, const Valor& v);

// Formatea un número para JSON con la representación más corta que lo reproduce
std::string numero_a_texto(double x);

} // namespace json
