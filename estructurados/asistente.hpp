#pragma once
// Asistente interactivo (en terminal) para definir un producto estructurado paso
// a paso. Pregunta con valores por defecto, valida cada respuesta al momento,
// muestra la ficha resultante y guarda el producto en un fichero JSON.

#include "json.hpp"

#include <iosfwd>
#include <string>

namespace est {

struct Cancelado {}; // el usuario escribió "salir" o se terminó la entrada

// Devuelve el documento del producto; en *ruta deja el fichero donde se guardó
// ("" si no se guardó). Si se pasa ruta_sugerida, se propone como nombre.
json::Valor asistente_nuevo_producto(std::istream& in, std::ostream& out, const std::string& ruta_sugerida,
                                     std::string* ruta);

// Utilidades de diálogo reutilizadas por la línea de comandos
bool preguntar_si_no(std::istream& in, std::ostream& out, const std::string& texto, bool defecto);

} // namespace est
