#pragma once
// Subcomandos de "estructurados fraccional": modelos fraccionarios y sus
// métodos numéricos (ver fraccional/fraccional.hpp y FRACCIONAL.md).

#include <iosfwd>
#include <string>
#include <vector>

namespace est {

void uso_fraccional(std::ostream& os);
int cmd_fraccional(const std::vector<std::string>& args);

} // namespace est
