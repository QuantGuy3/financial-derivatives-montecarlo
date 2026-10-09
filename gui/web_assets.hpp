#pragma once
// Recursos web embebidos en el binario (los genera cmake/embed_assets.cmake a partir de gui/web
// y third_party/echarts).
#include <cstddef>

namespace mc::gui {

struct EmbeddedAsset {
    const char* path;            // p. ej. "/index.html"
    const char* mime;
    const unsigned char* data;
    std::size_t size;
};

// Tabla de recursos embebidos (n recibe el nº de entradas).
const EmbeddedAsset* embedded_assets(std::size_t* n);

} // namespace mc::gui
