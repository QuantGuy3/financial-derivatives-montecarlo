#pragma once
// Conversión JSON <-> tipos del motor para la GUI (validación de rangos incluida).

#include "../engine/api.hpp"
#include "../reference_prices.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>

namespace mc::gui {

using json = nlohmann::json;

// Una petición de simulación ya validada
struct ParsedRun {
    ModelVariant   model;
    PayoffVariant  payoff;
    mc::RunSpec    spec;
    mc::RunOptions opt;          // backend / hilos / max_seconds (sin sink)
    std::string    model_type, payoff_type;
    json           echo;         // petición normalizada, para repetirla en los eventos
};

// Lanza std::invalid_argument con un mensaje en español si la petición no es válida.
ParsedRun parse_run_request(const json& body, int hw_threads, bool cuda_ok);

// Descripción de modelos, payoffs, métodos, límites y presets (para construir el formulario).
json capabilities_json(int hw_threads, bool cuda_ok);

json snapshot_json(const Snapshot& s);
json report_json(const ParsedRun& run, const mc::RunReport& rep, const std::optional<ReferencePrice>& ref);

// POST /api/paths
json paths_response(const json& body);

} // namespace mc::gui
