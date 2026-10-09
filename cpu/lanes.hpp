#pragma once
// Anchura del bloque "lockstep": nº de caminos que avanzan intercalados en los núcleos y en la
// generación de normales. Se fija al compilar (opción de CMake MC_LANES).

namespace mc::cpu {

#ifndef MC_LANES
#define MC_LANES 8
#endif
inline constexpr int kLanes = MC_LANES;

} // namespace mc::cpu
