// Tabla de vectores directores de Sobol (Joe-Kuo, 20000 dimensiones) en host.
// Se obtiene de la API de host de cuRAND, que no necesita GPU; permite QMC en el
// motor CPU con exactamente los mismos puntos que usa la GPU. En una compilación
// sin CUDA no está disponible y el motor CPU se limita a MC.

#include "valoracion.hpp"

#if defined(MC_CON_CURAND)
#include <curand.h>
#endif

namespace est {

bool direcciones_sobol(const uint32_t** dv, uint32_t* ndims) {
#if defined(MC_CON_CURAND)
    static const uint32_t* tabla = []() -> const uint32_t* {
        curandDirectionVectors32_t* h = nullptr;
        if (curandGetDirectionVectors32(&h, CURAND_DIRECTION_VECTORS_32_JOEKUO6) != CURAND_STATUS_SUCCESS)
            return nullptr;
        return reinterpret_cast<const uint32_t*>(h);
    }();
    if (!tabla) return false;
    *dv = tabla;
    *ndims = 20000u; // D_MAX_SOBOL: dimensiones de la tabla JOEKUO6 de cuRAND
    return true;
#else
    (void)dv;
    (void)ndims;
    return false;
#endif
}

} // namespace est
