// Sustituto del motor GPU para compilaciones sin CUDA.
#include "valoracion.hpp"

#include <stdexcept>

namespace est {

bool gpu_disponible(std::string* descripcion) {
    if (descripcion) *descripcion = "este ejecutable se compiló sin soporte CUDA";
    return false;
}

std::unique_ptr<Motor> crear_motor_gpu(const Producto&, uint32_t, bool) {
    throw std::runtime_error("motor GPU no disponible: el proyecto se compiló sin CUDA");
}

} // namespace est
