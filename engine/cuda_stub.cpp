// Símbolos de methods_cuda.cu para los builds SIN CUDA.
//
// engine/cuda_adapter.cpp (C++ plano) referencia estas funciones; en un build con nvcc las
// aporta mc_cuda y este fichero no se compila. Aquí lanzan una excepción clara: así el adaptador
// y los ejemplos se compilan y enlazan en cualquier máquina (y el compilador comprueba que las
// firmas siguen coincidiendo con methods_cuda.cuh).

#include "../methods_cuda.cuh"

#include <stdexcept>

namespace {
[[noreturn]] void no_cuda() {
    throw std::runtime_error("backend CUDA no disponible: este binario se compiló sin CUDA "
                             "(use --backend=cpu o recompile con nvcc: -DMC_ENABLE_CUDA=ON)");
}
} // namespace

bool cuda_device_available() { return false; }

DeviceBBData*  bb_upload(const BBData&) { no_cuda(); }
void           bb_free(DeviceBBData*) {}
DevicePCAData* pca_upload(const PCAData&) { no_cuda(); }
void           pca_free(DevicePCAData*) {}

MCResult run_mc_cuda(const ModelVariant&, const PayoffVariant&, double, int, const MCConfig&) { no_cuda(); }
MCResult run_mlmc_cuda(const ModelVariant&, const PayoffVariant&, double, const MLMCConfig&) { no_cuda(); }
MCResult run_qmc_cuda(const ModelVariant&, const PayoffVariant&, double, int, const QMCConfig&, NoiseMode,
                      DeviceBBData*, DevicePCAData*) { no_cuda(); }
MCResult run_mlqmc_cuda(const ModelVariant&, const PayoffVariant&, double, const MLMCConfig&, const QMCConfig&,
                        NoiseMode, std::vector<DeviceBBData*>, std::vector<DevicePCAData*>) { no_cuda(); }

CVPilot cv_pilot(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                 double, int, int, unsigned) { no_cuda(); }
MCResult run_mc_cv_cuda(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                        double, double, double, int, const MCConfig&) { no_cuda(); }
MCResult run_qmc_cv_cuda(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                         double, double, double, int, const QMCConfig&, NoiseMode,
                         DeviceBBData*, DevicePCAData*) { no_cuda(); }
MCResult run_mlmc_cv_cuda(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                          double, double, double, const MLMCConfig&) { no_cuda(); }
MCResult run_mlqmc_cv_cuda(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                           double, double, double, const MLMCConfig&, const QMCConfig&, NoiseMode,
                           std::vector<DeviceBBData*>, std::vector<DevicePCAData*>) { no_cuda(); }

MCResult run_is_cuda(const GBMParams&, const European&, double, double, const MCConfig&) { no_cuda(); }
MCResult run_qmc_is_cuda(const GBMParams&, const European&, double, double, const QMCConfig&, NoiseMode,
                         DeviceBBData*, DevicePCAData*) { no_cuda(); }
MCResult run_mlmc_is_cuda(const GBMParams&, const European&, double, double, const MLMCConfig&) { no_cuda(); }
MCResult run_mlqmc_is_cuda(const GBMParams&, const European&, double, double, const MLMCConfig&,
                           const QMCConfig&, NoiseMode, std::vector<DeviceBBData*>,
                           std::vector<DevicePCAData*>) { no_cuda(); }

std::pair<double, double> run_mc_fixed(const ModelVariant&, const PayoffVariant&, int, long long, unsigned) {
    no_cuda();
}
