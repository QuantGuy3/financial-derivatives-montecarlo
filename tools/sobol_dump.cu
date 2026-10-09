// Volcado de los primeros puntos de Sobol SIN scrambling que genera la API de dispositivo de cuRAND
// (curand_init(direction_vectors, offset) + curand()), para compararlos con el generador de la CPU
// (cpu/sobol.cpp, orden Gray-code, tablas Joe-Kuo). Solo se compila con CUDA (ver tools/colab_validate.sh).
//
// Salida (texto): una línea por dimensión muestreada: "dim w0 w1 ... w4095" con las palabras de 32 bits.
// Uso: sobol_dump salida.txt
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int kN = 4096;

__global__ void dump_kernel(const curandDirectionVectors32_t* dv, int dim, unsigned int* out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= kN) return;
    unsigned int v[32];
    for (int i = 0; i < 32; i++) v[i] = dv[dim][i];
    curandStateSobol32_t st;
    curand_init(v, (unsigned int)p, &st);
    out[p] = curand(&st);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "uso: %s salida.txt\n", argv[0]); return 2; }
    const int dims[] = {0, 1, 2, 63, 1000, 19999};
    curandDirectionVectors32_t* h_dv = nullptr;
    if (curandGetDirectionVectors32(&h_dv, CURAND_DIRECTION_VECTORS_32_JOEKUO6) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "curandGetDirectionVectors32 falló\n"); return 1;
    }
    curandDirectionVectors32_t* d_dv = nullptr;
    const size_t bytes = 20000 * sizeof(curandDirectionVectors32_t);
    cudaMalloc(&d_dv, bytes);
    cudaMemcpy(d_dv, h_dv, bytes, cudaMemcpyHostToDevice);
    unsigned int* d_out = nullptr;
    cudaMalloc(&d_out, kN * sizeof(unsigned int));
    std::vector<unsigned int> h_out(kN);

    FILE* f = std::fopen(argv[1], "w");
    if (!f) { std::fprintf(stderr, "no se pudo abrir %s\n", argv[1]); return 1; }
    for (int dim : dims) {
        dump_kernel<<<(kN + 255) / 256, 256>>>(d_dv, dim, d_out);
        cudaMemcpy(h_out.data(), d_out, kN * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        std::fprintf(f, "%d", dim);
        for (int i = 0; i < kN; i++) std::fprintf(f, " %u", h_out[i]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    cudaFree(d_out); cudaFree(d_dv);
    std::printf("volcado escrito en %s\n", argv[1]);
    return 0;
}
