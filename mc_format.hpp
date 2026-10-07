#pragma once
// Pequeño helper de formato basado en snprintf. Sustituye a std::format/<format> en el
// código nuevo y en las tablas de salida: <format> no existe en g++ < 13 (Ubuntu 22.04),
// y las tablas deben seguir siendo compatibles byte a byte con gen_informe.py.

#include <charconv>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define MC_PRINTF_LIKE(fmt_idx, first_arg) __attribute__((format(printf, fmt_idx, first_arg)))
#else
#define MC_PRINTF_LIKE(fmt_idx, first_arg)
#endif

inline std::string mc_sprintf(const char* fmt, ...) MC_PRINTF_LIKE(1, 2);
inline std::string mc_sprintf(const char* fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string out;
    if (n > 0) {
        out.resize(static_cast<size_t>(n));
        std::vsnprintf(out.data(), static_cast<size_t>(n) + 1, fmt, ap2);
    }
    va_end(ap2);
    return out;
}

// Destino opcional de las líneas impresas por MC_PRINTLN (barridos, tablas): por defecto
// stdout. Los tests y la GUI lo sustituyen para capturar la salida sin tocar stdout.
inline std::function<void(const std::string&)>& mc_output_sink() {
    static std::function<void(const std::string&)> sink;
    return sink;
}

// printf + '\n' + flush (equivalente al antiguo mc_println)
inline void mc_println_s(const std::string& s) {
    auto& sink = mc_output_sink();
    if (sink) { sink(s); return; }
    std::puts(s.c_str());
    std::fflush(stdout);
}

// Representación más corta que reproduce el double (igual que "{}" de std::format):
// 0.05 -> "0.05", 0.001 -> "0.001", 1e-4 -> "1e-04", 1e-5 -> "1e-05".
inline std::string mc_shortest(double x) {
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), x);
    return std::string(buf, r.ptr);
}

// printf con salto de línea y flush (los barridos largos deben verse en tiempo real).
#define MC_PRINTLN(...) mc_println_s(mc_sprintf(__VA_ARGS__))
