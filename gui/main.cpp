// mc_gui: interfaz gráfica local. Levanta el servidor (solo 127.0.0.1) y abre la interfaz web
// en una ventana de aplicación de Edge/Chrome (sin barra de direcciones). Se apaga solo cuando
// se cierra la ventana.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#include "server.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

bool file_exists(const std::string& p) {
#ifdef _WIN32
    DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    return std::system(("test -x '" + p + "'").c_str()) == 0;
#endif
}

// Abre la URL como ventana de aplicación (Edge/Chrome --app) o, si no hay, en el navegador por defecto.
bool open_ui(const std::string& url) {
#ifdef _WIN32
    const char* candidates[] = {
        "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    };
    for (const char* exe : candidates) {
        if (!file_exists(exe)) continue;
        std::string cmd = std::string("\"") + exe + "\" --app=\"" + url + "\" --window-size=1480,940 --new-window";
        STARTUPINFOA si{}; si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        std::string mutable_cmd = cmd;
        if (CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
            return true;
        }
    }
    return reinterpret_cast<INT_PTR>(ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL)) > 32;
#elif defined(__APPLE__)
    return std::system(("open '" + url + "'").c_str()) == 0;
#else
    for (const char* b : {"google-chrome", "chromium", "chromium-browser", "microsoft-edge"})
        if (std::system(("command -v " + std::string(b) + " >/dev/null 2>&1 && (" + b + " --app='" + url + "' >/dev/null 2>&1 &)").c_str()) == 0)
            return true;
    return std::system(("xdg-open '" + url + "' >/dev/null 2>&1 &").c_str()) == 0;
#endif
}

void usage(const char* exe) {
    std::printf(
        "uso: %s [opciones]\n"
        "  --port N          puerto (por defecto uno libre)\n"
        "  --no-browser      no abrir la interfaz; solo imprimir la URL\n"
        "  --web-dir RUTA    servir la interfaz desde disco (desarrollo) en vez de la embebida\n"
        "  --no-watchdog     no apagar el servidor al cerrar la ventana\n"
        "  --token T         token de acceso fijo (por defecto aleatorio)\n",
        exe);
}

} // namespace

int main(int argc, char** argv) {
    mc::gui::ServerOptions opt;
    bool open_browser = true;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--port") opt.port = std::atoi(next().c_str());
        else if (a == "--no-browser") open_browser = false;
        else if (a == "--web-dir") opt.web_dir = next();
        else if (a == "--no-watchdog") opt.watchdog = false;
        else if (a == "--token") opt.token = next();
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "argumento desconocido: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }

    mc::gui::GuiServer server(opt);
    const int port = server.start();
    if (port < 0) {
        std::fprintf(stderr, "no se pudo abrir el puerto %d\n", opt.port);
        return 1;
    }
    std::printf("Monte Carlo Studio en %s\n", server.url().c_str());
    std::fflush(stdout);
    if (open_browser && !open_ui(server.url()))
        std::fprintf(stderr, "no se pudo abrir el navegador; abra la URL manualmente\n");
    server.wait();
    return 0;
}
