#pragma once
// Servidor HTTP local de la GUI (solo 127.0.0.1): sirve la interfaz web embebida y expone la API
// REST + eventos (SSE o sondeo) para lanzar simulaciones, seguir su convergencia y muestrear trayectorias.

#include <memory>
#include <string>

namespace mc::gui {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int         port = 0;                 // 0 = puerto libre
    std::string web_dir;                  // vacío = interfaz embebida en el binario; si no, sirve desde disco
    std::string token;                    // vacío = se genera uno aleatorio
    bool        watchdog = true;          // se apaga solo cuando la ventana deja de hacer ping
    double      idle_timeout_s = 20.0;
    bool        quiet = false;
};

class GuiServer {
public:
    explicit GuiServer(ServerOptions opt = {});
    ~GuiServer();
    GuiServer(const GuiServer&) = delete;
    GuiServer& operator=(const GuiServer&) = delete;

    // Empieza a escuchar en un hilo propio. Devuelve el puerto real (o -1 si falla).
    int start();
    void stop();
    // Bloquea hasta que el servidor se detiene (stop(), /api/shutdown o watchdog).
    void wait();

    int port() const;
    const std::string& token() const;
    // http://127.0.0.1:PORT/#token=XXXX  (el token va en el fragmento: nunca viaja al servidor ni a otros sitios)
    std::string url() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace mc::gui
