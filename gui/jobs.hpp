#pragma once
// Cola de trabajos de la GUI: un único trabajo se ejecuta cada vez (los tiempos de la CPU siguen
// siendo representativos y el contexto de CUDA es global). Cada trabajo acumula un registro de
// eventos numerados (progress / result / error ...) que el cliente lee por SSE o por sondeo.

#include "json_codec.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace mc::gui {

struct Event {
    uint64_t seq = 0;       // 1, 2, 3... dentro del trabajo
    std::string type;       // started | progress | result | cancelled | error
    std::string data;       // JSON ya serializado
};

class Job {
public:
    explicit Job(std::string id) : id_(std::move(id)) {}

    const std::string& id() const { return id_; }
    void push(const std::string& type, const json& body);
    // Eventos con seq > after. Espera hasta `wait` si no hay ninguno y el trabajo sigue vivo.
    std::vector<Event> wait_events(uint64_t after, std::chrono::milliseconds wait);
    bool finished() const;
    std::string state() const;
    void set_state(const std::string& s);
    void finish(const std::string& final_state);

    std::atomic<bool> cancel{false};

private:
    std::string id_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::vector<Event> events_;
    std::string state_ = "queued";
    bool finished_ = false;
};

class JobManager {
public:
    // cuda_ok: si el backend CUDA está disponible; hw_threads: hilos del equipo.
    JobManager(int hw_threads, bool cuda_ok);
    ~JobManager();

    // Valida la petición (lanza std::invalid_argument) y la encola. Devuelve el trabajo.
    std::shared_ptr<Job> submit(const json& body);
    std::shared_ptr<Job> find(const std::string& id) const;
    bool cancel(const std::string& id);
    // ¿Hay algún trabajo en ejecución o en cola?
    bool busy() const;

private:
    void worker();
    void run_job(const std::shared_ptr<Job>& job, const ParsedRun& run);

    int hw_threads_;
    bool cuda_ok_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::pair<std::shared_ptr<Job>, ParsedRun>> queue_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
    std::deque<std::string> order_;
    std::atomic<bool> running_{false};
    bool stop_ = false;
    uint64_t counter_ = 0;
    std::thread worker_;
};

} // namespace mc::gui
