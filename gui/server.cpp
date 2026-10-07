#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include "server.hpp"

#include "jobs.hpp"
#include "web_assets.hpp"

#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace mc::gui {

namespace {

std::string random_token() {
    std::random_device rd;
    std::mt19937_64 g((uint64_t(rd()) << 32) ^ rd() ^ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
    static const char* hex = "0123456789abcdef";
    std::string t;
    for (int i = 0; i < 32; i++) t += hex[g() & 15];
    return t;
}

std::string mime_of(const std::string& path) {
    auto ends = [&](const char* e) { const std::string s = e; return path.size() >= s.size() && path.compare(path.size() - s.size(), s.size(), s) == 0; };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".js") || ends(".mjs")) return "text/javascript; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".json")) return "application/json; charset=utf-8";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".png")) return "image/png";
    if (ends(".ico")) return "image/x-icon";
    return "application/octet-stream";
}

} // namespace

struct GuiServer::Impl {
    ServerOptions opt;
    std::string token;
    int hw_threads = 1;
    bool cuda_ok = false;
    httplib::Server svr;
    std::unique_ptr<JobManager> jobs;
    std::thread listener;
    std::atomic<int> port{-1};
    std::atomic<bool> stopping{false};
    std::atomic<bool> ever_connected{false};
    std::atomic<long long> last_seen_ms{0};
    std::mutex wait_m;
    std::condition_variable wait_cv;
    bool stopped = true;
    std::thread watchdog;

    static long long now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    bool host_ok(const httplib::Request& req) const {
        const std::string h = req.get_header_value("Host");
        const std::string p = std::to_string(port.load());
        return h == "127.0.0.1:" + p || h == "localhost:" + p || h == "[::1]:" + p;
    }

    bool token_ok(const httplib::Request& req) const {
        std::string t = req.get_header_value("X-MC-Token");
        if (t.empty() && req.has_param("token")) t = req.get_param_value("token");
        return t == token;
    }

    static void json_out(httplib::Response& res, const json& j, int status = 200) {
        res.status = status;
        res.set_content(j.dump(), "application/json; charset=utf-8");
        res.set_header("Cache-Control", "no-store");
    }

    static void error_out(httplib::Response& res, int status, const std::string& msg) {
        json_out(res, json{{"error", msg}}, status);
    }

    void serve_asset(const std::string& raw_path, httplib::Response& res) {
        std::string path = raw_path == "/" ? "/index.html" : raw_path;
        if (path.find("..") != std::string::npos) { error_out(res, 400, "ruta inválida"); return; }
        if (!opt.web_dir.empty()) {
            std::ifstream f(opt.web_dir + path, std::ios::binary);
            if (!f) { error_out(res, 404, "no encontrado"); return; }
            std::stringstream ss; ss << f.rdbuf();
            res.set_content(ss.str(), mime_of(path));
            res.set_header("Cache-Control", "no-store");
            return;
        }
        std::size_t n = 0;
        const EmbeddedAsset* a = embedded_assets(&n);
        for (std::size_t i = 0; i < n; i++) {
            if (path == a[i].path) {
                res.set_content(reinterpret_cast<const char*>(a[i].data), a[i].size, a[i].mime);
                res.set_header("Cache-Control", "public, max-age=3600");
                return;
            }
        }
        error_out(res, 404, "no encontrado");
    }

    void setup_routes() {
        svr.new_task_queue = [] { return new httplib::ThreadPool(16); };

        // Seguridad: solo peticiones dirigidas a 127.0.0.1/localhost (anti DNS-rebinding) y, en /api/*,
        // con el token que viaja en el fragmento de la URL.
        svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
            last_seen_ms = now_ms();
            ever_connected = true;
            if (!host_ok(req)) { error_out(res, 403, "host no permitido"); return httplib::Server::HandlerResponse::Handled; }
            if (req.path.rfind("/api/", 0) == 0 && !token_ok(req)) {
                error_out(res, 401, "token inválido");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });

        svr.Get("/", [this](const httplib::Request&, httplib::Response& res) { serve_asset("/", res); });
        svr.Get(R"(/(assets|vendor)/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            serve_asset(req.path, res);
        });
        svr.Get("/favicon.svg", [this](const httplib::Request& req, httplib::Response& res) { serve_asset(req.path, res); });

        svr.Get("/api/ping", [](const httplib::Request&, httplib::Response& res) { json_out(res, json{{"ok", true}}); });

        svr.Get("/api/capabilities", [this](const httplib::Request&, httplib::Response& res) {
            json_out(res, capabilities_json(hw_threads, cuda_ok));
        });

        svr.Post("/api/run", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                json body = json::parse(req.body);
                auto job = jobs->submit(body);
                json_out(res, json{{"job_id", job->id()}}, 202);
            } catch (const json::parse_error& e) {
                error_out(res, 400, std::string("JSON inválido: ") + e.what());
            } catch (const std::invalid_argument& e) {
                error_out(res, 400, e.what());
            } catch (const std::exception& e) {
                error_out(res, 500, e.what());
            }
        });

        svr.Post(R"(/api/jobs/([^/]+)/cancel)", [this](const httplib::Request& req, httplib::Response& res) {
            const bool ok = jobs->cancel(req.matches[1]);
            json_out(res, json{{"ok", ok}}, ok ? 200 : 404);
        });

        svr.Get(R"(/api/jobs/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
            auto job = jobs->find(req.matches[1]);
            if (!job) { error_out(res, 404, "trabajo desconocido"); return; }
            json_out(res, json{{"job_id", job->id()}, {"state", job->state()}, {"finished", job->finished()}});
        });

        // Sondeo: {events:[...], next, finished}. ?after=N (último seq visto), ?wait=ms (espera larga, máx 10 s)
        svr.Get(R"(/api/jobs/([^/]+)/poll)", [this](const httplib::Request& req, httplib::Response& res) {
            auto job = jobs->find(req.matches[1]);
            if (!job) { error_out(res, 404, "trabajo desconocido"); return; }
            uint64_t after = req.has_param("after") ? std::strtoull(req.get_param_value("after").c_str(), nullptr, 10) : 0;
            int wait = req.has_param("wait") ? std::atoi(req.get_param_value("wait").c_str()) : 0;
            wait = std::clamp(wait, 0, 10000);
            auto evs = job->wait_events(after, std::chrono::milliseconds(wait));
            json arr = json::array();
            uint64_t next = after;
            for (const auto& e : evs) { arr.push_back(json::parse(e.data)); arr.back()["type"] = e.type; next = e.seq; }
            json_out(res, json{{"events", arr}, {"next", next}, {"finished", job->finished() && evs.empty()}});
        });

        // SSE
        svr.Get(R"(/api/jobs/([^/]+)/events)", [this](const httplib::Request& req, httplib::Response& res) {
            auto job = jobs->find(req.matches[1]);
            if (!job) { error_out(res, 404, "trabajo desconocido"); return; }
            uint64_t after = 0;
            if (req.has_header("Last-Event-ID")) after = std::strtoull(req.get_header_value("Last-Event-ID").c_str(), nullptr, 10);
            else if (req.has_param("after")) after = std::strtoull(req.get_param_value("after").c_str(), nullptr, 10);
            auto cursor = std::make_shared<uint64_t>(after);
            res.set_header("Cache-Control", "no-store");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider("text/event-stream",
                [job, cursor](size_t, httplib::DataSink& sink) {
                    auto evs = job->wait_events(*cursor, std::chrono::milliseconds(15000));
                    if (evs.empty()) {
                        if (job->finished()) { sink.done(); return true; }
                        static const char hb[] = ": heartbeat\n\n";
                        return sink.write(hb, sizeof(hb) - 1);
                    }
                    for (const auto& e : evs) {
                        const std::string msg = "id: " + std::to_string(e.seq) + "\nevent: " + e.type + "\ndata: " + e.data + "\n\n";
                        if (!sink.write(msg.data(), msg.size())) return false;
                        *cursor = e.seq;
                    }
                    return true;
                });
        });

        svr.Post("/api/paths", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                json_out(res, paths_response(json::parse(req.body)));
            } catch (const json::parse_error& e) {
                error_out(res, 400, std::string("JSON inválido: ") + e.what());
            } catch (const std::invalid_argument& e) {
                error_out(res, 400, e.what());
            } catch (const std::exception& e) {
                error_out(res, 500, e.what());
            }
        });

        svr.Post("/api/shutdown", [this](const httplib::Request&, httplib::Response& res) {
            json_out(res, json{{"ok", true}});
            std::thread([this] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); stop_impl(); }).detach();
        });
    }

    void stop_impl() {
        if (stopping.exchange(true)) return;
        svr.stop();
    }
};

GuiServer::GuiServer(ServerOptions opt) : p_(std::make_unique<Impl>()) {
    p_->opt = std::move(opt);
    p_->token = p_->opt.token.empty() ? random_token() : p_->opt.token;
    p_->hw_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    p_->cuda_ok = mc::cuda_available();
}

GuiServer::~GuiServer() { stop(); }

int GuiServer::start() {
    Impl& s = *p_;
    s.jobs = std::make_unique<JobManager>(s.hw_threads, s.cuda_ok);
    s.setup_routes();
    int port = s.opt.port > 0 ? (s.svr.bind_to_port(s.opt.host, s.opt.port) ? s.opt.port : -1)
                              : s.svr.bind_to_any_port(s.opt.host);
    if (port < 0) return -1;
    s.port = port;
    s.stopped = false;
    s.last_seen_ms = Impl::now_ms();
    s.listener = std::thread([&s] {
        s.svr.listen_after_bind();
        {
            std::lock_guard<std::mutex> lk(s.wait_m);
            s.stopped = true;
        }
        s.wait_cv.notify_all();
    });
    if (s.opt.watchdog) {
        s.watchdog = std::thread([&s] {
            while (!s.stopping.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (s.ever_connected.load() && !s.jobs->busy() &&
                    (Impl::now_ms() - s.last_seen_ms.load()) > (long long)(s.opt.idle_timeout_s * 1000)) {
                    s.stop_impl();
                    return;
                }
            }
        });
    }
    s.svr.wait_until_ready();
    return port;
}

void GuiServer::stop() {
    Impl& s = *p_;
    s.stop_impl();
    if (s.listener.joinable()) s.listener.join();
    if (s.watchdog.joinable()) s.watchdog.join();
    s.jobs.reset();
}

void GuiServer::wait() {
    Impl& s = *p_;
    std::unique_lock<std::mutex> lk(s.wait_m);
    s.wait_cv.wait(lk, [&] { return s.stopped; });
}

int GuiServer::port() const { return p_->port.load(); }
const std::string& GuiServer::token() const { return p_->token; }
std::string GuiServer::url() const {
    return "http://127.0.0.1:" + std::to_string(p_->port.load()) + "/#token=" + p_->token;
}

} // namespace mc::gui
