// API de la GUI con el servidor en proceso: seguridad, trabajos (sondeo y SSE), cancelación, trayectorias.
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "doctest.h"
#include "gui/server.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

using json = nlohmann::json;

namespace {

struct Fixture {
    mc::gui::GuiServer server;
    std::unique_ptr<httplib::Client> cli;
    int port = -1;

    Fixture() : server([] { mc::gui::ServerOptions o; o.watchdog = false; return o; }()) {
        port = server.start();
        REQUIRE(port > 0);
        cli = std::make_unique<httplib::Client>("127.0.0.1", port);
        cli->set_default_headers({{"X-MC-Token", server.token()}});
        cli->set_read_timeout(30, 0);
    }
    ~Fixture() { server.stop(); }

    json post(const std::string& path, const json& body, int* status = nullptr) {
        auto r = cli->Post(path, body.dump(), "application/json");
        REQUIRE(r);
        if (status) *status = r->status;
        return json::parse(r->body);
    }
    json get(const std::string& path, int* status = nullptr) {
        auto r = cli->Get(path);
        REQUIRE(r);
        if (status) *status = r->status;
        return json::parse(r->body);
    }

    // Sondea los eventos del trabajo hasta que termina; devuelve todos.
    std::vector<json> collect(const std::string& job_id, double timeout_s = 120.0) {
        std::vector<json> all;
        uint64_t after = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            json r = get("/api/jobs/" + job_id + "/poll?after=" + std::to_string(after) + "&wait=500");
            for (auto& e : r["events"]) all.push_back(e);
            after = r["next"].get<uint64_t>();
            if (r["finished"].get<bool>()) break;
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) break;
        }
        return all;
    }
};

json base_request(const char* family = "mc", double eps = 0.05) {
    return json{{"seed", 5},
                {"threads", 2},
                {"model", {{"type", "gbm"}, {"S0", 100}, {"mu", 0.05}, {"sigma", 0.2}, {"T", 1.0}}},
                {"payoff", {{"type", "european"}, {"K", 100}, {"r", 0.05}}},
                {"method", {{"family", family}, {"eps", eps}}}};
}

} // namespace

TEST_SUITE("fast") {

TEST_CASE("GUI: seguridad — token, Host y recursos estáticos") {
    Fixture f;
    // La interfaz se sirve sin token
    auto idx = f.cli->Get("/");
    REQUIRE(idx);
    CHECK(idx->status == 200);
    CHECK(idx->get_header_value("Content-Type").find("text/html") != std::string::npos);

    // Todos los recursos de la interfaz están embebidos en el binario
    for (const char* path : {"/assets/app.js", "/assets/charts.js", "/assets/api.js", "/assets/i18n.js",
                             "/assets/style.css", "/vendor/echarts.min.js", "/favicon.svg"}) {
        auto a = f.cli->Get(path);
        REQUIRE_MESSAGE(a, path);
        CHECK_MESSAGE(a->status == 200, path);
        CHECK_MESSAGE(a->body.size() > 100, path);
    }
    auto js = f.cli->Get("/vendor/echarts.min.js");
    REQUIRE(js);
    CHECK(js->get_header_value("Content-Type").find("javascript") != std::string::npos);
    CHECK(js->body.size() > 500000);
    const int trav = f.cli->Get("/assets/../../etc/passwd")->status;   // sin escapes de ruta
    CHECK((trav == 400 || trav == 404));
    CHECK(f.cli->Get("/nada.txt")->status == 404);

    httplib::Client anon("127.0.0.1", f.port);
    auto r1 = anon.Get("/api/capabilities");
    REQUIRE(r1);
    CHECK(r1->status == 401);
    auto r2 = anon.Get("/api/capabilities?token=" + f.server.token());
    REQUIRE(r2);
    CHECK(r2->status == 200);

    httplib::Client evil("127.0.0.1", f.port);
    evil.set_default_headers({{"X-MC-Token", f.server.token()}, {"Host", "evil.example.com"}});
    auto r3 = evil.Get("/api/capabilities");
    if (r3) CHECK(r3->status == 403);   // si el cliente respeta la cabecera Host manual
    CHECK(f.server.url().find("#token=") != std::string::npos);
    CHECK(f.server.url().find("127.0.0.1") != std::string::npos);
}

TEST_CASE("GUI: capacidades describen modelos, payoffs, presets y backends") {
    Fixture f;
    json caps = f.get("/api/capabilities");
    CHECK(caps["models"].size() == 4);
    CHECK(caps["payoffs"].size() == 6);
    CHECK(caps["presets"].size() == 11);
    CHECK(caps["backends"][0]["id"] == "cpu");
    CHECK(caps["backends"][0]["available"] == true);
    CHECK(caps["backends"][1]["available"] == false);     // este build no incluye CUDA
    for (const auto& m : caps["models"])
        for (const auto& p : m["params"]) {
            CHECK(p["min"].get<double>() <= p["default"].get<double>());
            CHECK(p["default"].get<double>() <= p["max"].get<double>());
        }
}

TEST_CASE("GUI: un trabajo MC produce started, progress... y result coherente con Black-Scholes") {
    Fixture f;
    int st = 0;
    json sub = f.post("/api/run", base_request("mc", 0.02), &st);
    CHECK(st == 202);
    auto ev = f.collect(sub["job_id"]);
    REQUIRE(ev.size() >= 3);
    CHECK(ev.front()["type"] == "started");
    CHECK(ev.front()["threads"] == 2);
    CHECK(ev.back()["type"] == "result");
    uint64_t prev = 0; long long prev_n = 0; int progress = 0;
    for (const auto& e : ev) {
        CHECK(e["seq"].get<uint64_t>() == prev + 1);
        prev = e["seq"].get<uint64_t>();
        if (e["type"] == "progress" && e["stage"] == "main") {
            ++progress;
            CHECK(e["n_done"].get<long long>() >= prev_n);
            prev_n = e["n_done"].get<long long>();
        }
    }
    CHECK(progress >= 3);
    const json& res = ev.back();
    const double bs = 10.4505835722;
    CHECK(std::abs(res["price"].get<double>() - bs) < 5.0 * res["std_error"].get<double>() + 0.03);
    CHECK(res["reference"]["value"].get<double>() == doctest::Approx(bs).epsilon(1e-9));
    CHECK(res["ci95"].size() == 2);
    CHECK(res["n_steps"].get<int>() >= 1);
    CHECK(res["mpaths_per_s"].get<double>() > 0.0);
    CHECK(res["within"].is_boolean());
}

TEST_CASE("GUI: QMC emite duplicaciones con las medias de réplica; MLMC emite niveles") {
    Fixture f;
    {
        json req = base_request("qmc", 0.01);
        req["method"]["noise"] = "bb"; req["method"]["n0"] = 256; req["method"]["R"] = 8;
        auto ev = f.collect(f.post("/api/run", req)["job_id"]);
        int dbl = 0;
        for (const auto& e : ev)
            if (e["type"] == "progress" && e["stage"] == "doubling") {
                ++dbl;
                CHECK(e["replica_means"].size() == 8);
            }
        CHECK(dbl >= 2);
        CHECK(ev.back()["type"] == "result");
    }
    {
        auto ev = f.collect(f.post("/api/run", base_request("mlmc", 0.02))["job_id"]);
        bool saw_levels = false;
        for (const auto& e : ev) if (e["type"] == "progress" && e.contains("levels")) saw_levels = true;
        CHECK(saw_levels);
        CHECK(ev.back()["type"] == "result");
        CHECK(ev.back()["levels"].size() >= 3);
    }
}

TEST_CASE("GUI: peticiones inválidas devuelven 400 con un mensaje claro") {
    Fixture f;
    int st = 0;
    auto expect_bad = [&](json req, const std::string& fragment) {
        json r = f.post("/api/run", req, &st);
        CHECK(st == 400);
        CHECK_MESSAGE(r["error"].get<std::string>().find(fragment) != std::string::npos, r.dump());
    };
    { auto r = base_request(); r["method"]["eps"] = 0.0; expect_bad(r, "eps"); }
    { auto r = base_request(); r["model"]["type"] = "foo"; expect_bad(r, "modelo desconocido"); }
    { auto r = base_request(); r["model"]["sigma"] = -1; expect_bad(r, "sigma"); }
    { auto r = base_request("mlmc"); r["model"] = {{"type", "basket"}, {"n", 5}}; r["payoff"] = {{"type", "basket"}, {"K", 100}};
      expect_bad(r, "MLMC"); }
    { auto r = base_request(); r["method"]["variance"] = "is"; r["payoff"]["type"] = "asian"; expect_bad(r, "importance sampling"); }
    { auto r = base_request(); r["backend"] = "cuda"; expect_bad(r, "CUDA"); }
    { auto r = base_request(); r["payoff"]["type"] = "basket"; expect_bad(r, "basket"); }
    auto r = f.cli->Post("/api/run", "{no es json", "application/json");
    REQUIRE(r);
    CHECK(r->status == 400);
    auto missing = f.cli->Get("/api/jobs/j999/poll");
    REQUIRE(missing);
    CHECK(missing->status == 404);
}

TEST_CASE("GUI: cancelar un trabajo largo devuelve 'cancelled' con resultado parcial") {
    Fixture f;
    json sub = f.post("/api/run", base_request("mc", 0.0005));        // pediría ~miles de millones de caminos
    const std::string id = sub["job_id"];
    // espera a que arranque
    for (int i = 0; i < 100; i++) {
        json r = f.get("/api/jobs/" + id + "/poll?after=0&wait=100");
        if (r["events"].size() >= 2) break;
    }
    json c = f.post("/api/jobs/" + id + "/cancel", json::object());
    CHECK(c["ok"] == true);
    auto ev = f.collect(id, 60.0);
    REQUIRE(!ev.empty());
    CHECK(ev.back()["type"] == "cancelled");
    CHECK(ev.back()["partial"].is_object());
    CHECK(f.get("/api/jobs/" + id)["state"] == "cancelled");
}

TEST_CASE("GUI: max_seconds devuelve resultado parcial marcado como truncado") {
    Fixture f;
    json req = base_request("mc", 0.0005);
    req["max_seconds"] = 0.4;
    auto ev = f.collect(f.post("/api/run", req)["job_id"], 60.0);
    CHECK(ev.back()["type"] == "result");
    CHECK(ev.back()["truncated"] == true);
}

TEST_CASE("GUI: dos trabajos seguidos se ejecutan en orden de uno en uno") {
    Fixture f;
    std::string a = f.post("/api/run", base_request("mc", 0.05))["job_id"];
    std::string b = f.post("/api/run", base_request("mc", 0.05))["job_id"];
    auto ea = f.collect(a), eb = f.collect(b);
    CHECK(ea.back()["type"] == "result");
    CHECK(eb.back()["type"] == "result");
    CHECK(ea.back()["price"] == eb.back()["price"]);        // determinista: misma semilla, mismo resultado
}

TEST_CASE("GUI: el flujo SSE entrega los mismos eventos que el sondeo") {
    Fixture f;
    std::string id = f.post("/api/run", base_request("mc", 0.05))["job_id"];
    std::string stream;
    auto r = f.cli->Get("/api/jobs/" + id + "/events", httplib::Headers{},
                        [&](const char* data, size_t len) { stream.append(data, len); return true; });
    REQUIRE(r);
    CHECK(r->status == 200);
    CHECK(stream.find("event: started") != std::string::npos);
    CHECK(stream.find("event: progress") != std::string::npos);
    CHECK(stream.find("event: result") != std::string::npos);
    CHECK(stream.find("id: 1\n") != std::string::npos);
}

TEST_CASE("GUI: /api/paths — forma, reproducibilidad, abanico y par acoplado") {
    Fixture f;
    json req = {{"model", {{"type", "heston"}}},
                {"payoff", {{"type", "european"}, {"K", 100}, {"r", 0.05}}},
                {"n_steps", 64}, {"n_paths", 5}, {"seed", 3}, {"fan", 400}};
    json a = f.post("/api/paths", req), b = f.post("/api/paths", req);
    CHECK(a == b);
    CHECK(a["paths"].size() == 5);
    CHECK(a["t"].size() == 65);
    CHECK(a["paths"][0]["S"].size() == 65);
    CHECK(a["paths"][0]["V"].size() == 65);
    CHECK(a["fan"]["p50"].size() == 65);
    CHECK(a["fan"]["hist"]["counts"].size() == 40);
    CHECK(a["marks"]["strike"] == 100.0);

    json g = {{"model", {{"type", "gbm"}}}, {"payoff", {{"type", "barrier"}, {"K", 100}, {"B", 130}}},
              {"n_steps", 64}, {"n_paths", 12}, {"sobol", true}, {"construction", "bb"},
              {"coupled", {{"level", 4}, {"M", 2}}}};
    json c = f.post("/api/paths", g);
    CHECK(c["marks"]["barrier"] == 130.0);
    CHECK(c["coupled"]["S_fine"].size() == 17);
    CHECK(c["coupled"]["S_coarse"].size() == 9);
    CHECK(c["paths"][0].contains("run"));

    int st = 0;
    g["n_paths"] = 5000;
    f.post("/api/paths", g, &st);
    CHECK(st == 400);
    g["n_paths"] = 5; g["n_steps"] = 60;
    json err = f.post("/api/paths", g, &st);       // BB exige potencia de 2
    CHECK(st == 400);
    CHECK(err["error"].get<std::string>().find("potencia de 2") != std::string::npos);
}

TEST_CASE("GUI: /api/shutdown detiene el servidor") {
    mc::gui::ServerOptions o; o.watchdog = false;
    mc::gui::GuiServer server(o);
    int port = server.start();
    REQUIRE(port > 0);
    httplib::Client cli("127.0.0.1", port);
    cli.set_default_headers({{"X-MC-Token", server.token()}});
    auto r = cli.Post("/api/shutdown", "", "application/json");
    REQUIRE(r);
    CHECK(r->status == 200);
    server.wait();    // retorna cuando el servidor se ha detenido
    CHECK(true);
}

} // TEST_SUITE
