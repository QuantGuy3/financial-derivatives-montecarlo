#include "jobs.hpp"

#include <algorithm>
#include <cmath>

namespace mc::gui {

// ---- Job --------------------------------------------------------------------------------------------

void Job::push(const std::string& type, const json& body) {
    std::lock_guard<std::mutex> lk(m_);
    Event e;
    e.seq = events_.size() + 1;
    e.type = type;
    json b = body;
    b["seq"] = e.seq;
    e.data = b.dump();
    events_.push_back(std::move(e));
    cv_.notify_all();
}

std::vector<Event> Job::wait_events(uint64_t after, std::chrono::milliseconds wait) {
    std::unique_lock<std::mutex> lk(m_);
    if (events_.size() <= after && !finished_ && wait.count() > 0)
        cv_.wait_for(lk, wait, [&] { return events_.size() > after || finished_; });
    std::vector<Event> out;
    for (size_t i = (size_t)after; i < events_.size(); i++) out.push_back(events_[i]);
    return out;
}

bool Job::finished() const { std::lock_guard<std::mutex> lk(m_); return finished_; }
std::string Job::state() const { std::lock_guard<std::mutex> lk(m_); return state_; }
void Job::set_state(const std::string& s) { std::lock_guard<std::mutex> lk(m_); state_ = s; }

void Job::finish(const std::string& final_state) {
    std::lock_guard<std::mutex> lk(m_);
    state_ = final_state;
    finished_ = true;
    cv_.notify_all();
}

// ---- Sink con regulación de eventos ------------------------------------------------------------------------

namespace {

// Regula la cadencia de eventos de progreso: como máximo unos ~400 puntos por ejecución, repartidos de
// forma aproximadamente logarítmica en n_done (las curvas de convergencia usan eje logarítmico) y
// con un mínimo de 40 ms entre eventos.
class JobSink : public ProgressSink {
public:
    JobSink(Job& job) : job_(job), t0_(std::chrono::steady_clock::now()) {}

    void on_snapshot(const Snapshot& s) override {
        const bool structural = (s.stage != Stage::Main);        // plan, piloto, niveles, duplicaciones, fin
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        if (!structural && !s.is_final) {
            const bool grew = last_n_ == 0 || (double)s.n_done >= 1.03 * (double)last_n_;
            if (!grew || now - last_t_ < 0.04) return;
        }
        last_n_ = s.n_done;
        last_t_ = now;
        json j = snapshot_json(s);
        job_.push("progress", j);
    }

    bool should_cancel() const override { return job_.cancel.load(); }

private:
    Job& job_;
    std::chrono::steady_clock::time_point t0_;
    long long last_n_ = 0;
    double last_t_ = -1.0;
};

} // namespace

// ---- JobManager -----------------------------------------------------------------------------------------------

JobManager::JobManager(int hw_threads, bool cuda_ok) : hw_threads_(hw_threads), cuda_ok_(cuda_ok) {
    worker_ = std::thread([this] { worker(); });
}

JobManager::~JobManager() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
        for (auto& [id, job] : jobs_) job->cancel = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    // Que los clientes SSE/sondeo pendientes vean el fin
    std::lock_guard<std::mutex> lk(m_);
    for (auto& [id, job] : jobs_) if (!job->finished()) job->finish("aborted");
}

std::shared_ptr<Job> JobManager::submit(const json& body) {
    ParsedRun run = parse_run_request(body, hw_threads_, cuda_ok_);   // valida antes de encolar
    std::lock_guard<std::mutex> lk(m_);
    auto job = std::make_shared<Job>("j" + std::to_string(++counter_));
    jobs_[job->id()] = job;
    order_.push_back(job->id());
    while (order_.size() > 60) {                                       // conserva los últimos 60
        const std::string old = order_.front();
        order_.pop_front();
        auto it = jobs_.find(old);
        if (it != jobs_.end() && it->second->finished()) jobs_.erase(it);
        else if (it != jobs_.end()) { order_.push_back(old); break; }
    }
    queue_.emplace_back(job, std::move(run));
    cv_.notify_all();
    return job;
}

std::shared_ptr<Job> JobManager::find(const std::string& id) const {
    std::lock_guard<std::mutex> lk(m_);
    auto it = jobs_.find(id);
    return it == jobs_.end() ? nullptr : it->second;
}

bool JobManager::cancel(const std::string& id) {
    std::lock_guard<std::mutex> lk(m_);
    auto it = jobs_.find(id);
    if (it == jobs_.end()) return false;
    it->second->cancel = true;
    return true;
}

bool JobManager::busy() const {
    std::lock_guard<std::mutex> lk(m_);
    return running_.load() || !queue_.empty();
}

void JobManager::worker() {
    for (;;) {
        std::pair<std::shared_ptr<Job>, ParsedRun> item;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            item = std::move(queue_.front());
            queue_.pop_front();
            running_ = true;
        }
        run_job(item.first, item.second);
        running_ = false;
    }
}

void JobManager::run_job(const std::shared_ptr<Job>& job, const ParsedRun& run) {
    if (job->cancel.load()) {
        job->push("cancelled", json{{"type", "cancelled"}, {"partial", nullptr}});
        job->finish("cancelled");
        return;
    }
    job->set_state("running");
    const int threads = run.opt.backend == mc::Backend::Cpu
        ? (run.opt.threads > 0 ? run.opt.threads : hw_threads_) : 1;
    job->push("started", json{{"type", "started"}, {"job_id", job->id()}, {"threads", threads},
                              {"backend", mc::backend_name(run.opt.backend)}, {"request", run.echo}});
    try {
        JobSink sink(*job);
        mc::RunOptions opt = run.opt;
        opt.sink = &sink;
        mc::RunReport rep = mc::run(run.model, run.payoff, run.spec, opt);

        std::optional<ReferencePrice> ref;
        try { ref = reference_price(run.model, run.payoff, rep.n_steps); } catch (...) {}
        json res = report_json(run, rep, ref);
        if (rep.cancelled) {
            res["type"] = "cancelled";
            job->push("cancelled", json{{"type", "cancelled"}, {"partial", res}});
            job->finish("cancelled");
        } else {
            job->push("result", res);
            job->finish("done");
        }
    } catch (const std::exception& e) {
        job->push("error", json{{"type", "error"}, {"message", e.what()}});
        job->finish("error");
    }
}

} // namespace mc::gui
