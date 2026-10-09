#include "thread_pool.hpp"

#include <algorithm>
#include <memory>

namespace mc::cpu {

namespace {
thread_local bool tl_in_pool_task = false;
}

ThreadPool::ThreadPool(int n_threads) : n_threads_(std::max(1, n_threads)) {
    workers_.reserve(n_threads_ - 1);
    for (int i = 1; i < n_threads_; i++)
        workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        shutdown_ = true;
    }
    cv_work_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::run_tasks(int worker) {
    const bool prev = tl_in_pool_task;
    tl_in_pool_task = true;
    for (;;) {
        if (failed_.load(std::memory_order_relaxed)) break;
        if (cancel_ && cancel_->load(std::memory_order_relaxed)) break;
        const std::size_t t = next_.fetch_add(1, std::memory_order_relaxed);
        if (t >= n_tasks_) break;
        try {
            (*fn_)(t, worker);
        } catch (...) {
            std::lock_guard<std::mutex> lk(m_);
            if (!error_) error_ = std::current_exception();
            failed_.store(true, std::memory_order_relaxed);
        }
    }
    tl_in_pool_task = prev;
}

void ThreadPool::worker_loop(int id) {
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_work_.wait(lk, [&] { return shutdown_ || generation_ != seen; });
            if (shutdown_) return;
            seen = generation_;
        }
        run_tasks(id);
        {
            std::lock_guard<std::mutex> lk(m_);
            if (--active_ == 0) cv_done_.notify_all();
        }
    }
}

bool ThreadPool::parallel_for(std::size_t n_tasks,
                              const std::function<void(std::size_t, int)>& fn,
                              const std::atomic<bool>* cancel) {
    if (n_tasks == 0) return true;

    // Anidado, pool de un solo hilo o una sola tarea: en serie en el hilo actual (despertar a los
    // demás hilos para que no encuentren nada cuesta más que la tarea de un piloto pequeño).
    if (tl_in_pool_task || n_threads_ == 1 || n_tasks == 1) {
        for (std::size_t t = 0; t < n_tasks; t++) {
            if (cancel && cancel->load(std::memory_order_relaxed)) return false;
            fn(t, 0);
        }
        return true;
    }

    std::lock_guard<std::mutex> call_lock(call_mutex_);
    {
        std::lock_guard<std::mutex> lk(m_);
        fn_ = &fn;
        cancel_ = cancel;
        n_tasks_ = n_tasks;
        next_.store(0, std::memory_order_relaxed);
        failed_.store(false, std::memory_order_relaxed);
        error_ = nullptr;
        active_ = n_threads_ - 1;
        ++generation_;
    }
    cv_work_.notify_all();

    run_tasks(0);   // el hilo que llama también trabaja

    std::exception_ptr err;
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [&] { return active_ == 0; });
        err = error_;
        fn_ = nullptr;
        cancel_ = nullptr;
    }
    if (err) std::rethrow_exception(err);
    return !(cancel && cancel->load(std::memory_order_relaxed));
}

// ---- pool global --------------------------------------------------------------------------

namespace {
std::mutex g_pool_mutex;
std::unique_ptr<ThreadPool> g_pool;

int resolve_threads(int n) {
    if (n > 0) return n;
    unsigned hc = std::thread::hardware_concurrency();
    return hc ? (int)hc : 1;
}
}

void set_num_threads(int n) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    const int want = resolve_threads(n);
    if (!g_pool || g_pool->threads() != want) g_pool = std::make_unique<ThreadPool>(want);
}

int num_threads() { return global_pool().threads(); }

ThreadPool& global_pool() {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    if (!g_pool) g_pool = std::make_unique<ThreadPool>(resolve_threads(0));
    return *g_pool;
}

} // namespace mc::cpu
