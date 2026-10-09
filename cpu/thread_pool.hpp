#pragma once
// Pool de hilos propio (sin OpenMP) para el motor CPU.
//
// `parallel_for(n_tasks, fn)` reparte los índices 0..n_tasks-1 dinámicamente entre los
// hilos (contador atómico) y bloquea hasta que todos terminan; el hilo que llama también
// trabaja. Cada tarea escribe SOLO en su propia ranura de salida (out[task]) y la
// reducción se hace después, EN ORDEN DE ÍNDICE, por el hilo que llama: el resultado no
// depende de qué hilo ejecutó qué tarea ni de cuántos hilos hay (ver reduce.hpp).
//
//  * Las excepciones lanzadas por las tareas se capturan (la primera) y se relanzan en el
//    hilo que llama tras terminar la ronda.
//  * Cancelación cooperativa: si `cancel` pasa a true, los hilos dejan de tomar tareas
//    nuevas; parallel_for devuelve false (algunas ranuras de salida quedan sin escribir).
//  * Llamadas anidadas (desde dentro de una tarea) se ejecutan en serie en ese hilo.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mc::cpu {

class ThreadPool {
public:
    // n_threads = paralelismo total incluyendo al hilo que llama (>= 1).
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int threads() const { return n_threads_; }

    // fn(task, worker): worker en [0, threads()).
    // Devuelve true si se ejecutaron todas las tareas (false si se canceló).
    bool parallel_for(std::size_t n_tasks,
                      const std::function<void(std::size_t task, int worker)>& fn,
                      const std::atomic<bool>* cancel = nullptr);

private:
    void worker_loop(int id);
    void run_tasks(int worker);

    int n_threads_;
    std::vector<std::thread> workers_;

    std::mutex call_mutex_;            // serializa llamadas concurrentes a parallel_for
    std::mutex m_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    bool shutdown_ = false;
    uint64_t generation_ = 0;
    int active_ = 0;                   // trabajadores aún dentro de la ronda actual

    // Estado de la ronda actual
    const std::function<void(std::size_t, int)>* fn_ = nullptr;
    const std::atomic<bool>* cancel_ = nullptr;
    std::size_t n_tasks_ = 0;
    std::atomic<std::size_t> next_{0};
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
};

// Pool global compartido por los motores. 0 = hardware_concurrency().
// No cambiar el nº de hilos mientras haya una ejecución en curso.
void set_num_threads(int n);
int  num_threads();
ThreadPool& global_pool();

} // namespace mc::cpu
