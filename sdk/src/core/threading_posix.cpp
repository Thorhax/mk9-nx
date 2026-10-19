/**
 * @file        core/threading_posix.cpp
 * @brief       POSIX platform threading implementations
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 * @license     BSD 3-Clause License
 */

#include <rex/platform.h>
#include <rex/thread.h>

static_assert(REX_PLATFORM_LINUX || REX_PLATFORM_MAC, "This file is POSIX-only");

#include <signal.h>

#include <atomic>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <ctime>
#include <deque>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <pthread.h>
#include <semaphore.h>

#if defined(__APPLE__)
#include <cstdio>

#include <fcntl.h>
#endif
#if defined(__SWITCH__)
// Horizon has no sys/syscall.h. The shim provides affinity, priority, cpu_set_t
// and the thread id. See core/threading_switch.h.
#include "threading_switch.h"
// The pool of thread stacks (switch_reaper below) allocates the blocks the way
// libnx would (memalign) and sizes them the way libnx carves them (struct _reent).
#include <malloc.h>
#include <sys/reent.h>
#else
#include <sys/syscall.h>
#endif
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <rex/assert.h>
#include <rex/chrono/chrono_steady_cast.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/sys_counters.h>
#include <rex/thread/timer_queue.h>

#include <sched.h>

#if REX_PLATFORM_ANDROID
#include <dlfcn.h>

#include <rex/main_android.h>
#include <rex/string.h>
#endif

#if REX_PLATFORM_LINUX
// SIGEV_THREAD_ID in timer_create(...) is a Linux extension
#define REX_HAS_SIGEV_THREAD_ID 1
#ifdef __GLIBC__
#define sigev_notify_thread_id _sigev_un._tid
#endif
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#define gettid() syscall(SYS_gettid)
#endif
#else
#define REX_HAS_SIGEV_THREAD_ID 0
#endif

REXCVAR_DEFINE_BOOL(thread_wait_blocking, true, "Threading",
                    "Event-driven waits: WaitMultiple parks on a wake point instead of polling every 1 ms, "
                    "and alertable waits are woken by the queued APC hint instead of a 1 ms poll slice. "
                    "false = the original 1 ms polling (A/B; also the fallback if a hang is suspected)");
REXCVAR_DEFINE_INT32(thread_alertable_slice_ms, 8, "Threading",
                     "With thread_wait_blocking: longest an alertable wait sleeps before it re-checks the "
                     "APC queue (safety net; the wake-up normally arrives through the queued callback). "
                     "Ignored (1 ms) with thread_wait_blocking = false");
REXCVAR_DEFINE_INT32(thread_multiwait_safety_ms, 20, "Threading",
                     "With thread_wait_blocking: a parked WaitMultiple re-polls its handles at least this "
                     "often even if nobody poked it (safety net against a state change that did not notify)");

namespace rex::thread {

#if REX_PLATFORM_ANDROID
// May be null if no dynamically loaded functions are required.
static void* android_libc_;
// API 26+.
static int (*android_pthread_getname_np_)(pthread_t pthread, char* buf, size_t n);

void AndroidInitialize() {
  if (rex::GetAndroidApiLevel() >= 26) {
    android_libc_ = dlopen("libc.so", RTLD_NOW);
    assert_not_null(android_libc_);
    if (android_libc_) {
      android_pthread_getname_np_ = reinterpret_cast<decltype(android_pthread_getname_np_)>(
          dlsym(android_libc_, "pthread_getname_np"));
      assert_not_null(android_pthread_getname_np_);
    }
  }
}

void AndroidShutdown() {
  android_pthread_getname_np_ = nullptr;
  if (android_libc_) {
    dlclose(android_libc_);
    android_libc_ = nullptr;
  }
}
#endif

template <typename _Rep, typename _Period>
inline timespec DurationToTimeSpec(std::chrono::duration<_Rep, _Period> duration) {
  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  auto div = ldiv(nanoseconds.count(), 1000000000L);
  return timespec{div.quot, div.rem};
}

// Thread interruption is done using user-defined signals
// This implementation uses the SIGRTMAX - SIGRTMIN to signal to a thread
// gdb tip, for SIG = SIGRTMIN + SignalType : handle SIG nostop
// lldb tip, for SIG = SIGRTMIN + SignalType : process handle SIG -s false
enum class SignalType {
  kThreadSuspend,
  kThreadUserCallback,
#if REX_PLATFORM_ANDROID
  // pthread_cancel is not available on Android, using a signal handler for
  // simplified PTHREAD_CANCEL_ASYNCHRONOUS-like behavior - not disabling
  // cancellation currently, so should be enough.
  kThreadTerminate,
#endif
  k_Count
};

int GetSystemSignal(SignalType num) {
#if defined(__APPLE__) || defined(__SWITCH__)
  // macOS has no realtime signals (SIGRTMIN..SIGRTMAX); map the runtime's
  // signals onto the two user-defined signals.
  switch (num) {
    case SignalType::kThreadSuspend:
      return SIGUSR1;
    case SignalType::kThreadUserCallback:
      return SIGUSR2;
    default:
      assert_unhandled_case(num);
      return SIGUSR1;
  }
#else
  auto result = SIGRTMIN + static_cast<int>(num);
  assert_true(result < SIGRTMAX);
  return result;
#endif
}

SignalType GetSystemSignalType(int num) {
#if defined(__APPLE__) || defined(__SWITCH__)
  switch (num) {
    case SIGUSR1:
      return SignalType::kThreadSuspend;
    case SIGUSR2:
      return SignalType::kThreadUserCallback;
    default:
      assert_unhandled_case(num);
      return SignalType::kThreadSuspend;
  }
#else
  return static_cast<SignalType>(num - SIGRTMIN);
#endif
}

std::array<std::atomic<bool>, static_cast<size_t>(SignalType::k_Count)> signal_handler_installed =
    {};

static void signal_handler(int signal, siginfo_t* info, void* context);

void install_signal_handler(SignalType type) {
#if defined(__SWITCH__)
  // Horizon does not deliver signals to user processes. Installing the handler
  // would not fail, it would simply never be called, and that is worse than not
  // doing it: it would suggest that signal-based suspension works.
  //
  // Real suspension goes through svcSetThreadActivity, in Suspend()/Resume().
  // See core/threading_switch.h.
  (void)type;
#else
  bool expected = false;
  if (!signal_handler_installed[static_cast<size_t>(type)].compare_exchange_strong(expected,
                                                                                   true)) {
    return;
  }
  struct sigaction action{};
  action.sa_flags = SA_SIGINFO | SA_RESTART;
  action.sa_sigaction = signal_handler;
  sigemptyset(&action.sa_mask);
  if (sigaction(GetSystemSignal(type), &action, nullptr) != 0) {
    signal_handler_installed[static_cast<size_t>(type)] = false;
  }
#endif
}

// TODO(dougvj)
void EnableAffinityConfiguration() {}

// uint64_t ticks() { return mach_absolute_time(); }

uint32_t current_thread_system_id() {
#if defined(__APPLE__)
  return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__SWITCH__)
  return RexSwitchCurrentThreadId();
#else
  return static_cast<uint32_t>(syscall(SYS_gettid));
#endif
}

void MaybeYield() {
  sched_yield();
  __sync_synchronize();
}

void SyncMemory() {
  __sync_synchronize();
}

void Sleep(std::chrono::microseconds duration) {
  timespec rqtp = DurationToTimeSpec(duration);
  timespec rmtp = {};
  auto p_rqtp = &rqtp;
  auto p_rmtp = &rmtp;
  int ret = 0;
  do {
    ret = nanosleep(p_rqtp, p_rmtp);
    // Swap requested for remaining in case of signal interruption
    // in which case, we start sleeping again for the remainder
    std::swap(p_rqtp, p_rmtp);
  } while (ret == -1 && errno == EINTR);
}

// AlertableSleep: defined below, after the condition classes (it parks on a wake point).
thread_local bool alertable_state_ = false;
bool DispatchCurrentThreadUserCallback();

TlsHandle AllocateTlsHandle() {
  auto key = static_cast<pthread_key_t>(-1);
  auto res = pthread_key_create(&key, nullptr);
  assert_zero(res);
  assert_true(key != static_cast<pthread_key_t>(-1));
  return static_cast<TlsHandle>(key);
}

bool FreeTlsHandle(TlsHandle handle) {
  return pthread_key_delete(static_cast<pthread_key_t>(handle)) == 0;
}

uintptr_t GetTlsValue(TlsHandle handle) {
  return reinterpret_cast<uintptr_t>(pthread_getspecific(static_cast<pthread_key_t>(handle)));
}

bool SetTlsValue(TlsHandle handle, uintptr_t value) {
  return pthread_setspecific(static_cast<pthread_key_t>(handle), reinterpret_cast<void*>(value)) ==
         0;
}

#if defined(__APPLE__)
// macOS does not implement unnamed POSIX semaphores (sem_init fails with
// ENOSYS). Emulate one with a named semaphore that is unlinked immediately
// after creation: the name is released but the descriptor stays valid until
// sem_close, and sem_wait/sem_post remain async-signal-safe (needed because
// WaitSuspended() runs from a signal handler).
static sem_t* RexCreateAnonymousSemaphore() {
  static std::atomic<uint64_t> counter{0};
  char name[64];
  std::snprintf(name, sizeof(name), "/rexsusp_%d_%llu", static_cast<int>(getpid()),
                static_cast<unsigned long long>(counter.fetch_add(1, std::memory_order_relaxed)));
  sem_t* sem = sem_open(name, O_CREAT | O_EXCL, 0600, 0);
  if (sem != SEM_FAILED) {
    sem_unlink(name);
  }
  return sem;
}
#endif  // __APPLE__

// A thread parked in a blocking WaitMultiple (or an alertable wait) sleeps on one of these. Handles that
// change state poke the wake points registered on them; QueueUserCallback pokes the one of the target thread.
// Lifetime: owned by the waiter's stack; it is unregistered from every handle (which takes that handle's
// registration mutex, so it also waits for a poker that is inside Poke) before it is destroyed.
struct WakePoint {
  std::mutex m;
  std::condition_variable cv;
  std::atomic<uint64_t> seq{0};

  void Poke() {
    std::lock_guard<std::mutex> lock(m);
    seq.fetch_add(1, std::memory_order_relaxed);
    cv.notify_all();
  }
};

class PosixConditionBase {
 public:
  PosixConditionBase() {
#if REX_PLATFORM_LINUX
    // Use robust mutexes so waits can recover if owner thread terminates.
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr) == 0) {
      if (pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST) == 0) {
        auto native_mutex = static_cast<pthread_mutex_t*>(mutex_.native_handle());
        pthread_mutex_destroy(native_mutex);
        pthread_mutex_init(native_mutex, &attr);
      }
      pthread_mutexattr_destroy(&attr);
    }
#endif
  }

  virtual ~PosixConditionBase() = default;
  virtual bool Signal() = 0;

  // Wake points of the WaitMultiple calls currently parked on this handle.
  void RegisterMulti(WakePoint* wp) {
    std::lock_guard<std::mutex> lock(multi_mutex_);
    multi_waiters_.push_back(wp);
    multi_count_.fetch_add(1, std::memory_order_seq_cst);
  }
  void UnregisterMulti(WakePoint* wp) {
    std::lock_guard<std::mutex> lock(multi_mutex_);
    for (auto it = multi_waiters_.begin(); it != multi_waiters_.end(); ++it) {
      if (*it == wp) {
        multi_waiters_.erase(it);
        break;
      }
    }
    multi_count_.fetch_sub(1, std::memory_order_seq_cst);
  }
  // Called by every state change that can satisfy a wait, AFTER the state was written (and its lock
  // released). One relaxed-cost load when nobody is parked.
  void NotifyMulti() {
    if (multi_count_.load(std::memory_order_seq_cst) == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(multi_mutex_);
    for (WakePoint* wp : multi_waiters_) {
      wp->Poke();
    }
  }
  // Wakes a thread blocked in Wait() on this condition without signalling it (alertable wait interrupted by
  // a queued callback). The waiter's predicate sees its interrupt flag.
  void InterruptWaiters() {
    std::lock_guard<std::mutex> lock(mutex_);
    cond_.notify_all();
  }

  // `interrupt` (optional): an alertable wait passes the flag "a user callback is queued for me"; the wait
  // then returns kTimeout early (the caller dispatches the callback and decides what to do next).
  WaitResult Wait(std::chrono::milliseconds timeout,
                  const std::atomic<bool>* interrupt = nullptr) {
    bool executed;
    auto predicate = [this, interrupt] {
      return this->signaled() || (interrupt && interrupt->load(std::memory_order_acquire));
    };
#if defined(__SWITCH__)
    // The timeout includes acquiring the internal lock. lock() ignored it, and a
    // zero-timeout query could block behind another thread indefinitely.
    const auto deadline = timeout == std::chrono::milliseconds::max()
                              ? std::chrono::steady_clock::time_point::max()
                              : std::chrono::steady_clock::now() + timeout;
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (timeout == std::chrono::milliseconds::max()) {
      lock.lock();
    } else {
      while (!lock.try_lock()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
          return WaitResult::kTimeout;
        }
        // Sleeping lets the owner make progress even with a lower priority.
        const auto remaining = std::chrono::ceil<std::chrono::microseconds>(deadline - now);
        std::this_thread::sleep_for(std::min(remaining, std::chrono::microseconds(100)));
      }
    }
#elif REX_PLATFORM_LINUX
    auto native_mutex = static_cast<pthread_mutex_t*>(mutex_.native_handle());
    int lock_result = pthread_mutex_lock(native_mutex);
    if (lock_result == EOWNERDEAD) {
      pthread_mutex_consistent(native_mutex);
    } else if (lock_result != 0) {
      return WaitResult::kFailed;
    }
    std::unique_lock<std::mutex> lock(mutex_, std::adopt_lock);
#else
    std::unique_lock<std::mutex> lock(mutex_);
#endif
    if (predicate()) {
      executed = true;
    } else {
      if (timeout == std::chrono::milliseconds::max()) {
        cond_.wait(lock, predicate);
        executed = true;  // Did not time out;
      } else {
#if defined(__SWITCH__)
        // Do not start another full timeout after acquiring the lock.
        executed = cond_.wait_until(lock, deadline, predicate);
#else
        executed = cond_.wait_for(lock, timeout, predicate);
#endif
      }
    }
    if (executed) {
      if (interrupt && !signaled()) {
        return WaitResult::kTimeout;  // interrupted by a queued user callback
      }
      post_execution();
      return WaitResult::kSuccess;
    } else {
      return WaitResult::kTimeout;
    }
  }

  // `wake` (optional): wake point registered by the caller (and, for alertable calls, with its thread). With
  // it the call parks instead of sleeping 1 ms between polls: it re-polls when a registered handle is
  // signalled (poke), when `interrupt` is raised, at the end of the timeout, or after the safety interval.
  static std::pair<WaitResult, size_t> WaitMultiple(std::vector<PosixConditionBase*>&& handles,
                                                    bool wait_all,
                                                    std::chrono::milliseconds timeout,
                                                    WakePoint* wake = nullptr,
                                                    const std::atomic<bool>* interrupt = nullptr) {
    assert_true(!handles.empty());

    if (handles.size() == 1) {
      auto result = handles[0]->Wait(timeout, interrupt);
      return std::make_pair(result, 0);
    }

    auto start_time = std::chrono::steady_clock::now();
    auto end_time = (timeout == std::chrono::milliseconds::max())
                        ? std::chrono::steady_clock::time_point::max()
                        : start_time + timeout;

    // Registered for the whole call; the guard unregisters on every exit path.
    struct MultiRegistration {
      std::vector<PosixConditionBase*>& handles;
      WakePoint* wake;
      MultiRegistration(std::vector<PosixConditionBase*>& h, WakePoint* w) : handles(h), wake(w) {
        if (wake) {
          for (auto* handle : handles) handle->RegisterMulti(wake);
        }
      }
      ~MultiRegistration() {
        if (wake) {
          for (auto* handle : handles) handle->UnregisterMulti(wake);
        }
      }
    } registration(handles, wake);
    const auto safety = std::chrono::milliseconds(std::max(1, REXCVAR_GET(thread_multiwait_safety_ms)));

    // Reuse the allocation between polls; clear() releases the locks, not the memory.
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(handles.size());
    while (true) {
      // Sampled BEFORE the poll: a handle signalled after this point bumps the sequence, so the park
      // below returns at once instead of missing it.
      const uint64_t seen = wake ? wake->seq.load(std::memory_order_acquire) : 0;
      size_t first_signaled = std::numeric_limits<size_t>::max();
      bool condition_met = false;
      bool all_locked = true;

      for (size_t i = 0; i < handles.size(); ++i) {
#if REX_PLATFORM_LINUX
        auto native_mutex = static_cast<pthread_mutex_t*>(handles[i]->mutex_.native_handle());
        int result = pthread_mutex_trylock(native_mutex);
        if (result == 0 || result == EOWNERDEAD) {
          if (result == EOWNERDEAD) {
            pthread_mutex_consistent(native_mutex);
          }
          locks.emplace_back(handles[i]->mutex_, std::adopt_lock);
        } else {
          all_locked = false;
          break;
        }
#else
        locks.emplace_back(handles[i]->mutex_, std::try_to_lock);
        if (!locks.back().owns_lock()) {
          all_locked = false;
          break;
        }
#endif
      }

      if (!all_locked) {
        locks.clear();
        // The timeout also expires when an object cannot be inspected.
        // The earlier continue skipped the only timeout check.
        const auto now = std::chrono::steady_clock::now();
        if (now >= end_time) {
          return std::make_pair<WaitResult, size_t>(WaitResult::kTimeout, 0);
        }
#if defined(__SWITCH__)
        // On Horizon yielding is useless against a lower-priority thread:
        // svcSleepThread(0) only yields to threads of the same priority. If the
        // lock holder has lower priority and is on this core, it would never get
        // to release the lock. A real sleep is needed.
        const auto pause = timeout == std::chrono::milliseconds::max()
                               ? std::chrono::microseconds(100)
                               : std::min(std::chrono::ceil<std::chrono::microseconds>(end_time - now),
                                          std::chrono::microseconds(100));
        std::this_thread::sleep_for(pause);
#else
        std::this_thread::yield();
#endif
        continue;
      }

      if (wait_all) {
        bool all_signaled = true;
        for (size_t i = 0; i < handles.size(); ++i) {
          if (!handles[i]->signaled()) {
            all_signaled = false;
            break;
          }
          if (first_signaled == std::numeric_limits<size_t>::max()) {
            first_signaled = i;
          }
        }
        condition_met = all_signaled;
      } else {
        for (size_t i = 0; i < handles.size(); ++i) {
          if (handles[i]->signaled()) {
            first_signaled = i;
            condition_met = true;
            break;
          }
        }
      }

      if (condition_met) {
        if (wait_all) {
          for (size_t i = 0; i < handles.size(); ++i) {
            handles[i]->post_execution();
          }
        } else {
          handles[first_signaled]->post_execution();
        }
        return std::make_pair(WaitResult::kSuccess, first_signaled);
      }

      locks.clear();

      auto now = std::chrono::steady_clock::now();
      if (now >= end_time) {
        return std::make_pair<WaitResult, size_t>(WaitResult::kTimeout, 0);
      }

      if (wake) {
        // An interrupt raised before we got here is handled by the caller (it dispatches the callback).
        if (interrupt && interrupt->load(std::memory_order_acquire)) {
          return std::make_pair<WaitResult, size_t>(WaitResult::kTimeout, 0);
        }
        auto wake_deadline = now + safety;
        if (end_time < wake_deadline) {
          wake_deadline = end_time;
        }
        REX_SYS_COUNT(rex::syscount::kMultiBlock, 1);
        std::unique_lock<std::mutex> park(wake->m);
        const bool woke = wake->cv.wait_until(park, wake_deadline, [&] {
          return wake->seq.load(std::memory_order_relaxed) != seen ||
                 (interrupt && interrupt->load(std::memory_order_acquire));
        });
        park.unlock();
        if (woke) {
          if (interrupt && interrupt->load(std::memory_order_acquire)) {
            return std::make_pair<WaitResult, size_t>(WaitResult::kTimeout, 0);
          }
          REX_SYS_COUNT(rex::syscount::kMultiWake, 1);
        } else if (std::chrono::steady_clock::now() < end_time) {
          REX_SYS_COUNT(rex::syscount::kMultiSafety, 1);
        }
        continue;
      }

      REX_SYS_COUNT(rex::syscount::kMultiPoll, 1);
      if (timeout == std::chrono::milliseconds::max()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        // In microseconds. In milliseconds, the remainder of a short wait was
        // truncated to 0 and sleep_for(0) does not sleep: the alertable wait asks
        // for 1 ms slices, so WaitAny spun for its whole timeout. On Switch the
        // audio thread (0x2B) ate a whole core this way.
        auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(end_time - now);
        auto sleep_time = std::min(remaining, std::chrono::microseconds(1000));
        std::this_thread::sleep_for(sleep_time);
      }
    }
  }

  virtual void* native_handle() const {
    return const_cast<std::condition_variable&>(cond_).native_handle();
  }

 protected:
  inline virtual bool signaled() const = 0;
  inline virtual void post_execution() = 0;
  std::condition_variable cond_;
  std::mutex mutex_;

 private:
  std::mutex multi_mutex_;
  std::vector<WakePoint*> multi_waiters_;
  std::atomic<uint32_t> multi_count_{0};
};

// There really is no native POSIX handle for a single wait/signal construct
// pthreads is at a lower level with more handles for such a mechanism.
// This simple wrapper class functions as our handle and uses conditional
// variables for waits and signals.
template <typename T>
class PosixCondition {};

template <>
class PosixCondition<Event> : public PosixConditionBase {
 public:
  PosixCondition(bool manual_reset, bool initial_state)
      : signal_(initial_state), manual_reset_(manual_reset) {}
  virtual ~PosixCondition() = default;

  bool Signal() override {
    {
      auto lock = std::unique_lock<std::mutex>(mutex_);
      signal_ = true;
    }
    // Notified after the lock is released: a woken waiter (on Horizon the XMA decoder runs above the guest
    // threads) would otherwise run straight into the mutex its signaller still holds and go back to sleep.
    cond_.notify_all();
    NotifyMulti();
    return true;
  }

  void Reset() {
    auto lock = std::unique_lock<std::mutex>(mutex_);
    signal_ = false;
  }

 private:
  inline bool signaled() const override { return signal_; }
  inline void post_execution() override {
    if (!manual_reset_) {
      signal_ = false;
    }
  }
  bool signal_;
  const bool manual_reset_;
};

template <>
class PosixCondition<Semaphore> : public PosixConditionBase {
 public:
  PosixCondition(uint32_t initial_count, uint32_t maximum_count)
      : count_(initial_count), maximum_count_(maximum_count) {}

  bool Signal() override { return Release(1, nullptr); }

  bool Release(uint32_t release_count, int* out_previous_count) {
    auto lock = std::unique_lock<std::mutex>(mutex_);
    if (release_count > maximum_count_ - count_) {
      return false;
    }
    if (out_previous_count) {
      *out_previous_count = count_;
    }
    count_ += release_count;
    lock.unlock();
    cond_.notify_all();
    NotifyMulti();
    return true;
  }

 private:
  inline bool signaled() const override { return count_ > 0; }
  inline void post_execution() override {
    count_--;
    cond_.notify_all();
  }
  uint32_t count_;
  const uint32_t maximum_count_;
};

template <>
class PosixCondition<Mutant> : public PosixConditionBase {
 public:
  explicit PosixCondition(bool initial_owner) : count_(0) {
    if (initial_owner) {
      count_ = 1;
      owner_ = std::this_thread::get_id();
    }
  }

  bool Signal() override { return Release(); }

  bool Release() {
    if (owner_ == std::this_thread::get_id() && count_ > 0) {
      bool released = false;
      {
        auto lock = std::unique_lock<std::mutex>(mutex_);
        --count_;
        // Free to be acquired by another thread
        released = count_ == 0;
      }
      if (released) {
        cond_.notify_all();
        NotifyMulti();
      }
      return true;
    }
    return false;
  }

  void* native_handle() const override { return const_cast<std::mutex&>(mutex_).native_handle(); }

 private:
  inline bool signaled() const override {
    return count_ == 0 || owner_ == std::this_thread::get_id();
  }
  inline void post_execution() override {
    count_++;
    owner_ = std::this_thread::get_id();
  }
  uint32_t count_;
  std::thread::id owner_;
};

template <>
class PosixCondition<Timer> : public PosixConditionBase {
 public:
  explicit PosixCondition(bool manual_reset)
      : callback_(nullptr), signal_(false), manual_reset_(manual_reset) {}

  virtual ~PosixCondition() { Cancel(); }

  bool Signal() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      signal_ = true;
    }
    cond_.notify_all();
    NotifyMulti();
    return true;
  }

  void SetOnce(std::chrono::steady_clock::time_point due_time, std::function<void()> opt_callback) {
    Cancel();

    std::lock_guard<std::mutex> lock(mutex_);

    callback_ = std::move(opt_callback);
    signal_ = false;
    wait_item_ = QueueTimerOnce(&CompletionRoutine, this, due_time);
  }

  void SetRepeating(std::chrono::steady_clock::time_point due_time,
                    std::chrono::milliseconds period, std::function<void()> opt_callback) {
    Cancel();

    std::lock_guard<std::mutex> lock(mutex_);

    callback_ = std::move(opt_callback);
    signal_ = false;
    wait_item_ = QueueTimerRecurring(&CompletionRoutine, this, due_time, period);
  }

  void Cancel() {
    if (auto wait_item = wait_item_.lock()) {
      wait_item->Disarm();
    }
  }

  void* native_handle() const override {
    assert_always();
    return nullptr;
  }

 private:
  static void CompletionRoutine(void* userdata) {
    assert_not_null(userdata);
    auto timer = reinterpret_cast<PosixCondition<Timer>*>(userdata);
    timer->Signal();
    // As the callback may reset the timer, store local.
    std::function<void()> callback;
    {
      std::lock_guard<std::mutex> lock(timer->mutex_);
      callback = timer->callback_;
    }
    if (callback) {
      callback();
    }
  }

 private:
  inline bool signaled() const override { return signal_; }
  inline void post_execution() override {
    if (!manual_reset_) {
      signal_ = false;
    }
  }
  std::weak_ptr<TimerQueueWaitItem> wait_item_;
  std::function<void()> callback_;
  bool signal_;  // Protected by mutex_
  const bool manual_reset_;
};

struct ThreadStartData {
  std::function<void()> start_routine;
  bool create_suspended;
  Thread* thread_obj;
};

#if defined(__SWITCH__)
/*
 * sa2-nx: the reaper of finished threads.
 *
 * libnx has no thread detach (newlib's pthread_detach calls __syscall_thread_detach, which libnx does not
 * implement), and a Horizon thread keeps its kernel thread slot and its stack (16 MB for a guest thread) until
 * pthread_join closes it. The SDK joined a thread only when guest code waited on it (post_execution). Sonic
 * Adventure 2 starts a one-shot "AsynDo" thread per asynchronous job and never waits on them: 58 of them piled up,
 * the process reached its thread limit (95 threads), and the level loader's CreateThread failed with
 * C0000017 - the loading screen then waited forever for a thread that did not exist.
 *
 * Each thread hands itself to this reaper as it finishes (ClaimJoin), and one reaper thread joins it a second
 * later, when the thread has surely exited and nothing still holds its handle for a call in flight. A wait that
 * comes after the claim no longer joins (the claim is a flag both sides take), so nothing is joined twice.
 *
 * One exception: when a creation fails, usually for want of what those finished threads still hold (a stack block,
 * a kernel slot), they are joined at once (JoinFinishedNow) instead of after their second. That is safe on Horizon:
 * pthread_join itself waits for the thread to exit, and no call made on another thread's pthread_t reads it there
 * (the shim looks handles up by value, pthread_cancel is a stub, and names are only touched before the thread is
 * marked finished, which comes before its claim).
 */
namespace switch_reaper {
struct Item {
  pthread_t thread;
  std::chrono::steady_clock::time_point due;
  void* stack;  // its stack block, from the pool below (null: libnx allocated that stack and frees it itself)
  size_t stack_bytes;
};
std::mutex g_mutex;
std::condition_variable g_cv;
std::deque<Item> g_queue;  // oldest first, hence in due order (JoinFinishedNow only brings due times in to now)
bool g_started = false;
std::atomic<uint64_t> g_reaped{0};
std::condition_variable g_drained;  // notified after each join, and when a stack block is kept in the pool
bool g_joining = false;             // Loop has taken an item and not joined it yet (under g_mutex)

/*
 * sa2-nx: the stacks of joined threads, kept for the next thread of the same size.
 *
 * Console, 2026-10-07 (the Tails boss stage and later): 77 creations of a guest thread failed with "heap has NO a
 * 16448 KB block" while the heap had 1.0-1.5 GB free, in ~10.8 k pieces. The only block that big was the stack of
 * the previous one-shot "AsynDo" thread, which went back to the heap when the reaper joined that thread, a second
 * after it ended, so each creation waited for that join: 0.3-1.0 s, a frame of ~1000 ms every time, 74 s in all. A
 * 16 MB block given back to the heap is also soon split by small allocations.
 *
 * So a big stack is a block of our own, from here or, when none is kept, from the heap. It goes to libnx through
 * pthread_attr_setstack: threadCreate maps it into the stack region without owning it (it carves the TLS and the
 * reent out of its top, as it does with its own allocation) and the join (threadClose) unmaps it. Then the block
 * comes back here instead of to the heap. At most kPoolMaxBlocks blocks and kPoolMaxBytes are kept; any other block
 * goes back to the heap.
 */
// Only the guest threads' 16 MB stacks. The SDK's own threads (4 MB by default, long-lived) stay with libnx as
// before, and their blocks never take the pool's places.
constexpr size_t kPoolMinStack = size_t(8) << 20;
constexpr size_t kPoolMaxBlocks = 4;
constexpr size_t kPoolMaxBytes = size_t(80) << 20;  // four guest stacks (16 MB each, with their TLS)
struct Stack {
  void* base;
  size_t bytes;
};
std::array<Stack, kPoolMaxBlocks> g_pool{};  // the first g_pool_count are kept blocks (under g_mutex)
size_t g_pool_count = 0;
size_t g_pool_bytes = 0;
std::atomic<uint64_t> g_stacks_new{0};     // stack blocks taken from the heap
std::atomic<uint64_t> g_stacks_reused{0};  // ... and from the pool
std::atomic<uint64_t> g_early_joins{0};    // creations that went through once JoinFinishedNow had joined threads

// libnx's linker script (switch.ld): the thread-local segment that threadCreate sets up in every new thread.
extern "C" char __tls_start[];
extern "C" char __tls_end[];
extern "C" size_t __tls_align;

// What threadCreate takes from the top of a stack it is handed (libnx, kernel/thread.c): the TLS segment and the
// newlib reent, rounded up to the TLS alignment, then 48 bytes of start arguments. Measured as libnx measures it.
size_t LibnxStackOverhead() {
  const size_t tls = (static_cast<size_t>(__tls_end - __tls_start) + 0xF) & ~size_t(0xF);
  const size_t reent = (sizeof(struct _reent) + 0xF) & ~size_t(0xF);
  const size_t align = std::max<size_t>(__tls_align, 2 * sizeof(void*));
  return ((tls + reent + align - 1) & ~(align - 1)) + 0x30;
}

// The block for a thread that asks for `stack_size`, or 0 when libnx allocates the stack as before. Whole pages
// (libnx requires it), and big enough that the thread keeps at least `stack_size` of stack (libnx's own allocation
// leaves it `stack_size` minus those 48 bytes). A TLS segment as big as a pooled stack is not expected: left to libnx.
size_t PooledStackBytes(size_t stack_size) {
  static const size_t overhead = LibnxStackOverhead();
  if (stack_size < kPoolMinStack || overhead > kPoolMinStack) {
    return 0;
  }
  return (stack_size + overhead + 0xFFF) & ~size_t(0xFFF);
}

// Takes a kept block of exactly `bytes` out of the pool, or returns null. Under g_mutex.
void* PopPooledLocked(size_t bytes) {
  for (size_t i = g_pool_count; i-- > 0;) {
    if (g_pool[i].bytes == bytes) {
      void* const base = g_pool[i].base;
      g_pool[i] = g_pool[--g_pool_count];
      g_pool_bytes -= bytes;
      return base;
    }
  }
  return nullptr;
}

// A block for a new thread's stack: a kept one, else a new one from the heap. Null when the heap has none either.
void* TakeStack(size_t bytes) {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (void* const base = PopPooledLocked(bytes)) {
      g_stacks_reused.fetch_add(1, std::memory_order_relaxed);
      return base;
    }
  }
  void* const base = memalign(0x1000, bytes);
  if (base != nullptr) {
    g_stacks_new.fetch_add(1, std::memory_order_relaxed);
  }
  return base;
}

// A block no thread runs on any more (its thread was joined, or never started): kept for the next thread while the
// pool has room, else given back to the heap.
void GiveStack(void* base, size_t bytes) {
  if (base == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_pool_count < kPoolMaxBlocks && g_pool_bytes + bytes <= kPoolMaxBytes) {
      g_pool[g_pool_count++] = {base, bytes};
      g_pool_bytes += bytes;
      g_drained.notify_all();  // a creation may be waiting for one (JoinFinishedNow)
      return;
    }
  }
  free(base);
}

void* Loop(void*) {
  std::unique_lock<std::mutex> lock(g_mutex);
  for (;;) {
    if (g_queue.empty()) {
      g_cv.wait(lock);
      continue;
    }
    // The oldest finished thread is joined when it is due, or sooner once JoinFinishedNow brought its due time in.
    const auto due = g_queue.front().due;
    if (std::chrono::steady_clock::now() < due) {
      g_cv.wait_until(lock, due);
      continue;
    }
    const Item item = g_queue.front();
    g_queue.pop_front();
    g_joining = true;
    lock.unlock();
    pthread_join(item.thread, nullptr);
    GiveStack(item.stack, item.stack_bytes);  // the join unmapped it
    const uint64_t reaped = g_reaped.fetch_add(1, std::memory_order_relaxed) + 1;
    if (reaped <= 4 || reaped % 16 == 0) {
      size_t kept = 0;
      size_t kept_bytes = 0;
      {
        std::lock_guard<std::mutex> pool_lock(g_mutex);
        kept = g_pool_count;
        kept_bytes = g_pool_bytes;
      }
      REXSYS_INFO("[threads] {} finished host threads released (kernel slot and stack); stacks: {} from the heap, "
                  "{} reused, {} kept ({} MB); {} creations helped by joining finished threads early",
                  reaped, g_stacks_new.load(std::memory_order_relaxed),
                  g_stacks_reused.load(std::memory_order_relaxed), kept, kept_bytes >> 20,
                  g_early_joins.load(std::memory_order_relaxed));
    }
    lock.lock();
    g_joining = false;
    g_drained.notify_all();
  }
  return nullptr;
}

// Finished threads that still hold their kernel slot and their stack.
size_t Pending() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_queue.size() + (g_joining ? 1 : 0);
}

// Waits, at most `timeout`, until every finished thread has been joined.
bool WaitDrained(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(g_mutex);
  return g_drained.wait_for(lock, timeout, [] { return g_queue.empty() && !g_joining; });
}

/*
 * For a creation that failed: has every finished thread joined now rather than a second after it ended, then waits
 * (at most `timeout`) until a block of `bytes` is kept in the pool, which it takes and returns, or until nothing is
 * left to join. `bytes` 0 (a stack libnx allocates): it only waits for the joins. `pending`: how many finished
 * threads were waiting to be joined.
 */
void* JoinFinishedNow(size_t bytes, std::chrono::milliseconds timeout, size_t* pending) {
  std::unique_lock<std::mutex> lock(g_mutex);
  *pending = g_queue.size() + (g_joining ? 1 : 0);
  const auto now = std::chrono::steady_clock::now();
  for (Item& item : g_queue) {
    item.due = std::min(item.due, now);
  }
  g_cv.notify_one();
  void* stack = nullptr;
  g_drained.wait_for(lock, timeout, [&] {
    stack = bytes != 0 ? PopPooledLocked(bytes) : nullptr;
    return stack != nullptr || (g_queue.empty() && !g_joining);
  });
  if (stack != nullptr) {
    g_stacks_reused.fetch_add(1, std::memory_order_relaxed);
  }
  return stack;
}

void Queue(pthread_t thread, void* stack, size_t stack_bytes) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_started) {
    pthread_attr_t attr;
    pthread_t reaper;
    if (pthread_attr_init(&attr) == 0) {
      pthread_attr_setstacksize(&attr, 64 * 1024);
      g_started = pthread_create(&reaper, &attr, Loop, nullptr) == 0;
      pthread_attr_destroy(&attr);
    }
  }
  g_queue.push_back({thread, std::chrono::steady_clock::now() + std::chrono::seconds(1), stack, stack_bytes});
  g_cv.notify_one();
}
}  // namespace switch_reaper
#endif

template <>
class PosixCondition<Thread> : public PosixConditionBase {
  enum class State {
    kUninitialized,
    kRunning,
    kSuspended,
    kFinished,
  };

 public:
  PosixCondition()
      : thread_(0),
        signaled_(false),
        exit_code_(0),
        state_(State::kUninitialized),
        suspend_count_(0) {
#if defined(__APPLE__)
    suspend_sem_ = RexCreateAnonymousSemaphore();
#else
    sem_init(&suspend_sem_, 0, 0);
#endif
#if REX_PLATFORM_ANDROID
    android_pre_api_26_name_[0] = '\0';
#endif
  }
  bool Initialize(Thread::CreationParameters params, ThreadStartData* start_data) {
    start_data->create_suspended = params.create_suspended;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0)
      return false;
    if (pthread_attr_setstacksize(&attr, params.stack_size) != 0) {
      pthread_attr_destroy(&attr);
      return false;
    }
    if (params.initial_priority != 0) {
      sched_param sched{};
      sched.sched_priority = params.initial_priority + 1;
      if (pthread_attr_setschedpolicy(&attr, SCHED_FIFO) != 0) {
        pthread_attr_destroy(&attr);
        return false;
      }
      if (pthread_attr_setschedparam(&attr, &sched) != 0) {
        pthread_attr_destroy(&attr);
        return false;
      }
    }
#if defined(__SWITCH__)
    /*
     * sa2-nx: a big stack (16 MB for a guest thread) runs on a block from switch_reaper's pool, which keeps the
     * stacks of joined threads for the next threads instead of giving them back to a fragmented heap (see there).
     */
    const size_t pooled_bytes = switch_reaper::PooledStackBytes(params.stack_size);
    // One try. `stack`: a block already taken from the pool, or null to take one here.
    auto create = [&](void* stack) -> int {
      if (pooled_bytes == 0) {
        return pthread_create(&thread_, &attr, ThreadStartRoutine, start_data);
      }
      if (stack == nullptr) {
        stack = switch_reaper::TakeStack(pooled_bytes);
        if (stack == nullptr) {
          return ENOMEM;  // what libnx returns when it cannot allocate the stack itself
        }
      }
      // Recorded before the thread exists: it hands the block to the reaper as it ends (ClaimJoin), which can
      // happen before pthread_create returns here.
      stack_ = stack;
      stack_bytes_ = pooled_bytes;
      int result = pthread_attr_setstack(&attr, stack, pooled_bytes);
      if (result == 0) {
        result = pthread_create(&thread_, &attr, ThreadStartRoutine, start_data);
      }
      if (result != 0) {
        stack_ = nullptr;
        stack_bytes_ = 0;
        switch_reaper::GiveStack(stack, pooled_bytes);  // the thread never ran on it
      }
      return result;
    };
    int created = create(nullptr);
    if (created != 0) {
      /*
       * sa2-nx: what is missing is usually held by finished threads the reaper has not joined yet. Each keeps its
       * stack and its kernel slot for a second after it ends, and the game starts its next one-shot thread long
       * before that (every ~35 ms on the Tails boss stage). They are joined now and the creation is tried again as
       * soon as one of their stacks is back, instead of waiting for the reaper; the retries below stay as the
       * fallback.
       */
      const auto asked = std::chrono::steady_clock::now();
      size_t pending = 0;
      void* const stack = switch_reaper::JoinFinishedNow(pooled_bytes, std::chrono::milliseconds(250), &pending);
      if (stack != nullptr || pending != 0) {
        const int first = created;
        created = create(stack);
        if (created == 0) {
          const uint64_t helped = switch_reaper::g_early_joins.fetch_add(1, std::memory_order_relaxed) + 1;
          if (helped <= 4) {
            REXSYS_INFO("[threads] creating a thread with a {} KB stack failed ({}): {} finished threads were joined "
                        "at once instead of a second after they ended, and the thread was created {:.2f} ms later "
                        "(early join {}; the reaper's report line counts the next ones)",
                        params.stack_size >> 10, first, pending,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - asked).count(),
                        helped);
          }
        }
      }
    }
    /*
     * sa2-nx: a failed creation ends a level load for good. The game's loader thread never runs and the loading
     * screen waits for it forever (console, 2026-10-06 22:41:12, the load after the Big Foot boss: "CreateThread
     * failed", C0000017). libnx turns every failure into ENOMEM, so the log says which resource is missing, and
     * the finished threads the reaper still holds (each keeps its stack and its kernel slot for a second) are
     * released before trying again.
     */
    for (int attempt = 1; created != 0 && attempt <= 3; ++attempt) {
      char why[320];
      RexSwitchDescribeThreadFailure(params.stack_size, why, sizeof(why));
      REXSYS_WARN("[threads] creating a thread with a {} KB stack failed ({}), attempt {} of 4: {}; {} finished "
                  "threads not released yet",
                  params.stack_size >> 10, created, attempt, static_cast<const char*>(why),
                  switch_reaper::Pending());
      switch_reaper::WaitDrained(std::chrono::milliseconds(1500));
      created = create(nullptr);
      if (created == 0) {
        REXSYS_WARN("[threads] the thread was created at attempt {}", attempt + 1);
      }
    }
#else
    int created = pthread_create(&thread_, &attr, ThreadStartRoutine, start_data);
#endif
    if (created != 0) {
      pthread_attr_destroy(&attr);
      return false;
    }
    pthread_attr_destroy(&attr);
    return true;
  }

  /// Constructor for existing thread. This should only happen once called by
  /// Thread::GetCurrentThread() on the main thread
  explicit PosixCondition(pthread_t thread)
      : thread_(thread),
        signaled_(false),
        exit_code_(0),
        state_(State::kRunning),
        suspend_count_(0) {
#if defined(__APPLE__)
    suspend_sem_ = RexCreateAnonymousSemaphore();
#else
    sem_init(&suspend_sem_, 0, 0);
#endif
#if REX_PLATFORM_ANDROID
    android_pre_api_26_name_[0] = '\0';
#endif
  }

  virtual ~PosixCondition() {
    // Match Canary/Edge behavior.
    // Force-cancel/join from the condition destructor can self-join/crash
    // depending on shutdown ordering, so threads must be stopped explicitly.
  }

  bool Signal() override { return true; }

  std::string name() const {
    WaitStarted();
    auto result = std::array<char, 17>{'\0'};
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (state_ != State::kUninitialized && state_ != State::kFinished) {
#if REX_PLATFORM_ANDROID
      // pthread_getname_np was added in API 26 - below that, store the name in
      // this object, which may be only modified through Xenia threading, but
      // should be enough in most cases.
      if (android_pthread_getname_np_) {
        if (android_pthread_getname_np_(thread_, result.data(), result.size() - 1) != 0) {
          assert_always();
        }
      } else {
        std::lock_guard<std::mutex> lock(android_pre_api_26_name_mutex_);
        rex::string::copy_truncating(result.data(), android_pre_api_26_name_, result.size());
      }
#else
      if (pthread_getname_np(thread_, result.data(), result.size() - 1) != 0) {
        assert_always();
      }
#endif
    }
    return std::string(result.data());
  }

  void set_name(const std::string& name) {
    WaitStarted();
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (state_ != State::kUninitialized && state_ != State::kFinished) {
#if defined(__APPLE__)
      // macOS pthread_setname_np() only names the calling thread.
      if (pthread_equal(thread_, pthread_self())) {
        pthread_setname_np(std::string(name).c_str());
      }
#else
      pthread_setname_np(thread_, std::string(name).c_str());
#endif
#if REX_PLATFORM_ANDROID
      SetAndroidPreApi26Name(name);
#endif
    }
  }

#if REX_PLATFORM_ANDROID
  void SetAndroidPreApi26Name(const std::string_view name) {
    if (android_pthread_getname_np_) {
      return;
    }
    std::lock_guard<std::mutex> lock(android_pre_api_26_name_mutex_);
    rex::string::copy_truncating(android_pre_api_26_name_, name,
                                 rex::countof(android_pre_api_26_name_));
  }
#endif

#if defined(__APPLE__)
  uint32_t system_id() const { return static_cast<uint32_t>(pthread_mach_thread_np(thread_)); }
#elif defined(__SWITCH__)
  // pthread_t is a pointer in newlib, not an integer: it has to be reinterpreted.
  uint32_t system_id() const {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(thread_));
  }
#else
  uint32_t system_id() const { return static_cast<uint32_t>(thread_); }
#endif

#if defined(__SWITCH__)
  // The finished thread's handle goes to the reaper (with its pooled stack): no call may use it once the join is
  // claimed.
  void ClaimJoin() {
    if (thread_ && !join_claimed_.exchange(true)) switch_reaper::Queue(thread_, stack_, stack_bytes_);
  }
  bool HandleGone() const { return join_claimed_.load(std::memory_order_acquire); }
#else
  bool HandleGone() const { return false; }
#endif

  uint64_t affinity_mask() {
    WaitStarted();
    if (HandleGone()) return 0;
#if defined(__APPLE__)
    // macOS doesn't expose Linux-style pthread affinity masks; report every
    // logical processor as available.
    auto cpu_count = std::min<uint32_t>(logical_processor_count(), 64);
    if (cpu_count == 64) {
      return std::numeric_limits<uint64_t>::max();
    }
    return (uint64_t(1) << cpu_count) - 1;
#else
    cpu_set_t cpu_set;
#if REX_PLATFORM_ANDROID
    if (sched_getaffinity(pthread_gettid_np(thread_), sizeof(cpu_set_t), &cpu_set) != 0) {
      assert_always();
    }
#else
    if (pthread_getaffinity_np(thread_, sizeof(cpu_set_t), &cpu_set) != 0) {
      assert_always();
    }
#endif
    uint64_t result = 0;
    auto cpu_count = std::min(CPU_SETSIZE, 64);
    for (auto i = 0u; i < cpu_count; i++) {
      auto set = CPU_ISSET(i, &cpu_set);
      result |= set << i;
    }
    return result;
#endif
  }

  void set_affinity_mask(uint64_t mask) {
    WaitStarted();
    if (HandleGone()) return;
#if defined(__APPLE__)
    // macOS doesn't expose Linux-style pthread affinity masks; keep this a
    // no-op so higher layers can build and run.
    (void)mask;
#else
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (auto i = 0u; i < 64; i++) {
      if (mask & (1ULL << i)) {
        CPU_SET(i, &cpu_set);
      }
    }
#if REX_PLATFORM_ANDROID
    if (sched_setaffinity(pthread_gettid_np(thread_), sizeof(cpu_set_t), &cpu_set) != 0) {
      assert_always();
    }
#else
    if (pthread_setaffinity_np(thread_, sizeof(cpu_set_t), &cpu_set) != 0) {
      assert_always();
    }
#endif
#endif
  }

  int priority() {
    WaitStarted();
    if (HandleGone()) return -1;
    int policy;
    sched_param param{};
    int ret = pthread_getschedparam(thread_, &policy, &param);
    if (ret != 0) {
      return -1;
    }

    return param.sched_priority;
  }

  void set_priority(int new_priority) {
    WaitStarted();
    if (HandleGone()) return;
    sched_param param{};
    param.sched_priority = new_priority;
    int result = pthread_setschedparam(thread_, SCHED_FIFO, &param);
    if (result != 0) {
      switch (result) {
        case EPERM:
          REXSYS_WARN("set_priority: permission denied");
          break;
        case EINVAL:
          assert_always();
          break;
        default:
          REXSYS_WARN("set_priority: pthread_setschedparam failed ({})", result);
          break;
      }
    }
  }

  void QueueUserCallback(std::function<void()> callback) {
    WaitStarted();
    {
      std::unique_lock<std::mutex> lock(callback_mutex_);
      user_callbacks_.push_back(std::move(callback));
      has_pending_user_callbacks_.store(true, std::memory_order_release);
    }

    // If the callback is queued on the current thread, don't self-signal.
    // Alertable waits drain this queue in normal thread context.
    if (pthread_equal(thread_, pthread_self())) {
      if (alertable_state_) {
        DispatchQueuedUserCallbacks();
      }
      return;
    }

    // An alertable wait of the target is parked until its timeout / a signal: wake it so it dispatches the
    // callback now (the flag above is what its wait predicate looks at).
    WakeAlertableWaiter();

#if defined(__SWITCH__)
    // Horizon does not deliver signals, so there is no way to interrupt another
    // thread to make it run the call. The callback is queued and runs when the
    // target thread passes an alertable point (DispatchCurrentThreadUserCallback),
    // which is where the SDK already drains them.
    //
    // A real difference in behavior: on the PC the callback arrives right away;
    // here it arrives when the thread gets to it. That is enough for guest APCs,
    // which is what the SDK uses this for.
    int result = 0;
#elif defined(__APPLE__)
    // macOS has no sigqueue/pthread_sigqueue; pthread_kill delivers the signal
    // without the sigval payload.
    int result = pthread_kill(thread_, GetSystemSignal(SignalType::kThreadUserCallback));
#else
    sigval value{};
    value.sival_ptr = this;
#if REX_PLATFORM_ANDROID
    int result = sigqueue(pthread_gettid_np(thread_),
                          GetSystemSignal(SignalType::kThreadUserCallback), value);
#else
    int result = pthread_sigqueue(thread_, GetSystemSignal(SignalType::kThreadUserCallback), value);
#endif
#endif
    if (result != 0) {
      REXSYS_WARN("QueueUserCallback: signal delivery failed ({})", result);
    }
  }

  // What the owning thread is parked on inside an alertable wait (exactly one of the two is set), or none.
  // Set/cleared by ScopedAlertWake; QueueUserCallback pokes it. Holding wake_mutex_ while poking keeps the
  // target alive (the waiter clears it, under the same mutex, before its wait object goes out of scope).
  void SetWakeTarget(PosixConditionBase* cond, WakePoint* point) {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    wake_cond_ = cond;
    wake_point_ = point;
  }
  void WakeAlertableWaiter() {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    if (wake_cond_) {
      wake_cond_->InterruptWaiters();
    }
    if (wake_point_) {
      wake_point_->Poke();
    }
  }
  const std::atomic<bool>* pending_user_callbacks_flag() const {
    return &has_pending_user_callbacks_;
  }

  bool DispatchQueuedUserCallbacks() {
    if (!has_pending_user_callbacks_.load(std::memory_order_acquire)) {
      return false;
    }

    std::function<void()> callback;
    {
      std::unique_lock<std::mutex> lock(callback_mutex_);
      if (user_callbacks_.empty()) {
        has_pending_user_callbacks_.store(false, std::memory_order_release);
        return false;
      }
      callback = std::move(user_callbacks_.front());
      user_callbacks_.pop_front();
      has_pending_user_callbacks_.store(!user_callbacks_.empty(), std::memory_order_release);
    }
    if (callback) {
      callback();
    }
    return true;
  }

  bool Resume(uint32_t* out_previous_suspend_count = nullptr) {
    if (out_previous_suspend_count) {
      *out_previous_suspend_count = 0;
    }
    WaitStarted();
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (suspend_count_ == 0) {
      return false;
    }
    if (out_previous_suspend_count) {
      *out_previous_suspend_count = suspend_count_;
    }
    --suspend_count_;
    if (suspend_count_ == 0 && state_ == State::kSuspended) {
      state_ = State::kRunning;
#if defined(__SWITCH__)
      // Counterpart of RexSwitchPauseThread in Suspend(). The semaphore below
      // wakes a thread that suspended itself from WaitSuspended(); this call
      // resumes a thread that was paused from outside. Both are needed,
      // because they are two different suspension paths.
      if (!pthread_equal(thread_, pthread_self()) && !HandleGone()) {
        RexSwitchResumeThread(thread_);
      }
#endif
      // Async-signal-safe wakeup for WaitSuspended() from signal handler path.
      sem_post(suspend_sem_ptr());
    }
    state_signal_.notify_all();
    return true;
  }

  bool Suspend(uint32_t* out_previous_suspend_count = nullptr) {
    if (out_previous_suspend_count) {
      *out_previous_suspend_count = 0;
    }
    WaitStarted();
    bool is_current_thread = pthread_self() == thread_;
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      if (out_previous_suspend_count) {
        *out_previous_suspend_count = suspend_count_;
      }
      state_ = State::kSuspended;
      ++suspend_count_;
    }

    if (is_current_thread) {
      WaitSuspended();
      return true;
    }

#if defined(__SWITCH__)
    // Careful: pthread_kill does not work here.
    //
    // On Horizon newlib accepts the call and returns 0, but the signal is never
    // delivered. The result would be the worst possible: the state would say
    // kSuspended, the function would return true, and the thread would keep
    // running happily. A bug that compiles, gives no warning and shows up later
    // as random corruption.
    //
    // svcSetThreadActivity is the system's real mechanism.
    return !HandleGone() && RexSwitchPauseThread(thread_) == 0;
#else
    int result = pthread_kill(thread_, GetSystemSignal(SignalType::kThreadSuspend));
    return result == 0;
#endif
  }

  void Terminate(int exit_code) {
    bool is_current_thread = pthread_self() == thread_;
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      if (state_ == State::kFinished) {
        if (is_current_thread) {
          // This is really bad. Some thread must have called Terminate() on us
          // just before we decided to terminate ourselves
          assert_always();
          for (;;) {
            // Wait for pthread_cancel() to actually happen.
          }
        }
        return;
      }
      state_ = State::kFinished;
    }
#if defined(__SWITCH__)
    if (is_current_thread) ClaimJoin();  // ExitThread: this thread will not return to ThreadStartRoutine
#endif

    {
      std::lock_guard<std::mutex> lock(mutex_);

      exit_code_ = exit_code;
      signaled_ = true;
      cond_.notify_all();
    }
    NotifyMulti();
    if (is_current_thread) {
      pthread_exit(reinterpret_cast<void*>(exit_code));
    } else {
#if REX_PLATFORM_ANDROID
      if (pthread_kill(thread_, GetSystemSignal(SignalType::kThreadTerminate)) != 0) {
        assert_always();
      }
#else
      if (pthread_cancel(thread_) != 0) {
        assert_always();
      }
#endif
    }
  }

  void WaitStarted() const {
    std::unique_lock<std::mutex> lock(state_mutex_);
    state_signal_.wait(lock, [this] { return state_ != State::kUninitialized; });
  }

  /// Uses sem_wait because it may be called from signal handler context.
  void WaitSuspended() {
    int ret;
    do {
      ret = sem_wait(suspend_sem_ptr());
    } while (ret == -1 && errno == EINTR);
  }

  void* native_handle() const override { return reinterpret_cast<void*>(thread_); }

 private:
  static void* ThreadStartRoutine(void* parameter);
  inline bool signaled() const override { return signaled_; }
  inline void post_execution() override {
    // The first one to claim the join does it (a wait here, or the Switch reaper); a second wait does nothing.
    if (thread_ && !join_claimed_.exchange(true)) {
      pthread_join(thread_, nullptr);
#if defined(__SWITCH__)
      switch_reaper::GiveStack(stack_, stack_bytes_);  // the join unmapped it: it can carry another thread
#endif
    }
#if defined(__APPLE__)
    if (suspend_sem_ != SEM_FAILED) {
      sem_close(suspend_sem_);
      suspend_sem_ = SEM_FAILED;
    }
#else
    sem_destroy(&suspend_sem_);
#endif
  }
  pthread_t thread_;
  std::atomic<bool> join_claimed_{false};  // somebody joins (or will join) thread_: see switch_reaper
#if defined(__SWITCH__)
  void* stack_ = nullptr;  // thread_'s stack block from switch_reaper's pool (null: libnx allocated the stack)
  size_t stack_bytes_ = 0;
#endif
  bool signaled_;
  int exit_code_;
  State state_;             // Protected by state_mutex_
  uint32_t suspend_count_;  // Protected by state_mutex_
#if defined(__APPLE__)
  sem_t* suspend_sem_ = SEM_FAILED;
  sem_t* suspend_sem_ptr() { return suspend_sem_; }
#else
  sem_t suspend_sem_;
  sem_t* suspend_sem_ptr() { return &suspend_sem_; }
#endif
  mutable std::mutex state_mutex_;
  mutable std::mutex callback_mutex_;
  mutable std::condition_variable state_signal_;
  std::deque<std::function<void()>> user_callbacks_;
  std::atomic<bool> has_pending_user_callbacks_{false};
  std::mutex wake_mutex_;
  PosixConditionBase* wake_cond_ = nullptr;
  WakePoint* wake_point_ = nullptr;
#if REX_PLATFORM_ANDROID
  // Name accessible via name() on Android before API 26 which added
  // pthread_getname_np.
  mutable std::mutex android_pre_api_26_name_mutex_;
  char android_pre_api_26_name_[16];
#endif
};

class PosixWaitHandle {
 public:
  virtual ~PosixWaitHandle();
  virtual PosixConditionBase& condition() = 0;
};

PosixWaitHandle::~PosixWaitHandle() = default;

thread_local PosixCondition<Thread>* current_thread_condition_ = nullptr;

bool DispatchCurrentThreadUserCallback() {
  if (!current_thread_condition_) {
    Thread::GetCurrentThread();
  }
  return current_thread_condition_ && current_thread_condition_->DispatchQueuedUserCallbacks();
}

namespace {

// Slice an alertable wait sleeps before re-checking the user-callback queue. With event-driven waits the queued
// callback wakes the waiter itself, so the slice is only a safety net; legacy mode polls every 1 ms.
std::chrono::milliseconds AlertablePollSlice() {
  if (!REXCVAR_GET(thread_wait_blocking)) {
    return std::chrono::milliseconds(1);
  }
  return std::chrono::milliseconds(std::clamp(REXCVAR_GET(thread_alertable_slice_ms), 1, 1000));
}

// Registers what the current thread parks on for the duration of an alertable wait, so QueueUserCallback
// (from another thread) can wake it. Inert when event-driven waits are off.
class ScopedAlertWake {
 public:
  ScopedAlertWake(PosixConditionBase* cond, WakePoint* point) {
    if (!REXCVAR_GET(thread_wait_blocking)) {
      return;
    }
    if (!current_thread_condition_) {
      Thread::GetCurrentThread();
    }
    thread_ = current_thread_condition_;
    if (thread_) {
      thread_->SetWakeTarget(cond, point);
    }
  }
  ~ScopedAlertWake() {
    if (thread_) {
      thread_->SetWakeTarget(nullptr, nullptr);
    }
  }
  // The "callback queued" flag the parked wait watches, or null when not event-driven.
  const std::atomic<bool>* flag() const {
    return thread_ ? thread_->pending_user_callbacks_flag() : nullptr;
  }

 private:
  PosixCondition<Thread>* thread_ = nullptr;
};

class ScopedAlertableState {
 public:
  explicit ScopedAlertableState(bool alertable) : alertable_(alertable) {
    if (alertable_) {
      alertable_state_ = true;
    }
  }
  ~ScopedAlertableState() {
    if (alertable_) {
      alertable_state_ = false;
    }
  }

 private:
  bool alertable_;
};

std::chrono::steady_clock::time_point ComputeAlertableDeadline(std::chrono::milliseconds timeout) {
  if (timeout == std::chrono::milliseconds::max()) {
    return std::chrono::steady_clock::time_point::max();
  }
  return std::chrono::steady_clock::now() + timeout;
}

bool HasAlertableTimeoutElapsed(std::chrono::steady_clock::time_point deadline) {
  return deadline != std::chrono::steady_clock::time_point::max() &&
         std::chrono::steady_clock::now() >= deadline;
}

std::chrono::milliseconds ComputeAlertableWaitTimeout(
    std::chrono::steady_clock::time_point deadline) {
  if (deadline == std::chrono::steady_clock::time_point::max()) {
    return AlertablePollSlice();
  }
  // A positive remainder under 1 ms must keep waiting. Truncating it to zero
  // turns the end of every alertable wait into a busy poll on Horizon.
  auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now());
  if (remaining <= std::chrono::milliseconds::zero()) {
    return std::chrono::milliseconds::zero();
  }
  return std::min(remaining, AlertablePollSlice());
}

}  // namespace

// This wraps a condition object as our handle because posix has no single
// native handle for higher level concurrency constructs such as semaphores
template <typename T>
class PosixConditionHandle : public T, public PosixWaitHandle {
 public:
  PosixConditionHandle() = default;
  explicit PosixConditionHandle(bool);
  explicit PosixConditionHandle(pthread_t thread);
  PosixConditionHandle(bool manual_reset, bool initial_state);
  PosixConditionHandle(uint32_t initial_count, uint32_t maximum_count);
  ~PosixConditionHandle() override = default;

  PosixCondition<T>& condition() override { return handle_; }
  void* native_handle() const override { return handle_.native_handle(); }

 protected:
  PosixCondition<T> handle_;
  friend PosixCondition<T>;
};

template <>
PosixConditionHandle<Semaphore>::PosixConditionHandle(uint32_t initial_count,
                                                      uint32_t maximum_count)
    : handle_(initial_count, maximum_count) {}

template <>
PosixConditionHandle<Mutant>::PosixConditionHandle(bool initial_owner) : handle_(initial_owner) {}

template <>
PosixConditionHandle<Timer>::PosixConditionHandle(bool manual_reset) : handle_(manual_reset) {}

template <>
PosixConditionHandle<Event>::PosixConditionHandle(bool manual_reset, bool initial_state)
    : handle_(manual_reset, initial_state) {}

template <>
PosixConditionHandle<Thread>::PosixConditionHandle(pthread_t thread) : handle_(thread) {}

WaitResult Wait(WaitHandle* wait_handle, bool is_alertable, std::chrono::milliseconds timeout) {
  auto posix_wait_handle = dynamic_cast<PosixWaitHandle*>(wait_handle);
  if (posix_wait_handle == nullptr) {
    return WaitResult::kFailed;
  }
  if (!is_alertable) {
    return posix_wait_handle->condition().Wait(timeout);
  }

  ScopedAlertableState alertable_state_guard(true);
  auto deadline = ComputeAlertableDeadline(timeout);
  // Event-driven: a queued callback interrupts the parked Wait() (flag + InterruptWaiters).
  ScopedAlertWake wake(&posix_wait_handle->condition(), nullptr);
  const std::atomic<bool>* interrupt = wake.flag();

  while (true) {
    if (DispatchCurrentThreadUserCallback()) {
      return WaitResult::kUserCallback;
    }
    auto result =
        posix_wait_handle->condition().Wait(ComputeAlertableWaitTimeout(deadline), interrupt);
    if (result != WaitResult::kTimeout) {
      return result;
    }
    // Even with a zero timeout the signaled object has to be checked once.
    if (HasAlertableTimeoutElapsed(deadline)) {
      return WaitResult::kTimeout;
    }
    REX_SYS_COUNT(interrupt && interrupt->load(std::memory_order_relaxed) ? rex::syscount::kAlertWake
                                                                          : rex::syscount::kAlertSlice,
                  1);
  }
}

WaitResult SignalAndWait(WaitHandle* wait_handle_to_signal, WaitHandle* wait_handle_to_wait_on,
                         bool is_alertable, std::chrono::milliseconds timeout) {
  auto result = WaitResult::kFailed;
  auto posix_wait_handle_to_signal = dynamic_cast<PosixWaitHandle*>(wait_handle_to_signal);
  auto posix_wait_handle_to_wait_on = dynamic_cast<PosixWaitHandle*>(wait_handle_to_wait_on);
  if (posix_wait_handle_to_signal == nullptr || posix_wait_handle_to_wait_on == nullptr) {
    return WaitResult::kFailed;
  }
  if (!posix_wait_handle_to_signal->condition().Signal()) {
    return WaitResult::kFailed;
  }

  if (!is_alertable) {
    return posix_wait_handle_to_wait_on->condition().Wait(timeout);
  }

  ScopedAlertableState alertable_state_guard(true);
  auto deadline = ComputeAlertableDeadline(timeout);
  ScopedAlertWake wake(&posix_wait_handle_to_wait_on->condition(), nullptr);
  const std::atomic<bool>* interrupt = wake.flag();
  while (true) {
    if (DispatchCurrentThreadUserCallback()) {
      return WaitResult::kUserCallback;
    }
    result = posix_wait_handle_to_wait_on->condition().Wait(ComputeAlertableWaitTimeout(deadline),
                                                            interrupt);
    if (result != WaitResult::kTimeout) {
      return result;
    }
    if (HasAlertableTimeoutElapsed(deadline)) {
      return WaitResult::kTimeout;
    }
  }
}

std::pair<WaitResult, size_t> WaitMultiple(WaitHandle* wait_handles[], size_t wait_handle_count,
                                           bool wait_all, bool is_alertable,
                                           std::chrono::milliseconds timeout) {
  std::vector<PosixConditionBase*> conditions;
  conditions.reserve(wait_handle_count);
  for (size_t i = 0u; i < wait_handle_count; ++i) {
    auto handle = dynamic_cast<PosixWaitHandle*>(wait_handles[i]);
    if (handle == nullptr) {
      return std::make_pair(WaitResult::kFailed, 0);
    }
    conditions.push_back(&handle->condition());
  }
  // Parked (event-driven) waits need a wake point; legacy mode keeps the 1 ms polling.
  const bool blocking = REXCVAR_GET(thread_wait_blocking);
  WakePoint wake_point;
  WakePoint* wake = blocking && conditions.size() > 1 ? &wake_point : nullptr;
  if (!is_alertable) {
    return PosixConditionBase::WaitMultiple(std::move(conditions), wait_all, timeout, wake, nullptr);
  }

  ScopedAlertableState alertable_state_guard(true);
  auto deadline = ComputeAlertableDeadline(timeout);
  // A single handle parks in Wait() itself; several park on the wake point.
  ScopedAlertWake alert_wake(blocking && conditions.size() == 1 ? conditions[0] : nullptr, wake);
  const std::atomic<bool>* interrupt = alert_wake.flag();
  while (true) {
    if (DispatchCurrentThreadUserCallback()) {
      return std::make_pair(WaitResult::kUserCallback, 0);
    }
    auto result = PosixConditionBase::WaitMultiple(std::vector<PosixConditionBase*>(conditions),
                                                   wait_all, ComputeAlertableWaitTimeout(deadline),
                                                   wake, interrupt);
    if (result.first != WaitResult::kTimeout) {
      return result;
    }
    if (HasAlertableTimeoutElapsed(deadline)) {
      return std::make_pair(WaitResult::kTimeout, 0);
    }
    REX_SYS_COUNT(interrupt && interrupt->load(std::memory_order_relaxed) ? rex::syscount::kAlertWake
                                                                          : rex::syscount::kAlertSlice,
                  1);
  }
}

// Sleeps (alertably) for `duration`; returns kAlerted when a queued user callback ran. Event-driven: parks on
// a wake point that QueueUserCallback pokes, instead of 1 ms nanosleep slices.
SleepResult AlertableSleep(std::chrono::microseconds duration) {
  alertable_state_ = true;
  auto deadline = std::chrono::steady_clock::now() + duration;
  const bool blocking = REXCVAR_GET(thread_wait_blocking);
  WakePoint wake_point;
  ScopedAlertWake alert_wake(nullptr, blocking ? &wake_point : nullptr);
  const std::atomic<bool>* interrupt = alert_wake.flag();
  const auto slice = std::chrono::duration_cast<std::chrono::microseconds>(AlertablePollSlice());
  while (true) {
    if (DispatchCurrentThreadUserCallback()) {
      alertable_state_ = false;
      return SleepResult::kAlerted;
    }
    auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      alertable_state_ = false;
      return SleepResult::kSuccess;
    }
    auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
    const auto nap = std::min(remaining, slice);
    if (blocking && interrupt) {
      std::unique_lock<std::mutex> park(wake_point.m);
      wake_point.cv.wait_until(park, now + nap, [&] {
        return interrupt->load(std::memory_order_acquire);
      });
    } else {
      Sleep(nap);
    }
    REX_SYS_COUNT(interrupt && interrupt->load(std::memory_order_relaxed) ? rex::syscount::kAlertWake
                                                                          : rex::syscount::kAlertSlice,
                  1);
  }
}

class PosixEvent : public PosixConditionHandle<Event> {
 public:
  PosixEvent(bool manual_reset, bool initial_state)
      : PosixConditionHandle(manual_reset, initial_state) {}
  ~PosixEvent() override = default;
  void Set() override { handle_.Signal(); }
  void Reset() override { handle_.Reset(); }
  void Pulse() override {
    using namespace std::chrono_literals;
    handle_.Signal();
    MaybeYield();
    Sleep(10us);
    handle_.Reset();
  }
};

std::unique_ptr<Event> Event::CreateManualResetEvent(bool initial_state) {
  return std::make_unique<PosixEvent>(true, initial_state);
}

std::unique_ptr<Event> Event::CreateAutoResetEvent(bool initial_state) {
  return std::make_unique<PosixEvent>(false, initial_state);
}

class PosixSemaphore : public PosixConditionHandle<Semaphore> {
 public:
  PosixSemaphore(int initial_count, int maximum_count)
      : PosixConditionHandle(static_cast<uint32_t>(initial_count),
                             static_cast<uint32_t>(maximum_count)) {}
  ~PosixSemaphore() override = default;
  bool Release(int release_count, int* out_previous_count) override {
    if (release_count < 1) {
      return false;
    }
    return handle_.Release(static_cast<uint32_t>(release_count), out_previous_count);
  }
};

std::unique_ptr<Semaphore> Semaphore::Create(int initial_count, int maximum_count) {
  if (initial_count < 0 || initial_count > maximum_count || maximum_count <= 0) {
    return nullptr;
  }
  return std::make_unique<PosixSemaphore>(initial_count, maximum_count);
}

class PosixMutant : public PosixConditionHandle<Mutant> {
 public:
  explicit PosixMutant(bool initial_owner) : PosixConditionHandle(initial_owner) {}
  ~PosixMutant() override = default;
  bool Release() override { return handle_.Release(); }
};

std::unique_ptr<Mutant> Mutant::Create(bool initial_owner) {
  return std::make_unique<PosixMutant>(initial_owner);
}

class PosixTimer : public PosixConditionHandle<Timer> {
  using WClock_ = Timer::WClock_;
  using GClock_ = Timer::GClock_;

 public:
  explicit PosixTimer(bool manual_reset) : PosixConditionHandle(manual_reset) {}
  ~PosixTimer() override = default;

  bool SetOnceAfter(rex::chrono::hundrednanoseconds rel_time,
                    std::function<void()> opt_callback = nullptr) override {
    return SetOnceAt(GClock_::now() + rel_time, std::move(opt_callback));
  }
  bool SetOnceAt(WClock_::time_point due_time,
                 std::function<void()> opt_callback = nullptr) override {
    return SetOnceAt(std::chrono::clock_cast<GClock_>(due_time), std::move(opt_callback));
  };
  bool SetOnceAt(GClock_::time_point due_time,
                 std::function<void()> opt_callback = nullptr) override {
    handle_.SetOnce(due_time, std::move(opt_callback));
    return true;
  }

  bool SetRepeatingAfter(rex::chrono::hundrednanoseconds rel_time, std::chrono::milliseconds period,
                         std::function<void()> opt_callback = nullptr) override {
    return SetRepeatingAt(GClock_::now() + rel_time, period, std::move(opt_callback));
  }
  bool SetRepeatingAt(WClock_::time_point due_time, std::chrono::milliseconds period,
                      std::function<void()> opt_callback = nullptr) override {
    return SetRepeatingAt(std::chrono::clock_cast<GClock_>(due_time), period,
                          std::move(opt_callback));
  }
  bool SetRepeatingAt(GClock_::time_point due_time, std::chrono::milliseconds period,
                      std::function<void()> opt_callback = nullptr) override {
    handle_.SetRepeating(due_time, period, std::move(opt_callback));
    return true;
  }
  bool Cancel() override {
    handle_.Cancel();
    return true;
  }
};

std::unique_ptr<Timer> Timer::CreateManualResetTimer() {
  return std::make_unique<PosixTimer>(true);
}

std::unique_ptr<Timer> Timer::CreateSynchronizationTimer() {
  return std::make_unique<PosixTimer>(false);
}

class PosixThread : public PosixConditionHandle<Thread> {
 public:
  PosixThread() = default;
  explicit PosixThread(pthread_t thread) : PosixConditionHandle(thread) {}
  ~PosixThread() override = default;

  bool Initialize(CreationParameters params, std::function<void()> start_routine) {
    auto start_data = new ThreadStartData({std::move(start_routine), false, this});
    return handle_.Initialize(params, start_data);
  }

  void set_name(std::string name) override {
    handle_.WaitStarted();
    Thread::set_name(name);
    if (name.length() > 15) {
      name = name.substr(0, 15);
    }
    handle_.set_name(name);
  }

  uint32_t system_id() const override { return handle_.system_id(); }

  uint64_t affinity_mask() override { return handle_.affinity_mask(); }
  void set_affinity_mask(uint64_t mask) override { handle_.set_affinity_mask(mask); }

  int priority() override { return handle_.priority(); }
  void set_priority(int new_priority) override { handle_.set_priority(new_priority); }

  void QueueUserCallback(std::function<void()> callback) override {
    handle_.QueueUserCallback(std::move(callback));
  }

  bool Resume(uint32_t* out_previous_suspend_count) override {
    return handle_.Resume(out_previous_suspend_count);
  }

  bool Suspend(uint32_t* out_previous_suspend_count) override {
    return handle_.Suspend(out_previous_suspend_count);
  }

  void Terminate(int exit_code) override { handle_.Terminate(exit_code); }

  void WaitSuspended() { handle_.WaitSuspended(); }
};

thread_local PosixThread* current_thread_ = nullptr;

void* PosixCondition<Thread>::ThreadStartRoutine(void* parameter) {
#if !REX_PLATFORM_ANDROID
  if (pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr) != 0) {
    assert_always();
  }
#endif
  set_current_thread_name("");

  auto start_data = static_cast<ThreadStartData*>(parameter);
  assert_not_null(start_data);
  assert_not_null(start_data->thread_obj);

  auto thread = dynamic_cast<PosixThread*>(start_data->thread_obj);
  auto start_routine = std::move(start_data->start_routine);
  auto create_suspended = start_data->create_suspended;
  delete start_data;

  current_thread_ = thread;
  current_thread_condition_ = &thread->handle_;
  {
    std::unique_lock<std::mutex> lock(thread->handle_.state_mutex_);
    thread->handle_.state_ = create_suspended ? State::kSuspended : State::kRunning;
    // Arm the suspend counter in the SAME critical section that publishes the
    // started state. Resume() only waits for state_ != kUninitialized, so if it
    // observed the state published here while suspend_count_ was still 0 it
    // would take the "not suspended" path, return false without decrementing,
    // and the wait below would then never be satisfied.
    if (create_suspended) {
      thread->handle_.suspend_count_ = 1;
    }
    thread->handle_.state_signal_.notify_all();
  }

  if (create_suspended) {
    std::unique_lock<std::mutex> lock(thread->handle_.state_mutex_);
    thread->handle_.state_signal_.wait(lock,
                                       [thread] { return thread->handle_.suspend_count_ == 0; });
  }

  start_routine();

  {
    std::unique_lock<std::mutex> lock(thread->handle_.state_mutex_);
    thread->handle_.state_ = State::kFinished;
  }
#if defined(__SWITCH__)
  thread->handle_.ClaimJoin();  // before signaling: a waiter woken below must not join it too
#endif

  {
    std::unique_lock<std::mutex> lock(thread->handle_.mutex_);
    thread->handle_.exit_code_ = 0;
    thread->handle_.signaled_ = true;
    thread->handle_.cond_.notify_all();
  }
  thread->handle_.NotifyMulti();

  current_thread_ = nullptr;
  current_thread_condition_ = nullptr;
  return nullptr;
}

std::unique_ptr<Thread> Thread::Create(CreationParameters params,
                                       std::function<void()> start_routine) {
  install_signal_handler(SignalType::kThreadSuspend);
  install_signal_handler(SignalType::kThreadUserCallback);
#if REX_PLATFORM_ANDROID
  install_signal_handler(SignalType::kThreadTerminate);
#endif
  auto thread = std::make_unique<PosixThread>();
  if (!thread->Initialize(params, std::move(start_routine)))
    return nullptr;
  assert_not_null(thread);
  return thread;
}

Thread* Thread::GetCurrentThread() {
  if (current_thread_) {
    return current_thread_;
  }

  // Threads not created by Thread::Create (typically main thread) still need
  // process-wide signal handlers used by suspend/APC machinery.
  install_signal_handler(SignalType::kThreadSuspend);
  install_signal_handler(SignalType::kThreadUserCallback);
#if REX_PLATFORM_ANDROID
  install_signal_handler(SignalType::kThreadTerminate);
#endif

  // Should take this route only for threads not created by Thread::Create.
  // The only thread not created by Thread::Create should be the main thread.
  pthread_t handle = pthread_self();

  current_thread_ = new PosixThread(handle);
  current_thread_condition_ = &current_thread_->condition();
  // TODO(bwrsandman): Disabling deleting thread_local current thread to prevent
  //                   assert in destructor. Since this is thread local, the
  //                   "memory leaking" is controlled.
  // atexit([] { delete current_thread_; });

  return current_thread_;
}

void Thread::Exit(int exit_code) {
  if (current_thread_) {
    current_thread_->Terminate(exit_code);
  } else {
    // Should only happen with the main thread
    pthread_exit(reinterpret_cast<void*>(exit_code));
  }
  // Function must not return
  assert_always();
}

void set_current_thread_name(const std::string_view name) {
#if defined(__APPLE__)
  pthread_setname_np(std::string(name).c_str());
#else
  pthread_setname_np(pthread_self(), std::string(name).c_str());
#endif
#if REX_PLATFORM_ANDROID
  if (!android_pthread_getname_np_ && current_thread_) {
    current_thread_->condition().SetAndroidPreApi26Name(name);
  }
#endif
}

static void signal_handler(int signal, siginfo_t* /*info*/, void* /*context*/) {
  switch (GetSystemSignalType(signal)) {
    case SignalType::kThreadSuspend: {
      if (!current_thread_) {
        return;
      }
      current_thread_->WaitSuspended();
    } break;
    case SignalType::kThreadUserCallback: {
      // Callbacks are drained from alertable waits in normal thread context.
      // This signal is only used as a wakeup hint.
    } break;
#if REX_PLATFORM_ANDROID
    case SignalType::kThreadTerminate: {
      pthread_exit(reinterpret_cast<void*>(-1));
    } break;
#endif
    default:
      assert_always();
  }
}

}  // namespace rex::thread
