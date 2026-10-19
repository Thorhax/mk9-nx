/**
 * @file        core/logging.cpp
 * @brief       Logging infrastructure implementation
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <algorithm>
#include <cctype>
#include <charconv>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <share.h>  // _fsopen (WriterFileSink)
#endif
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <spdlog/async.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <toml++/toml.hpp>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/platform/env.h>
#include <rex/thread.h>

#if REX_PLATFORM_SWITCH
#include "threading_switch.h"
#endif

#ifdef __SWITCH__
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <switch.h>
#include <spdlog/sinks/base_sink.h>
#endif

#if REX_PLATFORM_WIN32
#include <spdlog/sinks/msvc_sink.h>
#else
#include <spdlog/sinks/stdout_sinks.h>
#endif

REXCVAR_DEFINE_STRING(log_network, "", "Log",
                      "Switch: also send every log line (and stdout/stderr, e.g. NVK's messages) over TCP to this "
                      "host:port, for watching a run live (e.g. 192.168.1.76:28771). Empty = off");
REXCVAR_DEFINE_STRING(log_level, "info", "Log",
                      "Global log level: trace, debug, info, warn, error, critical, off")
    .allowed({"trace", "debug", "info", "warn", "error", "critical", "off"});

REXCVAR_DEFINE_STRING(log_file, "", "Log", "Log file path (empty = auto sequential naming)");
// sa2-nx: false = no log file (with log_file empty): nothing is written to logs/.
REXCVAR_DEFINE_BOOL(log_to_file, true, "Log", "Write the log to a file in logs/ (false = no log file)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_BOOL(log_verbose, false, "Log", "Enable verbose logging (sets level to trace)")
    .debug_only();

REXCVAR_DEFINE_BOOL(log_noisy, false, "Log", "Enable noisy/high-frequency log macros");

/*
 * Periodic flush once per second (0 turns it off).
 *
 * It goes together with flush_level being `warn` instead of `info` (see rex/logging/types.h):
 * normal lines are not written to the SD one by one from the thread that logs them. This spdlog
 * thread flushes them once per second. It is joined in its destructor, not detached, so it is
 * safe on Horizon.
 */
REXCVAR_DEFINE_INT32(log_flush_interval, 1, "Log", "Periodic flush interval in seconds (0 = off)")
    .range(0, 60)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(log_max_file_size_mb, 5, "Log", "Max log file size in MB before rotation")
    .range(1, 100)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(log_max_files, 20, "Log", "Max number of rotated log files to keep")
    .range(1, 100)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

/*
 * log_async: write the log from a separate thread. Off by default: it made stutter much worse.
 *
 * The log itself is a source of stutter. The periodic report writes ~250 long lines every 10 seconds
 * from the ring thread, the one that feeds the GPU. With the synchronous logger, that thread formats
 * the lines and writes them to the SD file itself. Measured over 191 race stutters: during the 2-3 s
 * the dump lasts, the rate of frames over 60 ms is 1.03/s, against 0.34/s outside it: three times as
 * high, with the GPU at its normal value (28.7 ms) and no work counter spiking. About 29 stutters per
 * session are self-inflicted.
 *
 * With log_async, the logging thread only enqueues: pattern formatting and the trip to the SD are
 * done by a separate spdlog thread. The queue holds 8,192 messages (log_async_queue) and the policy is
 * to block when it is full.
 *
 * ================================================================================================
 *  With log_async on, the console did much worse. Do not turn it on as it is.
 *
 *  953 stutters against 226 with the synchronous logger, whole stretches with 0 % of the frames
 *  under 33.3 ms, frames of 1,050 ms, and an unplayable stretch.
 *
 *  The telltale sign is that CPU usage drops while FPS sinks:
 *      CPU total   266 %  ->  179-233 %
 *      ring       75.7 %  ->  50-72 %
 *      game       68.9 %  ->  25-48 %
 *  and the stutters come with `GPU 0.0 real, record 0.4`: a whole second of frame with nobody
 *  working. That is a blocked thread, not work.
 *
 *  Two design mistakes:
 *   1. `async_overflow_policy::block` blocks the writing thread when the queue fills up. The
 *      writer is the ring thread, the one that feeds the GPU.
 *   2. The spdlog pool thread inherits the priority of the thread that creates it, and on Horizon
 *      only priority 0x3B is time-sliced: with 3 cores busy with audio (0x2B), presentation
 *      (0x2C) and the ring (0x2D), that thread may almost never run. Queue full -> every REXLOG
 *      call blocks. See docs/platform-notes.md (Threads).
 * ================================================================================================
 */
REXCVAR_DEFINE_BOOL(log_async, false, "Log",
                    "Write the log from a separate thread (the calling thread only enqueues). Removes the "
                    "hitches of the periodic flush; false loses the last messages if there is a hang")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(log_async_queue, 8192, "Log", "Messages that fit in the async log queue")
    .range(256, 65536)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

/*
 * log_file_writer: the file log is written by a thread of its own (WriterFileSink below).
 *
 * futari-nx console run of 2026-10-07 (every CPU enhancement on): one frame of 1,022 ms and one of 277 ms
 * came exactly while the ring thread wrote its 10-second report (~40 lines). Each line took 50-150 ms
 * while the game was streaming its music from the same SD card. During those frames the profiler caught
 * the ring thread inside fsdev_write, under spdlog's file sink, in 84-92 % of its samples, with the game
 * thread waiting for the ring. newlib's FILE buffer is 1 KB, so almost every long line became an SD write
 * of its own, made by whichever thread logged it. flush_level warn and the 1 s periodic flush do not
 * prevent that: those writes come from the full 1 KB buffer, not from a flush.
 *
 * With this sink the logging thread only formats the line and appends it to a memory buffer: a copy under
 * the sink's mutex, a few microseconds. The writer thread (priority 0x3B on the Switch, the time-sliced
 * band, see log_async) writes the whole buffer in one go once a second, or sooner when 64 KB pile up or a
 * warning or worse asks for a flush. Unlike log_async, nothing can block the logging thread. When the
 * writer cannot keep up, lines past log_file_writer_buffer_kb are dropped and counted, and the count goes
 * into the log once the writer catches up. FlushLogging (crash handlers, exit) still writes everything
 * before it returns. A Switch fatal error appends the lines not yet written to rex_crash.log
 * (RexLogUnwrittenTail).
 */
REXCVAR_DEFINE_BOOL(log_file_writer, true, "Log",
                    "Write the file log from a thread of its own: the logging thread only copies the line to "
                    "memory and never waits for the disk (false = spdlog's synchronous rotating file sink)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(log_file_writer_buffer_kb, 4096, "Log",
                     "log_file_writer: lines waiting for the writer thread beyond this many KB are dropped "
                     "(and counted), never waited for")
    .range(64, 65536)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace rex {

namespace {

std::vector<LogCategoryEntry> g_registry;
spdlog::sink_ptr g_console_sink;
spdlog::sink_ptr g_file_sink;
spdlog::sink_ptr g_early_sink;
std::vector<spdlog::sink_ptr> g_extra_sinks;
bool g_early_initialized = false;
bool g_initialized = false;
std::mutex g_mutex;
LogConfig g_config;

// log_file_writer (see the cvar): the rotating file log, written by a thread of its own. The logging thread
// formats the line and copies it to pending_; the writer thread swaps pending_ for an empty buffer and writes
// it with one fwrite. Rotation works like spdlog's rotating_file_sink (base.N.log, oldest dropped), but per
// batch instead of per line.
class WriterFileSink final : public spdlog::sinks::base_sink<std::mutex> {
 public:
  // nullptr when the file cannot be opened (the caller falls back to spdlog's sink, which reports the error).
  static std::shared_ptr<WriterFileSink> Open(const std::string& path, size_t max_size, size_t max_files,
                                              size_t buffer_limit) {
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent, ec);
    std::FILE* file = OpenFile(path, "ab");
    if (!file)
      return nullptr;
    return std::shared_ptr<WriterFileSink>(new WriterFileSink(path, file, max_size, max_files, buffer_limit));
  }

  ~WriterFileSink() override {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    wake_.notify_one();
    if (writer_.joinable())
      writer_.join();
    if (file_)
      std::fclose(file_);
  }

  // Everything logged so far, written and flushed before it returns (FlushLogging: crash handlers, exit). Gives up
  // after 2 s when the writer is stuck in the disk, or the crash happened on the writer while it held the file.
  void WriteNow() {
    if (std::this_thread::get_id() == writer_.get_id())
      return;  // a crash inside the writer itself
    for (int i = 0; i < 2000; ++i) {  // try_lock in a loop: no timed mutex on every platform
      if (io_mutex_.try_lock()) {
        std::lock_guard io(io_mutex_, std::adopt_lock);
        WriteBatch();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // For a fatal error: up to 'cap' bytes of the newest lines not written yet, without waiting for any lock.
  size_t UnwrittenTail(char* out, size_t cap) {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || cap == 0)
      return 0;
    const size_t n = std::min(cap, pending_.size());
    std::memcpy(out, pending_.data() + pending_.size() - n, n);
    return n;
  }

 protected:
  // The logging thread, with mutex_ held: a copy to memory, never any I/O.
  void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t text;
    formatter_->format(msg, text);
    if (pending_.size() + text.size() > buffer_limit_) {
      ++dropped_;
      return;
    }
    pending_.append(text.data(), text.size());
    if (pending_.size() >= kBatch)
      wake_.notify_one();
  }
  // flush_on (warn and worse) and the periodic flush only wake the writer.
  void flush_() override {
    flush_wanted_ = true;
    wake_.notify_one();
  }

 private:
  static constexpr size_t kBatch = 64 * 1024;

  // Like spdlog's file_helper: on Windows other programs may read (and delete) the log while the game runs.
  static std::FILE* OpenFile(const std::string& path, const char* mode) {
#if REX_PLATFORM_WIN32
    return _fsopen(path.c_str(), mode, _SH_DENYNO);
#else
    return std::fopen(path.c_str(), mode);
#endif
  }

  WriterFileSink(std::string path, std::FILE* file, size_t max_size, size_t max_files, size_t buffer_limit)
      : path_(std::move(path)), file_(file), max_size_(max_size), max_files_(max_files),
        buffer_limit_(buffer_limit) {
    std::setvbuf(file_, nullptr, _IOFBF, kBatch);  // a batch is one SD write, not one per KB
    std::error_code ec;
    const auto size = std::filesystem::file_size(path_, ec);
    file_size_ = ec ? 0 : static_cast<size_t>(size);
    pending_.reserve(kBatch * 2);
    batch_.reserve(kBatch * 2);
    writer_ = std::thread([this] { Run(); });
  }

  void Run() {
    rex::thread::set_current_thread_name("Log writer");
#if REX_PLATFORM_SWITCH
    // Not above the guests: the writer must never take a core from the ring, audio or presentation threads.
    RexSwitchSetCurrentThreadPriority(REX_SWITCH_PRIO_GUEST);
#endif
    for (;;) {
      bool stop;
      {
        std::unique_lock lock(mutex_);
        wake_.wait_for(lock, std::chrono::seconds(1),
                       [this] { return stop_ || flush_wanted_ || pending_.size() >= kBatch; });
        stop = stop_;
      }
      {
        std::lock_guard io(io_mutex_);
        WriteBatch();
      }
      if (stop)
        return;
    }
  }

  // io_mutex_ held. The swap is done under io_mutex_ too, so batches reach the file in order.
  void WriteBatch() {
    uint64_t dropped;
    {
      std::lock_guard lock(mutex_);
      batch_.swap(pending_);
      flush_wanted_ = false;
      dropped = dropped_;
      dropped_ = 0;
    }
    if (dropped)
      batch_ += fmt::format("[log] {} lines dropped: the log writer thread fell behind (log_file_writer_buffer_kb)\n",
                            dropped);
    if (batch_.empty() || !file_)
      return;
    if (file_size_ > 0 && file_size_ + batch_.size() > max_size_)
      Rotate();
    if (file_) {
      std::fwrite(batch_.data(), 1, batch_.size(), file_);
      std::fflush(file_);
      file_size_ += batch_.size();
    }
    batch_.clear();
  }

  // spdlog's rotating_file_sink::rotate_: base.log -> base.1.log -> ... -> base.<max_files>.log (dropped).
  void Rotate() {
    std::fclose(file_);
    file_ = nullptr;
    std::error_code ec;
    for (size_t i = max_files_; i > 0; --i) {
      const auto src = spdlog::sinks::rotating_file_sink_mt::calc_filename(path_, i - 1);
      if (!std::filesystem::exists(src, ec))
        continue;
      const auto target = spdlog::sinks::rotating_file_sink_mt::calc_filename(path_, i);
      std::filesystem::remove(target, ec);
      std::filesystem::rename(src, target, ec);
    }
    file_ = OpenFile(path_, "wb");
    if (file_)
      std::setvbuf(file_, nullptr, _IOFBF, kBatch);
    file_size_ = 0;
  }

  const std::string path_;
  std::FILE* file_;
  const size_t max_size_, max_files_, buffer_limit_;
  size_t file_size_ = 0;
  std::string pending_;  // mutex_
  std::string batch_;    // io_mutex_
  uint64_t dropped_ = 0;  // mutex_
  bool flush_wanted_ = false, stop_ = false;  // mutex_
  std::condition_variable wake_;
  std::mutex io_mutex_;
  std::thread writer_;
};

std::shared_ptr<WriterFileSink> g_file_writer;  // the same sink as g_file_sink when log_file_writer is on

std::filesystem::path NextSequentialLogPath(const std::filesystem::path& logs_dir,
                                            std::string_view app_name) {
  std::filesystem::create_directories(logs_dir);

  int max_seq = 0;
  std::string prefix = std::string(app_name) + "_";
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(logs_dir, ec)) {
    if (!entry.is_regular_file())
      continue;
    auto stem = entry.path().stem().string();
    if (stem.starts_with(prefix)) {
      auto num_str = stem.substr(prefix.size());
      int num = 0;
      auto [ptr, parse_ec] = std::from_chars(num_str.data(), num_str.data() + num_str.size(), num);
      if (parse_ec == std::errc() && ptr == num_str.data() + num_str.size())
        max_seq = std::max(max_seq, num);
    }
  }

  return logs_dir / fmt::format("{}_{:03d}.log", app_name, max_seq + 1);
}

#ifdef __SWITCH__
// log_network: one TCP connection to a listener on the development machine. Never blocks the game: lines are
// sent with MSG_DONTWAIT and dropped when the socket is full; after an error the sink goes quiet.
class TcpLogSink : public spdlog::sinks::base_sink<std::mutex> {
 public:
  explicit TcpLogSink(int fd) : fd_(fd) {}

 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    if (fd_ < 0) return;
    spdlog::memory_buf_t text;
    formatter_->format(msg, text);
    if (send(fd_, text.data(), text.size(), MSG_DONTWAIT) < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
      fd_ = -1;
    }
  }
  void flush_() override {}

 private:
  int fd_;
};

int ConnectLogNetwork(const std::string& target) {
  const size_t colon = target.rfind(':');
  if (colon == std::string::npos) return -1;
  if (R_FAILED(socketInitializeDefault())) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(uint16_t(std::stoi(target.substr(colon + 1))));
  if (inet_pton(AF_INET, target.substr(0, colon).c_str(), &addr.sin_addr) != 1) return -1;
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }
  // printf/fprintf(stderr) of the libraries (NVK's "[nvk] ..." lines) go to the same connection.
  dup2(fd, STDOUT_FILENO);
  dup2(fd, STDERR_FILENO);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
  return fd;
}

spdlog::sink_ptr g_network_sink;
#endif

std::vector<spdlog::sink_ptr> BuildDefaultSinks() {
  std::vector<spdlog::sink_ptr> sinks;
  if (g_early_sink)
    sinks.push_back(g_early_sink);
  if (g_console_sink)
    sinks.push_back(g_console_sink);
  if (g_file_sink)
    sinks.push_back(g_file_sink);
  for (auto& s : g_extra_sinks)
    sinks.push_back(s);
#ifdef __SWITCH__
  if (g_network_sink)
    sinks.push_back(g_network_sink);
#endif
  return sinks;
}

// Build sinks for a specific category (handles per-category sinks from config)
std::vector<spdlog::sink_ptr> BuildCategorySinks(const std::string& name) {
  auto it = g_config.category_sinks.find(name);
  if (it != g_config.category_sinks.end()) {
    if (g_config.category_sinks_exclusive) {
      // Replace default sinks entirely
      return it->second;
    }
    // Additive: default sinks + category-specific sinks
    auto sinks = BuildDefaultSinks();
    for (auto& s : it->second)
      sinks.push_back(s);
    return sinks;
  }
  return BuildDefaultSinks();
}

// Resolve per-category level from config, or return default
spdlog::level::level_enum ResolveCategoryLevel(const std::string& name) {
  auto it = g_config.category_levels.find(name);
  if (it != g_config.category_levels.end())
    return it->second;
  return g_config.default_level;
}

// true once the spdlog thread pool is up (see log_async).
bool g_async_ready = false;

// Create a logger and register it
std::shared_ptr<spdlog::logger> CreateCategoryLogger(const std::string& name) {
  auto sinks = BuildCategorySinks(name);
  std::shared_ptr<spdlog::logger> logger;
  if (g_async_ready) {
    // The logging thread only enqueues; formatting and writing are done by the spdlog thread.
    logger = std::make_shared<spdlog::async_logger>(name, sinks.begin(), sinks.end(),
                                                    spdlog::thread_pool(),
                                                    spdlog::async_overflow_policy::block);
  } else {
    logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
  }
  logger->set_level(ResolveCategoryLevel(name));
  logger->flush_on(g_config.flush_level);
  spdlog::register_logger(logger);
  return logger;
}

}  // namespace

void InitLoggingEarly() {
  std::lock_guard lock(g_mutex);
  if (g_early_initialized || g_initialized)
    return;

#if REX_PLATFORM_WIN32
  auto sink = std::make_shared<spdlog::sinks::msvc_sink_mt>();
#else
  auto sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
#endif
  sink->set_level(spdlog::level::trace);
  sink->set_pattern("[%l] [%n] %v");
  g_early_sink = sink;

  g_config.default_level = kDefaultLogLevel;

  // Create loggers for any categories already registered during static init
  for (auto& entry : g_registry) {
    if (!entry.name.empty() && !entry.logger)
      entry.logger = CreateCategoryLogger(entry.name);
  }

  // Set default logger to "core" if registered
  for (auto& entry : g_registry) {
    if (entry.name == "core") {
      spdlog::set_default_logger(entry.logger);
      break;
    }
  }
  g_early_initialized = true;
}

void InitLogging(const LogConfig& config) {
  std::lock_guard lock(g_mutex);

  if (g_initialized) {
    g_config = config;
    for (auto& entry : g_registry) {
      if (!entry.logger)
        continue;
      entry.logger->set_level(ResolveCategoryLevel(entry.name));
      entry.logger->flush_on(config.flush_level);
      if (g_config.category_levels.count(entry.name))
        entry.has_explicit_level = true;
    }
    return;
  }

  g_config = config;

  // Early sink handling:
  //   Windows: the early msvc_sink is the persistent debug channel for GUI
  //     apps and does not conflict with the stdout console sink, so keep it.
  //   Non-Windows: drop the early stdout sink unconditionally so file-only
  //     configs don't leak to stdout and console configs don't duplicate.
#if !REX_PLATFORM_WIN32
  if (g_early_sink) {
    for (auto& entry : g_registry) {
      if (entry.logger)
        std::erase(entry.logger->sinks(), g_early_sink);
    }
    g_early_sink.reset();
  }
#endif

  // Console sink (stdout, colored). Intended for console-subsystem processes.
  if (config.log_to_console) {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_level(spdlog::level::trace);
    sink->set_pattern(config.console_pattern);
    g_console_sink = sink;
  }

  // File sink (rotating) with sequential naming fallback
  std::string resolved_path;
  if (config.log_file) {
    resolved_path = config.log_file;
  } else if (!config.app_name.empty()) {
    auto log_dir = config.log_dir.empty() ? std::filesystem::current_path() / "logs"
                                          : std::filesystem::path(config.log_dir);
    resolved_path = NextSequentialLogPath(log_dir, config.app_name).string();
  }
  if (!resolved_path.empty()) {
    const size_t max_size = static_cast<size_t>(REXCVAR_GET(log_max_file_size_mb)) * 1024 * 1024;
    const size_t max_files = static_cast<size_t>(REXCVAR_GET(log_max_files));
    spdlog::sink_ptr sink;
    if (REXCVAR_GET(log_file_writer)) {
      g_file_writer = WriterFileSink::Open(resolved_path, max_size, max_files,
                                           static_cast<size_t>(REXCVAR_GET(log_file_writer_buffer_kb)) * 1024);
      sink = g_file_writer;
    }
    if (!sink)
      sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(resolved_path, max_size, max_files, false);
    sink->set_level(spdlog::level::trace);
    sink->set_pattern(config.file_pattern);
    g_file_sink = sink;
  }

  g_extra_sinks = config.extra_sinks;

#ifdef __SWITCH__
  if (!g_network_sink && !REXCVAR_GET(log_network).empty()) {
    const int fd = ConnectLogNetwork(REXCVAR_GET(log_network));
    if (fd >= 0) {
      auto sink = std::make_shared<TcpLogSink>(fd);
      sink->set_level(spdlog::level::trace);
      sink->set_pattern(config.file_pattern);
      g_network_sink = sink;
    }
  }
#endif

  // The spdlog pool, before the loggers are rebuilt (see log_async). A single thread: log lines have to
  // keep their usual order.
  if (REXCVAR_GET(log_async) && !g_async_ready) {
    spdlog::init_thread_pool(static_cast<size_t>(REXCVAR_GET(log_async_queue)), 1);
    g_async_ready = true;
  }

  // Rebuild all loggers with new sinks
  for (auto& entry : g_registry) {
    if (entry.name.empty())
      continue;
    if (entry.logger)
      spdlog::drop(entry.name);
    entry.logger = CreateCategoryLogger(entry.name);
    if (g_config.category_levels.count(entry.name))
      entry.has_explicit_level = true;
  }

  // Set default logger to "core"
  for (auto& entry : g_registry) {
    if (entry.name == "core") {
      spdlog::set_default_logger(entry.logger);
      break;
    }
  }
  g_initialized = true;

  // Periodic flush
  int flush_interval = REXCVAR_GET(log_flush_interval);
  if (flush_interval > 0)
    spdlog::flush_every(std::chrono::seconds(flush_interval));
}

void InitLogging(const char* log_file, spdlog::level::level_enum level) {
  LogConfig config;
  config.log_file = log_file;
  config.default_level = level;
  InitLogging(config);
}

void ShutdownLogging() {
  std::lock_guard lock(g_mutex);
  if (!g_initialized && !g_early_initialized)
    return;

  for (auto& entry : g_registry)
    if (entry.logger)
      entry.logger->flush();
  if (g_file_writer)
    g_file_writer->WriteNow();

  spdlog::shutdown();  // also joins the async pool thread
  g_async_ready = false;
  g_registry.clear();
  g_console_sink.reset();
  g_file_sink.reset();
  g_file_writer.reset();
  g_early_sink.reset();
  g_extra_sinks.clear();
  g_initialized = false;
  g_early_initialized = false;
}

void FlushLogging() {
  std::lock_guard lock(g_mutex);
  for (auto& entry : g_registry)
    if (entry.logger)
      entry.logger->flush();
  if (g_file_writer)
    g_file_writer->WriteNow();  // flush() above only wakes the writer thread
}

LogCategoryId RegisterLogCategory(const char* name) {
  std::lock_guard lock(g_mutex);

  // Check for duplicates
  for (size_t i = 0; i < g_registry.size(); ++i) {
    if (g_registry[i].name == name) {
      return LogCategoryId{static_cast<uint16_t>(i)};
    }
  }

  uint16_t id = static_cast<uint16_t>(g_registry.size());
  LogCategoryEntry entry;
  entry.name = name;
  if (g_initialized || g_early_initialized) {
    entry.logger = CreateCategoryLogger(name);
  }
  g_registry.push_back(std::move(entry));
  return LogCategoryId{id};
}

LogCategoryId RegisterLogSubcategory(const char* name, LogCategoryId parent) {
  std::lock_guard lock(g_mutex);

  std::string parent_name;
  if (parent.id < g_registry.size())
    parent_name = g_registry[parent.id].name;
  std::string full_name = parent_name + "." + name;

  for (size_t i = 0; i < g_registry.size(); ++i) {
    if (g_registry[i].name == full_name)
      return LogCategoryId{static_cast<uint16_t>(i)};
  }

  uint16_t id = static_cast<uint16_t>(g_registry.size());
  LogCategoryEntry entry;
  entry.name = full_name;
  entry.parent = parent;
  if (g_initialized || g_early_initialized) {
    entry.logger = CreateCategoryLogger(full_name);
    if (g_config.category_levels.count(full_name))
      entry.has_explicit_level = true;
  }
  g_registry.push_back(std::move(entry));
  return LogCategoryId{id};
}

std::optional<LogCategoryId> FindCategory(const std::string& name) {
  std::lock_guard lock(g_mutex);
  for (size_t i = 0; i < g_registry.size(); ++i) {
    if (g_registry[i].name == name) {
      return LogCategoryId{static_cast<uint16_t>(i)};
    }
  }
  return std::nullopt;
}

std::span<const LogCategoryEntry> GetAllCategories() {
  // No lock: only safe to call from main thread or after init.
  return {g_registry.data(), g_registry.size()};
}

spdlog::logger* GetLoggerRaw(LogCategoryId category) {
  if (!g_initialized && !g_early_initialized)
    InitLoggingEarly();
  if (category.id < g_registry.size()) {
    return g_registry[category.id].logger.get();
  }
  return nullptr;
}

std::shared_ptr<spdlog::logger> GetLogger(LogCategoryId category) {
  if (!g_initialized && !g_early_initialized)
    InitLoggingEarly();
  if (category.id < g_registry.size()) {
    return g_registry[category.id].logger;
  }
  return nullptr;
}

std::shared_ptr<spdlog::logger> GetLogger() {
  return GetLogger(rex::log::core());
}

void SetCategoryLevel(LogCategoryId category, spdlog::level::level_enum level) {
  if (category.id < g_registry.size()) {
    if (auto& logger = g_registry[category.id].logger) {
      logger->set_level(level);
    }
  }
}

void SetAllLevels(spdlog::level::level_enum level) {
  for (auto& entry : g_registry) {
    if (entry.logger)
      entry.logger->set_level(level);
  }
}

void SetRootLevel(LogCategoryId root, spdlog::level::level_enum level) {
  SetCategoryLevel(root, level);
  for (auto& entry : g_registry) {
    if (entry.parent.has_value() && entry.parent->id == root.id && !entry.has_explicit_level &&
        entry.logger)
      entry.logger->set_level(level);
  }
}

void RegisterLogLevelCallback() {
  rex::cvar::RegisterChangeCallback("log_level", [](std::string_view, std::string_view value) {
    if (auto level = ParseLogLevel(std::string(value))) {
      for (size_t i = 0; i < g_registry.size(); ++i) {
        auto& entry = g_registry[i];
        if (!entry.parent.has_value() && entry.logger && !entry.has_explicit_level)
          SetRootLevel(LogCategoryId{static_cast<uint16_t>(i)}, *level);
      }
    }
  });
}

void AddSink(spdlog::sink_ptr sink) {
  std::lock_guard lock(g_mutex);
  g_extra_sinks.push_back(sink);
  for (auto& entry : g_registry) {
    if (entry.logger)
      entry.logger->sinks().push_back(sink);
  }
}

void AddSink(LogCategoryId category, spdlog::sink_ptr sink) {
  std::lock_guard lock(g_mutex);
  if (category.id < g_registry.size()) {
    if (auto& logger = g_registry[category.id].logger) {
      logger->sinks().push_back(sink);
    }
  }
}

void RemoveSink(spdlog::sink_ptr sink) {
  std::lock_guard lock(g_mutex);
  std::erase(g_extra_sinks, sink);
  for (auto& entry : g_registry) {
    if (entry.logger)
      std::erase(entry.logger->sinks(), sink);
  }
}

void RemoveSink(LogCategoryId category, spdlog::sink_ptr sink) {
  std::lock_guard lock(g_mutex);
  if (category.id < g_registry.size()) {
    if (auto& logger = g_registry[category.id].logger) {
      std::erase(logger->sinks(), sink);
    }
  }
}

void ReplaceConsoleSink(spdlog::sink_ptr sink) {
  std::lock_guard lock(g_mutex);
  for (auto& entry : g_registry) {
    if (!entry.logger)
      continue;
    auto& sinks = entry.logger->sinks();
    if (g_console_sink)
      std::erase(sinks, g_console_sink);
    if (sink)
      sinks.push_back(sink);
  }
  g_console_sink = sink;
}

void SetConsolePattern(const std::string& pattern) {
  if (g_console_sink)
    g_console_sink->set_pattern(pattern);
}

void SetFilePattern(const std::string& pattern) {
  if (g_file_sink)
    g_file_sink->set_pattern(pattern);
}

// ==========================================================================
// CLI Helpers
// ==========================================================================

std::optional<spdlog::level::level_enum> ParseLogLevel(const std::string& level_str) {
  static const std::unordered_map<std::string, spdlog::level::level_enum> level_map = {
      {"trace", spdlog::level::trace},  {"debug", spdlog::level::debug},
      {"info", spdlog::level::info},    {"warn", spdlog::level::warn},
      {"warning", spdlog::level::warn}, {"error", spdlog::level::err},
      {"err", spdlog::level::err},      {"critical", spdlog::level::critical},
      {"off", spdlog::level::off},
  };

  std::string lower = level_str;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  auto it = level_map.find(lower);
  if (it != level_map.end())
    return it->second;
  return std::nullopt;
}

spdlog::level::level_enum ParseLogLevelOr(const std::string& level_str,
                                          spdlog::level::level_enum default_level) {
  return ParseLogLevel(level_str).value_or(default_level);
}

LogConfig BuildLogConfig(const char* log_file, const std::string& cli_level,
                         const std::map<std::string, std::string>& category_levels) {
  LogConfig config;
  config.log_file = log_file;

  // Build-type default
  config.default_level = kDefaultLogLevel;

  // Environment variable
  if (auto env_level = rex::platform::env::get("REX_LOG_LEVEL")) {
    if (auto level = ParseLogLevel(*env_level))
      config.default_level = *level;
  } else if (auto env_level2 = rex::platform::env::get("SPDLOG_LEVEL")) {
    if (auto level = ParseLogLevel(*env_level2))
      config.default_level = *level;
  }

  // CLI global level overrides environment
  if (!cli_level.empty()) {
    if (auto level = ParseLogLevel(cli_level))
      config.default_level = *level;
  }

  // Per-category CLI levels (string-keyed)
  for (const auto& [cat_name, level_str] : category_levels) {
    if (level_str.empty())
      continue;
    if (auto level = ParseLogLevel(level_str)) {
      config.category_levels[cat_name] = *level;
    }
  }

  return config;
}

std::map<std::string, std::string> ParseCategoryLevelsFromConfig(
    const std::filesystem::path& config_path) {
  std::map<std::string, std::string> result;
  if (!std::filesystem::exists(config_path))
    return result;

  try {
    auto config = toml::parse_file(config_path.string());
    auto* log_table = config["log"].as_table();
    if (!log_table)
      return result;
    auto* levels_table = (*log_table)["levels"].as_table();
    if (!levels_table)
      return result;

    for (const auto& [key, value] : *levels_table) {
      if (value.is_string())
        result[std::string(key)] = value.as_string()->get();
    }
  } catch (const toml::parse_error&) {}

  return result;
}

}  // namespace rex

// log_file_writer: the newest lines the writer thread has not written yet, for the Switch fatal-error report
// (exception_handler_switch.cpp appends them to rex_crash.log). Takes no lock it would have to wait for.
extern "C" size_t RexLogUnwrittenTail(char* out, size_t cap) {
  auto writer = rex::g_file_writer;
  return writer ? writer->UnwrittenTail(out, cap) : 0;
}
