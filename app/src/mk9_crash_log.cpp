// Fatal host exceptions/signals that the SDK does not report, logged with the module and offset so the crash can be
// symbolized afterwards. Windows: sa2-nx's vectored handler (app/src/sa2_crash_log.cpp there). Linux: a signal handler.

#if defined(_WIN32)

#include <windows.h>

#include <cstring>

#include <rex/logging.h>
#include <rex/logging/api.h>

namespace {

LONG CALLBACK LogFatalException(EXCEPTION_POINTERS* info) {
  const DWORD code = info->ExceptionRecord->ExceptionCode;
  switch (code) {
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_PRIV_INSTRUCTION:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case STATUS_STACK_BUFFER_OVERRUN:
      break;
    default:
      return EXCEPTION_CONTINUE_SEARCH;
  }
  const auto pc = reinterpret_cast<uintptr_t>(info->ExceptionRecord->ExceptionAddress);
  HMODULE module = nullptr;
  char path[MAX_PATH] = "?";
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(pc), &module) && module) {
    GetModuleFileNameA(module, path, MAX_PATH);
  }
  const char* name = std::strrchr(path, '\\');
  REXLOG_CRITICAL("[crash] host exception {:08X} at {:016X} ({}+0x{:X}) on thread {}", uint32_t(code), pc,
                  name ? name + 1 : path, pc - reinterpret_cast<uintptr_t>(module), GetCurrentThreadId());
  rex::FlushLogging();
  return EXCEPTION_CONTINUE_SEARCH;
}

// Installed before main, first in the chain; it only logs and never handles.
const PVOID g_handler = AddVectoredExceptionHandler(1, LogFatalException);

}  // namespace

#endif

// Linux: logs fatal host signals that the SDK does not report (illegal instruction from a __builtin_trap in the
// generated code, divide by zero, ...) with the executable offset, so the crash can be symbolized afterwards:
// llvm-symbolizer --obj=mk9 0x<offset> (or addr2line -e mk9 -f -C 0x<offset>). SIGSEGV/SIGBUS are left alone: the
// SDK handles those itself (guest memory, MMIO) and reports the unhandled ones. The Linux counterpart of sa2-nx's
// Windows vectored handler (app/src/sa2_crash_log.cpp there).

#if defined(__linux__)

#include <dlfcn.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>

#include <rex/logging.h>
#include <rex/logging/api.h>

namespace {

struct sigaction g_previous[32];

void LogFatalSignal(int sig, siginfo_t* info, void* context) {
  uintptr_t pc = 0;
#if defined(__x86_64__)
  pc = static_cast<uintptr_t>(static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
  pc = static_cast<uintptr_t>(static_cast<ucontext_t*>(context)->uc_mcontext.pc);
#endif
  Dl_info dl{};
  const char* name = "?";
  uintptr_t base = 0;
  if (dladdr(reinterpret_cast<void*>(pc), &dl) && dl.dli_fname) {
    name = std::strrchr(dl.dli_fname, '/') ? std::strrchr(dl.dli_fname, '/') + 1 : dl.dli_fname;
    base = reinterpret_cast<uintptr_t>(dl.dli_fbase);
  }
  REXLOG_CRITICAL("[crash] host signal {} ({}) at {:016X} ({}+0x{:X}), fault address {:016X}, thread {}", sig,
                  strsignal(sig), pc, name, pc - base, reinterpret_cast<uintptr_t>(info->si_addr), gettid());
  rex::FlushLogging();
  // Hand over to whatever was installed before (or the default action: core dump / exit).
  sigaction(sig, &g_previous[sig], nullptr);
  raise(sig);
}

bool Install() {
  for (int sig : {SIGILL, SIGFPE, SIGTRAP, SIGABRT}) {
    struct sigaction sa {};
    sa.sa_sigaction = LogFatalSignal;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, &g_previous[sig]);
  }
  return true;
}

// Installed before main; it only logs, then re-raises with the previous action.
const bool g_installed = Install();

}  // namespace

#endif
