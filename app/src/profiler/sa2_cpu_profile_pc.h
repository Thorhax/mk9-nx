// sa2-nx (from nfsmw-nx via banjo-nx) - sampling CPU profiler and stack dumper for the PC build (testing only).
//
// The console has its own profiler with stacks (switch_perf.cpp), but using it requires a console build. This
// one does the same on the PC: with sa2_cpu_profile_from_s > 0, from that second of process lifetime and
// for sa2_cpu_profile_duration_s, it suspends the busy threads every ~1 ms, records their program counter
// and a few stack addresses, and writes logs/perfil_pc.csv. The addresses are symbolized offline with
// llvm-symbolizer. On the Switch it does nothing.
//
// VolcarPilas is for hangs: the full stacks of every thread, including blocked ones, in logs/pilas_N.txt.

#pragma once

namespace nfsmw::perfil_pc {

#if defined(_WIN32)
// Starts the profiler thread and the timed stack dump thread (sa2_stack_dump_s) if the cvars ask
// for them.
void Arrancar();
// Stops the threads (idempotent).
void Parar();
// Dumps the stacks of every thread in the process except the caller. Used by the watchdog in nfsmw_app.h
// when it sees the game stalled.
void VolcarPilas(const char* motivo);
#else
inline void Arrancar() {}
inline void Parar() {}
inline void VolcarPilas(const char*) {}
#endif

}  // namespace nfsmw::perfil_pc
