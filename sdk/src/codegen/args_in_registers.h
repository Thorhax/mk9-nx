/**
 * @file        codegen/args_in_registers.h
 * @brief       Arguments in registers: recompiled functions take r3-r10/f1-f13 (and a few scratch
 *              registers and lr) as C++ values instead of through PPCContext
 *
 * Every recompiled function is `void sub_X(PPCContext& ctx, uint8_t* base)` and passes arguments,
 * the return value and every other register through the PPCContext in memory. This pass rewrites the
 * emitted bodies so that direct calls between ordinary functions pass those registers as C++
 * arguments and return r3 as the C++ return value:
 *
 *   sub_X(ctx, base)          stable ABI entry (dispatch table, indirect calls, hooks, runtime): a thin
 *                             wrapper that unpacks ctx, calls __fast_sub_X and stores r3 back. A function
 *                             that nobody calls directly keeps one body (the wrapper loads from ctx).
 *   __fast_sub_X(ctx, base,   fast entry, used by direct calls whose callee is not excluded (hooked)
 *                p_r3, ...)
 *
 * Which registers a function receives by value (Par), which ones it still needs valid in ctx at entry
 * (Fw: forwarded untouched to an unknown callee), which ones it exports through ctx at its exits (Ex)
 * and whether it returns r3 by value (Val) come from an interprocedural dataflow over the emitted code.
 * The pass works on the C++ text of the function bodies exactly as FunctionNode::emitCpp emits them, so
 * it sees every register access the builders produce. The same algorithm exists as the post-codegen script
 * tools/args_in_registers.py in this repository (reference implementation; the two outputs are
 * compared in tests).
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace rex::codegen {

struct ArgsInRegistersOptions {
  /// Functions that keep the old ABI because something else replaces them (hooks), by emitted name
  /// (sub_82XXXXXX). Direct calls to them stay `name(ctx, base)`.
  std::unordered_set<std::string> excluded;
  /// Emit ME_ARGS_COUNT(name, kind) at every entry (call counters for validation builds).
  bool instrument = false;
  /// Emit ME_ARGS_POISON(mask) before fast calls (tripwire for stale ctx reads, validation builds).
  bool poison = false;
};

struct ArgsInRegistersStats {
  size_t functions = 0;
  size_t eligible = 0;
  size_t combined = 0;      ///< converted functions without direct callers (one body)
  size_t fastSites = 0;     ///< direct fast call sites
  size_t unknownSites = 0;  ///< calls that keep the old ABI (indirect, hooked, imports)
  size_t argsByValue = 0;
};

/**
 * Rewrites `bodies` in place. names[i] is the symbol of bodies[i] (as in its DEFINE_REX_FUNC); bodies
 * that are not plain recompiled functions are left alone. declarations[i] receives the fast-entry
 * prototypes (full `extern "C" ...;` lines) that the translation unit holding bodies[i] needs; they
 * belong next to the DECLARE_REX_FUNC lines of the file's header.
 */
void ApplyArgsInRegisters(const std::vector<std::string>& names, std::vector<std::string>& bodies,
                          std::vector<std::vector<std::string>>& declarations,
                          const ArgsInRegistersOptions& options, ArgsInRegistersStats* stats = nullptr);

}  // namespace rex::codegen
