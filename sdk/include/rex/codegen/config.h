/**
 * @file        rexcodegen/internal/config.h
 * @brief       Recompiler configuration types
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <toml++/toml.hpp>

#include <rex/codegen/function_graph.h>  // For JumpTable

namespace rex::codegen {

struct MidAsmHook {
  std::string name;
  std::vector<std::string> registers;

  bool ret = false;
  bool returnOnTrue = false;
  bool returnOnFalse = false;

  uint32_t jumpAddress = 0;
  uint32_t jumpAddressOnTrue = 0;
  uint32_t jumpAddressOnFalse = 0;

  bool afterInstruction = false;
};

// Unified function/chunk configuration
// A "chunk" is simply a function entry with a non-zero parent field
struct FunctionConfig {
  uint32_t size = 0;    // Explicit size in bytes (mutually exclusive with end)
  uint32_t end = 0;     // End address, exclusive (mutually exclusive with size)
  std::string name;     // Custom symbol name (empty = auto-generate sub_XXXXXXXX)
  uint32_t parent = 0;  // Parent function address (0 = standalone, non-zero = chunk)

  // Keep non-volatiles in ctx instead of localizing them. Marks an MSVC SEH
  // funclet, which reads the registers its owner left live and would see a zero
  // from a local. Callers sync their localized copies across the call site.
  bool shareRegisters = false;

  // Get effective size (prefers size over end)
  uint32_t getSize(uint32_t address) const {
    return size ? size : (end > address ? end - address : 0);
  }
  // Returns true if this is a discontinuous chunk belonging to a parent function
  bool isChunk() const { return parent != 0; }
};

// Section info for analysis output
struct SectionInfo {
  std::string name;
  uint64_t address = 0;
  uint64_t size = 0;
  std::string flags;  // "rx", "rw", "r" etc.
};

// Function entry for analysis output
struct FunctionEntry {
  uint64_t address = 0;
  uint64_t size = 0;
  std::string name;  // optional, defaults to "sub_XXXXXXXX"
};

struct RecompilerConfig {
  // === Required user-provided fields ===
  std::string projectName = "rex";  ///< Project name for output files
  std::string filePath;             ///< Path to XEX/ELF file
  std::string outDirectoryPath;     ///< Output directory for generated code
  std::string templateDir;          ///< Optional custom template directory for overrides

  /// Every TOML that fed this config, in load order. LoadFromTable() gets a
  /// parsed table, so a manifest-embedded entry omits the manifest itself.
  std::vector<std::string> loadedFiles;

  // === Code generation options (optional) ===
  bool skipLr = false;
  /// Codegen diet: `ctx.lr = <return address>;` stores before bl/bctrl/blrl are dropped (nothing but a stack
  /// slot reads the guest LR), except at the return addresses listed in lrKeepReturns (hooks that compare
  /// ctx.lr) and at PIC `bl` to a local label (the LR is then the data).
  bool elideLrStores = false;
  /// Codegen diet: D-form loads/stores pass the register and the displacement apart
  /// (REX_LOAD_D_*(reg, disp) = base + zext(reg) + disp) instead of (u32)(reg + disp).
  bool dformDispSplit = false;
  /// Codegen diet: fctiwz as an fcvtzs plus a NaN select (rex::ppc::fctiwz).
  bool fctiwzInline = false;
  /// sa2-nx round 7: no host flush-mode switch (an `msr fpcr` on AArch64, very slow on the A57) for floating-point
  /// moves: lfd/stfd and their indexed/update forms, stfiwx, fmr, fabs, fnabs and fneg only move bits. And in VMX
  /// mode (flush on), lfs/stfs and their forms convert through rex::ppc::SingleToDoubleAnyMode /
  /// DoubleToSingleAnyMode, which give the IEEE result in either mode, instead of switching the mode off and on again.
  bool fpMovesModeless = false;
  std::unordered_set<uint32_t> lrKeepReturns;
  bool ctrAsLocalVariable = false;
  bool xerAsLocalVariable = false;
  bool reservedRegisterAsLocalVariable = false;
  bool skipMsr = false;
  bool crRegistersAsLocalVariables = false;
  bool nonArgumentRegistersAsLocalVariables = false;
  /// sa2-nx: with non_argument_as_local, the functions that keep r0, r2, r11, r12, f0 and v32-v63 in ctx
  /// (`non_argument_as_local_keep = [0x82XXXXXX, ...]`, additive): callees that take an argument there (the stack
  /// probe reads the frame size in r12) and callers that hand one over through an indirect call (the C++ EH
  /// dispatcher passes the frame to a funclet in r12 and calls it through the LR). Everything else in them stays
  /// localized. A function that localizes these registers hands its values over around a call to one of them and
  /// takes back what it leaves (emit_call_handing_non_arguments). The SDK's __save* / __rest* helpers (r12 and r11
  /// arguments) keep them in ctx without being listed.
  std::unordered_set<uint32_t> nonArgumentKeep;
  bool nonVolatileRegistersAsLocalVariables = false;
  bool generateExceptionHandlers = false;  ///< Generate SEH exception handler wrappers

  /// Arguments in registers (codegen/args_in_registers.h): direct calls between recompiled functions pass
  /// r3-r10, f1-f13, r0/r11/r12/f0 and lr as C++ arguments and return r3 as the C++ return value. The
  /// functions named in `argsInRegistersExcludeFile` (hooked ones) keep the old ABI.
  bool argsInRegisters = false;
  std::string argsInRegistersExcludeFile;  ///< one emitted function name (sub_82XXXXXX) per line
  bool argsInRegistersInstrument = false;  ///< ME_ARGS_COUNT entries (validation builds)
  bool argsInRegistersPoison = false;      ///< ME_ARGS_POISON before fast calls (validation builds)

  // === Analysis tuning (optional) ===
  uint32_t maxJumpExtension = 65536;  ///< Max bytes to extend function for jump table targets
  uint32_t dataRegionThreshold = 16;  ///< Consecutive invalid instructions to mark as data region
  uint32_t largeFunctionThreshold = 1048576;  ///< 1MB - warn if function exceeds this size

  // Optional override for DLL module flag. If unset, the orchestrator infers
  // from the module's position in the manifest (entrypoint = false, modules = true).
  std::optional<bool> isDll;

  // === Manual overrides ===
  std::unordered_map<uint32_t, FunctionConfig> functions;  ///< Function/chunk configuration
  std::unordered_map<uint32_t, JumpTable> switchTables;
  std::unordered_map<uint32_t, MidAsmHook> midAsmHooks;
  uint32_t longJmpAddress = 0;
  uint32_t setJmpAddress = 0;

  // === rexcrt: CRT function address overrides ===
  // Maps function name -> guest address (e.g. "CreateFileA" -> 0x8248B780)
  // Parsed from [rexcrt] TOML table. Codegen generates rexcrt_<Name> entries.
  std::unordered_map<std::string, uint32_t> rexcrtFunctions;

  // === User hints (merged with analysis results in AnalysisState) ===
  std::unordered_map<uint32_t, uint32_t> invalidInstructionHints;  ///< addr -> size
  std::unordered_set<uint32_t>
      knownIndirectCallHints;  ///< bctr addresses that are vtable/computed calls
  std::vector<uint32_t> exceptionHandlerFuncHints;  ///< Additional exception handler addresses

  /**
   * Load configuration from a TOML file.
   *
   * Supports an optional `includes` array for layered config. Paths in
   * `includes` resolve relative to the including file's directory.  Merge
   * semantics: scalars last-wins, keyed tables additive (same key = last
   * wins), arrays-of-tables deduplicated by primary key, sets additive.
   *
   * @param configFilePath Path to the TOML config file
   * @return true on success, false on error (parse failure, circular
   *         include, depth exceeded)
   */
  bool Load(const std::string_view& configFilePath);

  /**
   * Load configuration from an in-memory TOML table (e.g. an inline binary
   * entry inside a manifest). Includes referenced from the table resolve
   * relative to `base_dir`. Same merge semantics as Load().
   */
  bool LoadFromTable(const toml::table& tbl, const std::filesystem::path& base_dir);

  /// Validation result containing warnings and errors.
  struct ValidationResult {
    bool valid = true;                  ///< true if no errors (warnings OK)
    std::vector<std::string> warnings;  ///< Non-fatal issues
    std::vector<std::string> errors;    ///< Fatal issues that block codegen

    explicit operator bool() const { return valid; }
  };

  /**
   * Validate the loaded configuration.
   * Checks address alignment, required fields, and sanity constraints.
   * @return ValidationResult with warnings and errors
   */
  ValidationResult Validate() const;
};

}  // namespace rex::codegen
