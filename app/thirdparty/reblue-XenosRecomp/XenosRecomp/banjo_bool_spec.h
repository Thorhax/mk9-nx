// banjo-nx (2026-10-17, package 27): the bool constants as specialization constants, for Renut's translator (banjo-nx's
// banjo_gpu_bool_spec, rewritten for this translator's output; docs/renut-renderer-port.md).
//
// The translator tests a Xenos bool constant as BOOL_BIT(N) (N the unified index: vertex shader bools 0-127, pixel
// shader bools 128-255), a read of the draw's SharedConstants. The tests are uniform branches around the light blocks
// and the like (41,090 in the library's pixel shaders): the driver compiles every block, and allocates registers for
// the largest path. Here each bool a shader tests gets a specialization constant of its own, 1000 + N:
//
//     [[vk::constant_id(1005)]] const uint g_banjoBool5 = 2;
//     BOOL_BIT(5)   ->   BANJO_BOOL_5 = select(g_banjoBool5 == 2u, BOOL_BIT(5), g_banjoBool5 != 0u)
//
// Value 2 (the default, a pipeline created without the constant) reads the bool at run time, as before; 0 or 1 is the
// bool's value in the pipeline, which the driver folds, dropping the block the draw does not run. The renderer gives
// a pipeline the values of the draws it serves (the pipeline key holds them); the result is the same either way. SPIR-V
// only: DXIL has no specialization constants and keeps BOOL_BIT. Off with BANJO_RENUT_BOOL_SPEC=0 in the build's
// environment (the shader cache is rebuilt).
#pragma once

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <set>
#include <string>

namespace banjo_bool_spec {

// Shaders given specialization constants, and the constants, over the run (main.cpp prints them).
inline std::atomic<size_t> g_shaders{0}, g_constants{0};

// Rewrites 'hlsl' (the shader's text up to the end of main's body). Returns the number of bools specialised.
inline size_t Specialise(std::string &hlsl) {
    if (const char *off = std::getenv("BANJO_RENUT_BOOL_SPEC"); off && off[0] == '0')
        return 0;
    const size_t main = hlsl.find("\nvoid main(");
    if (main == std::string::npos)
        return 0;
    static const std::string kCall = "BOOL_BIT(";
    std::set<unsigned> used;
    std::string out;
    out.reserve(hlsl.size() + 256);
    size_t at = 0;
    for (;;) {
        const size_t k = hlsl.find(kCall, at);
        if (k == std::string::npos)
            break;
        size_t e = k + kCall.size();
        unsigned n = 0;
        size_t digits = 0;
        while (e < hlsl.size() && std::isdigit(static_cast<unsigned char>(hlsl[e]))) {
            n = n * 10 + unsigned(hlsl[e] - '0');
            ++e;
            ++digits;
        }
        // Only a literal index (the macro's own definition, BOOL_BIT(n), is left alone) of a real bool.
        const bool literal = digits > 0 && e < hlsl.size() && hlsl[e] == ')' && n < 256 &&
                             (k == 0 || !(std::isalnum(static_cast<unsigned char>(hlsl[k - 1])) || hlsl[k - 1] == '_'));
        out.append(hlsl, at, k - at);
        if (literal) {
            used.insert(n);
            out += "BANJO_BOOL_" + std::to_string(n);
            at = e + 1;
        } else {
            out += kCall;
            at = k + kCall.size();
        }
    }
    if (used.empty())
        return 0;
    out.append(hlsl, at, std::string::npos);
    std::string decls = "\n// banjo-nx: bool constants as specialization constants (banjo_bool_spec.h)\n#ifdef __spirv__\n";
    for (const unsigned n : used) {
        const std::string s = std::to_string(n);
        decls += "[[vk::constant_id(" + std::to_string(1000 + n) + ")]] const uint g_banjoBool" + s + " = 2;\n";
        decls += "#define BANJO_BOOL_" + s + " select(g_banjoBool" + s + " == 2u, BOOL_BIT(" + s + "), g_banjoBool" + s +
                 " != 0u)\n";
    }
    decls += "#else\n";
    for (const unsigned n : used)
        decls += "#define BANJO_BOOL_" + std::to_string(n) + " BOOL_BIT(" + std::to_string(n) + ")\n";
    decls += "#endif\n";
    // Before the first use: the named bool defines the translator writes ahead of main may already refer to them, but
    // a macro is expanded where it is used, so in front of main is early enough for every use.
    const size_t insert = out.find("\nvoid main(");
    out.insert(insert, decls);
    hlsl.swap(out);
    g_shaders.fetch_add(1, std::memory_order_relaxed);
    g_constants.fetch_add(used.size(), std::memory_order_relaxed);
    return used.size();
}

}  // namespace banjo_bool_spec
