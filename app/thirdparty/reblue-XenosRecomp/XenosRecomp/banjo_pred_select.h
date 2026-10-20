// banjo-nx (2026-10-17): predicated blocks without a branch, for Renut's translator (banjo-nx's banjo_gpu_pred_select
// and banjo_gpu_pred_select_ps, rewritten for this translator's output; docs/renut-renderer-port.md).
//
// The translator writes each predicated instruction, and each predicated exec block, as `if (p0)` / `if (!p0)` around
// its statements. NAK keeps those branches (a heavy pixel shader has a few hundred), and a branch costs more than the
// arithmetic it skips. A block whose statements are all plain register arithmetic becomes straight-line code:
//
//     if (p0) { r6.xy = A; ps = B; }   ->   { const bool pb1 = p0;
//                                             r6.xy = select(pb1, (float2)(A), r6.xy);
//                                             ps = select(pb1, (float)(B), ps); }
//
// Exact: each statement still runs in order when the predicate holds, and each target keeps its value when it does
// not (so later statements read what they read before). The predicate is taken once, at the block's entry, as the
// `if` took it; a block nested in a converted one takes `and(outer, p0)`, the current p0 (an earlier statement of the
// outer block may have set it) under the outer block's own condition. Only side-effect-free statements qualify: the
// targets are temporaries (rN, ps, p0), and anything that fetches, loads, kills, writes an output or the address
// register, takes a derivative or indexes keeps its branch (the block stays as written; blocks inside it are still
// converted). At most kMaxStatements statements per converted block, as banjo-nx's pixel shader rule.
#pragma once

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace banjo_pred_select {

constexpr size_t kMaxStatements = 12;

// Blocks converted, over every shader of the run (main.cpp prints it), and memory fetches among their statements.
inline std::atomic<size_t> g_blocks{0}, g_fetches{0};

inline std::string_view Trim(std::string_view s) {
    const size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string_view::npos)
        return {};
    const size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

struct Statement {
    std::string lhs, rhs, type;
};

// banjo-nx (package 27): a vertex fetch from memory (bone palettes, instance data) may run unconditionally in its
// clamped form (vfetchMemoryClamped in shader_common.h: the address stays inside the draw's fetch data, so a fetch
// whose predicate does not hold reads something there and the select drops it), as banjo-nx's bone padding lets its
// translator do. Off with BANJO_RENUT_PRED_FETCH=0 in the build's environment.
inline bool FetchesAllowed() {
    static const bool on = [] {
        const char *off = std::getenv("BANJO_RENUT_PRED_FETCH");
        return !(off && off[0] == '0');
    }();
    return on;
}

// The rhs with its memory fetches' names and their slot arguments taken out, for the forbidden-token test below.
inline std::string WithoutMemoryFetches(std::string_view rhs) {
    std::string s(rhs);
    for (const char *name : {"vfetchMemory(", "g_VertexFetchSlot("}) {
        const size_t n = std::strlen(name);
        for (size_t k = s.find(name); k != std::string::npos; k = s.find(name, k))
            s.replace(k, n, "(");
    }
    return s;
}

// `lhs = rhs;` with lhs rN, rN.mask, ps or p0 and a side-effect-free rhs.
inline bool ParseStatement(std::string_view t, Statement &st) {
    if (t.size() < 6 || t.back() != ';')
        return false;
    const size_t eq = t.find(" = ");
    if (eq == std::string_view::npos)
        return false;
    const std::string_view lhs = t.substr(0, eq);
    const std::string_view rhs = t.substr(eq + 3, t.size() - eq - 4);
    if (rhs.empty())
        return false;
    if (lhs == "ps") {
        st.type = "float";
    } else if (lhs == "p0") {
        st.type = "bool";
    } else {
        if (lhs.size() < 2 || lhs[0] != 'r' || !std::isdigit(static_cast<unsigned char>(lhs[1])))
            return false;
        size_t i = 1;
        while (i < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[i])))
            ++i;
        if (i == lhs.size()) {
            st.type = "float4";
        } else {
            if (lhs[i] != '.')
                return false;
            const size_t n = lhs.size() - i - 1;
            if (n < 1 || n > 4)
                return false;
            for (size_t k = i + 1; k < lhs.size(); ++k)
                if (!std::strchr("xyzw", lhs[k]))
                    return false;
            st.type = n == 1 ? std::string("float") : "float" + std::to_string(n);
        }
    }
    // Anything that is not register arithmetic on constants and inputs keeps its branch.
    const std::string checked = FetchesAllowed() ? WithoutMemoryFetches(rhs) : std::string(rhs);
    static constexpr std::string_view kForbidden[] = {
        "fetch", "Fetch", "Sample", "Load", "getTexture", "getCompTex", "getWeights", "texelToUV", "cubeMap",
        "clip", "discard", "ddx", "ddy", "fwidth", "a0", "aL", "oC", "oPos", "oD", "oT", "oKill", "oPts", "oFog",
        "emExport", "Interlocked", "?", "++", "--", "+=", "-=", "*=", "/=", "{", "}", "[", ";", "return", "break"};
    for (const std::string_view f : kForbidden)
        if (checked.find(f) != std::string::npos)
            return false;
    // No assignment inside the rhs (comparisons only).
    for (size_t k = 0; k < rhs.size(); ++k) {
        if (rhs[k] != '=')
            continue;
        const char before = k ? rhs[k - 1] : 0, after = k + 1 < rhs.size() ? rhs[k + 1] : 0;
        if (!(before == '=' || before == '!' || before == '<' || before == '>' || after == '='))
            return false;
    }
    st.lhs = std::string(lhs);
    st.rhs = std::string(rhs);
    return true;
}

struct Node {
    enum Kind { Raw, Group, PredBlock } kind = Raw;
    std::string line;            // Raw: the line; PredBlock: the `if` line (its indentation is kept)
    std::string openLine, closeLine;  // Group / PredBlock: the brace lines as written
    bool negated = false;
    bool keep = false;           // PredBlock: an `else` follows (a control-flow region): the branch stays
    std::vector<Node> children;
};

// Parses until the closing brace of the current group (consumed) or the end.
inline void ParseLevel(const std::vector<std::string_view> &lines, size_t &i, std::vector<Node> &out, bool inGroup,
                       std::string *closeLine) {
    while (i < lines.size()) {
        const std::string_view t = Trim(lines[i]);
        if (t == "}" && inGroup) {
            if (closeLine)
                *closeLine = std::string(lines[i]);
            ++i;
            return;
        }
        if ((t == "if (p0)" || t == "if (!p0)") && i + 1 < lines.size() && Trim(lines[i + 1]) == "{") {
            Node b;
            b.kind = Node::PredBlock;
            b.negated = t[4] == '!';
            b.line = std::string(lines[i]);
            b.openLine = std::string(lines[i + 1]);
            i += 2;
            ParseLevel(lines, i, b.children, true, &b.closeLine);
            b.keep = i < lines.size() && Trim(lines[i]).substr(0, 4) == "else";
            out.push_back(std::move(b));
            continue;
        }
        if (t == "{") {
            Node g;
            g.kind = Node::Group;
            g.openLine = std::string(lines[i]);
            ++i;
            ParseLevel(lines, i, g.children, true, &g.closeLine);
            out.push_back(std::move(g));
            continue;
        }
        Node r;
        r.line = std::string(lines[i]);
        out.push_back(std::move(r));
        ++i;
    }
}

// Statements in a block made only of statements and convertible blocks; SIZE_MAX if it cannot be converted.
inline size_t ConvertibleStatements(const Node &b) {
    if (b.keep)
        return SIZE_MAX;
    size_t n = 0;
    for (const Node &c : b.children) {
        if (c.kind == Node::PredBlock) {
            const size_t m = ConvertibleStatements(c);
            if (m == SIZE_MAX)
                return SIZE_MAX;
            n += m;
        } else if (c.kind == Node::Raw) {
            Statement st;
            if (!ParseStatement(Trim(c.line), st))
                return SIZE_MAX;
            ++n;
        } else {
            return SIZE_MAX;
        }
        if (n > kMaxStatements)
            return SIZE_MAX;
    }
    return n;
}

inline std::string IndentOf(const std::string &line) {
    return line.substr(0, line.find_first_not_of('\t') == std::string::npos ? line.size()
                                                                              : line.find_first_not_of('\t'));
}

struct Emitter {
    std::string out;
    size_t next = 0;
    size_t blocks = 0;

    // A convertible block, with the predicate of the converted block around it ('outer'; empty at a real branch).
    void Converted(const Node &b, const std::string &outer) {
        const std::string indent = IndentOf(b.line);
        const std::string name = "pb" + std::to_string(++next);
        const std::string pred = b.negated ? "!p0" : "p0";
        out += indent + "{\n";
        out += indent + "\tconst bool " + name + " = " + (outer.empty() ? pred : "and(" + outer + ", " + pred + ")") +
               ";\n";
        for (const Node &c : b.children) {
            if (c.kind == Node::PredBlock) {
                Converted(c, name);
                continue;
            }
            Statement st;
            ParseStatement(Trim(c.line), st);
            // A memory fetch runs in its clamped form (see FetchesAllowed).
            for (size_t k = st.rhs.find("vfetchMemory("); k != std::string::npos;
                 k = st.rhs.find("vfetchMemory(", k + 1)) {
                st.rhs.replace(k, 13, "vfetchMemoryClamped(");
                g_fetches.fetch_add(1, std::memory_order_relaxed);
            }
            out += indent + "\t" + st.lhs + " = select(" + name + ", (" + st.type + ")(" + st.rhs + "), " + st.lhs +
                   ");\n";
        }
        out += indent + "}\n";
        ++blocks;
    }

    void Emit(const std::vector<Node> &nodes) {
        for (const Node &n : nodes) {
            switch (n.kind) {
            case Node::Raw:
                out += n.line;
                out += '\n';
                break;
            case Node::Group:
                out += n.openLine + "\n";
                Emit(n.children);
                out += n.closeLine + "\n";
                break;
            case Node::PredBlock:
                if (ConvertibleStatements(n) != SIZE_MAX) {
                    Converted(n, std::string());
                } else {
                    out += n.line + "\n" + n.openLine + "\n";
                    Emit(n.children);
                    out += n.closeLine + "\n";
                }
                break;
            }
        }
    }
};

// Rewrites the body of `void main(...)` in 'hlsl' (everything after its opening brace line, which the translator has
// not closed yet). Returns the number of blocks converted.
inline size_t Flatten(std::string &hlsl) {
    const size_t main = hlsl.find("\nvoid main(");
    if (main == std::string::npos)
        return 0;
    const size_t open = hlsl.find("\n{\n", main);
    if (open == std::string::npos)
        return 0;
    const size_t bodyStart = open + 3;
    std::string body = hlsl.substr(bodyStart);
    const bool trailingNewline = !body.empty() && body.back() == '\n';
    std::vector<std::string_view> lines;
    size_t at = 0;
    while (at < body.size()) {
        size_t end = body.find('\n', at);
        if (end == std::string::npos)
            end = body.size();
        lines.push_back(std::string_view(body).substr(at, end - at));
        at = end + 1;
    }
    std::vector<Node> nodes;
    size_t i = 0;
    ParseLevel(lines, i, nodes, false, nullptr);
    Emitter e;
    e.Emit(nodes);
    if (!trailingNewline && !e.out.empty() && e.out.back() == '\n')
        e.out.pop_back();
    hlsl.replace(bodyStart, std::string::npos, e.out);
    return e.blocks;
}

}  // namespace banjo_pred_select
