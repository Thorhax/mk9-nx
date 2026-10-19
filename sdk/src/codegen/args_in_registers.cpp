/**
 * @file        codegen/args_in_registers.cpp
 * @brief       Arguments in registers (see args_in_registers.h)
 *
 * Soundness rules, in short (the argument is written up in the project documentation):
 *  - Par: registers read before written (by the function or by a fast callee) on some path from the entry
 *    arrive by value.
 *  - A call to anything that is not a fast function (indirect call, hooked function, kernel import, an
 *    ineligible function) reads and may write every argument register and lr. The locals that differ from
 *    ctx are stored before it, the ones that are live afterwards are reloaded. Registers that still hold
 *    their ENTRY value at such a call are forwarded through ctx (Fw): callers store them.
 *  - Scratch registers (r0, r11, r12, f0) are not read by unknown callees; they are synchronised only when
 *    live across the call.
 *  - r3 is returned by value when the function may define it itself (Val); everything else a direct caller
 *    reads after the call, and f1/r3 for indirect callers, is exported through ctx at the exits (Ex).
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include "args_in_registers.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <map>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include <fmt/format.h>

namespace rex::codegen {
namespace {

// ------------------------------------------------------------------------------------------ registers
constexpr int kNR = 26;
constexpr uint32_t kAll = (1u << kNR) - 1;
constexpr uint32_t kTemp = (1u << 21) | (1u << 22) | (1u << 23) | (1u << 24);
constexpr uint32_t kFwd = kAll & ~kTemp;  // r3-r10, f1-f13 and lr are forwarded to unknown callees
constexpr uint32_t kR3 = 1u;
constexpr uint32_t kF1 = 1u << 8;

const char* const kRegName[kNR] = {"r3", "r4",  "r5",  "r6",  "r7",  "r8",  "r9", "r10", "f1",
                                   "f2", "f3",  "f4",  "f5",  "f6",  "f7",  "f8", "f9",  "f10",
                                   "f11", "f12", "f13", "r0",  "r11", "r12", "f0", "lr"};

inline bool IsWord(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}
inline bool IsHex(char c) {
  return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F');
}

int RegIndex(std::string_view id) {
  if (id == "lr")
    return 25;
  if (id.size() < 2 || (id[0] != 'r' && id[0] != 'f'))
    return -1;
  int n = 0;
  for (size_t i = 1; i < id.size(); ++i) {
    if (id[i] < '0' || id[i] > '9')
      return -1;
    n = n * 10 + (id[i] - '0');
    if (n > 99)
      return -1;
  }
  if (id.size() > 2 && id[1] == '0')
    return -1;  // leading zero
  if (id[0] == 'r') {
    if (n >= 3 && n <= 10)
      return n - 3;
    if (n == 0)
      return 21;
    if (n == 11)
      return 22;
    if (n == 12)
      return 23;
    return -1;
  }
  if (n >= 1 && n <= 13)
    return 7 + n;
  if (n == 0)
    return 24;
  return -1;
}

std::string RegAccess(int idx) {  // r3.u64 / f1.f64 / lr
  if (idx == 25)
    return "lr";
  return fmt::format("{}.{}", kRegName[idx], kRegName[idx][0] == 'f' ? "f64" : "u64");
}
bool IsFp(int idx) {
  return kRegName[idx][0] == 'f';
}

template <typename F>
void ForEachRef(std::string_view s, F&& fn) {
  size_t pos = 0;
  while ((pos = s.find("ctx.", pos)) != std::string_view::npos) {
    if (pos > 0 && IsWord(s[pos - 1])) {
      pos += 4;
      continue;
    }
    size_t e = pos + 4;
    while (e < s.size() && IsWord(s[e]))
      ++e;
    int idx = RegIndex(s.substr(pos + 4, e - pos - 4));
    if (idx >= 0)
      fn(idx);
    pos = e;
  }
}

bool HasRef(std::string_view s) {
  bool found = false;
  ForEachRef(s, [&](int) { found = true; });
  return found;
}

/// (use, full-def, partial-def) masks of one statement.
void StmtMasks(std::string_view s, uint32_t& use, uint32_t& dfull, uint32_t& dpart) {
  use = dfull = dpart = 0;
  size_t restStart = 0;
  if (s.size() > 4 && s.substr(0, 4) == "ctx.") {
    size_t e = 4;
    while (e < s.size() && IsWord(s[e]))
      ++e;
    int idx = RegIndex(s.substr(4, e - 4));
    if (idx >= 0) {
      size_t p = e;
      auto skipWs = [&](size_t q) {
        while (q < s.size() && (s[q] == ' ' || s[q] == '\t'))
          ++q;
        return q;
      };
      if (idx == 25) {
        size_t q = skipWs(p);
        if (q < s.size() && s[q] == '=' && !(q + 1 < s.size() && s[q + 1] == '=')) {
          dfull = 1u << idx;
          restStart = q + 1;
        }
      } else if (p < s.size() && s[p] == '.') {
        size_t m0 = p + 1;
        size_t m1 = m0;
        while (m1 < s.size() && IsWord(s[m1]))
          ++m1;
        std::string_view member = s.substr(m0, m1 - m0);
        size_t q = skipWs(m1);
        // full definition: ctx.rN.u64 = / s64 = / f64 =
        if ((member == "u64" || member == "s64" || member == "f64") && q < s.size() && s[q] == '=' &&
            !(q + 1 < s.size() && s[q + 1] == '=')) {
          dfull = 1u << idx;
          restStart = q + 1;
        } else if (!member.empty()) {
          // any other width or a compound assignment: read-modify-write
          size_t r = q;
          if (r < s.size() && (s[r] == '-' || s[r] == '+' || s[r] == '|' || s[r] == '&' || s[r] == '^'))
            ++r;
          else if (r + 1 < s.size() && ((s[r] == '<' && s[r + 1] == '<') || (s[r] == '>' && s[r + 1] == '>')))
            r += 2;
          if (r < s.size() && s[r] == '=' && !(r + 1 < s.size() && s[r + 1] == '=')) {
            dpart = 1u << idx;
            use |= dpart;
            restStart = r + 1;
          }
        }
      }
    }
  }
  ForEachRef(s.substr(restStart), [&](int idx) { use |= 1u << idx; });
}

// ------------------------------------------------------------------------------------------ statements
enum Kind : uint8_t {
  kPlain, kLabel, kGoto, kIfGoto, kIfRet, kRet, kIfOpen, kClose, kBlock, kSwitch, kCase, kDefault,
  kStop, kCallF, kCallU
};

std::string_view Trim(std::string_view s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r'))
    ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r'))
    --b;
  return s.substr(a, b - a);
}

bool StartsWith(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.substr(0, p.size()) == p;
}
bool EndsWith(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}

bool IsLabelName(std::string_view s) {  // loc_[0-9A-F]+
  if (!StartsWith(s, "loc_") || s.size() <= 4)
    return false;
  for (size_t i = 4; i < s.size(); ++i)
    if (!IsHex(s[i]))
      return false;
  return true;
}

bool IsDecl(std::string_view s) {
  static const char* kTypes[] = {"PPCRegister", "PPCXERRegister", "PPCCRRegister", "PPCVRegister",
                                 "PPCFPRegister", "uint32_t", "uint64_t", "PPCContext"};
  if (!EndsWith(s, "{};"))
    return false;
  for (const char* t : kTypes) {
    size_t n = std::strlen(t);
    if (StartsWith(s, t) && (s.size() == n || !IsWord(s[n])))
      return true;
  }
  return false;
}

Kind Classify(std::string_view s, std::string_view& target, std::string_view& callName) {
  target = {};
  callName = {};
  if (s == "return;")
    return kRet;
  if (EndsWith(s, ":") && IsLabelName(s.substr(0, s.size() - 1))) {
    target = s.substr(0, s.size() - 1);
    return kLabel;
  }
  if (StartsWith(s, "goto ") && EndsWith(s, ";") && IsLabelName(s.substr(5, s.size() - 6))) {
    target = s.substr(5, s.size() - 6);
    return kGoto;
  }
  if (StartsWith(s, "if (") && EndsWith(s, ";")) {
    size_t g = s.rfind(") goto ");
    if (g != std::string_view::npos && IsLabelName(s.substr(g + 7, s.size() - g - 8))) {
      target = s.substr(g + 7, s.size() - g - 8);
      return kIfGoto;
    }
  }
  if (StartsWith(s, "if (") && EndsWith(s, ") return;"))
    return kIfRet;
  if (StartsWith(s, "if (") && EndsWith(s, ") {"))
    return kIfOpen;
  if (StartsWith(s, "switch (") && EndsWith(s, ") {"))
    return kSwitch;
  if (s == "}")
    return kClose;
  if (s == "{")
    return kBlock;
  if (StartsWith(s, "case ") && EndsWith(s, ":")) {
    std::string_view num = s.substr(5, s.size() - 6);
    size_t i = (!num.empty() && num[0] == '-') ? 1 : 0;
    bool ok = i < num.size();
    for (; i < num.size() && ok; ++i)
      ok = num[i] >= '0' && num[i] <= '9';
    if (ok)
      return kCase;
  }
  if (s == "default:")
    return kDefault;
  if (StartsWith(s, "__builtin_trap()") || StartsWith(s, "REX_FATAL(") || StartsWith(s, "ppc_longjmp("))
    return kStop;
  if (EndsWith(s, "(ctx, base);")) {
    std::string_view name = s.substr(0, s.size() - 12);
    bool ok = !name.empty() && !(name[0] >= '0' && name[0] <= '9');
    for (char c : name)
      ok = ok && IsWord(c);
    if (ok) {
      if (StartsWith(name, "__imp__") && name.size() > 7)
        name = name.substr(7);
      callName = name;
      return kCallF;
    }
  }
  if (s.find("REX_CALL_INDIRECT_FUNC(") != std::string_view::npos)
    return kCallU;
  return kPlain;
}

struct Fn {
  std::string name;
  size_t bodyIndex = 0;
  bool eligible = false;
  int n = 0;  // statements including the synthetic return
  std::vector<uint8_t> kind;
  std::vector<uint32_t> use, dfull, dpart;
  std::vector<int32_t> callee;   // function index for kCallF with a known function, else -1
  std::vector<std::string_view> callName;  // kCallF: called symbol (without __imp__)
  std::vector<uint32_t> succOff; // CSR successor lists, node n = virtual exit
  std::vector<int32_t> succ;
  std::vector<int32_t> rawLine;  // index into the body lines, -1 for the synthetic return
  std::vector<std::string_view> lines;  // body lines (views into the body text)
  size_t firstStmtLine = 0;
};

// ------------------------------------------------------------------------------------------ summaries
struct Site {
  int32_t stmt;
  int32_t callee;  // function index; -1 unknown; -2 exit
  uint32_t ein;
};

struct Summary {
  uint32_t dany = 0;
  bool val = false;
  uint32_t fw = 0;
  uint32_t par = 0;
  uint32_t ex = 0;
  uint32_t dem = 0;
  std::vector<Site> calls;
};

struct Pass {
  const ArgsInRegistersOptions& opt;
  std::vector<Fn> fns;
  std::vector<Summary> S;
  std::unordered_map<std::string, int> index;
  std::vector<uint32_t> legacy;      // per function: TEMP registers its legacy body reads (ineligible only)
  std::vector<uint8_t> direct;       // has at least one direct fast caller
  std::vector<std::vector<int>> callers;
  explicit Pass(const ArgsInRegistersOptions& o) : opt(o) {}

  int Callee(const Fn& f, int i) const { return f.callee[i]; }
  bool IsFast(int g) const { return g >= 0 && fns[g].eligible; }
};

// ------------------------------------------------------------------------------------------ parsing
void SplitLines(std::string_view text, std::vector<std::string_view>& out) {
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t e = text.find('\n', pos);
    if (e == std::string_view::npos) {
      out.push_back(text.substr(pos));
      break;
    }
    out.push_back(text.substr(pos, e - pos));
    pos = e + 1;
  }
}

/// Parses one emitted function (text begins at or before `DEFINE_REX_FUNC(name) {`). Returns false when the
/// text is not a plain function body we can handle.
bool ParseFunction(Pass& P, Fn& f, std::string_view text, bool excluded) {
  std::vector<std::string_view> all;
  SplitLines(text, all);
  // locate the DEFINE line and the closing brace
  size_t def = 0;
  while (def < all.size() && !StartsWith(all[def], "DEFINE_REX_FUNC("))
    ++def;
  if (def == all.size())
    return false;
  size_t close = all.size();
  while (close > def + 1 && all[close - 1] != "}")
    --close;
  if (close <= def + 1)
    return false;
  --close;  // index of the closing brace
  for (size_t i = def + 1; i < close; ++i)
    f.lines.push_back(all[i]);

  struct Raw {
    std::string_view text;
    int32_t line;
  };
  std::vector<Raw> stmts;
  for (size_t i = 0; i < f.lines.size(); ++i) {
    std::string_view s = Trim(f.lines[i]);
    if (s.empty() || StartsWith(s, "//") || StartsWith(s, "REX_FUNC_PROLOGUE"))
      continue;
    size_t c = s.find("//");
    if (c != std::string_view::npos)
      s = Trim(s.substr(0, c));
    if (s.empty())
      continue;
    if (stmts.empty() && IsDecl(s))
      continue;
    stmts.push_back({s, static_cast<int32_t>(i)});
  }
  int n = static_cast<int>(stmts.size());
  std::vector<std::string_view> target(n), callName(n);
  f.kind.assign(n, kPlain);
  bool ineligible = excluded;
  for (int i = 0; i < n; ++i) {
    f.kind[i] = Classify(stmts[i].text, target[i], callName[i]);
    if (f.kind[i] == kPlain) {
      std::string_view s = stmts[i].text;
      bool setjmp = s.find("env") != std::string_view::npos &&
                    (s.find("env = ctx") != std::string_view::npos || s.find("ctx = env") != std::string_view::npos ||
                     StartsWith(s, "env"));
      if (setjmp || s.find("ppc_setjmp") != std::string_view::npos)
        ineligible = true;
      else if (StartsWith(s, "ppc_trap(ctx") && !StartsWith(s, "ppc_trap(ctx, base, 0)"))
        ineligible = true;
      else if (std::count(s.begin(), s.end(), ';') > 1 && HasRef(s))
        ineligible = true;
    }
  }
  // a body that can fall off its end gets a synthetic `return;`
  if (n == 0 || (f.kind[n - 1] != kRet && f.kind[n - 1] != kGoto && f.kind[n - 1] != kStop)) {
    stmts.push_back({std::string_view(), -1});
    f.kind.push_back(kRet);
    target.push_back({});
    callName.push_back({});
    ++n;
  }
  f.n = n;
  std::unordered_map<std::string_view, int> labels;
  for (int i = 0; i < n; ++i)
    if (f.kind[i] == kLabel)
      labels[target[i]] = i;
  // brace matching
  std::vector<int> stack;
  std::vector<int> closeOf(n, -1);
  std::map<int, std::vector<int>> switchCases;
  std::vector<int> curSw;
  for (int i = 0; i < n; ++i) {
    Kind k = static_cast<Kind>(f.kind[i]);
    if (k == kIfOpen || k == kBlock || k == kSwitch) {
      stack.push_back(i);
      if (k == kSwitch) {
        curSw.push_back(i);
        switchCases[i];
      }
    } else if (k == kClose) {
      if (!stack.empty()) {
        int o = stack.back();
        stack.pop_back();
        closeOf[o] = i;
        if (!curSw.empty() && curSw.back() == o)
          curSw.pop_back();
      }
    } else if ((k == kCase || k == kDefault) && !curSw.empty()) {
      switchCases[curSw.back()].push_back(i);
    }
  }
  f.succOff.assign(n + 1, 0);
  f.succ.clear();
  for (int i = 0; i < n; ++i) {
    f.succOff[i] = static_cast<uint32_t>(f.succ.size());
    Kind k = static_cast<Kind>(f.kind[i]);
    int nxt = (i + 1 < n) ? i + 1 : n;
    auto lab = [&](std::string_view t) -> int {
      auto it = labels.find(t);
      return it == labels.end() ? -1 : it->second;
    };
    if (k == kGoto) {
      int l = lab(target[i]);
      if (l >= 0)
        f.succ.push_back(l);
    } else if (k == kIfGoto) {
      f.succ.push_back(nxt);
      int l = lab(target[i]);
      if (l >= 0)
        f.succ.push_back(l);
    } else if (k == kIfRet) {
      f.succ.push_back(nxt);
      f.succ.push_back(n);
    } else if (k == kRet) {
      f.succ.push_back(n);
    } else if (k == kStop) {
    } else if (k == kSwitch) {
      auto& cs = switchCases[i];
      if (cs.empty())
        f.succ.push_back(nxt);
      else
        for (int c : cs)
          f.succ.push_back(c);
    } else if (k == kIfOpen) {
      f.succ.push_back(nxt);
      if (closeOf[i] >= 0)
        f.succ.push_back(closeOf[i] + 1 < n ? closeOf[i] + 1 : n);
    } else {
      f.succ.push_back(nxt);
    }
  }
  f.succOff[n] = static_cast<uint32_t>(f.succ.size());
  f.use.assign(n, 0);
  f.dfull.assign(n, 0);
  f.dpart.assign(n, 0);
  f.callee.assign(n, -1);
  f.callName.assign(callName.begin(), callName.end());
  f.rawLine.assign(n, -1);
  for (int i = 0; i < n; ++i) {
    f.rawLine[i] = stmts[i].line;
    Kind k = static_cast<Kind>(f.kind[i]);
    if (k == kPlain || k == kIfGoto || k == kIfRet || k == kIfOpen || k == kSwitch || k == kStop)
      StmtMasks(stmts[i].text, f.use[i], f.dfull[i], f.dpart[i]);
  }
  f.firstStmtLine = f.lines.size();
  for (int i = 0; i < n; ++i)
    if (f.rawLine[i] >= 0) {
      f.firstStmtLine = static_cast<size_t>(f.rawLine[i]);
      break;
    }
  (void)P;
  f.eligible = !ineligible;
  return true;
}

// ------------------------------------------------------------------------------------------ dataflow
void Phase1(Pass& P) {
  size_t nf = P.fns.size();
  std::vector<std::vector<int>> callees(nf);
  std::vector<uint32_t> own(nf, 0);
  for (size_t fi = 0; fi < nf; ++fi) {
    Fn& f = P.fns[fi];
    if (!f.eligible)
      continue;
    uint32_t o = 0;
    bool unk = false;
    std::vector<int>& cs = callees[fi];
    for (int i = 0; i < f.n; ++i) {
      o |= f.dfull[i] | f.dpart[i];
      Kind k = static_cast<Kind>(f.kind[i]);
      if (k == kCallU)
        unk = true;
      else if (k == kCallF) {
        int g = f.callee[i];
        if (P.IsFast(g)) {
          if (std::find(cs.begin(), cs.end(), g) == cs.end())
            cs.push_back(g);
        } else {
          unk = true;
        }
      }
    }
    Summary& s = P.S[fi];
    s.dany = unk ? kAll : o;
    s.val = (o & kR3) != 0;
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t fi = 0; fi < nf; ++fi) {
      if (!P.fns[fi].eligible)
        continue;
      Summary& s = P.S[fi];
      uint32_t d = s.dany;
      bool v = s.val;
      for (int g : callees[fi]) {
        d |= P.S[g].dany;
        v = v || P.S[g].val;
      }
      if (d != s.dany || v != s.val) {
        s.dany = d;
        s.val = v;
        changed = true;
      }
    }
  }
}

void EntryReach(Pass& P, int fi) {
  Fn& f = P.fns[fi];
  int n = f.n;
  std::vector<uint32_t> in(n + 1, 0);
  std::vector<uint8_t> inq(n + 1, 0);
  in[0] = kAll;
  std::deque<int> work{0};
  inq[0] = 1;
  while (!work.empty()) {
    int i = work.front();
    work.pop_front();
    inq[i] = 0;
    if (i >= n)
      continue;
    Kind k = static_cast<Kind>(f.kind[i]);
    uint32_t kill;
    if (k == kCallU)
      kill = kAll;
    else if (k == kCallF)
      kill = P.IsFast(f.callee[i]) ? P.S[f.callee[i]].dany : kAll;
    else
      kill = f.dfull[i];
    uint32_t out = in[i] & ~kill;
    for (uint32_t e = f.succOff[i]; e < f.succOff[i + 1]; ++e) {
      int j = f.succ[e];
      if (j <= n && (in[j] | out) != in[j]) {
        in[j] |= out;
        if (!inq[j]) {
          inq[j] = 1;
          work.push_back(j);
        }
      }
    }
  }
  Summary& s = P.S[fi];
  s.calls.clear();
  for (int i = 0; i < n; ++i) {
    Kind k = static_cast<Kind>(f.kind[i]);
    if (k == kCallU)
      s.calls.push_back({i, -1, in[i]});
    else if (k == kCallF)
      s.calls.push_back({i, P.IsFast(f.callee[i]) ? f.callee[i] : -1, in[i]});
    else if (k == kRet || k == kIfRet)
      s.calls.push_back({i, -2, in[i]});
  }
}

inline uint32_t FwEff(const Summary& G) {
  return G.fw & ~G.par;
}

struct EvalResult {
  uint32_t newpar = 0, newfw = 0, newld = 0;
  std::vector<std::pair<int, uint32_t>> dem;
  std::vector<uint32_t> cv, H, A;
};

void Evaluate(Pass& P, int fi, EvalResult& r, bool direct) {
  Fn& f = P.fns[fi];
  Summary& s = P.S[fi];
  int n = f.n;
  r.dem.clear();
  std::vector<const Summary*> gs(n, nullptr);
  for (int i = 0; i < n; ++i)
    if (static_cast<Kind>(f.kind[i]) == kCallF && P.IsFast(f.callee[i]))
      gs[i] = &P.S[f.callee[i]];
  bool val = s.val;
  uint32_t ex = s.ex;
  std::vector<uint32_t>& cv = r.cv;
  cv.assign(n + 1, kAll);
  cv[0] = direct ? (kAll & ~s.par) : kAll;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = 0; i < n; ++i) {
      Kind k = static_cast<Kind>(f.kind[i]);
      uint32_t x = cv[i];
      uint32_t out;
      if (k == kCallU)
        out = kAll;
      else if (k == kCallF) {
        const Summary* G = gs[i];
        if (!G)
          out = kAll;
        else {
          out = x | FwEff(*G) | G->dany;
          if (G->val)
            out &= ~kR3;
        }
      } else {
        out = x & ~(f.dfull[i] | f.dpart[i]);
      }
      for (uint32_t e = f.succOff[i]; e < f.succOff[i + 1]; ++e) {
        int j = f.succ[e];
        if (j < n) {
          uint32_t nv = cv[j] & out;
          if (nv != cv[j]) {
            cv[j] = nv;
            changed = true;
          }
        }
      }
    }
  }
  std::vector<uint32_t>& H = r.H;
  std::vector<uint32_t>& A = r.A;
  H.assign(n + 1, 0);
  A.assign(n + 1, 0);
  uint32_t exitVal = (val && direct) ? kR3 : 0;
  uint32_t exitSoft = (val && !direct) ? kR3 : 0;
  changed = true;
  while (changed) {
    changed = false;
    for (int i = n - 1; i >= 0; --i) {
      uint32_t oh = 0, oa = 0;
      for (uint32_t e = f.succOff[i]; e < f.succOff[i + 1]; ++e) {
        oh |= H[f.succ[e]];
        oa |= A[f.succ[e]];
      }
      Kind k = static_cast<Kind>(f.kind[i]);
      uint32_t uh = 0, us = 0, kill = 0;
      if (k == kCallU) {
        us = (kFwd | oa) & ~cv[i];
        kill = kAll;
      } else if (k == kCallF) {
        const Summary* G = gs[i];
        if (!G) {
          uint32_t leg = f.callee[i] >= 0 ? P.legacy[f.callee[i]] : 0;
          us = (kFwd | leg | oa) & ~cv[i];
          kill = kAll;
        } else {
          us = G->par | (FwEff(*G) & ~cv[i]);
          kill = G->dany;
        }
      } else if (k == kRet) {
        uh = exitVal;
        us = (ex | exitSoft) & ~cv[i];
      } else if (k == kIfRet) {
        uh = f.use[i] | exitVal;
        us = (ex | exitSoft) & ~cv[i];
      } else {
        uh = f.use[i];
        kill = f.dfull[i];
      }
      uint32_t nh = uh | (oh & ~kill);
      uint32_t na = uh | us | (oa & ~kill);
      if (nh != H[i] || na != A[i]) {
        H[i] = nh;
        A[i] = na;
        changed = true;
      }
    }
  }
  r.newpar = H[0];
  r.newld = A[0] & ~H[0];
  r.newfw = r.newld;
  for (const Site& st : s.calls) {
    if (st.callee == -1)
      r.newfw |= st.ein & kFwd;
    else if (st.callee == -2)
      r.newfw |= st.ein & ex;
    else
      r.newfw |= st.ein & FwEff(P.S[st.callee]);
  }
  std::map<int, uint32_t> dem;
  for (int i = 0; i < n; ++i) {
    const Summary* G = gs[i];
    if (!G)
      continue;
    uint32_t la = 0;
    for (uint32_t e = f.succOff[i]; e < f.succOff[i + 1]; ++e)
      la |= A[f.succ[e]];
    uint32_t m = la & G->dany;
    if (G->val)
      m &= ~kR3;
    if (m)
      dem[f.callee[i]] |= m;
  }
  for (auto& kv : dem)
    r.dem.push_back(kv);
}

void Phase3(Pass& P, const std::vector<int>& order) {
  size_t nf = P.fns.size();
  P.callers.assign(nf, {});
  std::vector<std::vector<int>> callersSet(nf);
  for (size_t fi = 0; fi < nf; ++fi) {
    const Fn& f = P.fns[fi];
    if (!f.eligible)
      continue;
    for (int i = 0; i < f.n; ++i)
      if (static_cast<Kind>(f.kind[i]) == kCallF && P.IsFast(f.callee[i])) {
        auto& v = callersSet[f.callee[i]];
        if (std::find(v.begin(), v.end(), static_cast<int>(fi)) == v.end())
          v.push_back(static_cast<int>(fi));
      }
  }
  P.direct.assign(nf, 0);
  for (size_t g = 0; g < nf; ++g)
    P.direct[g] = !callersSet[g].empty();
  for (auto& v : callersSet)
    std::sort(v.begin(), v.end(), [&](int a, int b) { return P.fns[a].name < P.fns[b].name; });
  for (size_t fi = 0; fi < nf; ++fi) {
    if (!P.fns[fi].eligible)
      continue;
    EntryReach(P, static_cast<int>(fi));
    Summary& s = P.S[fi];
    s.dem = 0;
    s.ex = (kR3 | kF1) & s.dany & ~(s.val ? kR3 : 0u);
  }
  std::deque<int> work(order.begin(), order.end());
  std::vector<uint8_t> inq(nf, 0);
  for (int i : order)
    inq[i] = 1;
  EvalResult r;
  while (!work.empty()) {
    int fi = work.front();
    work.pop_front();
    inq[fi] = 0;
    Summary& s = P.S[fi];
    Evaluate(P, fi, r, P.direct[fi] != 0);
    bool chUp = false;
    if (r.newpar & ~s.par) {
      s.par |= r.newpar;
      chUp = true;
    }
    if (r.newfw & ~s.fw) {
      s.fw |= r.newfw;
      chUp = true;
    }
    for (auto& [g, m] : r.dem) {
      Summary& G = P.S[g];
      if (m & ~G.dem) {
        G.dem |= m;
        uint32_t nex = (G.dem | kR3 | kF1) & G.dany & ~(G.val ? kR3 : 0u);
        if (nex != G.ex) {
          G.ex = nex;
          if (!inq[g]) {
            inq[g] = 1;
            work.push_back(g);
          }
        }
      }
    }
    if (chUp) {
      for (int c : callersSet[fi])
        if (!inq[c]) {
          inq[c] = 1;
          work.push_back(c);
        }
      if (!inq[fi]) {
        inq[fi] = 1;
        work.push_back(fi);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------ emitter
std::vector<std::string> RegNames(uint32_t mask) {
  std::vector<std::string> out;
  for (int i = 0; i < kNR; ++i)
    if (mask >> i & 1)
      out.push_back(kRegName[i]);
  return out;
}

std::string Prototype(const std::string& name, const Summary& G) {
  std::string ps;
  for (int i = 0; i < kNR; ++i)
    if (G.par >> i & 1)
      ps += fmt::format(", {} p_{}", IsFp(i) ? "double" : "uint64_t", kRegName[i]);
  return fmt::format("extern \"C\" {} __fast_{}(PPCContext& __restrict ctx, uint8_t* base{})",
                     G.val ? "uint64_t" : "void", name, ps);
}

void StoreLines(uint32_t mask, std::string_view ind, std::vector<std::string>& out) {
  for (int i = 0; i < kNR; ++i)
    if (mask >> i & 1)
      out.push_back(fmt::format("{}ctx.{} = {};", ind, RegAccess(i), RegAccess(i)));
}
void ReloadLines(uint32_t mask, std::string_view ind, std::vector<std::string>& out) {
  for (int i = 0; i < kNR; ++i)
    if (mask >> i & 1)
      out.push_back(fmt::format("{}{} = ctx.{};", ind, RegAccess(i), RegAccess(i)));
}

/// Replaces ctx.<tracked register> by the local of the same name in one source line.
std::string ReplaceRefs(std::string_view line) {
  std::string out;
  out.reserve(line.size());
  size_t pos = 0;
  while (pos < line.size()) {
    size_t p = line.find("ctx.", pos);
    if (p == std::string_view::npos) {
      out.append(line.substr(pos));
      break;
    }
    if (p > 0 && IsWord(line[p - 1])) {
      out.append(line.substr(pos, p + 4 - pos));
      pos = p + 4;
      continue;
    }
    size_t e = p + 4;
    while (e < line.size() && IsWord(line[e]))
      ++e;
    int idx = RegIndex(line.substr(p + 4, e - p - 4));
    out.append(line.substr(pos, p - pos));
    if (idx >= 0)
      out.append(line.substr(p + 4, e - p - 4));  // drop the `ctx.`
    else
      out.append(line.substr(p, e - p));
    pos = e;
  }
  return out;
}

std::string Join(const std::vector<std::string>& lines) {
  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) {
    out += lines[i];
    out += '\n';
  }
  return out;
}

struct Emitted {
  std::string text;
  std::vector<std::string> protos;
};

}  // namespace

// ------------------------------------------------------------------------------------------ public entry
namespace {

Emitted EmitFunction(Pass& P, int fi, std::string_view prefixLines, bool direct, ArgsInRegistersStats* stats) {
  Fn& f = P.fns[fi];
  Summary& s = P.S[fi];
  EvalResult r;
  Evaluate(P, fi, r, direct);
  if ((r.newpar & ~s.par) || (r.newfw & ~s.fw))
    throw std::runtime_error("args_in_registers: summaries of " + f.name + " are not a fixpoint");
  const std::vector<uint32_t>& cv = r.cv;
  const std::vector<uint32_t>& A = r.A;
  int n = f.n;
  uint32_t ld = r.newld;
  uint32_t touched = s.par | ld;
  for (int i = 0; i < n; ++i)
    touched |= f.use[i] | f.dfull[i] | f.dpart[i];
  bool exitVal = s.val;

  std::vector<std::string> outHead, outBody;
  Emitted em;
  auto exitCode = [&](int i, std::string_view ind, std::vector<std::string>& out) {
    uint32_t st = (s.ex | ((exitVal && !direct) ? kR3 : 0u)) & ~cv[i];
    touched |= st;
    if (exitVal && direct)
      touched |= kR3;
    StoreLines(st, ind, out);
    out.push_back(std::string(ind) + ((exitVal && direct) ? "return r3.u64;" : "return;"));
  };
  std::vector<int> rawToStmt(f.lines.size(), -1);
  for (int i = 0; i < n; ++i)
    if (f.rawLine[i] >= 0)
      rawToStmt[f.rawLine[i]] = i;
  for (size_t rl = 0; rl < f.lines.size(); ++rl) {
    std::string_view line = f.lines[rl];
    if (rl < f.firstStmtLine) {
      outHead.push_back(std::string(line));
      continue;
    }
    int i = rawToStmt[rl];
    if (i < 0) {
      outBody.push_back(std::string(line));
      continue;
    }
    Kind k = static_cast<Kind>(f.kind[i]);
    size_t indEnd = 0;
    while (indEnd < line.size() && (line[indEnd] == '\t' || line[indEnd] == ' '))
      ++indEnd;
    std::string ind(line.substr(0, indEnd));
    if (k == kRet) {
      exitCode(i, ind, outBody);
    } else if (k == kIfRet) {
      std::string_view t = Trim(line);
      std::string cond(t.substr(4, t.size() - 4 - 9));  // between "if (" and ") return;"
      cond = ReplaceRefs(cond);
      outBody.push_back(fmt::format("{}if ({}) {{", ind, cond));
      exitCode(i, ind + "\t", outBody);
      outBody.push_back(ind + "}");
    } else if (k == kCallU || (k == kCallF && !P.IsFast(f.callee[i]))) {
      uint32_t la = 0;
      for (uint32_t e = f.succOff[i]; e < f.succOff[i + 1]; ++e)
        la |= A[f.succ[e]];
      uint32_t leg = (k == kCallF && f.callee[i] >= 0) ? P.legacy[f.callee[i]] : 0;
      uint32_t stores = (kFwd | leg | la) & ~cv[i];
      touched |= stores | la;
      if (stats)
        ++stats->unknownSites;
      StoreLines(stores, ind, outBody);
      outBody.push_back(std::string(line));
      ReloadLines(la, ind, outBody);
    } else if (k == kCallF) {
      const Summary& G = P.S[f.callee[i]];
      const std::string& gname = P.fns[f.callee[i]].name;
      uint32_t stores = FwEff(G) & ~cv[i];
      uint32_t la = 0;
      for (uint32_t e = f.succOff[i]; e < f.succOff[i + 1]; ++e)
        la |= A[f.succ[e]];
      uint32_t rl2 = la & G.dany & ~(G.val ? kR3 : 0u);
      if (rl2 & ~G.ex)
        throw std::runtime_error("args_in_registers: reload after call to " + gname + " not exported");
      touched |= stores | rl2 | G.par;
      std::string call = fmt::format("__fast_{}(ctx, base", gname);
      for (int q = 0; q < kNR; ++q)
        if (G.par >> q & 1) {
          call += ", ";
          call += RegAccess(q);
        }
      call += ")";
      if (stats) {
        ++stats->fastSites;
        for (int q = 0; q < kNR; ++q)
          stats->argsByValue += (G.par >> q & 1);
      }
      StoreLines(stores, ind, outBody);
      if (P.opt.poison) {
        uint32_t pm = G.dany & ~FwEff(G);
        outBody.push_back(fmt::format("{}ME_ARGS_POISON(0x{:X});", ind, pm));
      }
      if (G.val && (la & kR3)) {
        touched |= kR3;
        outBody.push_back(fmt::format("{}r3.u64 = {};", ind, call));
      } else {
        outBody.push_back(fmt::format("{}{};", ind, call));
      }
      ReloadLines(rl2, ind, outBody);
    } else {
      outBody.push_back(ReplaceRefs(line));
    }
  }
  if (f.rawLine[n - 1] < 0)
    exitCode(n - 1, "\t", outBody);

  // lint: no statement may still address a tracked register through ctx except the sync copies
  for (const std::string& ln : outBody) {
    if (!HasRef(ln))
      continue;
    // sync copies: ctx.X = X;  or  X = ctx.X;
    std::string_view t = Trim(ln);
    bool ok = false;
    if (StartsWith(t, "ctx.") && EndsWith(t, ";")) {
      size_t eq = t.find(" = ");
      if (eq != std::string_view::npos) {
        std::string_view l = t.substr(4, eq - 4);
        std::string_view rr = t.substr(eq + 3, t.size() - eq - 4);
        auto root = [](std::string_view x) { return x.substr(0, x.find('.')); };
        ok = root(l) == root(rr) && rr.find("ctx.") == std::string_view::npos;
      }
    } else if (EndsWith(t, ";")) {
      size_t eq = t.find(" = ctx.");
      if (eq != std::string_view::npos) {
        std::string_view l = t.substr(0, eq);
        std::string_view rr = t.substr(eq + 7, t.size() - eq - 8);
        auto root = [](std::string_view x) { return x.substr(0, x.find('.')); };
        ok = root(l) == root(rr);
      }
    }
    if (!ok)
      throw std::runtime_error("args_in_registers: " + f.name + ": statement still uses ctx: " + ln);
  }

  std::vector<std::string> decls;
  for (int i = 0; i < kNR; ++i)
    if (touched >> i & 1)
      decls.push_back(i == 25 ? "\tuint64_t lr{};" : fmt::format("\tPPCRegister {}{{}};", kRegName[i]));
  std::vector<std::string> init;
  uint32_t paramRegs = direct ? s.par : 0;
  if (!direct)
    ld |= s.par;
  for (int i = 0; i < kNR; ++i)
    if (paramRegs >> i & 1)
      init.push_back(fmt::format("\t{} = p_{};", RegAccess(i), kRegName[i]));
  for (int i = 0; i < kNR; ++i)
    if ((ld & ~paramRegs) >> i & 1)
      init.push_back(fmt::format("\t{} = ctx.{};", RegAccess(i), RegAccess(i)));
  if (stats && !direct)
    ++stats->combined;

  auto instrument = [&](std::vector<std::string>& head, int kindId) {
    if (P.opt.instrument)
      head.insert(head.begin() + (head.empty() ? 0 : 1), fmt::format("\tME_ARGS_COUNT(\"{}\", {});", f.name, kindId));
  };
  std::string result(prefixLines);
  if (!direct) {
    std::vector<std::string> head = outHead;
    instrument(head, 0);
    result += fmt::format("DEFINE_REX_FUNC({}) {{\n", f.name);
    result += Join(head) + Join(decls) + Join(init) + Join(outBody) + "}\n";
    em.text = std::move(result);
    return em;
  }
  std::string wargs;
  for (int i = 0; i < kNR; ++i)
    if (s.par >> i & 1)
      wargs += ", ctx." + RegAccess(i);
  std::vector<std::string> wrapper = {fmt::format("DEFINE_REX_FUNC({}) {{", f.name), "\tREX_FUNC_PROLOGUE();"};
  if (P.opt.instrument)
    wrapper.push_back(fmt::format("\tME_ARGS_COUNT(\"{}\", 1);", f.name));
  if (s.val)
    wrapper.push_back(fmt::format("\tctx.r3.u64 = __fast_{}(ctx, base{});", f.name, wargs));
  else
    wrapper.push_back(fmt::format("\t__fast_{}(ctx, base{});", f.name, wargs));
  wrapper.push_back("}");
  std::vector<std::string> head = outHead;
  instrument(head, 2);
  // Compiler barrier: without it GCC (guest loads are not volatile on the Switch) infers that a fast function which
  // only reads guest memory is `pure` and hoists / merges calls to it out of polling loops (the old ABI wrote ctx,
  // which prevented that).
  head.insert(head.begin() + (head.empty() ? 0 : 1), "\tasm volatile(\"\" ::: \"memory\");");
  result += Join(wrapper) + "\n" + Prototype(f.name, s) + " {\n" + Join(head) + Join(decls) + Join(init) +
            Join(outBody) + "}\n";
  em.text = std::move(result);
  return em;
}

}  // namespace

void ApplyArgsInRegisters(const std::vector<std::string>& names, std::vector<std::string>& bodies,
                          std::vector<std::vector<std::string>>& declarations,
                          const ArgsInRegistersOptions& options, ArgsInRegistersStats* stats) {
  Pass P(options);
  size_t total = bodies.size();
  declarations.assign(total, {});
  // pass 1: parse
  std::vector<int> fnOfBody(total, -1);
  std::vector<std::string> prefix(total);
  std::vector<std::vector<std::string>> callNames(total);
  P.fns.reserve(total);
  for (size_t b = 0; b < total; ++b) {
    const std::string& text = bodies[b];
    if (text.empty() || text.find("DEFINE_REX_FUNC(") == std::string::npos)
      continue;
    Fn f;
    f.name = names[b];
    f.bodyIndex = b;
    bool excluded = options.excluded.count(f.name) != 0;
    if (!ParseFunction(P, f, text, excluded))
      continue;
    // the prefix is everything before the DEFINE line (stubs, extern declarations of mid-asm hooks)
    size_t d = text.find("DEFINE_REX_FUNC(");
    prefix[b] = text.substr(0, d);
    fnOfBody[b] = static_cast<int>(P.fns.size());
    P.index[f.name] = static_cast<int>(P.fns.size());
    P.fns.push_back(std::move(f));
  }
  // resolve callee indices (second parse of the call names is cheap: names come from the statement lines)
  P.S.assign(P.fns.size(), Summary());
  P.legacy.assign(P.fns.size(), 0);
  for (size_t fi = 0; fi < P.fns.size(); ++fi) {
    Fn& f = P.fns[fi];
    for (int i = 0; i < f.n; ++i) {
      if (static_cast<Kind>(f.kind[i]) != kCallF)
        continue;
      std::string_view cn = f.callName[i];
      auto it = P.index.find(std::string(cn));
      f.callee[i] = it == P.index.end() ? -1 : it->second;
    }
  }
  // ineligible functions: TEMP registers their legacy body reads
  for (size_t fi = 0; fi < P.fns.size(); ++fi)
    if (!P.fns[fi].eligible) {
      uint32_t m = 0;
      for (int i = 0; i < P.fns[fi].n; ++i)
        m |= P.fns[fi].use[i];
      P.legacy[fi] = m & kTemp;
    }
  Phase1(P);
  // phase 3 in name order (same as the reference implementation)
  std::vector<int> order;
  for (size_t fi = 0; fi < P.fns.size(); ++fi)
    if (P.fns[fi].eligible)
      order.push_back(static_cast<int>(fi));
  std::sort(order.begin(), order.end(), [&](int a, int b) { return P.fns[a].name < P.fns[b].name; });
  Phase3(P, order);
  if (stats) {
    stats->functions = P.fns.size();
    stats->eligible = order.size();
  }
  // emission
  std::vector<std::unordered_set<std::string>> protoSeen(total);
  for (size_t fi = 0; fi < P.fns.size(); ++fi) {
    Fn& f = P.fns[fi];
    if (!f.eligible)
      continue;
    size_t b = f.bodyIndex;
    bool direct = P.direct[fi] != 0;
    Emitted em = EmitFunction(P, static_cast<int>(fi), prefix[b], direct, stats);
    auto need = [&](int g) {
      if (g < 0 || !P.fns[g].eligible || !P.direct[g])
        return;
      if (protoSeen[b].insert(P.fns[g].name).second)
        declarations[b].push_back(Prototype(P.fns[g].name, P.S[g]) + ";");
    };
    if (direct)
      need(static_cast<int>(fi));
    for (int i = 0; i < f.n; ++i)
      if (static_cast<Kind>(f.kind[i]) == kCallF && P.IsFast(f.callee[i]))
        need(f.callee[i]);
    // keep the blank line that separates functions
    std::string text = std::move(em.text);
    text += "\n";
    bodies[b] = std::move(text);
  }
}

}  // namespace rex::codegen
