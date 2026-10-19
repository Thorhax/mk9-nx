#pragma once

namespace me::native {
// Prefer an identity-proven loaded program over a legacy object candidate.
// When both are proven, permit the caller's existing container/DEF choice;
// this decision alone does not establish container ABI compatibility.
constexpr bool ShouldReplaceVertexCandidate(bool has_current, bool current_proven,
                                            bool has_candidate, bool candidate_proven,
                                            bool legacy_allowed) {
  if (!has_candidate) return false;
  if (candidate_proven) return true;
  if (has_current && current_proven) return false;
  return legacy_allowed;
}
}  // namespace me::native
