#!/usr/bin/env python3
"""Discover and classify runtime invalid-function failures from one or more logs.

All unique observed targets are classified in one find_gaps.py scan. Discovery is
read-only by default. Applying remains deliberately conservative: --apply may
append exactly one proven target, selected explicitly when several are eligible.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

FATAL_RE = re.compile(
    r"Call to invalid or unregistered function at guest address (0x[0-9A-Fa-f]+)"
)


@dataclass(frozen=True)
class GapEvidence:
    address: str
    gap_size: int
    gap_start: str
    gap_end: str
    previous: str
    following: str

    @property
    def offset(self) -> int:
        return int(self.address, 16) - int(self.gap_start, 16)

    def summary(self) -> str:
        return (
            f"+0x{self.offset:X} in {self.gap_size}-byte gap "
            f"{self.gap_start}-{self.gap_end} "
            f"(after {self.previous}, before {self.following})"
        )


def canonical_address(value: str) -> str:
    return f"0x{int(value, 16):08X}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        action="append",
        required=True,
        help="ReXGlue runtime log; repeat to batch logs into one discovery pass",
    )
    parser.add_argument("--generated", type=Path, default=ROOT / "app" / "generated" / "default")
    parser.add_argument("--overrides", type=Path, default=ROOT / "app" / "overrides.toml")
    parser.add_argument(
        "--find-gaps",
        type=Path,
        default=Path(__file__).resolve().parent / "find_gaps.py",
        help="path to tools/find_gaps.py",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "out" / "gaps" / "runtime-report.txt",
        help="combined find_gaps.py evidence report",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "out" / "gaps" / "runtime-candidates.toml",
        help="unreviewed find_gaps.py gap-start inventory",
    )
    parser.add_argument(
        "--address",
        help="with --apply, select one observed eligible target from a batch",
    )
    parser.add_argument("--apply", action="store_true", help="append one proven entry")
    return parser.parse_args()


def die(message: str) -> None:
    print(f"refused: {message}", file=sys.stderr)
    raise SystemExit(2)


def observed_targets(logs: list[Path]) -> tuple[list[str], int]:
    ordered: list[str] = []
    seen: set[str] = set()
    occurrences = 0
    for log in logs:
        for raw_address in FATAL_RE.findall(log.read_text(errors="replace")):
            occurrences += 1
            address = canonical_address(raw_address)
            if address not in seen:
                seen.add(address)
                ordered.append(address)
    return ordered, occurrences


def override_addresses(overrides_text: str) -> set[str]:
    return {
        canonical_address(match)
        for match in re.findall(
            r'^"(0x[0-9A-Fa-f]+)"\s*=', overrides_text, re.MULTILINE
        )
    }


def classify_report(report_text: str, address: str) -> tuple[GapEvidence | None, str]:
    section = re.search(
        rf"{re.escape(address)}\s+function start:\s+(\S+)(.*?)(?:\n\s*\n|\Z)",
        report_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not section:
        return None, "find_gaps.py did not classify the address"
    if section.group(1).upper() != "NO":
        return None, "already a discovered function start"

    gap = re.search(
        r"falls in a gap of (\d+) bytes:\s*(0x[0-9A-Fa-f]+)\s*-\s*(0x[0-9A-Fa-f]+)",
        section.group(2),
        re.IGNORECASE,
    )
    neighbors = re.search(
        r"after function (0x[0-9A-Fa-f]+), before (0x[0-9A-Fa-f]+)",
        section.group(2),
        re.IGNORECASE,
    )
    if not gap or not neighbors:
        return None, "not proven to lie in an unclaimed code gap"

    gap_size, gap_start, gap_end = gap.groups()
    previous, following = neighbors.groups()
    return (
        GapEvidence(
            address=address,
            gap_size=int(gap_size),
            gap_start=canonical_address(gap_start),
            gap_end=canonical_address(gap_end),
            previous=canonical_address(previous),
            following=canonical_address(following),
        ),
        "",
    )


def select_apply_candidate(
    eligible: list[GapEvidence], requested: str | None
) -> GapEvidence:
    if requested is not None:
        try:
            selected_address = canonical_address(requested)
        except ValueError:
            die(f"invalid --address value: {requested}")
        for evidence in eligible:
            if evidence.address == selected_address:
                return evidence
        die(f"{selected_address} is not an observed eligible candidate in this pass")

    if not eligible:
        die("no observed target is eligible to apply")
    if len(eligible) != 1:
        choices = ", ".join(evidence.address for evidence in eligible)
        die(f"multiple eligible targets ({choices}); select exactly one with --address")
    return eligible[0]


def main() -> int:
    args = parse_args()
    addresses, occurrences = observed_targets(args.log)
    if not addresses:
        die("runtime logs have no invalid/unregistered-function fatal")

    overrides_text = args.overrides.read_text()
    existing = override_addresses(overrides_text)
    command = [
        sys.executable,
        str(args.find_gaps),
        "--gen",
        str(args.generated),
        "--min",
        "8",
        "--check",
        *addresses,
        "--report",
        str(args.report),
        "--toml",
        str(args.candidates),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        die(f"find_gaps.py exited with {result.returncode}{suffix}")

    report_text = args.report.read_text(errors="replace")
    eligible: list[GapEvidence] = []
    print(
        f"discovery: {occurrences} occurrence(s), {len(addresses)} unique target(s), "
        f"{len(args.log)} log(s), one find_gaps.py scan"
    )
    for address in addresses:
        if address in existing:
            print(f"existing  {address}: already present in {args.overrides}")
            continue
        evidence, rejection = classify_report(report_text, address)
        if evidence is None:
            print(f"rejected  {address}: {rejection}")
            continue
        eligible.append(evidence)
        print(f"candidate {address}: {evidence.summary()}")

    print(
        f"summary: {len(eligible)} eligible, "
        f"{len(addresses) - len(eligible)} skipped/rejected"
    )
    print(f"evidence: {args.report}")

    if not args.apply:
        print("read-only: use --apply for one candidate; add --address when several qualify")
        return 0

    selected = select_apply_candidate(eligible, args.address)
    location = (
        "Exact start" if selected.offset == 0 else f"Observed entry +0x{selected.offset:X} inside"
    )
    addition = (
        "\n# Runtime auto-resolve candidate, accepted from an observed indirect-call "
        "failure.\n"
        f"# {location} the {selected.gap_size}-byte gap "
        f"{selected.gap_start}-{selected.gap_end}; only this address is declared.\n"
        f'"{selected.address}" = {{ }}\n'
    )
    args.overrides.write_text(overrides_text.rstrip() + "\n" + addition)
    print(f"applied one entry to {args.overrides}: {selected.address}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
