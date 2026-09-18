#!/usr/bin/env python3
"""Export NVSHMEM module state from whole-program PTX."""

from __future__ import annotations

import argparse
import re

from pathlib import Path

REQUIRED_SYMBOLS = (
    "nvshmemi_device_state_d",
    "nvshmemi_ibgda_device_state_d",
    "nvshmemi_device_lib_version_d",
    "liger_cute_sm90_nonrdc_build_fingerprint",
    "liger_cute_sm90_nonrdc_transport_mode",
    "_ZN10liger_cute6detail12g_dest_tableE",
    "_ZN10liger_cute6detail12g_rank_tableE",
)


def export_symbols(ptx: str) -> str:
    if ".target sm_90a" not in ptx:
        raise ValueError("non-RDC MoE PTX must target sm_90a")
    if "setmaxnreg.dec.sync.aligned.u32" not in ptx or "setmaxnreg.inc.sync.aligned.u32" not in ptx:
        raise ValueError("non-RDC MoE PTX is missing the SETMAXNREG build probe")

    lines = ptx.splitlines(keepends=True)
    for symbol in REQUIRED_SYMBOLS:
        pattern = re.compile(rf"^\.const\b.*\b{re.escape(symbol)}(?:\[|;|\s*=)")
        matches = [index for index, line in enumerate(lines) if pattern.search(line)]
        if len(matches) != 1:
            raise ValueError(f"expected one PTX declaration for {symbol}, found {len(matches)}")
        index = matches[0]
        lines[index] = f".visible {lines[index]}"
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output PTX paths must differ")
    args.output.write_text(export_symbols(args.input.read_text()))


if __name__ == "__main__":
    main()
