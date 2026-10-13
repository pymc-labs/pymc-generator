"""Fail-closed aggregate reservation ledger for one local invocation.

A crash consumes the full reservation; there is no automatic retry. The ledger
path is fixed for this work envelope and must not be deleted/reset between runs.
Only one holder can launch a benchmark. The 60-second initial charge and 16-MiB
initial retention reservation cover the already published preflight conservatively.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = Path("/tmp/pymc-generator-comparison-budget.json")
LIMIT = 7200
OUTPUT_LIMIT = 2 * 1024**3


def write_ledger(value):
    temp = LEDGER.with_suffix(".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(LEDGER)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True, type=int)
    p.add_argument("--repeat", required=True, type=int)
    p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args()
    with LEDGER.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = (
            json.loads(LEDGER.read_text())
            if LEDGER.exists()
            else {"charged_s": 60, "retained_bytes": 16 * 1024**2, "invocations": []}
        )
        remaining = LIMIT - ledger["charged_s"]
        free = OUTPUT_LIMIT - ledger["retained_bytes"] - 65536
        if remaining < 30 or free < 1024**2:
            raise RuntimeError("approved aggregate envelope exhausted")
        reservation = min(900, remaining)
        dest = args.destination.resolve()
        dest.mkdir()
        entry = {
            "workload": args.workload,
            "repeat": args.repeat,
            "destination": str(dest),
            "reserved_s": reservation,
            "status": "reserved",
        }
        ledger["charged_s"] += reservation
        ledger["invocations"].append(entry)
        # Reserve remaining output capacity as well: a supervisor crash must not
        # let a future process forget unpublished outputs. Refund after completion.
        ledger["retained_bytes"] += free + 65536
        write_ledger(ledger)
        env = {
            **os.environ,
            "ROOT": str(ROOT),
            "DEST": str(dest),
            "SCRATCH_BYTES": str(free),
            "INVOCATION_SECONDS": str(reservation - 15),
            "WORKLOAD": str(args.workload),
            "REPEAT": str(args.repeat),
        }
        start = time.monotonic()
        result = subprocess.run(["bash", str(ROOT / "benchmarks/comparison_envelope.sh")], env=env)
        elapsed = math.ceil(time.monotonic() - start)
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        ledger["charged_s"] += elapsed - reservation
        ledger["retained_bytes"] += size - free
        entry.update(
            status="passed" if result.returncode == 0 else "failed",
            exit_code=result.returncode,
            charged_s=elapsed,
            retained_bytes=size,
        )
        write_ledger(ledger)
        (dest / "budget-after.json").write_text(json.dumps(ledger, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
