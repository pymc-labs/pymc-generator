"""Separate, immutable approval of one sealed run; never changes report evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .artifacts import digest
from .config import ContractError
from .reporting import read_report


def _review_path(report: Path) -> Path:
    return report.parent / f"{report.name}-review"


def _id(meta: dict) -> str:
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


def create_review(
    report: Path, interpretation: Path, reviewer: str, approved_run: str, *, out: Path | None = None
) -> dict:
    """Explicit approval bound to exact verified report ID, external to its receipt."""
    report = report.resolve()
    target = (out or _review_path(report)).resolve()
    if target != _review_path(report) or target.exists():
        raise ContractError("review must use a new canonical sibling <report>-review directory")
    evidence = read_report(report)
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer != reviewer.strip():
        raise ContractError("reviewer must be a nonempty explicit name")
    if approved_run != evidence["run_id"]:
        raise ContractError("approved run ID does not match the sealed evidence")
    text = interpretation.read_text()
    if not text.strip() or "pending review" in text.lower():
        raise ContractError("reviewed interpretation must be nonempty and must not be pending")
    if interpretation.resolve().is_relative_to(report) or interpretation.resolve().is_relative_to(
        target
    ):
        raise ContractError("review text must be prepared outside sealed report/review roots")
    meta = {
        "schema": 1,
        "report_run_id": evidence["run_id"],
        "reviewer": reviewer,
        "decision": "approved",
        "report_relative": os.path.relpath(report, target),
        "evidence_sha256": digest(report / "evidence.json"),
        "report_manifest_sha256": digest(report / "manifest.json"),
        "report_receipt_sha256": digest(report / "completion.json"),
        "interpretation_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    meta["review_id"] = _id(meta)
    target.mkdir(parents=True)
    (target / "interpretation.md").write_text(text)
    (target / "review.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": 1,
        "outputs": {name: digest(target / name) for name in ("review.json", "interpretation.md")},
    }
    (target / "review-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return read_review(report, target)


def read_review(report: Path, review: Path) -> dict:
    """Fail closed on wrong run, changed evidence, review body or approval receipt."""
    report, review = report.resolve(), review.resolve()
    if review != _review_path(report):
        raise ContractError("review must be the canonical sibling of its report")
    evidence = read_report(report)
    try:
        meta = json.loads((review / "review.json").read_text())
        receipt = json.loads((review / "review-receipt.json").read_text())
        text = (review / "interpretation.md").read_text()
    except (OSError, ValueError) as exc:
        raise ContractError("missing or invalid reviewed interpretation/receipt") from exc
    if not isinstance(meta, dict) or not isinstance(receipt, dict):
        raise ContractError("invalid reviewed interpretation/receipt tables")
    check = {k: value for k, value in meta.items() if k != "review_id"}
    expected_files = {name: digest(review / name) for name in ("review.json", "interpretation.md")}
    if (
        meta.get("schema") != 1
        or meta.get("review_id") != _id(check)
        or meta.get("report_run_id") != evidence["run_id"]
        or meta.get("decision") != "approved"
        or not isinstance(meta.get("reviewer"), str)
        or not meta["reviewer"].strip()
        or (review / meta.get("report_relative", "missing")).resolve() != report
        or meta.get("evidence_sha256") != digest(report / "evidence.json")
        or meta.get("report_manifest_sha256") != digest(report / "manifest.json")
        or meta.get("report_receipt_sha256") != digest(report / "completion.json")
        or meta.get("interpretation_sha256") != hashlib.sha256(text.encode()).hexdigest()
        or not text.strip()
        or "pending review" in text.lower()
        or receipt != {"schema": 1, "outputs": expected_files}
        or set(review.iterdir())
        != {review / "review.json", review / "interpretation.md", review / "review-receipt.json"}
    ):
        raise ContractError("review identity, approval or sealed output receipt mismatch")
    return {**meta, "interpretation": text}


def attached_review(report: Path) -> dict | None:
    """Return pending when absent; never hide a present but invalid review."""
    review = _review_path(report.resolve())
    return read_review(report, review) if review.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Approve one run-bound interpretation; no data processing"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--interpretation", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--approve-run", required=True, help="Exact run ID displayed in verified report"
    )
    args = parser.parse_args()
    try:
        result = create_review(args.report, args.interpretation, args.reviewer, args.approve_run)
    except ContractError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {key: result[key] for key in ("report_run_id", "reviewer", "decision", "review_id")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
