"""Execute a sealed report notebook read-only; write the executed copy OUTSIDE its evidence root."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .config import ContractError
from .reporting import read_report


def execute_report(report_dir: Path, out: Path) -> dict:
    report_dir, out = report_dir.resolve(), out.resolve()
    if out.suffix != ".ipynb" or out.exists() or out.is_relative_to(report_dir):
        raise ContractError("executed notebook must be a new .ipynb outside the sealed report")
    evidence = read_report(report_dir)
    inputs = [(report_dir / path).resolve() for path in evidence["input_paths"].values()]
    if any(out == path or out.is_relative_to(path.parent) for path in inputs):
        raise ContractError("executed notebook cannot overwrite original evidence")
    book = nbformat.read(report_dir / "report.ipynb", as_version=4)
    executed = NotebookClient(
        book,
        timeout=120,
        kernel_name="recovery-pipeline",
        resources={"metadata": {"path": str(report_dir)}},
    ).execute()
    if read_report(report_dir)["run_id"] != evidence["run_id"]:
        raise ContractError("report changed during Run All")
    outputs = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    ]
    html = sum("text/html" in output.get("data", {}) for output in outputs)
    images = sum("image/png" in output.get("data", {}) for output in outputs)
    if (
        any(output.output_type == "error" for output in outputs)
        or "widgets" in executed.metadata
        or html < 5
        or images < 1
    ):
        raise ContractError("Run All did not persist ordinary static tables and PNG figures")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, out)
    return {
        "run_id": evidence["run_id"],
        "executed_notebook": str(out),
        "code_cells": sum(
            cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code"
        ),
        "html_tables": html,
        "png_figures": images,
    }


def main() -> int:
    import json

    parser = argparse.ArgumentParser(
        description="Run All only from sealed recovery report artifacts"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute_report(args.report, args.out)
    except ContractError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
