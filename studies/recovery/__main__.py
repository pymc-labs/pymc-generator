"""Explicit generation, fitting, native analysis and static reporting commands."""

import argparse
import json
from pathlib import Path

from .config import ContractError, StudyConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Current native recovery pipeline (no legacy imports)"
    )
    parser.add_argument("stage", choices=("generate", "fit", "analyze", "report"))
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="TOML study config with config-relative full prior JSON",
    )
    parser.add_argument("--out", type=Path, help="Generation/fit root or new report root")
    parser.add_argument("--saved", type=Path, help="Completed native generation/fit root")
    parser.add_argument(
        "--method", choices=("ols", "advi", "mcmc"), help="Fit only this enabled method"
    )
    parser.add_argument(
        "--report-method",
        choices=("ols", "advi", "mcmc"),
        nargs="+",
        help="Select complete already-fitted methods for report or analyze",
    )
    args = parser.parse_args()
    try:
        config = StudyConfig.load(args.config)
        if args.stage in ("generate", "fit"):
            if (
                args.out is None
                or args.saved is not None
                or args.report_method is not None
                or (args.stage == "generate" and args.method is not None)
            ):
                parser.error(
                    "generate/fit require --out, and forbid --saved/--report-method (generate also forbids --method)"
                )
            from .runner import fit, generate

            result = (
                generate(config, args.out)
                if args.stage == "generate"
                else fit(config, args.out, args.method)
            )
        elif args.stage == "analyze":
            if args.saved is None or args.out is not None or args.method is not None:
                parser.error("analyze requires --saved and forbids --out/--method")
            from .reporting import analyze

            result = analyze(
                config, args.saved, tuple(args.report_method) if args.report_method else None
            )
        else:
            if args.saved is None or args.out is None or args.method is not None:
                parser.error("report requires --saved and new --out, forbids --method")
            from .reporting import create_report

            result = create_report(
                config,
                args.saved,
                args.out,
                methods=tuple(args.report_method) if args.report_method else None,
            )
    except ContractError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
