"""Explicit study stages; analysis remains read-only and inference opt-in."""

import argparse
import json
from pathlib import Path

from .analysis import saved_summary
from .config import ContractError, StudyConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Recovery study (explicit stages)")
    parser.add_argument("stage", choices=("generate", "fit", "analyze", "report", "matched-replay"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--saved", type=Path, help="Existing saved-result directory for analysis")
    parser.add_argument("--format", choices=("tv_nuts", "bounded_ols", "smoke", "native"))
    parser.add_argument(
        "--report-method",
        choices=("ols", "advi", "mcmc"),
        nargs="+",
        help="Explicit enabled method subset for native/historical reporting",
    )
    parser.add_argument(
        "--historical", action="store_true", help="Explicit recorded-identity smoke reporting only"
    )
    parser.add_argument(
        "--out", type=Path, help="New study artifact root for generation or fitting"
    )
    parser.add_argument(
        "--method", choices=("ols", "advi", "mcmc"), help="Fit only this enabled method"
    )
    parser.add_argument("--source", type=Path, help="Recorded source corpus for fixed-RV replay")
    parser.add_argument("--paired", type=Path, help="Existing paired fixed-RV/trajectory fixtures")
    parser.add_argument(
        "--worlds", nargs="+", type=int, help="Explicit existing world IDs for deterministic replay"
    )
    args = parser.parse_args()
    config = StudyConfig.load(args.config)
    if args.historical and args.stage != "report":
        parser.error("--historical is only supported for explicit recorded-identity report")
    if args.report_method is not None and args.stage != "report":
        parser.error("--report-method is only supported by report")
    if args.stage == "matched-replay":
        if args.source is None or args.paired is None or args.out is None or not args.worlds:
            parser.error("matched-replay requires --source, --paired, --out, and --worlds")
        from .matched import replay

        try:
            result = replay(config, args.source, args.paired, args.out, worlds=tuple(args.worlds))
        except ContractError as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))
        return 0
    if args.source is not None or args.paired is not None or args.worlds is not None:
        parser.error("--source/--paired/--worlds are only for matched-replay")
    if args.stage in ("generate", "fit"):
        if args.out is None or args.saved is not None or args.format is not None:
            parser.error(f"{args.stage} requires --out, and forbids --saved/--format")
        from .runner import fit, generate

        try:
            result = (
                generate(config, args.out)
                if args.stage == "generate"
                else fit(config, args.out, args.method)
            )
        except ContractError as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))
        return 0
    if args.stage == "report":
        if args.saved is None or args.format is None or args.out is None or args.method is not None:
            parser.error("report requires --saved, --format, --out and forbids --method")
        from .reporting import create_report

        try:
            result = create_report(
                config,
                args.saved,
                args.format,
                args.out,
                historical=args.historical,
                methods=tuple(args.report_method) if args.report_method is not None else None,
            )
        except ContractError as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))
        return 0
    if args.out is not None or args.method is not None or args.historical or args.format == "smoke":
        parser.error("analyze forbids --out/--method/--historical/smoke")
    if args.saved is None or args.format is None:
        parser.error("analyze requires --saved and --format")
    print(
        json.dumps(
            saved_summary(
                args.saved,
                kind=args.format,
                config=config,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
