"""``prior-generator`` command-line interface.

Generates the five-scenario inspection bundles (the human-audit layout) —
one numbered folder per scenario with CSVs, a plain-text description, the
DAG as ``.dot`` + ``.png``, and diagnostic figures.

Usage::

    prior-generator [--out inspection-datasets] [--seed 20260712] [--t 104]
                    [--require-path-to-y] [--no-plots]
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prior-generator",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default="inspection-datasets", help="output root directory")
    ap.add_argument("--seed", type=int, default=20260712, help="base seed (scenario idx adds idx)")
    ap.add_argument("--t", type=int, default=104, help="weeks per world")
    ap.add_argument(
        "--require-path-to-y",
        action="store_true",
        help=(
            "force full connectivity in EVERY scenario (overrides connect_all; "
            "removes the deliberate isolated-null traps from channel_halo/kitchen_sink)"
        ),
    )
    ap.add_argument("--no-plots", action="store_true", help="skip PNG rendering (text/CSV only)")
    args = ap.parse_args(argv)

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")

    from .bundles import write_scenario_bundles

    written = write_scenario_bundles(
        args.out,
        n_time_steps=args.t,
        seed=args.seed,
        require_path_to_y=args.require_path_to_y,
        plots=not args.no_plots,
    )
    print(f"README + {len(written)} datasets under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
