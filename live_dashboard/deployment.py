"""Names and limits of the Modal deployment, shared by the app, the launcher and the writer.

Every Modal object lives in the ``pfn`` environment of the workspace. Stdlib only: the
Modal CLI imports this module with its own interpreter.
"""

from __future__ import annotations

from live_dashboard.publication import RunRequest

ENVIRONMENT = "pfn"
APP_NAME = "pymc-generator-live-dashboard"
LAUNCHER_APP_NAME = "pymc-generator-live-dashboard-launch"
VOLUME_NAME = "pymc-generator-live-dashboard"
SECRET_NAME = "pymc-generator-dashboard-auth"
DASHBOARD_LABEL = "pymc-generator-dashboard"
GENERATE_FUNCTION = "generate_run"
DASHBOARD_FUNCTION = "dashboard"
REVISION_ENV = "LIVE_DASHBOARD_CODE_REVISION"

GENERATION_TIMEOUT_S = 1800
# Demo bounds. One attempt covers roughly 260-420 worlds on Modal (4-6.5 s of wall-clock
# time per world); a larger run stops before the timeout and finishes with
# `launch.py --resume`.
DEMO_MAX_WORLDS = 400
DEMO_MAX_WORLDS_PER_TASK = 20

# The generation worker's reservation and its Modal list price (modal.com/pricing, October
# 2026: $0.0000131 per physical core-second and $0.00000222 per GiB-second; Modal bills
# the larger of reservation and use). Each attempt records this hourly price, and the
# dashboard's cost card multiplies it by the attempt's recorded time.
GENERATION_CPU = 2.0
GENERATION_MEMORY_MIB = 4096
GENERATION_USD_PER_HOUR = 3600 * (
    GENERATION_CPU * 0.0000131 + GENERATION_MEMORY_MIB / 1024 * 0.00000222
)


def check_demo_bounds(request: RunRequest) -> None:
    """Refuse requests too large for one bounded demo worker attempt."""
    if request.worlds > DEMO_MAX_WORLDS:
        raise ValueError(f"at most {DEMO_MAX_WORLDS} worlds per demo run, got {request.worlds}")
    if request.worlds_per_task > DEMO_MAX_WORLDS_PER_TASK:
        raise ValueError(
            f"at most {DEMO_MAX_WORLDS_PER_TASK} worlds per task, got {request.worlds_per_task}"
        )
