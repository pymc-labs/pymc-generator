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

GPU = "A100"
GENERATION_TIMEOUT_S = 1200
# A demo run must fit comfortably inside one worker timeout (about 1-3 s per world).
DEMO_MAX_WORLDS = 400
DEMO_MAX_WORLDS_PER_TASK = 20


def check_demo_bounds(request: RunRequest) -> None:
    """Refuse requests too large for one bounded demo worker attempt."""
    if request.worlds > DEMO_MAX_WORLDS:
        raise ValueError(f"at most {DEMO_MAX_WORLDS} worlds per demo run, got {request.worlds}")
    if request.worlds_per_task > DEMO_MAX_WORLDS_PER_TASK:
        raise ValueError(
            f"at most {DEMO_MAX_WORLDS_PER_TASK} worlds per task, got {request.worlds_per_task}"
        )
