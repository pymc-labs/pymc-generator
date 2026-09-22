import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

RECIPE = Path(__file__).parents[1] / "conda" / "recipes" / "pymc-generator" / "meta.yaml"
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def _project_runtime_requirements() -> set[str]:
    with PYPROJECT.open("rb") as pyproject:
        project = tomllib.load(pyproject)["project"]
    return {
        canonicalize_name(Requirement(dependency).name) for dependency in project["dependencies"]
    }


def _recipe_runtime_requirements() -> set[str]:
    recipe = RECIPE.read_text()
    runtime = recipe.split("\n  run:\n", 1)[1].split("\n\ntest:", 1)[0]
    return {
        line.strip()[2:].split()[0] for line in runtime.splitlines() if line.strip().startswith("-")
    }


def test_native_conda_recipe_declares_nutpie_runtime_dependency():
    assert "nutpie" in _project_runtime_requirements()
    assert "nutpie" in _recipe_runtime_requirements()
