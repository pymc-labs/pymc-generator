import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

RECIPE = Path(__file__).parents[1] / "conda" / "recipes" / "pymc-generator" / "meta.yaml"
CONDA_FORGE_RECIPE = Path(__file__).parents[1] / "conda" / "recipes" / "conda-forge" / "meta.yaml"
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def _project_runtime_requirements() -> set[str]:
    with PYPROJECT.open("rb") as pyproject:
        project = tomllib.load(pyproject)["project"]
    return {
        canonicalize_name(Requirement(dependency).name) for dependency in project["dependencies"]
    }


def _recipe_runtime_requirements(recipe_path: Path) -> set[str]:
    recipe = recipe_path.read_text()
    runtime = recipe.split("\n  run:\n", 1)[1].split("\n\ntest:", 1)[0]
    return {
        line.strip()[2:].split()[0] for line in runtime.splitlines() if line.strip().startswith("-")
    }


@pytest.mark.parametrize("recipe_path", [RECIPE, CONDA_FORGE_RECIPE])
def test_active_conda_recipes_declare_nutpie_runtime_dependency(recipe_path: Path):
    assert "nutpie" in _project_runtime_requirements()
    assert "nutpie" in _recipe_runtime_requirements(recipe_path)


def test_conda_recipe_contract_keeps_native_and_submission_roles_distinct():
    native = RECIPE.read_text()
    conda_forge = CONDA_FORGE_RECIPE.read_text()
    assert "scripts/smoke_install.py" in native
    assert "scripts/smoke_install.py" not in conda_forge
    assert 'url: {{ environ["PYMC_GENERATOR_SDIST"] }}' in native
    assert "url: https://pypi.io/packages/source/" in conda_forge
