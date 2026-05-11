import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _package_version() -> str:
    tree = ast.parse((ROOT / "src/seele_scholar_agent/__init__.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError("__version__ is missing from src/seele_scholar_agent/__init__.py")


def _locked_package_version() -> str:
    lock_data = tomllib.loads((ROOT / "uv.lock").read_text())
    for package in lock_data["package"]:
        if package["name"] == "seele-scholar-agent":
            return package["version"]
    raise AssertionError("seele-scholar-agent is missing from uv.lock")


def test_versions_are_consistent() -> None:
    project_data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project_version = project_data["project"]["version"]

    assert _package_version() == project_version
    assert _locked_package_version() == project_version
