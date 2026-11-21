"""Static rules of M11 (versioning; AGENTS.md §12, docs/mlops.md §7).

    1. DVC and MLflow are in dependency groups of their own, not among the
       runtime or development dependencies (the pipeline image installs neither).
    2. The repository is public: no bucket URL is in any file of the code tree,
       and the tracked DVC configuration names the remote without its URL.
    3. The new code stays where it belongs: hmrec.tracking loads neither torch
       nor lightgbm and imports MLflow only inside hmrec.tracking.store; no
       other code imports hmrec.tracking.

Files are read as text; nothing of the checked code is imported.
"""

import ast
import configparser
import os
import re
import tomllib
from pathlib import Path

from hmrec.code_tree import CODE_TREE_ENTRIES, code_tree_files
from hmrec.git_state import IMAGE_COMMIT_VARIABLE

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hmrec"
# A Cloud Storage URL with a real bucket name. "gs://<bucket>/" (the placeholder the
# documents use) does not match: "<" is not a letter or a digit.
BUCKET_URL = re.compile("gs" + r"://[A-Za-z0-9]")
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".csv", ".txt", ".lock", ""}
VERSIONING_GROUP = "versioning"
TRACKING_GROUP = "tracking"
NEW_MODULES = (
    SRC / "data_manifest.py",
    SRC / "bootstrap_state.py",
    SRC / "run_manifest.py",
    *sorted((SRC / "tracking").glob("*.py")),
)
MODEL_LIBRARIES = ("torch", "lightgbm")


def _requirement_names(requirements: list[str]) -> set[str]:
    """Package names of a requirement list ("dvc[gs]>=3" -> "dvc")."""
    return {
        re.split(r"[\[<>=!~ ;]", requirement, maxsplit=1)[0].lower() for requirement in requirements
    }


def _imports(path: Path) -> set[str]:
    """Top-level names of the modules a Python file imports, with the dotted names."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_dvc_and_mlflow_are_in_groups_of_their_own() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    groups = project["dependency-groups"]
    assert _requirement_names(groups[VERSIONING_GROUP]) == {"dvc"}
    assert any(requirement.startswith("dvc[gs]") for requirement in groups[VERSIONING_GROUP])
    assert _requirement_names(groups[TRACKING_GROUP]) == {"mlflow"}
    # Not runtime dependencies and not in the dev group: `uv sync` (the laptop's
    # `make setup` and the image build) installs neither.
    runtime_and_dev = _requirement_names(project["project"]["dependencies"] + groups["dev"])
    assert not runtime_and_dev & {"dvc", "mlflow"}
    assert "default-groups" not in project.get("tool", {}).get("uv", {})


def test_no_bucket_url_in_the_code_tree() -> None:
    files = [*code_tree_files(REPO, CODE_TREE_ENTRIES), *code_tree_files(REPO, ("results",))]
    found: list[str] = []
    for name in files:
        path = REPO / name
        if path.suffix not in TEXT_SUFFIXES:
            continue
        if BUCKET_URL.search(path.read_text(encoding="utf-8", errors="replace")) is not None:
            found.append(name)
    assert found == [], f"a bucket URL is written in: {found}"


def test_the_tracked_dvc_configuration_names_no_bucket() -> None:
    tracked_config = REPO / ".dvc" / "config"
    if tracked_config.is_file():
        # A checkout (the laptop). The tracked file holds the remote's NAME; the URL is in
        # .dvc/config.local, which git ignores.
        text = tracked_config.read_text(encoding="utf-8")
        assert BUCKET_URL.search(text) is None and "url" not in text
        parser = configparser.ConfigParser()
        parser.read_string(text)
        assert parser.sections() == ["core"]
        assert parser["core"]["remote"] == "storage"
        assert parser["core"]["analytics"] == "false"
        ignore_lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "/.dvc/config.local" in ignore_lines
        # The pointer files are let into git; the data are not ("?*", see .gitignore).
        assert "/data/?*" in ignore_lines and "!/data/*.dvc" in ignore_lines
        assert "/data/*" not in ignore_lines and "/artifacts/*" not in ignore_lines
        # The environment that holds DVC and MLflow is never committed either.
        assert ".venv-tools/" in ignore_lines
    else:
        # Inside an image: .dvc/ is not on the allow-list of .dockerignore, so no DVC
        # configuration (tracked or local) and no pointer file can be in the image.
        assert IMAGE_COMMIT_VARIABLE in os.environ
        assert not (REPO / ".dvc").exists()
        assert not (REPO / ".dvcignore").exists()


def test_the_new_modules_load_no_model_library() -> None:
    for module in NEW_MODULES:
        imported = _imports(module)
        for library in MODEL_LIBRARIES:
            assert not any(
                name == library or name.startswith(f"{library}.") for name in imported
            ), f"{module.name} imports {library}"
        # MLflow is not installed everywhere: it is never imported statically.
        assert not any(name == "mlflow" or name.startswith("mlflow.") for name in imported)


def test_only_the_view_and_its_script_import_the_tracking_package() -> None:
    allowed = {path for path in NEW_MODULES if path.parent.name == "tracking"}
    allowed.add(REPO / "scripts" / "m11_mlflow.py")
    sources = [*SRC.rglob("*.py"), *(REPO / "scripts").rglob("*.py")]
    importers = {
        path
        for path in sources
        if any(
            name == "hmrec.tracking" or name.startswith("hmrec.tracking.")
            for name in _imports(path)
        )
    }
    assert importers <= allowed, sorted(str(path) for path in importers - allowed)
