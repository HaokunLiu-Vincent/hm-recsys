"""Static rules of the recommendation service's code (stage 3, M14; AGENTS.md §12).

The service's image holds numpy, FastAPI, uvicorn and five source files, so:

    1. hmrec.serving, hmrec.serving_set and hmrec.structured_logs import only the
       standard library, numpy, FastAPI and each other. No torch, no LightGBM, no
       polars, no other module of the project.
    2. The Dockerfile of the service copies every project module the service
       imports (checked in a checkout; an image holds no Dockerfile).
    3. No other code imports hmrec.serving: the pipeline never depends on the
       service. The pipeline may use the two bottom-layer modules.
    4. The dependency group `serving` is exactly fastapi, uvicorn and numpy.
    5. No customer id can reach a log line: the service writes the route's
       template, and uvicorn's access log is off wherever the server is started.

The source files are read as text; nothing is imported.
"""

import ast
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hmrec"
SERVING = SRC / "serving"
SERVICE_MODULES = (SERVING / "__init__.py", SERVING / "store.py", SERVING / "app.py")
BOTTOM_MODULES = (SRC / "serving_set.py", SRC / "structured_logs.py")
ALLOWED_THIRD_PARTY = {"numpy", "fastapi"}
ALLOWED_PROJECT = {"hmrec.serving", "hmrec.serving_set", "hmrec.structured_logs"}


def imported_modules(path: Path) -> set[str]:
    """The modules a Python file imports ("from a.b import c" gives a.b)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_the_module_list_is_complete() -> None:
    # A new module of the service must be added to SERVICE_MODULES above.
    assert sorted(SERVING.glob("*.py")) == sorted(SERVICE_MODULES)


def test_the_service_imports_only_what_its_image_holds() -> None:
    for path in (*SERVICE_MODULES, *BOTTOM_MODULES):
        for module in imported_modules(path):
            top = module.split(".")[0]
            if top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY:
                continue
            assert any(
                module == allowed or module.startswith(allowed + ".") for allowed in ALLOWED_PROJECT
            ), f"{path.name} imports {module}"
    # The two bottom-layer modules do not import the service or FastAPI.
    for path in BOTTOM_MODULES:
        imports = imported_modules(path)
        assert not any(name.startswith(("hmrec.serving.", "fastapi")) for name in imports), path
        assert "hmrec.serving" not in imports


def test_the_service_image_copies_every_module_the_service_imports() -> None:
    dockerfile = REPO / "Dockerfile.serving"
    if not dockerfile.is_file():
        # Inside an image: no Dockerfile is copied into it, so there is nothing to compare.
        assert not (REPO / "Dockerfile").exists()
        return
    text = dockerfile.read_text(encoding="utf-8")
    copies = [line.split()[1:-1] for line in text.splitlines() if line.startswith("COPY src/")]
    copied = sorted(source for sources in copies for source in sources)
    assert copied == [
        "src/hmrec/__init__.py",
        "src/hmrec/serving/",
        "src/hmrec/serving_set.py",
        "src/hmrec/structured_logs.py",
    ]
    # Every project module the service imports is one of the copied files.
    needed: set[str] = set()
    for path in (*SERVICE_MODULES, *BOTTOM_MODULES):
        needed |= {name for name in imported_modules(path) if name.startswith("hmrec")}
    for module in needed:
        relative = "src/" + module.replace(".", "/")
        assert relative + ".py" in copied or relative.startswith("src/hmrec/serving"), module


def test_no_other_code_imports_the_service() -> None:
    files = [*SRC.rglob("*.py"), *(REPO / "scripts").glob("*.py")]
    for path in files:
        if SERVING in path.parents:
            continue
        imports = imported_modules(path)
        assert not any(
            name == "hmrec.serving" or name.startswith("hmrec.serving.") for name in imports
        ), path


def test_the_serving_group_is_fastapi_uvicorn_and_numpy() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    group = project["dependency-groups"]["serving"]
    names = sorted(requirement.split(">=")[0] for requirement in group)
    assert names == ["fastapi", "numpy", "uvicorn"]
    # The tests of the service run in the development environment and in the pipeline
    # image, which install the group explicitly; it is not a default group.
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "\nsetup:\n\tuv sync --group serving\n" in makefile
    dockerfile = REPO / "Dockerfile"
    if dockerfile.is_file():  # a checkout; an image holds no Dockerfile
        synced = [line for line in dockerfile.read_text().splitlines() if "uv sync" in line]
        assert len(synced) == 2 and all("--group serving" in line for line in synced)
    # numpy is a dependency of the project too: the group adds no package to the lock.
    assert any(dependency.startswith("numpy") for dependency in project["project"]["dependencies"])


def test_no_customer_id_can_reach_a_log_line() -> None:
    app = (SERVING / "app.py").read_text(encoding="utf-8")
    # The request line is built from the route's template and from request.state, never
    # from the path or from the customer id.
    log_function = app[app.index("async def log_request(") : app.index("@application.get(")]
    for forbidden in ("customer_id", "request.url", ".path_params", 'scope["path"]'):
        assert forbidden not in log_function, forbidden
    assert '"route": route_template(request)' in log_function
    store = (SERVING / "store.py").read_text(encoding="utf-8")
    assert "customer_id" not in store
    # uvicorn's access log prints the path: it is off wherever the server is started.
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    serve = makefile[makefile.index("\nserve:") :]
    assert "--no-access-log" in serve[: serve.index("\n\n")]
