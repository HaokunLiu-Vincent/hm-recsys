"""Static rules of the cloud scripts and the image workflow (stage 3, M13; AGENTS.md §12).

    1. The repository is public: the names of the cloud project (infra/.env, which
       git ignores) are in no tracked file.
    2. infra/setup.sh grants nothing at the level of Owner or Editor, creates no
       key file, makes nothing public, and every grant has a comment that says
       what it allows and why.
    3. infra/operate.sh runs every gcloud command as the operator account.
    4. The image workflow uses Workload Identity Federation with the least
       permissions, pins every action by commit hash, and takes the names of the
       cloud project from repository secrets, never from the file.

The files are read as text; nothing is run and no cloud API is called.

infra/ and .github/ are not copied into a container image (.dockerignore is an
allow-list), so inside an image there is nothing to check: every test has an
explicit branch for that case and says why it holds there.
"""

import os
import re
import subprocess
from pathlib import Path

import yaml

from hmrec.git_state import IMAGE_COMMIT_VARIABLE

REPO = Path(__file__).resolve().parents[2]
INFRA = REPO / "infra"
WORKFLOW = REPO / ".github" / "workflows" / "image.yml"
SCRIPTS = ("setup.sh", "teardown.sh", "operate.sh")
# The values of infra/.env that must never be published, and the shortest value that is
# searched for (a short one, like a region, would match by accident).
PRIVATE_KEYS = ("PROJECT_ID", "BUCKET", "OPERATOR_SERVICE_ACCOUNT")
MIN_PRIVATE_LENGTH = 8
PINNED_ACTION = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


def _in_image() -> bool:
    """True inside a container image: no infra/, no .github/, no .git (see the module docstring)."""
    return IMAGE_COMMIT_VARIABLE in os.environ


def _env_values(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of an env file (comments and empty lines are skipped)."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _code_lines(path: Path) -> list[str]:
    """The lines of a shell script that are not comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if not line.lstrip().startswith("#")]


def test_the_cloud_scripts_exist_and_the_example_holds_placeholders_only() -> None:
    if _in_image():
        # The image holds no infra/ folder: the scripts are run from a checkout only.
        assert not INFRA.exists()
        return
    for name in (*SCRIPTS, "env.example", "README.md"):
        assert (INFRA / name).is_file(), name
    example = _env_values(INFRA / "env.example")
    assert example["PROJECT_ID"] == "your-project-id"
    assert example["BUCKET"] == "your-bucket-name"
    assert example["OPERATOR_SERVICE_ACCOUNT"].endswith("@your-project-id.iam.gserviceaccount.com")
    assert example["GITHUB_REPOSITORY"] == "owner/repository"
    # Every name a script needs is in the example, so a copy of it is a complete .env.
    needed: set[str] = set()
    for name in SCRIPTS:
        text = (INFRA / name).read_text(encoding="utf-8")
        needed |= set(re.findall(r'"\$\{([A-Z_]+):\?\}"', text))
    assert needed and needed <= set(example), sorted(needed - set(example))


def test_the_names_of_the_cloud_project_are_in_no_tracked_file() -> None:
    env_file = INFRA / ".env"
    if _in_image() or not env_file.is_file():
        # Inside an image and in CI there is no infra/.env (git ignores it), so there is
        # no private value to look for; the laptop, where the file exists, runs the check.
        assert not env_file.exists()
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "infra/.env"], cwd=REPO, check=False
    )
    assert ignored.returncode == 0, "infra/.env must be ignored by git"
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True, text=True
    )
    values = _env_values(env_file)
    private = [
        values[key] for key in PRIVATE_KEYS if len(values.get(key, "")) >= MIN_PRIVATE_LENGTH
    ]
    assert private, "infra/.env holds no project id or bucket name to look for"
    found: list[str] = []
    for name in listing.stdout.split("\0"):
        path = REPO / name
        if name == "" or not path.is_file():
            continue
        content = path.read_bytes()
        for value in private:
            if value.encode("utf-8") in content:
                # The value itself is not put into the message: a test log may be published.
                found.append(f"{name} holds a private value of infra/.env")
    assert found == []


def test_setup_grants_little_and_explains_every_grant() -> None:
    if _in_image():
        assert not INFRA.exists()
        return
    script = INFRA / "setup.sh"
    code = "\n".join(_code_lines(script))
    for forbidden in (
        "roles/owner",
        "roles/editor",
        "allUsers",
        "allAuthenticatedUsers",
        "keys create",
        "--impersonate-service-account",
    ):
        assert forbidden not in code, forbidden
    # Every command that adds a role binding comes right after a comment block that
    # starts with "# GRANT:" (what it allows) and says "WHY" (why it is needed).
    lines = script.read_text(encoding="utf-8").splitlines()
    grants = _grant_commands(lines)
    assert len(grants) == 13
    for first_line in grants:
        block: list[str] = []
        above = first_line - 1
        while above >= 0 and lines[above].startswith("#"):
            block.insert(0, lines[above])
            above -= 1
        text = " ".join(block)
        assert "GRANT:" in text and "WHY" in text, f"setup.sh line {first_line + 1}"


def _grant_commands(lines: list[str]) -> list[int]:
    """Index of the first line of every top-level command of setup.sh that adds a role binding.

    A command starts at a line without indentation and goes on while lines end
    with a backslash. It is a grant if it calls the helper `project_role` or
    contains `add-iam-policy-binding`.
    """
    grants: list[int] = []
    index = 0
    while index < len(lines):
        first = index
        command = lines[index]
        while lines[index].endswith("\\") and index + 1 < len(lines):
            index += 1
            command += " " + lines[index]
        index += 1
        top_level = lines[first].startswith(("project_role ", "run gcloud "))
        if top_level and ("project_role " in command or "add-iam-policy-binding" in command):
            grants.append(first)
    return grants


def test_teardown_never_deletes_the_bucket_or_the_dvc_data() -> None:
    if _in_image():
        assert not INFRA.exists()
        return
    code = "\n".join(_code_lines(INFRA / "teardown.sh"))
    assert "buckets delete" not in code and "projects delete" not in code
    # Objects are removed only below the pipeline's own prefixes, and only when asked.
    removals = [line for line in code.splitlines() if "storage rm" in line]
    removal = 'attempt gcloud storage rm --recursive "gs://${BUCKET}/${prefix}/"'
    assert [line.strip() for line in removals] == [f'{removal} --project "$PROJECT_ID"']
    assert (
        'for prefix in "$INPUTS_PREFIX" "$STATE_PREFIX" "${STATE_PREFIX}-calibration"; do' in code
    )
    assert "dvc" not in "\n".join(line for line in code.splitlines() if "storage rm" in line)


def test_the_operator_script_runs_every_command_as_the_operator() -> None:
    if _in_image():
        assert not INFRA.exists()
        return
    code = _code_lines(INFRA / "operate.sh")
    # gcloud is called in exactly one place: the function g, which impersonates.
    calls = [line.strip() for line in code if re.search(r"(^|\s)gcloud\s", line)]
    impersonating = 'gcloud --impersonate-service-account "$OPERATOR_SERVICE_ACCOUNT"'
    assert calls == [f'{impersonating} --project "$PROJECT_ID" "$@"']
    text = "\n".join(code)
    for forbidden in ("add-iam-policy-binding", "set-iam-policy", "services enable", "billing"):
        assert forbidden not in text, forbidden
    # The job has no automatic retry and the scheduler job is paused when it is created.
    assert "--max-retries 0" in text
    assert text.index("scheduler jobs create http") < text.index("scheduler jobs pause")


def test_the_image_workflow_has_least_permissions_pinned_actions_and_no_project_name() -> None:
    if _in_image():
        # .github/ is not in the image either.
        assert not WORKFLOW.exists()
        return
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    # PyYAML reads the key "on" as the boolean True.
    triggers = workflow[True]
    assert triggers == {"push": {"tags": ["image-*"]}, "workflow_dispatch": None}
    assert workflow["permissions"] == {"contents": "read", "id-token": "write"}
    steps = workflow["jobs"]["image"]["steps"]
    actions = [step["uses"] for step in steps if "uses" in step]
    assert len(actions) == 4
    for action in actions:
        assert PINNED_ACTION.fullmatch(action), action
    # The names of the project come from secrets (masked in the public log), never from
    # repository variables (printed in the head of every step) or from the file.
    assert "${{ vars." not in text
    assert set(re.findall(r"secrets\.([A-Z_]+)", text)) == {
        "GCP_WORKLOAD_IDENTITY_PROVIDER",
        "GCP_DEPLOYER_SERVICE_ACCOUNT",
        "GCP_IMAGE_REPOSITORY",
    }
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "gserviceaccount.com" not in code and "docker.pkg.dev" not in code
    assert "credentials_json" not in code
    # The build is the one of the CI workflow: same platform, same build argument.
    build = next(step for step in steps if step.get("id") == "build")["with"]
    ci = yaml.safe_load((WORKFLOW.parent / "ci.yml").read_text(encoding="utf-8"))
    ci_build = next(
        step for step in ci["jobs"]["image"]["steps"] if "build-push-action" in step.get("uses", "")
    )["with"]
    for key in ("context", "platforms", "build-args", "cache-from", "cache-to"):
        assert build[key] == ci_build[key], key
    assert build["push"] is True
