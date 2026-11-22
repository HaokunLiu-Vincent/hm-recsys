# infra/: the cloud project of stage 3

Scripts of plain `gcloud` commands that set up, operate and remove what the weekly pipeline
needs on Google Cloud (`docs/mlops.md` §5, §11, §12; decision D7). There is no Terraform and
no state file: every script can be read from top to bottom and run again.

| File | Who runs it | What it does |
|---|---|---|
| `env.example` | — | The names the scripts need, with placeholders. Copy it to `infra/.env` |
| `setup.sh` | **The owner of the project**, once | Enables APIs, creates the image repository, three service accounts, one custom role and Workload Identity Federation, and gives the operator account its roles |
| `operate.sh` | The operator (a person or a coding agent, by impersonation) | Uploads the inputs, creates and updates the job, starts and watches executions, creates the paused scheduler job |
| `teardown.sh` | **The owner of the project**, at the end | Removes everything `setup.sh` and `operate.sh` created; keeps the bucket and the DVC data |

**The repository is public.** The project id, the bucket name and the service-account
addresses are written only in `infra/.env`, which git ignores. They are in no tracked
file, no commit message, no workflow file and no published log
(`tests/unit/test_infra_rules.py` checks the tracked files on a machine that has an
`infra/.env`).

## Before anything: the budget

A billing budget with alerts must exist on the project before the first job runs
(`docs/mlops.md` §12, guardrail 1). No script here creates or changes a budget.

## 1. `infra/.env`

```bash
cp infra/env.example infra/.env
```

Fill in the first five values: the project id, the region, the bucket, the address of
the operator service account, and the GitHub repository as `owner/name`. The other
values are names of things the scripts create and can stay as they are.

## 2. `setup.sh` (owner)

```bash
gcloud auth login
bash infra/setup.sh
```

It prints every command before it runs it and ends with `setup finished` and the three
values of step 3. It can be run again at any time.

**What exists afterwards, and who may do what:**

| Identity | Created by | May | May not |
|---|---|---|---|
| Runtime account of the job | `setup.sh` | Read, write and delete objects of the one bucket; write log entries | Anything else. The job mounts only the inputs and state prefixes, so it never sees `dvc/` |
| Scheduler account | `setup.sh` | Start a Cloud Run job of the project (a custom role with the single permission `run.jobs.run`) | Change a job, call a service, read anything |
| Deployer account | `setup.sh` | Push and read images of the one repository. Only runs of this GitHub repository can act as it (Workload Identity Federation with a repository condition) | Deploy, read the bucket, anything else |
| Operator account | before M13 | Create, update and run the job; create, pause and run the scheduler job; read logs and metrics; read images; read and write objects of the bucket | Change IAM, billing or APIs; create service accounts |

Nothing has the Owner or Editor role, no key file exists, and nothing is public.

**Why a custom role for the scheduler account.** The job does not exist when `setup.sh`
runs, so the permission cannot be given on the job itself, and the predefined role that
contains it (`roles/run.invoker`) also allows calling Cloud Run services. The custom role
holds one permission. The project has one job.

## 3. Three GitHub repository secrets

The image workflow (`.github/workflows/image.yml`) reads the names of the project from
repository **secrets**. They are names, not credentials; they are secrets because the
workflow log of a public repository is public, and GitHub masks secrets in it (a
repository variable is printed in the head of every step).

`setup.sh` prints the three values at its end.

| Secret | Value |
|---|---|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/<number>/locations/global/workloadIdentityPools/<pool>/providers/<provider>` |
| `GCP_DEPLOYER_SERVICE_ACCOUNT` | the address of the deployer account |
| `GCP_IMAGE_REPOSITORY` | `<region>-docker.pkg.dev/<project>/<repository>/<image>` |

In the browser: the repository's **Settings → Secrets and variables → Actions → Secrets →
New repository secret**. With the GitHub CLI: `gh secret set NAME` (it asks for the value).

## 4. The image

A tag `image-<anything>` on a commit, or a manual run of the workflow, builds the
linux/amd64 image from that commit as CI does, pushes it under the commit hash, and
publishes its digest as a notice on the run page. **The job is deployed from the digest.**

```bash
git tag image-<name> <commit>
git push origin image-<name>
```

## 5. `operate.sh` (operator)

Every command runs as the operator account (`--impersonate-service-account`).

```bash
bash infra/operate.sh upload-inputs                  # data/processed, data/bootstrap, category codes
bash infra/operate.sh create-job sha256:<digest>     # 6 vCPU, 24 GiB, no retries, 6 h time limit
bash infra/operate.sh execute scripts/m13_volume_check.py --megabytes 1024 --hash-inputs
bash infra/operate.sh execute scripts/weekly.py --anchor 2020-07-29 --mode eval
bash infra/operate.sh wait <execution>
bash infra/operate.sh create-scheduler               # weekly, and paused at once
bash infra/operate.sh run-scheduler                  # the job's own arguments: --next --mode eval
bash infra/operate.sh download-records               # manifests and small tables to artifacts/m13_cloud/
bash infra/operate.sh memory <start> <end>           # highest memory use of the instance (Cloud Monitoring)
bash infra/operate.sh state-size
bash infra/operate.sh status
```

`run-scheduler` resumes the scheduler job, runs it and pauses it again: Cloud Scheduler
refuses to run a paused job by hand. The job is enabled for a few seconds.

**The bucket** (`docs/mlops.md` §7.2):

```
gs://<bucket>/
  dvc/                     the DVC remote of M11; never mounted, never touched
  inputs/                  data/processed/, data/bootstrap/, artifacts/features/category_codes.json
  pipeline/                the state: state/, weeks/, predictions/, runs/, and results/ (the tables)
  pipeline-calibration/    the same layout, written only by a calibration tick
```

**The job** mounts three prefixes with Cloud Storage FUSE (`only-dir`), so the container
sees nothing else of the bucket: `inputs/` read-only, `pipeline/` and
`pipeline-calibration/` writable. Two environment variables tell the pipeline where they
are (`HMREC_PIPELINE_INPUTS_ROOT`, `HMREC_PIPELINE_STATE_ROOT`); a third holds the image
digest, which every run manifest records (`HMREC_IMAGE_DIGEST`). The job's own arguments
are `scripts/weekly.py --next --mode eval`: the anchor comes from the state, so the
scheduler starts it with the same arguments every week. One execution at a time;
`operate.sh` refuses to start a second one.

## 6. `teardown.sh` (owner)

```bash
bash infra/teardown.sh                    # asks for the project id before it removes anything
bash infra/teardown.sh --delete-objects   # also inputs/, pipeline/ and pipeline-calibration/
```

It never deletes the bucket, the `dvc/` prefix, the operator account or the budget, and it
leaves the APIs enabled.

## What costs money at rest

| Resource | At rest |
|---|---|
| Cloud Run job (no execution running) | nothing |
| Cloud Scheduler job, paused or not | nothing (the first three jobs of a billing account are free) |
| Service accounts, the custom role, Workload Identity Federation | nothing |
| Images in Artifact Registry | storage, about 0.10 USD per GB and month above the first 0.5 GB |
| Objects in the bucket | storage, about 0.02 USD per GB and month (Standard, `us-central1`) |

Prices as read from the pricing pages on 2026-11-22 (Cloud Storage Standard in the job's
region: 0.020 USD per GiB-month; Artifact Registry: the first 0.5 GiB of a billing account
free, then about 0.10 USD per GiB-month; Cloud Scheduler: three jobs of a billing account
free, and a paused job counts as a job).

**After M13** (`docs/mlops.md` §12): the bucket holds 40.5 GiB (state 27.2, DVC remote 5.6,
calibration state 5.4, inputs 2.3), about 0.81 USD a month; the two images of 635 MB cost at
most 0.07 USD a month; the job, the paused scheduler job and the accounts cost nothing.

## What M13 found out about this set-up

- **The mounted bucket is enough**: the state lies directly on it, and no step needed an
  in-memory working copy. It refuses to rename a folder, so the pipeline commits a step file by
  file with the step's record last (`hmrec.weekly.storage.MountedStateStore`).
- The image's user is not root (uid 1000); the volumes are mounted with `uid=1000;gid=1000`.
- An execution takes between one and two minutes to start (image pull and three mounts).
- A paused scheduler job cannot be run by hand; see `run-scheduler` above.
- The memory metric of Cloud Monitoring is sampled once a minute and misses short peaks; the
  true peak of every step is in its step record and in the run manifest.
