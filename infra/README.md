# infra/: the cloud project of stage 3

Scripts of plain `gcloud` commands that set up, operate and remove what the weekly pipeline
needs on Google Cloud (`docs/mlops.md` §5, §11, §12; decision D7). There is no Terraform and
no state file: every script can be read from top to bottom and run again.

| File | Who runs it | What it does |
|---|---|---|
| `env.example` | — | The names the scripts need, with placeholders. Copy it to `infra/.env` |
| `setup.sh` | **The owner of the project**, once (and again after M14) | Enables APIs, creates the image repository, four service accounts, one custom role and Workload Identity Federation, gives the operator and the deployer their roles, and creates the email notification channel and five alert policies |
| `operate.sh` | The operator (a person or a coding agent, by impersonation) | Uploads the inputs, creates and updates the job, starts and watches executions, creates the paused scheduler job; creates, updates and calls the recommendation service; moves the `current` pointer |
| `teardown.sh` | **The owner of the project**, at the end | Removes everything `setup.sh` and `operate.sh` created; keeps the bucket and the DVC data |

**The repository is public.** The project id, the bucket name and the service-account
addresses are written only in `infra/.env`, which git ignores. They are in no tracked
file, no commit message, no workflow file and no published log
(`tests/unit/test_infra_rules.py` checks the tracked files on a machine that has an
`infra/.env`).

## The demo state (from 2026-11-23)

Stage 3 was closed on 2026-11-23 **without tearing the cloud down** (the user's decision,
`docs/decisions.md`): what was built is kept as a small live demo and is torn down later with
`teardown.sh` (section 7). Nothing bills by the hour: no execution runs, the service scales
to zero, the scheduler job is paused, and the billing budget with its alerts stays.

**What exists** (listed on 2026-11-23, after the calibration state and the drill states were
deleted):

| Resource | State | Cost per month at rest |
|---|---|---|
| Cloud Run job of the weekly pipeline (16 executions) | no execution running | nothing |
| Cloud Run service of the recommendation API, minimum instances 0 | no instance without a request | nothing |
| Cloud Scheduler job, weekly | **paused** | nothing |
| Bucket: `inputs/` 2.27 GiB, `pipeline/` 27.31 GiB, `dvc/` 5.59 GiB (35.17 GiB) | kept | about 0.70 USD (0.020 USD per GiB-month) |
| Images: five of the pipeline (635 to 640 MB each) and two of the service (70 MB each), 3.33 GB as the sum | kept | at most 0.28 USD (0.10 USD per GB-month above the first 0.5 GB, if no layer were shared) |
| One email notification channel, five alert policies, the billing budget | enabled | nothing |
| Five service accounts, one custom role, a workload identity pool and provider | kept | nothing |

**At most about 1 USD a month**, computed from these sizes and the list prices read on
2026-11-22; the bill was not read. A request to the service starts an instance for some
minutes (a fraction of a cent); an execution of the job costs about 0.25 USD per `eval`
tick. Do not start a tick or resume the scheduler job for a demo: the state is the record of
the replay.

**How to show it** (as the operator; every command only reads, except the call, which starts
an instance of the service):

```bash
bash infra/operate.sh call "/v1/recommendations/$(uv run python -c "import polars as pl; print(pl.read_parquet('data/processed/customer_id_map.parquet')['customer_id'][0])")"
bash infra/operate.sh call /v1/recommendations/nobody     # an unknown customer: "source": "fallback"
bash infra/operate.sh call /v1/health                     # the served set, its size, the image's commit
bash infra/operate.sh status                              # the job and its 16 executions, the service, the paused scheduler job, the images
bash infra/operate.sh pointer                             # state/current.json: the prediction set of 2020-09-23
bash infra/operate.sh state-size                          # the prefixes of the bucket
bash infra/operate.sh alert-lines 30d                     # the alert lines of the drills, while Cloud Logging keeps them
bash infra/operate.sh incidents                           # the incidents the alert policies opened
make mlflow-ui                                            # the local MLflow view: the runs of both replays, the model registry mirror
```

The alert policies and the notification channel are under Monitoring → Alerting in the
console of the project; the workflow runs (CI, image, deploy) are on the repository's
Actions page. The layout of the bucket is in section 5; what each tick wrote is in
`docs/mlops.md` §7.2.

**Images that are not deployed.** The job and the service run one image each; the other
five are kept only as history. `bash infra/operate.sh images-not-deployed` lists them
(read-only). The owner can delete them with:

```bash
( . infra/.env; bash infra/operate.sh images-not-deployed | while read -r image; do gcloud artifacts docker images delete "$image" --project "$PROJECT_ID" --delete-tags --quiet; done )
```

One of the five is the image that ran the replay on Cloud Run (commit `07220a7`): the run
manifests of the replay name its digest, and after this command that digest can no longer
be pulled. With only the two deployed images left, the images cost about 0.02 USD a month.
A new tag `deploy-*` or `v*` builds two new images and makes the two that are deployed now
"not deployed".

**How to tear it down later**: section 7 (`bash infra/teardown.sh`, and with
`--delete-objects` also the pipeline's prefixes of the bucket). The bucket itself, the DVC
data, the operator account and the budget are not removed by any script; delete the bucket
or the project by hand when nothing of it is needed any more.

## Before anything: the budget

A billing budget with alerts must exist on the project before the first job runs
(`docs/mlops.md` §12, guardrail 1). No script here creates or changes a budget.

## 1. `infra/.env`

```bash
cp infra/env.example infra/.env
```

Fill in the first six values: the project id, the region, the bucket, the address of
the operator service account, the GitHub repository as `owner/name`, and the address
that receives the alert emails (`ALERT_EMAIL`, M14). The other values are names of
things the scripts create and can stay as they are.

## 2. `setup.sh` (owner)

```bash
gcloud auth login
bash infra/setup.sh
```

It prints every command before it runs it and ends with `setup finished` and the three
values of step 3. It can be run again at any time: a thing that exists is left alone.
After M14 the last lines say `1 email notification channel and 5 alert policies exist
(5 expected)`.

**What exists afterwards, and who may do what:**

| Identity | Created by | May | May not |
|---|---|---|---|
| Runtime account of the job | `setup.sh` | Read, write and delete objects of the one bucket; write log entries | Anything else. The job mounts only the inputs and state prefixes, so it never sees `dvc/` |
| Scheduler account | `setup.sh` | Start a Cloud Run job of the project (a custom role with the single permission `run.jobs.run`) | Change a job, call a service, read anything |
| Runtime account of the service (M14) | `setup.sh` | Read objects of the one bucket; write log entries | Write or delete an object, anything else. The service mounts only the state prefix, read-only |
| Deployer account | `setup.sh` | Push and read images of the one repository; from M14 update the Cloud Run job and service (`roles/run.developer`) and attach their two runtime accounts. Only runs of this GitHub repository can act as it (Workload Identity Federation with a repository condition) | Read the bucket, change who may call a service, anything else |
| Operator account | before M13 | Create, update and run the job; create, pause and run the scheduler job; read logs and metrics; read images; read and write objects of the bucket; from M14 create and update the service and call it (`roles/run.invoker`) | Change IAM, billing or APIs; create service accounts, notification channels or alert policies |

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

## 6. The recommendation service, rollback and deployment (M14)

```bash
bash infra/operate.sh serving-set 2020-09-23        # once per prediction set published before M14
bash infra/operate.sh create-service                # private, min 0 / max 2 instances, state read-only
git tag deploy-<name> <commit> && git push origin deploy-<name>   # build both images, deploy job and service
bash infra/operate.sh call /v1/health
bash infra/operate.sh call /v1/recommendations/<customer id>
bash infra/operate.sh rollback previous             # point `current` at the previous prediction set
bash infra/operate.sh rollback next                 # and forward again
bash infra/operate.sh pointer                       # state/current.json as it is in the bucket
bash infra/operate.sh service-logs hmrec_alert      # the service's alert lines
```

- **The service** (`docs/mlops.md` §9) is a Cloud Run service with authenticated calls
  only, minimum instances 0 and a small maximum, 1 vCPU, and ONE volume: the state
  prefix, read-only. It runs as an account that can only read objects and write logs. It
  loads the prediction set `state/current.json` names and follows the pointer without a
  restart.
- **`create-service` without a digest** starts the service from Google's placeholder
  image; the first run of the deploy workflow puts the real image in. With a digest
  (`create-service sha256:...`, `update-service sha256:...`) it runs that image of
  `<image>-serving`.
- **Deployment on a tag** (`.github/workflows/deploy.yml`): a tag `deploy-*` or `v*`, or a
  manual run, builds and pushes the pipeline image and the service image and points the
  job and the service at the two digests. It needs **no new secret**: the three of
  step 3 are enough. It changes images only; the job and the service must exist.
- **Rollback** is one execution of the job (it has the state mounted and checks the
  target's checksums there): `scripts/m14_rollback.py`. A target that does not exist or
  fails a check is refused and the pointer stays. The service switches at its next look
  at the pointer (every 60 seconds while it has CPU).
- **Health**: call `/v1/health` through Cloud Run. Google's front end answers `/healthz`
  on a `run.app` address itself (a 404 page); the service's `/healthz` is reachable only
  where the server is reached directly.
- **Measured in M14** (`docs/mlops.md` §9.1): cold start 6.0 and 6.3 s, warm p95 0.77 ms
  inside the service and 236 ms from a laptop on another continent, 0.19 GiB of the 1 GiB;
  a pointer move reached the service in 41 s and in under 60 s.
- **Alerting** (`docs/mlops.md` §10): `setup.sh` creates one email channel
  (`ALERT_EMAIL`) and five policies: four match the alert lines of the pipeline and the
  service (`jsonPayload.hmrec_alert` = `step_failed`, `input_check_failed`,
  `quality_alert`, `prediction_set_load_failed`), one a failed execution of a job. The
  operator creates no channel and no policy.

**Alert drills** (`docs/mlops.md` §10.1). A drill never runs in the replay's state: it
has a job, a service and a prefix of the bucket of their own (`<job>-drill`,
`<service>-drill`, `<state prefix>-drill/`), and every alert line of a drill carries a
note. The commands that prepare a drill state can write only below the drill prefix.

```bash
bash infra/operate.sh create-drill-job sha256:<digest of the pipeline image>
bash infra/operate.sh drill-copy weeks/2020-08-19 quality/weeks/2020-08-19      # inside the bucket
bash infra/operate.sh drill-upload <local file> <key below the drill prefix>
bash infra/operate.sh execute-drill <case> inputs scripts/weekly.py --anchor 2020-08-19 --mode eval
bash infra/operate.sh create-drill-service sha256:<digest of the service image> <case>
bash infra/operate.sh alert-lines && bash infra/operate.sh incidents
bash infra/operate.sh delete-drill                                              # the drill job and service
```

## 7. `teardown.sh` (owner)

```bash
bash infra/teardown.sh                    # asks for the project id before it removes anything
bash infra/teardown.sh --delete-objects   # also inputs/, pipeline/ and pipeline-calibration/
```

It never deletes the bucket, the `dvc/` prefix, the operator account or the budget, and it
leaves the APIs enabled. From M14 it also removes the service, the alert policies and the
notification channel.

## What costs money at rest

| Resource | At rest |
|---|---|
| Cloud Run job (no execution running) | nothing |
| Cloud Scheduler job, paused or not | nothing (the first three jobs of a billing account are free) |
| Cloud Run service with minimum instances 0 (M14) | nothing while no request arrives |
| Notification channel, alert policies | nothing |
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

**After M15**: see "The demo state" at the top (35.17 GiB, about 0.70 USD a month, after the
calibration state and the drill states were deleted).

**After M14** (`docs/mlops.md` §12): the bucket held 43.8 GiB (state 27.3, DVC remote 5.6,
calibration state 5.4, drill states 3.2, inputs 2.3), about 0.88 USD a month; five pipeline
images and two service images cost at most 0.3 USD a month; the job, the service without
an instance, the paused scheduler job, the channel, the policies and the accounts cost
nothing. M14 used about 0.11 USD of compute.

## What M13 found out about this set-up

- **The mounted bucket is enough**: the state lies directly on it, and no step needed an
  in-memory working copy. It refuses to rename a folder, so the pipeline commits a step file by
  file with the step's record last (`hmrec.weekly.storage.MountedStateStore`).
- The image's user is not root (uid 1000); the volumes are mounted with `uid=1000;gid=1000`.
- An execution takes between one and two minutes to start (image pull and three mounts).
- A paused scheduler job cannot be run by hand; see `run-scheduler` above.
- The memory metric of Cloud Monitoring is sampled once a minute and misses short peaks; the
  true peak of every step is in its step record and in the run manifest.
