#!/usr/bin/env bash
# infra/setup.sh: everything the weekly pipeline needs in the cloud project, except the
# job itself (docs/mlops.md §11, §12; M13).
#
# WHO RUNS IT: the owner of the project, signed in with `gcloud auth login`. Nobody else:
# this script changes IAM and enables APIs, which the operator account may never do.
#
#     cp infra/env.example infra/.env      # then fill in the first five values
#     bash infra/setup.sh
#
# WHAT IT IS: plain `gcloud` commands, one after another, each printed before it runs.
# It can be run again at any time: a thing that exists is left alone, and adding a role
# binding that exists changes nothing.
#
# WHAT IT CREATES
#   1. the APIs the pipeline uses, enabled;
#   2. one Docker repository in Artifact Registry;
#   3. the RUNTIME service account of the job: objects of the bucket, and writing logs;
#   4. the SCHEDULER service account: it may start a Cloud Run job, nothing else;
#   5. Workload Identity Federation for GitHub Actions, for this one repository, and the
#      DEPLOYER service account: it may push images to the repository of 2, nothing else;
#   6. the roles the existing OPERATOR account needs to create, update and run the job
#      and the scheduler job, and to read logs and metrics.
#
# WHAT IT NEVER DOES: no Owner or Editor role, no key file, nothing public, no change to
# the bucket's contents, no change to billing or budgets. It creates no resource that
# costs money while it is idle (an empty Artifact Registry repository is free).
#
# BEFORE THE FIRST BILLABLE RESOURCE a budget alert must exist on the project
# (docs/mlops.md §12, guardrail 1). This script cannot see budgets; it reminds you.

set -euo pipefail

cd "$(dirname "$0")"
if [ ! -f .env ]; then
  echo "infra/.env is missing. Create it:  cp infra/env.example infra/.env  and fill it in."
  exit 1
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a

# Every name comes from infra/.env; a missing one stops the script here.
: "${PROJECT_ID:?}" "${REGION:?}" "${BUCKET:?}" "${OPERATOR_SERVICE_ACCOUNT:?}"
: "${GITHUB_REPOSITORY:?}" "${ARTIFACT_REPOSITORY:?}" "${IMAGE_NAME:?}"
: "${RUNTIME_SERVICE_ACCOUNT_NAME:?}" "${SCHEDULER_SERVICE_ACCOUNT_NAME:?}"
: "${DEPLOYER_SERVICE_ACCOUNT_NAME:?}" "${JOB_STARTER_ROLE:?}" "${WIF_POOL:?}" "${WIF_PROVIDER:?}"

RUNTIME_SA="${RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="${SCHEDULER_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="${DEPLOYER_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

section() { printf '\n==== %s\n' "$*"; }
# Print a command, then run it. A new service account or role takes a few seconds to be
# known everywhere, so a command that fails is tried again (5 times, 10 seconds apart)
# before the script stops.
run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$@"; then
      return 0
    fi
    echo "  (attempt ${attempt} failed; trying again in 10 seconds)"
    sleep 10
  done
  echo "the command above failed 5 times; setup stops here. It can be run again."
  return 1
}
# Create a service account unless it exists.
service_account() {
  local name="$1" display="$2"
  if gcloud iam service-accounts describe "${name}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "service account ${name}: exists"
  else
    run gcloud iam service-accounts create "$name" --project "$PROJECT_ID" \
      --display-name "$display"
  fi
}
# Add one role to one member on the project. --condition=None: the binding has no
# condition (gcloud asks otherwise as soon as any binding of the project has one).
project_role() {
  local member="$1" role="$2"
  run gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "$member" --role "$role" --condition=None --format=none
}

section "0. Who and where"
ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
echo "signed in as: ${ACCOUNT}"
echo "project:      ${PROJECT_ID} (number ${PROJECT_NUMBER}), region ${REGION}"
# The bucket and the operator account were created before M13; this script does not create them.
run gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT_ID" \
  --format='value(name,location,uniform_bucket_level_access)'
run gcloud iam service-accounts describe "$OPERATOR_SERVICE_ACCOUNT" --project "$PROJECT_ID" \
  --format='value(email)'
echo "REMINDER: a billing budget with alerts must exist on this project before the first"
echo "          job runs (docs/mlops.md 12, guardrail 1). This script cannot check it."

section "1. APIs"
# run: Cloud Run jobs.   artifactregistry: the image registry.   cloudscheduler: the weekly
# trigger.   iam, iamcredentials, sts: service accounts, impersonation and Workload Identity
# Federation.   logging, monitoring: the job's logs and its memory and CPU metrics.
run gcloud services enable --project "$PROJECT_ID" \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com

section "2. Artifact Registry: one Docker repository"
if gcloud artifacts repositories describe "$ARTIFACT_REPOSITORY" --project "$PROJECT_ID" \
  --location "$REGION" >/dev/null 2>&1; then
  echo "repository ${ARTIFACT_REPOSITORY}: exists"
else
  run gcloud artifacts repositories create "$ARTIFACT_REPOSITORY" --project "$PROJECT_ID" \
    --location "$REGION" --repository-format docker \
    --description "Images of the weekly pipeline (pushed by GitHub Actions)"
fi

section "3. The runtime service account of the job"
service_account "$RUNTIME_SERVICE_ACCOUNT_NAME" "Weekly pipeline: the Cloud Run job runs as this"
# GRANT: read, list, create and delete OBJECTS of this one bucket (not the bucket itself,
# not its IAM policy). WHY: the job mounts two prefixes of the bucket as volumes: it reads
# its inputs and writes its state there. Delete is needed because a rename on a bucket is
# a copy and a delete. The job's mounts show it only the inputs and the state prefixes, so
# it never sees the DVC data under dvc/.
run gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --project "$PROJECT_ID" \
  --member "serviceAccount:${RUNTIME_SA}" --role roles/storage.objectUser --format=none
# GRANT: write log entries. WHY: the job's output goes to Cloud Logging. Nothing can be
# read with this role.
project_role "serviceAccount:${RUNTIME_SA}" roles/logging.logWriter

section "4. The service account of Cloud Scheduler"
service_account "$SCHEDULER_SERVICE_ACCOUNT_NAME" "Weekly pipeline: Cloud Scheduler starts the job as this"
# A custom role with ONE permission: run.jobs.run, "start an execution of a Cloud Run job
# with the arguments the job already has". WHY a custom role: the predefined role with this
# permission (roles/run.invoker) also allows calling Cloud Run services, and a role on the
# job itself cannot be given here because the job does not exist yet. The project has one
# job, so "start a job of this project" is "start the job".
if gcloud iam roles describe "$JOB_STARTER_ROLE" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "role ${JOB_STARTER_ROLE}: exists"
else
  run gcloud iam roles create "$JOB_STARTER_ROLE" --project "$PROJECT_ID" \
    --title "Start a Cloud Run job" \
    --description "The single permission run.jobs.run, for the Cloud Scheduler account" \
    --permissions run.jobs.run --stage GA
fi
# GRANT: the custom role above. WHY: the scheduler's weekly request starts the job.
project_role "serviceAccount:${SCHEDULER_SA}" "projects/${PROJECT_ID}/roles/${JOB_STARTER_ROLE}"

section "5. GitHub Actions: Workload Identity Federation and the deployer account"
# A pool and a provider let a GitHub Actions run prove who it is with a short-lived token
# GitHub signs; no key is stored anywhere. A pool that was deleted is kept for 30 days and
# its name cannot be created again in that time: it is restored instead.
POOL_STATE="$(gcloud iam workload-identity-pools describe "$WIF_POOL" --project "$PROJECT_ID" \
  --location global --format='value(state)' 2>/dev/null || true)"
if [ "$POOL_STATE" = "ACTIVE" ]; then
  echo "workload identity pool ${WIF_POOL}: exists"
elif [ "$POOL_STATE" = "DELETED" ]; then
  run gcloud iam workload-identity-pools undelete "$WIF_POOL" --project "$PROJECT_ID" \
    --location global
else
  run gcloud iam workload-identity-pools create "$WIF_POOL" --project "$PROJECT_ID" \
    --location global --display-name "GitHub Actions"
fi
PROVIDER_STATE="$(gcloud iam workload-identity-pools providers describe "$WIF_PROVIDER" \
  --project "$PROJECT_ID" --location global --workload-identity-pool "$WIF_POOL" \
  --format='value(state)' 2>/dev/null || true)"
if [ "$PROVIDER_STATE" = "ACTIVE" ]; then
  echo "workload identity provider ${WIF_PROVIDER}: exists"
elif [ "$PROVIDER_STATE" = "DELETED" ]; then
  run gcloud iam workload-identity-pools providers undelete "$WIF_PROVIDER" \
    --project "$PROJECT_ID" --location global --workload-identity-pool "$WIF_POOL"
else
  # --attribute-condition: tokens of ANY other repository are refused by the provider
  # itself, before any role is looked at.
  run gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
    --project "$PROJECT_ID" --location global --workload-identity-pool "$WIF_POOL" \
    --display-name "GitHub Actions" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository == '${GITHUB_REPOSITORY}'"
fi
service_account "$DEPLOYER_SERVICE_ACCOUNT_NAME" "Weekly pipeline: GitHub Actions pushes images as this"
# GRANT: write (push) and read images of THIS repository only. WHY: the image workflow
# pushes the image it built. The role is given on the repository, not on the project.
run gcloud artifacts repositories add-iam-policy-binding "$ARTIFACT_REPOSITORY" \
  --project "$PROJECT_ID" --location "$REGION" \
  --member "serviceAccount:${DEPLOYER_SA}" --role roles/artifactregistry.writer --format=none
# GRANT: runs of this one GitHub repository may act as the deployer account. WHY: this is
# the link between the provider above and the account; without it a token of the provider
# can do nothing.
run gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" --project "$PROJECT_ID" \
  --role roles/iam.workloadIdentityUser --format=none \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_REPOSITORY}"

section "6. The operator account (${OPERATOR_SERVICE_ACCOUNT})"
OPERATOR="serviceAccount:${OPERATOR_SERVICE_ACCOUNT}"
# GRANT: create, update, run and read Cloud Run jobs and their executions. WHY: the
# operator creates the job, changes its image, and starts and watches every tick. This is
# the smallest predefined role that can create a job; it also allows deleting a job and
# managing Cloud Run services (there are none). It cannot change who may call them.
project_role "$OPERATOR" roles/run.developer
# GRANT: attach the runtime account to a job. WHY: creating a job that runs as another
# account requires permission to act as that account. Given on that one account only.
run gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --project "$PROJECT_ID" \
  --member "$OPERATOR" --role roles/iam.serviceAccountUser --format=none
# GRANT: create, update, pause, resume and run Cloud Scheduler jobs. WHY: the operator
# creates the weekly trigger, keeps it paused, and runs it by hand for one tick. There is
# no smaller predefined role that can create a scheduler job.
project_role "$OPERATOR" roles/cloudscheduler.admin
# GRANT: attach the scheduler account to a scheduler job. WHY: as for the runtime account.
run gcloud iam service-accounts add-iam-policy-binding "$SCHEDULER_SA" --project "$PROJECT_ID" \
  --member "$OPERATOR" --role roles/iam.serviceAccountUser --format=none
# GRANT: read log entries. WHY: the operator reads the job's logs to see why a tick stopped.
project_role "$OPERATOR" roles/logging.viewer
# GRANT: read metrics. WHY: the highest memory use of the job's instance is recorded for
# every tick (docs/mlops.md 8.4).
project_role "$OPERATOR" roles/monitoring.viewer
# GRANT: use the project's enabled APIs for quota. WHY: some APIs (Cloud Scheduler,
# Monitoring) refuse a caller that may not "use" the project they are called for. It
# enables nothing and reads nothing.
project_role "$OPERATOR" roles/serviceusage.serviceUsageConsumer
# GRANT: read images of the repository. WHY: the operator reads the digest of an image and
# deploys the job from it. It cannot push.
run gcloud artifacts repositories add-iam-policy-binding "$ARTIFACT_REPOSITORY" \
  --project "$PROJECT_ID" --location "$REGION" \
  --member "$OPERATOR" --role roles/artifactregistry.reader --format=none
# The operator's access to the bucket's objects existed before M13 and is not changed here.

section "Done"
cat <<EOF
setup finished: APIs enabled; repository, 3 service accounts, 1 custom role,
workload identity pool and provider exist; the operator's roles are in place.

Three GitHub repository SECRETS for the image workflow (.github/workflows/image.yml).
Set them at  https://github.com/${GITHUB_REPOSITORY}/settings/secrets/actions
("New repository secret"), or with the gh CLI. They are names, not keys; they are stored
as secrets so that GitHub masks them in the public workflow log.

  GCP_WORKLOAD_IDENTITY_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}
  GCP_DEPLOYER_SERVICE_ACCOUNT   = ${DEPLOYER_SA}
  GCP_IMAGE_REPOSITORY           = ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}
EOF
