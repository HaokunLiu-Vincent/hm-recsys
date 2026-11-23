#!/usr/bin/env bash
# infra/setup.sh: everything the weekly pipeline needs in the cloud project, except the
# job itself (docs/mlops.md §11, §12; M13).
#
# WHO RUNS IT: the owner of the project, signed in with `gcloud auth login`. Nobody else:
# this script changes IAM and enables APIs, which the operator account may never do.
#
#     cp infra/env.example infra/.env      # then fill in the first six values
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
#      and the scheduler job, and to read logs and metrics;
#   7. (M14) the RUNTIME service account of the recommendation service: it may read
#      objects of the bucket and write logs, nothing else; and what the operator needs
#      to create the service and to call it;
#   8. (M14) what the DEPLOYER account needs to update the job and the service to a new
#      image when a tag is pushed (.github/workflows/deploy.yml);
#   9. (M14) alerting: ONE email notification channel (ALERT_EMAIL of infra/.env) and
#      five alert policies: one for each kind of alert line the pipeline and the service
#      write (hmrec.structured_logs), and one for a failed execution of the job.
#
# WHAT IT NEVER DOES: no Owner or Editor role, no key file, nothing public, no change to
# the bucket's contents, no change to billing or budgets. It creates no resource that
# costs money while it is idle (an empty Artifact Registry repository is free; a
# notification channel and alert policies are free).
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
: "${SERVICE_RUNTIME_SERVICE_ACCOUNT_NAME:?}" "${ALERT_CHANNEL_NAME:?}" "${ALERT_EMAIL:?}"

RUNTIME_SA="${RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="${SCHEDULER_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="${DEPLOYER_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SERVICE_RUNTIME_SA="${SERVICE_RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
# Every alert policy this script creates has a display name that starts with this, so
# that it is found again (here and in teardown.sh).
ALERT_POLICY_PREFIX="hmrec: "
MONITORING_API="https://monitoring.googleapis.com/v3/projects/${PROJECT_ID}"

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

section "7. The recommendation service: its runtime account, and the operator (M14)"
service_account "$SERVICE_RUNTIME_SERVICE_ACCOUNT_NAME" "Recommendation service: the Cloud Run service runs as this"
# GRANT: read and list OBJECTS of this one bucket. It cannot write, delete or change one.
# WHY: the service mounts the state prefix of the bucket READ-ONLY and loads the current
# prediction set from it. The mount shows it only that prefix.
run gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --project "$PROJECT_ID" \
  --member "serviceAccount:${SERVICE_RUNTIME_SA}" --role roles/storage.objectViewer --format=none
# GRANT: write log entries. WHY: the service's request lines and alert lines go to Cloud
# Logging. Nothing can be read with this role.
project_role "serviceAccount:${SERVICE_RUNTIME_SA}" roles/logging.logWriter
# GRANT: attach the service's runtime account to a service. WHY: the operator creates
# the Cloud Run service, which runs as that account. Given on that one account only.
run gcloud iam service-accounts add-iam-policy-binding "$SERVICE_RUNTIME_SA" --project "$PROJECT_ID" \
  --member "$OPERATOR" --role roles/iam.serviceAccountUser --format=none
# GRANT: call Cloud Run services of the project with an identity token. WHY: the service
# accepts authenticated calls only (nothing is public); the operator calls it to measure
# latency and to check its answers. Given on the project because the service does not
# exist yet when this script runs; the project has one service. The role also allows
# starting a job, which the operator may do already.
project_role "$OPERATOR" roles/run.invoker

section "8. Deployment on a tag: what the deployer account may do (M14)"
# GRANT: update Cloud Run jobs and services (and create or delete them; there is no
# smaller predefined role that can change the image of a job and of a service). It
# cannot change who may call a service and cannot read the bucket. WHY: the deploy
# workflow (.github/workflows/deploy.yml) points the job and the service at the images
# it has just pushed. Only runs of this one GitHub repository can act as this account.
project_role "serviceAccount:${DEPLOYER_SA}" roles/run.developer
# GRANT: deploy a job that runs as the job's runtime account. WHY: updating a job needs
# permission to act as the account the job runs as. Given on that one account only.
run gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --project "$PROJECT_ID" \
  --member "serviceAccount:${DEPLOYER_SA}" --role roles/iam.serviceAccountUser --format=none
# GRANT: deploy a service that runs as the service's runtime account. WHY: as above.
run gcloud iam service-accounts add-iam-policy-binding "$SERVICE_RUNTIME_SA" --project "$PROJECT_ID" \
  --member "serviceAccount:${DEPLOYER_SA}" --role roles/iam.serviceAccountUser --format=none

section "9. Alerting: one email channel and five alert policies (M14)"
# The Monitoring API is called with the owner's own token (curl): this version of gcloud
# has no released command for notification channels. The token is given to curl on its
# standard input, so it is not in a command line and not printed. The quota project
# header is needed for a person's credentials.
monitoring_api() {
  local method="$1" url="$2" body="${3:-}"
  local token
  token="$(gcloud auth print-access-token)"
  if [ -n "$body" ]; then
    printf 'header = "Authorization: Bearer %s"\n' "$token" | curl --silent --show-error --fail \
      --config - --request "$method" --header "x-goog-user-project: ${PROJECT_ID}" \
      --header "Content-Type: application/json" --data "$body" "$url"
  else
    printf 'header = "Authorization: Bearer %s"\n' "$token" | curl --silent --show-error --fail \
      --config - --request "$method" --header "x-goog-user-project: ${PROJECT_ID}" "$url"
  fi
}
# The resource name of the email channel with our display name ("" if there is none).
alert_channel() {
  monitoring_api GET "${MONITORING_API}/notificationChannels" | python3 -c '
import json, sys
wanted = sys.argv[1]
channels = json.load(sys.stdin).get("notificationChannels", [])
names = [c["name"] for c in channels if c.get("displayName") == wanted and c.get("type") == "email"]
print(names[0] if names else "")' "$ALERT_CHANNEL_NAME"
}
CHANNEL="$(alert_channel)"
if [ -n "$CHANNEL" ]; then
  echo "notification channel ${ALERT_CHANNEL_NAME}: exists"
else
  echo "+ create the email notification channel ${ALERT_CHANNEL_NAME} (address: ALERT_EMAIL of infra/.env)"
  CHANNEL_BODY="$(python3 -c '
import json, sys
print(json.dumps({"type": "email", "displayName": sys.argv[1], "enabled": True,
                  "labels": {"email_address": sys.argv[2]}}))' "$ALERT_CHANNEL_NAME" "$ALERT_EMAIL")"
  monitoring_api POST "${MONITORING_API}/notificationChannels" "$CHANNEL_BODY" >/dev/null
  CHANNEL="$(alert_channel)"
fi
if [ -z "$CHANNEL" ]; then
  echo "the notification channel was not found after it was created; setup stops here."
  exit 1
fi

# Create one alert policy unless a policy with its display name exists. A policy that
# exists is left as it is: to change one, delete it in the console and run this again.
alert_policy() {
  local name="$1" file="$2"
  local existing
  existing="$(gcloud monitoring policies list --project "$PROJECT_ID" \
    --filter "displayName=\"${name}\"" --format 'value(name)')"
  if [ -n "$existing" ]; then
    echo "alert policy ${name}: exists"
  else
    run gcloud monitoring policies create --project "$PROJECT_ID" --policy-from-file "$file" \
      --format 'value(displayName)'
  fi
}
# A policy that opens an incident when a LOG LINE with the fixed field
# jsonPayload.hmrec_alert = <kind> appears (hmrec.structured_logs). At most one
# notification in 5 minutes for one policy; an incident closes itself after 30 minutes.
POLICY_DIR="$(mktemp -d)"
log_alert_policy() {
  local kind="$1" title="$2" meaning="$3"
  local name="${ALERT_POLICY_PREFIX}${title}"
  python3 - "$name" "$kind" "$meaning" "$CHANNEL" > "${POLICY_DIR}/${kind}.json" <<'PY'
import json, sys
name, kind, meaning, channel = sys.argv[1:5]
documentation = (
    meaning + "\n\n"
    "The log line that opened this incident holds the details (fields of jsonPayload). "
    "Note of the line, if it has one: ${log.extracted_label.note}\n\n"
    "Runbook: README of the repository, section on the weekly pipeline in the cloud. "
    "No alert changes a configuration."
)
policy = {
    "displayName": name,
    "documentation": {"content": documentation, "mimeType": "text/markdown"},
    "combiner": "OR",
    "conditions": [
        {
            "displayName": "a log line with hmrec_alert = " + kind,
            "conditionMatchedLog": {
                "filter": 'jsonPayload.hmrec_alert="' + kind + '"',
                "labelExtractors": {"note": "EXTRACT(jsonPayload.note)"},
            },
        }
    ],
    "alertStrategy": {"notificationRateLimit": {"period": "300s"}, "autoClose": "1800s"},
    "notificationChannels": [channel],
}
print(json.dumps(policy, indent=2))
PY
  alert_policy "$name" "${POLICY_DIR}/${kind}.json"
}
log_alert_policy step_failed "a tick stopped" \
  "A step of the weekly pipeline failed, or a check stopped a tick. The run manifest of the tick says which step and why; the tick is resumed with the same arguments once the cause is known."
log_alert_policy input_check_failed "the input data check stopped a tick" \
  "The purchase lines of the new week are outside the expected band (row count or date range), so the tick stopped after step C1 and published nothing."
log_alert_policy quality_alert "a quality flag in 3 consecutive weeks" \
  "A labelled quality rule (recall of a channel, union recall, MAP@12 against the RRF baseline) was flagged in 3 closed weeks in a row. The tick went on and published; a person decides what to do."
log_alert_policy prediction_set_load_failed "the service could not load a prediction set" \
  "The recommendation service could not load the prediction set the current pointer names. It keeps serving the set it had (or answers 503 if it had none)."
# A policy on a METRIC: executions of a Cloud Run job that ended as failed.
FAILED_EXECUTION_NAME="${ALERT_POLICY_PREFIX}an execution of the pipeline job failed"
python3 - "$FAILED_EXECUTION_NAME" "$CHANNEL" > "${POLICY_DIR}/failed_execution.json" <<'PY'
import json, sys
name, channel = sys.argv[1:3]
metric_filter = (
    'resource.type = "cloud_run_job" AND '
    'metric.type = "run.googleapis.com/job/completed_execution_count" AND '
    'metric.labels.result = "failed"'
)
policy = {
    "displayName": name,
    "documentation": {
        "content": "An execution of a Cloud Run job of the project ended as failed: a tick "
        "stopped, or the container could not start. The execution's log says why.",
        "mimeType": "text/markdown",
    },
    "combiner": "OR",
    "conditions": [
        {
            "displayName": "failed executions in a minute > 0",
            "conditionThreshold": {
                "filter": metric_filter,
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_DELTA"}],
                "comparison": "COMPARISON_GT",
                "thresholdValue": 0,
                "duration": "0s",
                "trigger": {"count": 1},
                "evaluationMissingData": "EVALUATION_MISSING_DATA_INACTIVE",
            },
        }
    ],
    "alertStrategy": {"autoClose": "1800s"},
    "notificationChannels": [channel],
}
print(json.dumps(policy, indent=2))
PY
alert_policy "$FAILED_EXECUTION_NAME" "${POLICY_DIR}/failed_execution.json"
rm -r "$POLICY_DIR"
N_POLICIES="$(gcloud monitoring policies list --project "$PROJECT_ID" \
  --filter "displayName:\"${ALERT_POLICY_PREFIX}\"" --format 'value(name)' | wc -l | tr -d ' ')"

section "Done"
cat <<EOF
setup finished: APIs enabled; repository, 4 service accounts, 1 custom role,
workload identity pool and provider exist; the operator's and the deployer's roles are
in place; 1 email notification channel and ${N_POLICIES} alert policies exist (5 expected).

Three GitHub repository SECRETS for the image workflow (.github/workflows/image.yml)
and the deploy workflow (.github/workflows/deploy.yml). M14 needs NO new secret.
Set them at  https://github.com/${GITHUB_REPOSITORY}/settings/secrets/actions
("New repository secret"), or with the gh CLI. They are names, not keys; they are stored
as secrets so that GitHub masks them in the public workflow log.

  GCP_WORKLOAD_IDENTITY_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}
  GCP_DEPLOYER_SERVICE_ACCOUNT   = ${DEPLOYER_SA}
  GCP_IMAGE_REPOSITORY           = ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}
EOF
