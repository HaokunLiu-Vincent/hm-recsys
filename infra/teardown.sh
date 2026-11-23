#!/usr/bin/env bash
# infra/teardown.sh: remove what infra/setup.sh and infra/operate.sh created
# (docs/mlops.md §12 guardrail 6; M15 runs it).
#
# WHO RUNS IT: the owner of the project, like setup.sh.
#
#     bash infra/teardown.sh                    # jobs, service, images, accounts, roles, alerts
#     bash infra/teardown.sh --delete-objects   # also the pipeline's objects in the bucket
#
# WHAT IT REMOVES
#   - the Cloud Scheduler job, the Cloud Run jobs (with their executions) and the Cloud
#     Run services (the recommendation service; a job or a service of a drill, if one is left);
#   - the Artifact Registry repository with every image in it;
#   - the runtime, scheduler, deployer and service-runtime accounts and their role bindings;
#   - the custom role, the workload identity provider and pool;
#   - the roles setup.sh gave the operator account;
#   - the alert policies of setup.sh and the email notification channel;
#   - with --delete-objects: the inputs, state, calibration and drill prefixes of the bucket.
#
# WHAT IT NEVER TOUCHES: the bucket itself, the DVC data under dvc/, the operator account
# and its access to the bucket, the budget, and the enabled APIs (an enabled API costs
# nothing, and another use of the project may need it).
#
# A thing that does not exist is skipped, so the script can be run again.

set -euo pipefail

cd "$(dirname "$0")"
if [ ! -f .env ]; then
  echo "infra/.env is missing; nothing is removed."
  exit 1
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a
: "${PROJECT_ID:?}" "${REGION:?}" "${BUCKET:?}" "${OPERATOR_SERVICE_ACCOUNT:?}"
: "${ARTIFACT_REPOSITORY:?}" "${RUNTIME_SERVICE_ACCOUNT_NAME:?}"
: "${SCHEDULER_SERVICE_ACCOUNT_NAME:?}" "${DEPLOYER_SERVICE_ACCOUNT_NAME:?}"
: "${JOB_STARTER_ROLE:?}" "${WIF_POOL:?}" "${WIF_PROVIDER:?}" "${JOB_NAME:?}"
: "${SCHEDULER_JOB_NAME:?}" "${INPUTS_PREFIX:?}" "${STATE_PREFIX:?}"
: "${SERVICE_NAME:?}" "${SERVICE_RUNTIME_SERVICE_ACCOUNT_NAME:?}" "${ALERT_CHANNEL_NAME:?}"

DELETE_OBJECTS=false
if [ "${1:-}" = "--delete-objects" ]; then
  DELETE_OBJECTS=true
elif [ -n "${1:-}" ]; then
  echo "unknown argument: $1"
  exit 1
fi

RUNTIME_SA="${RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="${SCHEDULER_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="${DEPLOYER_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SERVICE_RUNTIME_SA="${SERVICE_RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
OPERATOR="serviceAccount:${OPERATOR_SERVICE_ACCOUNT}"
# The display names of the alert policies of setup.sh start with this.
ALERT_POLICY_PREFIX="hmrec: "
MONITORING_API="https://monitoring.googleapis.com/v3/projects/${PROJECT_ID}"

section() { printf '\n==== %s\n' "$*"; }
# Print a command and run it; a failure (most often: the thing does not exist) is shown
# and the script goes on, so that one missing resource does not keep the others alive.
attempt() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@" || echo "  (skipped: the command failed, most likely because this does not exist)"
}

echo "This removes the pipeline's cloud resources from project ${PROJECT_ID}."
if [ "$DELETE_OBJECTS" = true ]; then
  echo "It ALSO deletes gs://${BUCKET}/${INPUTS_PREFIX}/, ${STATE_PREFIX}/, ${STATE_PREFIX}-calibration/ and ${STATE_PREFIX}-drill/."
fi
echo "The bucket, its dvc/ prefix and the operator account are kept."
printf 'Type the project id to go on: '
read -r answer
if [ "$answer" != "$PROJECT_ID" ]; then
  echo "not confirmed; nothing was removed."
  exit 1
fi

section "1. The scheduler job, the Cloud Run jobs and the Cloud Run services"
attempt gcloud scheduler jobs delete "$SCHEDULER_JOB_NAME" --project "$PROJECT_ID" \
  --location "$REGION" --quiet
# "-drill": the job and the service of a drill (infra/operate.sh), if one was left behind.
for job in "$JOB_NAME" "${JOB_NAME}-drill"; do
  attempt gcloud run jobs delete "$job" --project "$PROJECT_ID" --region "$REGION" --quiet
done
for service in "$SERVICE_NAME" "${SERVICE_NAME}-drill"; do
  attempt gcloud run services delete "$service" --project "$PROJECT_ID" --region "$REGION" --quiet
done

section "2. The image repository (with every image)"
attempt gcloud artifacts repositories delete "$ARTIFACT_REPOSITORY" --project "$PROJECT_ID" \
  --location "$REGION" --quiet

section "3. The roles of the operator account"
for role in roles/run.developer roles/run.invoker roles/cloudscheduler.admin \
  roles/logging.viewer roles/monitoring.viewer roles/serviceusage.serviceUsageConsumer; do
  attempt gcloud projects remove-iam-policy-binding "$PROJECT_ID" --member "$OPERATOR" \
    --role "$role" --condition=None --format=none
done

section "4. The service accounts (their bindings on the bucket and the project first)"
attempt gcloud storage buckets remove-iam-policy-binding "gs://${BUCKET}" --project "$PROJECT_ID" \
  --member "serviceAccount:${RUNTIME_SA}" --role roles/storage.objectUser --format=none
attempt gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${RUNTIME_SA}" --role roles/logging.logWriter \
  --condition=None --format=none
attempt gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SCHEDULER_SA}" \
  --role "projects/${PROJECT_ID}/roles/${JOB_STARTER_ROLE}" --condition=None --format=none
attempt gcloud storage buckets remove-iam-policy-binding "gs://${BUCKET}" --project "$PROJECT_ID" \
  --member "serviceAccount:${SERVICE_RUNTIME_SA}" --role roles/storage.objectViewer --format=none
attempt gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SERVICE_RUNTIME_SA}" --role roles/logging.logWriter \
  --condition=None --format=none
attempt gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${DEPLOYER_SA}" --role roles/run.developer \
  --condition=None --format=none
# Deleting an account also removes the "may act as it" bindings that were set on it.
for account in "$RUNTIME_SA" "$SCHEDULER_SA" "$DEPLOYER_SA" "$SERVICE_RUNTIME_SA"; do
  attempt gcloud iam service-accounts delete "$account" --project "$PROJECT_ID" --quiet
done

section "5. The custom role and Workload Identity Federation"
attempt gcloud iam roles delete "$JOB_STARTER_ROLE" --project "$PROJECT_ID" --quiet
attempt gcloud iam workload-identity-pools providers delete "$WIF_PROVIDER" \
  --project "$PROJECT_ID" --location global --workload-identity-pool "$WIF_POOL" --quiet
attempt gcloud iam workload-identity-pools delete "$WIF_POOL" --project "$PROJECT_ID" \
  --location global --quiet

section "6. The alert policies and the email notification channel"
# Policies first: a channel that a policy still uses cannot be deleted.
for policy in $(gcloud monitoring policies list --project "$PROJECT_ID" \
  --filter "displayName:\"${ALERT_POLICY_PREFIX}\"" --format 'value(name)' 2>/dev/null); do
  attempt gcloud monitoring policies delete "$policy" --project "$PROJECT_ID" --quiet
done
# The channel is removed through the Monitoring API (this version of gcloud has no
# released command for channels). The token goes to curl on its standard input.
monitoring_api() {
  local method="$1" url="$2"
  printf 'header = "Authorization: Bearer %s"\n' "$(gcloud auth print-access-token)" |
    curl --silent --show-error --fail --config - --request "$method" \
      --header "x-goog-user-project: ${PROJECT_ID}" "$url"
}
CHANNELS="$(monitoring_api GET "${MONITORING_API}/notificationChannels" | python3 -c '
import json, sys
wanted = sys.argv[1]
for channel in json.load(sys.stdin).get("notificationChannels", []):
    if channel.get("displayName") == wanted and channel.get("type") == "email":
        print(channel["name"])' "$ALERT_CHANNEL_NAME" || true)"
for channel in $CHANNELS; do
  echo "+ delete the notification channel ${ALERT_CHANNEL_NAME}"
  monitoring_api DELETE "https://monitoring.googleapis.com/v3/${channel}" >/dev/null ||
    echo "  (skipped: the channel could not be deleted)"
done

if [ "$DELETE_OBJECTS" = true ]; then
  section "7. The pipeline's objects in the bucket (never dvc/)"
  for prefix in "$INPUTS_PREFIX" "$STATE_PREFIX" "${STATE_PREFIX}-calibration" "${STATE_PREFIX}-drill"; do
    attempt gcloud storage rm --recursive "gs://${BUCKET}/${prefix}/" --project "$PROJECT_ID"
  done
fi

section "What is left"
echo "Kept on purpose: the bucket gs://${BUCKET} (dvc/ and, without --delete-objects, the"
echo "pipeline's prefixes), the operator account, the budget, the enabled APIs."
echo "Check that nothing billable remains:"
attempt gcloud run jobs list --project "$PROJECT_ID" --region "$REGION"
attempt gcloud run services list --project "$PROJECT_ID" --region "$REGION"
attempt gcloud scheduler jobs list --project "$PROJECT_ID" --location "$REGION"
attempt gcloud artifacts repositories list --project "$PROJECT_ID" --location "$REGION"
attempt gcloud storage du --summarize "gs://${BUCKET}" --project "$PROJECT_ID"
