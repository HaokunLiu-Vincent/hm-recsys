#!/usr/bin/env bash
# infra/operate.sh: what the OPERATOR does with the cloud project (docs/mlops.md §6, §12; M13).
#
# WHO RUNS IT: someone who may impersonate the operator service account of infra/.env.
# EVERY gcloud command below runs as that account (--impersonate-service-account); the
# person's own account is never used for a cloud action. The operator can create, update
# and run the job and the scheduler job, read logs and metrics, and read and write
# objects of the bucket. It cannot change IAM, billing or APIs: that is infra/setup.sh,
# which only the owner of the project runs.
#
#     bash infra/operate.sh upload-inputs               copy the pipeline's inputs to the bucket
#     bash infra/operate.sh list-inputs                 name, size and md5 of every uploaded input
#     bash infra/operate.sh create-job sha256:<digest>  the Cloud Run job, from an image digest
#     bash infra/operate.sh update-job sha256:<digest>  the same job with another image
#     bash infra/operate.sh execute <arguments>         start one execution; prints its name
#     bash infra/operate.sh wait <execution>            wait until an execution ended
#     bash infra/operate.sh describe <execution>        start, end, state of an execution
#     bash infra/operate.sh logs <execution> [text]     its log lines (optionally those with a text)
#     bash infra/operate.sh create-scheduler            the weekly trigger, paused at once
#     bash infra/operate.sh run-scheduler               start the job through the scheduler now
#     bash infra/operate.sh pause-scheduler             pause the weekly trigger
#     bash infra/operate.sh memory <start> <end>        highest memory use of the job's instance
#                                                       between two UTC times, as a share of its limit
#     bash infra/operate.sh download-records [calibration]   the SMALL records of the state
#                                                       (manifests, step records, tables) to the laptop
#     bash infra/operate.sh state-size                  bytes under each prefix of the bucket
#     bash infra/operate.sh status                      every resource that exists
#     bash infra/operate.sh token                       an access token of the operator (for curl)
#
# `execute` takes the arguments of the container, a script first, for example
#     bash infra/operate.sh execute scripts/m13_volume_check.py --megabytes 1024 --hash-inputs
#     bash infra/operate.sh execute scripts/weekly.py --anchor 2020-07-29 --mode eval
# The job's own arguments (what the scheduler starts) are: scripts/weekly.py --next --mode eval
#
# One execution at a time: `execute` and `run-scheduler` refuse while one is running.

set -euo pipefail

cd "$(dirname "$0")/.."
if [ ! -f infra/.env ]; then
  echo "infra/.env is missing. Create it:  cp infra/env.example infra/.env  and fill it in."
  exit 1
fi
set -a
# shellcheck disable=SC1091
. infra/.env
set +a
: "${PROJECT_ID:?}" "${REGION:?}" "${BUCKET:?}" "${OPERATOR_SERVICE_ACCOUNT:?}"
: "${ARTIFACT_REPOSITORY:?}" "${IMAGE_NAME:?}" "${RUNTIME_SERVICE_ACCOUNT_NAME:?}"
: "${SCHEDULER_SERVICE_ACCOUNT_NAME:?}" "${JOB_NAME:?}" "${SCHEDULER_JOB_NAME:?}"
: "${INPUTS_PREFIX:?}" "${STATE_PREFIX:?}"

RUNTIME_SA="${RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="${SCHEDULER_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE_REPOSITORY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}"

# Where the three prefixes of the bucket appear inside the container. The pipeline is told
# the first two through its environment; the calibration state is the folder next to the
# state (hmrec.paths.calibration_state_dir).
MOUNT_ROOT=/mnt/hmrec
INPUTS_MOUNT="${MOUNT_ROOT}/${INPUTS_PREFIX}"
STATE_MOUNT="${MOUNT_ROOT}/${STATE_PREFIX}"
CALIBRATION_MOUNT="${MOUNT_ROOT}/${STATE_PREFIX}-calibration"
# The user of the image (Dockerfile: useradd --uid 1000). A mounted bucket belongs to
# root unless the mount says otherwise, and the image does not run as root.
IMAGE_UID=1000
# The shape of docs/mlops.md decision D10, and a time limit long enough for a `serve` tick.
JOB_CPU=6
JOB_MEMORY=24Gi
JOB_TIMEOUT=6h
# The weekly trigger: Wednesday 06:00 UTC (an anchor is a Wednesday). It stays paused.
SCHEDULE="0 6 * * 3"

# gcloud as the operator, in the project.
g() {
  gcloud --impersonate-service-account "$OPERATOR_SERVICE_ACCOUNT" --project "$PROJECT_ID" "$@"
}

require_digest() {
  if ! printf '%s' "${1:-}" | grep -Eq '^sha256:[0-9a-f]{64}$'; then
    echo "give the image digest: sha256:<64 hex characters>"
    exit 1
  fi
}

# Stop if an execution of the job is running: one execution at a time.
require_idle() {
  local running
  running="$(g run jobs executions list --job "$JOB_NAME" --region "$REGION" \
    --filter '-status.completionTime:*' --format 'value(metadata.name)' 2>/dev/null || true)"
  if [ -n "$running" ]; then
    echo "an execution is still running: ${running}. One execution at a time."
    exit 1
  fi
}

# A mounted prefix of the bucket. only-dir: the container sees this prefix and nothing
# else of the bucket (never dvc/). uid, gid: the files belong to the image's user.
volume() {
  local name="$1" prefix="$2" readonly="$3"
  printf 'name=%s,type=cloud-storage,bucket=%s,readonly=%s,mount-options=only-dir=%s;uid=%s;gid=%s' \
    "$name" "$BUCKET" "$readonly" "$prefix" "$IMAGE_UID" "$IMAGE_UID"
}

# The flags that define the job; create and update use the same ones.
job_flags() {
  local digest="$1"
  JOB_FLAGS=(
    --region "$REGION"
    --image "${IMAGE_REPOSITORY}@${digest}"
    --service-account "$RUNTIME_SA"
    --cpu "$JOB_CPU" --memory "$JOB_MEMORY"
    --tasks 1 --parallelism 1
    # No automatic retry: a tick that stopped is looked at, then resumed by hand.
    --max-retries 0
    --task-timeout "$JOB_TIMEOUT"
    --add-volume "$(volume inputs "$INPUTS_PREFIX" true)"
    --add-volume-mount "volume=inputs,mount-path=${INPUTS_MOUNT}"
    --add-volume "$(volume state "$STATE_PREFIX" false)"
    --add-volume-mount "volume=state,mount-path=${STATE_MOUNT}"
    --add-volume "$(volume calibration "${STATE_PREFIX}-calibration" false)"
    --add-volume-mount "volume=calibration,mount-path=${CALIBRATION_MOUNT}"
    --set-env-vars "HMREC_PIPELINE_INPUTS_ROOT=${INPUTS_MOUNT},HMREC_PIPELINE_STATE_ROOT=${STATE_MOUNT},HMREC_IMAGE_DIGEST=${digest}"
    --command python
    # "--args=..." in one word: the list starts with a script, but holds words with "--".
    "--args=scripts/weekly.py,--next,--mode,eval"
  )
}

command="${1:-}"
shift || true

case "$command" in
  upload-inputs)
    # rsync copies only what is missing or different, so an interrupted upload is simply
    # started again. Composite uploads are off: an object uploaded in one piece has an md5,
    # which `list-inputs` shows and configs/data_manifest.json records.
    export CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=False
    # --exclude: the Finder's .DS_Store files are not inputs.
    g storage rsync --recursive --exclude '.*\.DS_Store$' \
      data/processed "gs://${BUCKET}/${INPUTS_PREFIX}/data/processed"
    g storage rsync --recursive --exclude '.*\.DS_Store$' \
      data/bootstrap "gs://${BUCKET}/${INPUTS_PREFIX}/data/bootstrap"
    g storage cp --no-clobber artifacts/features/category_codes.json \
      "gs://${BUCKET}/${INPUTS_PREFIX}/artifacts/features/category_codes.json"
    ;;
  list-inputs)
    g storage objects list "gs://${BUCKET}/${INPUTS_PREFIX}/**" \
      --format 'csv[no-heading](name,size,md5_hash)'
    ;;
  create-job)
    require_digest "${1:-}"
    job_flags "$1"
    g run jobs create "$JOB_NAME" "${JOB_FLAGS[@]}"
    ;;
  update-job)
    require_digest "${1:-}"
    job_flags "$1"
    g run jobs update "$JOB_NAME" "${JOB_FLAGS[@]}"
    ;;
  execute)
    if [ "$#" -eq 0 ]; then
      echo "give the container's arguments, a script first (see the head of this file)"
      exit 1
    fi
    require_idle
    # The arguments replace the job's own for this one execution only.
    arguments="$(IFS=,; printf '%s' "$*")"
    g run jobs execute "$JOB_NAME" --region "$REGION" "--args=${arguments}" --async \
      --format 'value(metadata.name)'
    ;;
  wait)
    execution="${1:?give the name of the execution}"
    while true; do
      finished="$(g run jobs executions describe "$execution" --region "$REGION" \
        --format 'value(status.completionTime)')"
      if [ -n "$finished" ]; then
        break
      fi
      sleep 60
    done
    g run jobs executions describe "$execution" --region "$REGION" \
      --format 'value(metadata.name,status.startTime,status.completionTime,status.succeededCount,status.failedCount)'
    ;;
  describe)
    execution="${1:?give the name of the execution}"
    g run jobs executions describe "$execution" --region "$REGION" \
      --format 'yaml(metadata.name,status.startTime,status.completionTime,status.succeededCount,status.failedCount,status.conditions)'
    ;;
  logs)
    execution="${1:?give the name of the execution}"
    filter="resource.type=\"cloud_run_job\" AND labels.\"run.googleapis.com/execution_name\"=\"${execution}\""
    if [ -n "${2:-}" ]; then
      filter="${filter} AND textPayload:\"${2}\""
    fi
    # --freshness: without it only the last day is searched.
    g logging read "$filter" --freshness 60d --order asc --limit 5000 \
      --format 'value(textPayload)'
    ;;
  create-scheduler)
    # The scheduler calls the Cloud Run API "run this job" as the scheduler account, which
    # holds the single permission to do that. An empty request ("{}"): the job runs with
    # its own arguments (scripts/weekly.py --next --mode eval), the same every week.
    g scheduler jobs create http "$SCHEDULER_JOB_NAME" --location "$REGION" \
      --schedule "$SCHEDULE" --time-zone "Etc/UTC" \
      --uri "https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_NAME}:run" \
      --http-method POST \
      --headers "Content-Type=application/json" --message-body "{}" \
      --oauth-service-account-email "$SCHEDULER_SA" \
      --max-retry-attempts 0 \
      --description "Weekly tick of the pipeline: the job's own arguments, the anchor from the state"
    # Paused at once: the replay is driven by hand, one tick at a time.
    g scheduler jobs pause "$SCHEDULER_JOB_NAME" --location "$REGION"
    ;;
  run-scheduler)
    require_idle
    # "Force run": the scheduler sends its request now, also while it is paused.
    g scheduler jobs run "$SCHEDULER_JOB_NAME" --location "$REGION"
    ;;
  pause-scheduler)
    g scheduler jobs pause "$SCHEDULER_JOB_NAME" --location "$REGION"
    ;;
  memory)
    start="${1:?give the start, e.g. 2026-11-22T09:00:00Z}"
    end="${2:?give the end, e.g. 2026-11-22T09:20:00Z}"
    # Cloud Monitoring: the memory use of the job's container as a share of its limit, the
    # 99th percentile of every minute; the highest of those minutes is printed. The request
    # goes to the Monitoring API of the project with a token of the operator.
    curl --silent --fail --get \
      --header "Authorization: Bearer $(g auth print-access-token 2>/dev/null)" \
      --data-urlencode "filter=metric.type=\"run.googleapis.com/container/memory/utilizations\" AND resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"${JOB_NAME}\"" \
      --data-urlencode "interval.startTime=${start}" \
      --data-urlencode "interval.endTime=${end}" \
      --data-urlencode "aggregation.alignmentPeriod=60s" \
      --data-urlencode "aggregation.perSeriesAligner=ALIGN_PERCENTILE_99" \
      "https://monitoring.googleapis.com/v3/projects/${PROJECT_ID}/timeSeries" |
      python3 -c 'import json, sys
series = json.load(sys.stdin).get("timeSeries", [])
values = [point["value"]["doubleValue"] for one in series for point in one["points"]]
print(max(values) if values else "no data")'
    ;;
  download-records)
    # Only what is small: JSON records and CSV tables. Candidates, features, samples,
    # models, vectors and logs stay in the bucket (docs/mlops.md 12, guardrail 5).
    prefix="$STATE_PREFIX"
    if [ "${1:-}" = "calibration" ]; then
      prefix="${STATE_PREFIX}-calibration"
    fi
    target="artifacts/m13_cloud/${prefix}"
    mkdir -p "$target"
    g storage rsync --recursive \
      --exclude '.*\.parquet$|.*\.npy$|.*\.pt$|.*\.bin$|.*\.txt$|.*\.log$|.*/\.staging/.*|.*/\.failed/.*|^\.volume-check/.*' \
      "gs://${BUCKET}/${prefix}" "$target"
    echo "records of ${prefix} are in ${target}"
    ;;
  state-size)
    for prefix in "$INPUTS_PREFIX" "$STATE_PREFIX" "${STATE_PREFIX}-calibration" dvc; do
      g storage du --summarize "gs://${BUCKET}/${prefix}" || true
    done
    ;;
  status)
    echo "== Cloud Run jobs"
    g run jobs list --region "$REGION" || true
    echo "== executions of ${JOB_NAME}"
    g run jobs executions list --job "$JOB_NAME" --region "$REGION" --limit 30 || true
    echo "== Cloud Scheduler jobs"
    g scheduler jobs list --location "$REGION" || true
    echo "== images"
    g artifacts docker images list "$IMAGE_REPOSITORY" --include-tags || true
    ;;
  token)
    g auth print-access-token
    ;;
  *)
    sed -n '2,31p' "$0"
    exit 1
    ;;
esac
