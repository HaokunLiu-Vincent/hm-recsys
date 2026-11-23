#!/usr/bin/env bash
# infra/operate.sh: what the OPERATOR does with the cloud project (docs/mlops.md §6, §9, §12;
# M13, M14).
#
# WHO RUNS IT: someone who may impersonate the operator service account of infra/.env.
# EVERY gcloud command below runs as that account (--impersonate-service-account); the
# person's own account is never used for a cloud action. The operator can create, update
# and run the job and the scheduler job, create, update and call the recommendation
# service, read logs and metrics, and read and write objects of the bucket. It cannot
# change IAM, billing or APIs: that is infra/setup.sh, which only the owner of the
# project runs.
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
#                                                       (resume, run, pause again)
#     bash infra/operate.sh pause-scheduler             pause the weekly trigger
#     bash infra/operate.sh memory <start> <end>        highest memory use of the job's instance
#                                                       between two UTC times, as a share of its limit
#     bash infra/operate.sh download-records [calibration]   the SMALL records of the state
#                                                       (manifests, step records, tables) to the laptop
#     bash infra/operate.sh state-size                  bytes under each prefix of the bucket
#     bash infra/operate.sh status                      every resource that exists
#     bash infra/operate.sh token                       an access token of the operator (for curl)
#
# The recommendation service (M14):
#     bash infra/operate.sh create-service [sha256:<digest>]   the Cloud Run service; without a
#                                                       digest it starts from Google's placeholder
#                                                       image and the deploy workflow puts the
#                                                       real image in (.github/workflows/deploy.yml)
#     bash infra/operate.sh update-service sha256:<digest>     the same service with another image
#     bash infra/operate.sh service-url                 the URL of the service
#     bash infra/operate.sh call <path>                 GET a path of the service with an identity
#                                                       token of the operator, e.g. /healthz
#     bash infra/operate.sh service-logs [text]         log lines of the service (optionally with a text)
#     bash infra/operate.sh pointer                     state/current.json as it is in the bucket
#     bash infra/operate.sh serving-set <anchor>        build the serving form of a prediction set
#                                                       that was published before M14 (one execution)
#     bash infra/operate.sh rollback <previous|next|YYYY-MM-DD>   move the `current` pointer
#                                                       (one execution; the service follows it)
#     bash infra/operate.sh service-request-logs <start> <end>   the service's request lines
#                                                       between two UTC times, as JSON
#     bash infra/operate.sh service-metric <metric> <aligner> <start> <end> [service]
#                                                       one Cloud Monitoring metric of the service,
#                                                       a value per minute (memory, instances)
#
# Alert drills (M14): a job, a service and a state prefix of their own ("-drill"), so that
# the replay's state is never touched. Every alert line of a drill carries a note.
#     bash infra/operate.sh create-drill-job sha256:<digest>
#     bash infra/operate.sh execute-drill <case> <inputs|drill-inputs> <arguments>
#                                                       one execution with the state
#                                                       <drill prefix>/<case>; "drill-inputs"
#                                                       reads <drill prefix>/<case>-inputs
#                                                       instead of the real inputs
#     bash infra/operate.sh create-drill-service sha256:<digest> <case>
#     bash infra/operate.sh call-drill <path>
#     bash infra/operate.sh delete-drill                the drill job and the drill service
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
: "${SERVICE_NAME:?}" "${SERVICE_RUNTIME_SERVICE_ACCOUNT_NAME:?}"

RUNTIME_SA="${RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="${SCHEDULER_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SERVICE_RUNTIME_SA="${SERVICE_RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE_REPOSITORY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}"
# The service's image lies next to the pipeline's, under "<image>-serving" (the deploy
# workflow pushes it there).
SERVICE_IMAGE_REPOSITORY="${IMAGE_REPOSITORY}-serving"

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

# The recommendation service (docs/mlops.md §9). No always-on resource: it scales to zero
# and to at most SERVICE_MAX_INSTANCES. 1 vCPU; the memory holds the `serve` prediction
# set (about 160 MB in memory, twice that for the moment of a switch) with room to spare.
SERVICE_CPU=1
SERVICE_MEMORY=1Gi
SERVICE_MAX_INSTANCES=2
# Seconds between two looks of the service at the `current` pointer.
SERVICE_POLL_SECONDS=60
# The port of Dockerfile.serving.
SERVICE_PORT=8080
# What a service is created from before the first deployment: Google's sample container.
# It listens on the port and ignores the mounts; the deploy workflow replaces it.
PLACEHOLDER_IMAGE="us-docker.pkg.dev/cloudrun/container/hello"

# The alert drills of M14 (docs/mlops.md §10): a job, a service and a prefix of the
# bucket of their own. The note is added to every alert line a drill writes
# (hmrec.structured_logs; no comma in it: gcloud separates variables with commas).
DRILL_JOB_NAME="${JOB_NAME}-drill"
DRILL_SERVICE_NAME="${SERVICE_NAME}-drill"
DRILL_PREFIX="${STATE_PREFIX}-drill"
DRILL_MOUNT="${MOUNT_ROOT}/${DRILL_PREFIX}"
DRILL_NOTE="provoked on purpose - alert drill of M14 - not a real failure"
DRILL_TIMEOUT=30m

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
  local job="${1:-$JOB_NAME}"
  local running
  running="$(g run jobs executions list --job "$job" --region "$REGION" \
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

# The flags that define the service; create and update use the same ones.
service_flags() {
  local image="$1"
  SERVICE_FLAGS=(
    --region "$REGION"
    --image "$image"
    --service-account "$SERVICE_RUNTIME_SA"
    # Authenticated calls only: nothing is public (docs/mlops.md §9).
    --no-allow-unauthenticated
    --cpu "$SERVICE_CPU" --memory "$SERVICE_MEMORY"
    --min-instances 0 --max-instances "$SERVICE_MAX_INSTANCES"
    --port "$SERVICE_PORT"
    # A mounted bucket needs the second-generation execution environment.
    --execution-environment gen2
    # More CPU while an instance starts: the prediction set is loaded then.
    --cpu-boost
    # The state prefix, READ-ONLY: the service can change nothing of the state.
    --add-volume "$(volume state "$STATE_PREFIX" true)"
    --add-volume-mount "volume=state,mount-path=${STATE_MOUNT}"
    --set-env-vars "HMREC_SERVING_STATE_ROOT=${STATE_MOUNT},HMREC_SERVING_POLL_SECONDS=${SERVICE_POLL_SECONDS}"
  )
}

service_url() {
  g run services describe "$SERVICE_NAME" --region "$REGION" --format 'value(status.url)'
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
    # Cloud Scheduler runs a job by hand only while it is ENABLED (a paused job answers
    # "Job.state must be ENABLED for RunJob"). So: resume, send the request now, pause
    # again. The job is enabled for a few seconds; its weekly time is days away.
    g scheduler jobs resume "$SCHEDULER_JOB_NAME" --location "$REGION" --format none
    status=0
    g scheduler jobs run "$SCHEDULER_JOB_NAME" --location "$REGION" || status=$?
    g scheduler jobs pause "$SCHEDULER_JOB_NAME" --location "$REGION" --format 'value(state)'
    exit "$status"
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
  create-service)
    # `run deploy` creates the service, or changes it if it exists. With a digest the
    # service runs that image of the service's repository; without one it starts from
    # the placeholder, and the first run of the deploy workflow puts the real image in.
    if [ -n "${1:-}" ]; then
      require_digest "$1"
      service_flags "${SERVICE_IMAGE_REPOSITORY}@${1}"
    else
      service_flags "$PLACEHOLDER_IMAGE"
    fi
    g run deploy "$SERVICE_NAME" "${SERVICE_FLAGS[@]}"
    ;;
  update-service)
    require_digest "${1:-}"
    service_flags "${SERVICE_IMAGE_REPOSITORY}@${1}"
    g run deploy "$SERVICE_NAME" "${SERVICE_FLAGS[@]}"
    ;;
  service-url)
    service_url
    ;;
  call)
    path="${1:?give a path of the service, e.g. /healthz}"
    url="$(service_url)"
    # An identity token of the operator for this one service. It is given to curl on its
    # standard input, so it is not in a command line.
    printf 'header = "Authorization: Bearer %s"\n' "$(g auth print-identity-token --audiences "$url" 2>/dev/null)" |
      curl --silent --show-error --config - "${url}${path}"
    echo
    ;;
  service-logs)
    filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\""
    if [ -n "${1:-}" ]; then
      filter="${filter} AND \"${1}\""
    fi
    g logging read "$filter" --freshness 7d --order asc --limit 2000 \
      --format 'value(timestamp,jsonPayload,textPayload)'
    ;;
  pointer)
    g storage cat "gs://${BUCKET}/${STATE_PREFIX}/state/current.json"
    ;;
  serving-set)
    anchor="${1:?give the anchor of the prediction set, e.g. 2020-09-23}"
    exec bash "$0" execute scripts/m14_serving_set.py --anchor "$anchor"
    ;;
  rollback)
    target="${1:?give the target: previous, next or an anchor YYYY-MM-DD}"
    # One execution of the job, which has the state mounted: it checks the target's
    # checksums there and replaces state/current.json. The service follows the pointer.
    exec bash "$0" execute scripts/m14_rollback.py --to "$target"
    ;;
  service-request-logs)
    start="${1:?give the start, e.g. 2026-11-23T09:00:00Z}"
    end="${2:?give the end, e.g. 2026-11-23T09:20:00Z}"
    service="${3:-$SERVICE_NAME}"
    filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${service}\""
    filter="${filter} AND jsonPayload.hmrec_log=\"request\""
    filter="${filter} AND timestamp>=\"${start}\" AND timestamp<=\"${end}\""
    g logging read "$filter" --order asc --limit 20000 --format json 2>/dev/null
    ;;
  service-metric)
    metric="${1:?give the metric, e.g. run.googleapis.com/container/memory/utilizations}"
    aligner="${2:?give the aligner, e.g. ALIGN_PERCENTILE_99 or ALIGN_MAX}"
    start="${3:?give the start, e.g. 2026-11-23T09:00:00Z}"
    end="${4:?give the end, e.g. 2026-11-23T09:20:00Z}"
    service="${5:-$SERVICE_NAME}"
    # One value per minute and time series of the metric, with the series' labels.
    printf 'header = "Authorization: Bearer %s"\n' "$(g auth print-access-token 2>/dev/null)" |
      curl --silent --fail --get --config - \
        --data-urlencode "filter=metric.type=\"${metric}\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${service}\"" \
        --data-urlencode "interval.startTime=${start}" \
        --data-urlencode "interval.endTime=${end}" \
        --data-urlencode "aggregation.alignmentPeriod=60s" \
        --data-urlencode "aggregation.perSeriesAligner=${aligner}" \
        "https://monitoring.googleapis.com/v3/projects/${PROJECT_ID}/timeSeries" |
      python3 -c 'import json, sys
for series in json.load(sys.stdin).get("timeSeries", []):
    labels = {**series.get("metric", {}).get("labels", {})}
    labels["revision"] = series["resource"]["labels"].get("revision_name", "")
    for point in reversed(series["points"]):
        value = point["value"]
        number = value.get("doubleValue", value.get("int64Value"))
        print(point["interval"]["endTime"], number, json.dumps(labels, sort_keys=True))'
    ;;
  create-drill-job)
    require_digest "${1:-}"
    # The real inputs read-only and the drill prefix writable; NOT the state of the
    # replay. The state root is given per execution (execute-drill).
    g run jobs deploy "$DRILL_JOB_NAME" \
      --region "$REGION" \
      --image "${IMAGE_REPOSITORY}@${1}" \
      --service-account "$RUNTIME_SA" \
      --cpu "$JOB_CPU" --memory "$JOB_MEMORY" \
      --tasks 1 --parallelism 1 --max-retries 0 \
      --task-timeout "$DRILL_TIMEOUT" \
      --add-volume "$(volume inputs "$INPUTS_PREFIX" true)" \
      --add-volume-mount "volume=inputs,mount-path=${INPUTS_MOUNT}" \
      --add-volume "$(volume drill "$DRILL_PREFIX" false)" \
      --add-volume-mount "volume=drill,mount-path=${DRILL_MOUNT}" \
      --set-env-vars "HMREC_PIPELINE_INPUTS_ROOT=${INPUTS_MOUNT},HMREC_PIPELINE_STATE_ROOT=${DRILL_MOUNT}/unset,HMREC_IMAGE_DIGEST=${1},HMREC_ALERT_NOTE=${DRILL_NOTE}" \
      --command python \
      "--args=scripts/weekly.py,--next,--mode,eval"
    ;;
  execute-drill)
    case_name="${1:?give the name of the drill case (a folder of the drill prefix)}"
    inputs="${2:?give inputs (the real inputs) or drill-inputs (<case>-inputs of the drill prefix)}"
    shift 2
    if [ "$#" -eq 0 ]; then
      echo "give the container's arguments, a script first"
      exit 1
    fi
    require_idle "$DRILL_JOB_NAME"
    variables="HMREC_PIPELINE_STATE_ROOT=${DRILL_MOUNT}/${case_name}"
    if [ "$inputs" = "drill-inputs" ]; then
      variables="${variables},HMREC_PIPELINE_INPUTS_ROOT=${DRILL_MOUNT}/${case_name}-inputs"
    elif [ "$inputs" != "inputs" ]; then
      echo "the second argument is inputs or drill-inputs"
      exit 1
    fi
    arguments="$(IFS=,; printf '%s' "$*")"
    g run jobs execute "$DRILL_JOB_NAME" --region "$REGION" "--args=${arguments}" \
      --update-env-vars "$variables" --async --format 'value(metadata.name)'
    ;;
  create-drill-service)
    require_digest "${1:-}"
    case_name="${2:?give the name of the drill case (a folder of the drill prefix)}"
    # The service's own flags, but the DRILL prefix mounted (read-only) and a state root
    # inside it: this service never sees the replay's state.
    g run deploy "$DRILL_SERVICE_NAME" \
      --region "$REGION" \
      --image "${SERVICE_IMAGE_REPOSITORY}@${1}" \
      --service-account "$SERVICE_RUNTIME_SA" \
      --no-allow-unauthenticated \
      --cpu "$SERVICE_CPU" --memory "$SERVICE_MEMORY" \
      --min-instances 0 --max-instances 1 \
      --port "$SERVICE_PORT" \
      --execution-environment gen2 \
      --add-volume "$(volume drill "$DRILL_PREFIX" true)" \
      --add-volume-mount "volume=drill,mount-path=${DRILL_MOUNT}" \
      --set-env-vars "HMREC_SERVING_STATE_ROOT=${DRILL_MOUNT}/${case_name},HMREC_SERVING_POLL_SECONDS=${SERVICE_POLL_SECONDS},HMREC_ALERT_NOTE=${DRILL_NOTE}"
    ;;
  call-drill)
    path="${1:?give a path of the drill service, e.g. /healthz}"
    url="$(g run services describe "$DRILL_SERVICE_NAME" --region "$REGION" --format 'value(status.url)')"
    printf 'header = "Authorization: Bearer %s"\n' "$(g auth print-identity-token --audiences "$url" 2>/dev/null)" |
      curl --silent --show-error --config - "${url}${path}"
    echo
    ;;
  delete-drill)
    g run services delete "$DRILL_SERVICE_NAME" --region "$REGION" --quiet || true
    g run jobs delete "$DRILL_JOB_NAME" --region "$REGION" --quiet || true
    ;;
  status)
    echo "== Cloud Run jobs"
    g run jobs list --region "$REGION" || true
    echo "== Cloud Run services"
    g run services list --region "$REGION" || true
    echo "== executions of ${JOB_NAME}"
    g run jobs executions list --job "$JOB_NAME" --region "$REGION" --limit 30 || true
    echo "== Cloud Scheduler jobs"
    g scheduler jobs list --location "$REGION" || true
    echo "== images"
    g artifacts docker images list "$IMAGE_REPOSITORY" --include-tags || true
    g artifacts docker images list "$SERVICE_IMAGE_REPOSITORY" --include-tags || true
    ;;
  token)
    g auth print-access-token
    ;;
  *)
    sed -n '2,71p' "$0"
    exit 1
    ;;
esac
