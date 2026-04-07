#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TEI_GPUS="${TEI_GPUS:-}"
QWN_GPUS="${QWN_GPUS:-}"
TEI_MACHINE_URL="${TEI_MACHINE_URL:-http://127.0.0.1:28800}"
QWN_MACHINE_URL="${QWN_MACHINE_URL:-http://127.0.0.1:27800}"
TEI_MACHINE_PORT="${TEI_MACHINE_PORT:-28800}"
QWN_MACHINE_PORT="${QWN_MACHINE_PORT:-27800}"
TEI_BACKEND_BASE_PORT="${TEI_BACKEND_BASE_PORT:-28880}"
QWN_BACKEND_BASE_PORT="${QWN_BACKEND_BASE_PORT:-27880}"
TEI_LSH_INPUTS="${TEI_LSH_INPUTS:-240}"
TEI_LSH_ROUNDS="${TEI_LSH_ROUNDS:-3}"
TEI_BENCH_SAMPLES="${TEI_BENCH_SAMPLES:-0}"
QWN_BENCH_SAMPLES="${QWN_BENCH_SAMPLES:-20}"
QWN_BENCH_MAX_TOKENS="${QWN_BENCH_MAX_TOKENS:-128}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-600}"
QWN_PROJECT_NAME="${QWN_PROJECT_NAME:-qwn-uniform-awq}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/runs/recovery/results}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TEI_MACHINE_LOG="$RESULTS_DIR/tei_machine_${TIMESTAMP}.log"
TEI_MACHINE_PID_FILE="$RESULTS_DIR/tei_machine_launcher_${TIMESTAMP}.pid"
RECOVERY_SUCCESS=0
TEI_STACK_STARTED=0
QWN_STACK_STARTED=0

mkdir -p "$RESULTS_DIR" "$REPO_ROOT/runs/teis/results" "$REPO_ROOT/runs/qwns/results"

log() {
	printf '[recovery] %s\n' "$*"
}

cleanup_on_exit() {
	local exit_code=$?
	if [[ "$RECOVERY_SUCCESS" -eq 1 && "$exit_code" -eq 0 ]]; then
		return
	fi

	log "Recovery failed or was interrupted; cleaning up partial TEI/QWN state"
	qwn_cmd machine stop >/dev/null 2>&1 || true
	if [[ "$QWN_STACK_STARTED" -eq 1 ]]; then
		qwn_cmd compose down -j "$QWN_PROJECT_NAME" >/dev/null 2>&1 || true
		qwn_cmd compose down >/dev/null 2>&1 || true
	fi

	if [[ -f "$TEI_MACHINE_PID_FILE" ]]; then
		local tei_pid
		tei_pid="$(cat "$TEI_MACHINE_PID_FILE" 2>/dev/null || true)"
		if [[ -n "$tei_pid" ]] && kill -0 "$tei_pid" 2>/dev/null; then
			kill "$tei_pid" >/dev/null 2>&1 || true
		fi
	fi

	if [[ "$TEI_STACK_STARTED" -eq 1 ]]; then
		tei_cmd compose down >/dev/null 2>&1 || true
	fi
}

trap cleanup_on_exit EXIT INT TERM

tei_cmd() {
	if command -v tei >/dev/null 2>&1; then
		tei "$@"
	else
		python -m tfmx.teis.cli "$@"
	fi
}

qwn_cmd() {
	if command -v qwn >/dev/null 2>&1; then
		qwn "$@"
	else
		python -m tfmx.qwns.cli "$@"
	fi
}

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		printf '[recovery] missing required command: %s\n' "$1" >&2
		exit 1
	fi
}

count_csv_items() {
	local csv="$1"
	local count=0
	local value
	IFS=',' read -r -a values <<< "$csv"
	for value in "${values[@]}"; do
		value="${value//[[:space:]]/}"
		[[ -z "$value" ]] && continue
		count=$((count + 1))
	done
	printf '%s\n' "$count"
}

detect_visible_gpu_csv() {
	nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | paste -sd, -
}

preflight_guest_gpu_state() {
	local first_gpu="$1"
	local smi_out
	local probe_out

	log "Preflight: checking guest NVIDIA state before fresh restart"
	smi_out="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>&1 || true)"
	if printf '%s' "$smi_out" | grep -Eqi 'Unknown Error|Unable to determine the device handle'; then
		printf '%s\n' "$smi_out" >&2
		printf '[recovery] guest NVIDIA state is degraded; fresh CUDA init is currently unsafe. Aborting before starting TEI/QWN to avoid CPU fallback and crash loops.\n' >&2
		printf '[recovery] current evidence shows that even explicit single-GPU container probes fail while this broken device remains visible to the guest.\n' >&2
		return 1
	fi

	probe_out="$(nvidia-smi -i "$first_gpu" --query-gpu=name,pci.bus_id --format=csv,noheader 2>&1 || true)"
	if printf '%s' "$probe_out" | grep -Eqi 'Unknown Error|Unable to determine the device handle|No devices were found'; then
		printf '%s\n' "$probe_out" >&2
		printf '[recovery] selected GPU %s is not healthy enough for a fresh restart.\n' "$first_gpu" >&2
		return 1
	fi
}

wait_backend_health() {
	local label="$1"
	local base_port="$2"
	local gpu_csv="$3"
	local timeout_sec="$4"
	python3 - "$label" "$base_port" "$gpu_csv" "$timeout_sec" <<'PY'
import sys
import time
import urllib.request

label = sys.argv[1]
base_port = int(sys.argv[2])
gpu_csv = sys.argv[3]
timeout_sec = float(sys.argv[4])
gpus = [int(part.strip()) for part in gpu_csv.split(',') if part.strip()]
ports = [base_port + gpu for gpu in gpus]
deadline = time.time() + timeout_sec
last_state = None
last_print_at = 0.0

while time.time() < deadline:
	statuses = []
	all_ok = True
	for port in ports:
		ok = False
		try:
			with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as resp:
				ok = 200 <= resp.status < 300
		except Exception:
			ok = False
		statuses.append(f'{port}:{"ok" if ok else "wait"}')
		all_ok = all_ok and ok

	state = ' '.join(statuses)
	now = time.time()
	if state != last_state or (now - last_print_at) >= 10:
		print(f'[wait] {label} backend health {state}', flush=True)
		last_state = state
		last_print_at = now

	if all_ok:
		print(f'[okay] {label} backends healthy', flush=True)
		sys.exit(0)

	time.sleep(2)

print(f'[error] {label} backend health timed out after {timeout_sec:.0f}s', file=sys.stderr)
sys.exit(1)
PY
}

wait_machine_health() {
	local label="$1"
	local health_url="$2"
	local expected_total="$3"
	local timeout_sec="$4"
	python3 - "$label" "$health_url" "$expected_total" "$timeout_sec" <<'PY'
import json
import sys
import time
import urllib.request

label = sys.argv[1]
health_url = sys.argv[2]
expected_total = int(sys.argv[3])
timeout_sec = float(sys.argv[4])
deadline = time.time() + timeout_sec
last_state = None
last_print_at = 0.0

while time.time() < deadline:
	try:
		with urllib.request.urlopen(health_url, timeout=5) as resp:
			body = json.loads(resp.read().decode())
		state = f"{body.get('healthy')}/{body.get('total')} {body.get('status')}"
		now = time.time()
		if state != last_state or (now - last_print_at) >= 10:
			print(f'[wait] {label} machine health {state}', flush=True)
			last_state = state
			last_print_at = now
		if (
			body.get('status') == 'healthy'
			and body.get('healthy') == expected_total
			and body.get('total') == expected_total
		):
			print(f'[okay] {label} machine healthy {expected_total}/{expected_total}', flush=True)
			sys.exit(0)
	except Exception as exc:
		state = f'wait:{type(exc).__name__}'
		now = time.time()
		if state != last_state or (now - last_print_at) >= 10:
			print(f'[wait] {label} machine health {state}', flush=True)
			last_state = state
			last_print_at = now

	time.sleep(2)

print(f'[error] {label} machine health timed out after {timeout_sec:.0f}s', file=sys.stderr)
sys.exit(1)
PY
}

validate_tei_lsh() {
	local machine_url="$1"
	local n_inputs="$2"
	local rounds="$3"
	python3 - "$machine_url" "$n_inputs" "$rounds" <<'PY'
import json
import sys
import urllib.request

machine_url = sys.argv[1]
n_inputs = int(sys.argv[2])
rounds = int(sys.argv[3])
inputs = [f'live-embed-sample-{index}-' + ('abc' * (1 + index % 7)) for index in range(n_inputs)]

for round_index in range(1, rounds + 1):
	payload = json.dumps(
		{
			'inputs': inputs,
			'bitn': 2048,
			'normalize': True,
			'truncate': True,
		}
	).encode()
	request = urllib.request.Request(
		f'{machine_url}/lsh',
		data=payload,
		headers={'Content-Type': 'application/json'},
	)
	with urllib.request.urlopen(request, timeout=300) as resp:
		hashes = json.loads(resp.read().decode())
	if len(hashes) != n_inputs:
		raise SystemExit(
			f'TEI /lsh returned {len(hashes)} hashes, expected {n_inputs}'
		)
	print(f'[okay] TEI /lsh round {round_index}/{rounds}: {len(hashes)} hashes', flush=True)
PY
}

validate_qwn_chat() {
	local machine_url="$1"
	python3 - "$machine_url" <<'PY'
import json
import sys
import urllib.request

machine_url = sys.argv[1]
with urllib.request.urlopen(f'{machine_url}/v1/models', timeout=10) as resp:
	models = json.loads(resp.read().decode())

model_id = models['data'][0]['id']
payload = {
	'model': model_id,
	'messages': [
		{'role': 'system', 'content': 'You are concise.'},
		{'role': 'user', 'content': 'Reply with exactly: QWN live validation ok'},
	],
	'max_tokens': 32,
	'temperature': 0,
}
request = urllib.request.Request(
	f'{machine_url}/v1/chat/completions',
	data=json.dumps(payload).encode(),
	headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(request, timeout=120) as resp:
	body = json.loads(resp.read().decode())

message = body['choices'][0]['message']['content'].strip()
if 'QWN live validation ok' not in message:
	raise SystemExit(f'Unexpected QWN chat response: {message}')

print(f'[okay] QWN models: {len(models["data"])} entries', flush=True)
print(f'[okay] QWN chat reply: {message}', flush=True)
PY
}

start_tei_machine() {
	log "Starting TEI machine on port ${TEI_MACHINE_PORT}"
	if command -v setsid >/dev/null 2>&1; then
		if command -v tei >/dev/null 2>&1; then
			setsid tei machine run --perf-track --on-conflict replace < /dev/null >"$TEI_MACHINE_LOG" 2>&1 &
		else
			setsid python -m tfmx.teis.cli machine run --perf-track --on-conflict replace < /dev/null >"$TEI_MACHINE_LOG" 2>&1 &
		fi
	elif command -v tei >/dev/null 2>&1; then
		nohup tei machine run --perf-track --on-conflict replace < /dev/null >"$TEI_MACHINE_LOG" 2>&1 &
	else
		nohup python -m tfmx.teis.cli machine run --perf-track --on-conflict replace < /dev/null >"$TEI_MACHINE_LOG" 2>&1 &
	fi
	echo "$!" > "$TEI_MACHINE_PID_FILE"
}

run_optional_benchmarks() {
	local tei_output=""
	local qwn_output=""
	local tei_log=""

	if [[ "$TEI_BENCH_SAMPLES" -gt 0 ]]; then
		tei_output="$REPO_ROOT/runs/teis/results/recovery_${TIMESTAMP}.json"
	fi
	if [[ "$QWN_BENCH_SAMPLES" -gt 0 ]]; then
		qwn_output="$REPO_ROOT/runs/qwns/results/recovery_${TIMESTAMP}.json"
	fi

	if [[ "$TEI_BENCH_SAMPLES" -gt 0 && "$QWN_BENCH_SAMPLES" -gt 0 ]]; then
		tei_log="$RESULTS_DIR/tei_benchmark_${TIMESTAMP}.log"
		log "Running TEI benchmark in background: ${TEI_BENCH_SAMPLES} samples"
		(
			cd "$REPO_ROOT"
			tei_cmd benchmark run -E "$TEI_MACHINE_URL" -n "$TEI_BENCH_SAMPLES" -o "$tei_output"
		) >"$tei_log" 2>&1 &
		local tei_pid=$!

		log "Running QWN benchmark in foreground: ${QWN_BENCH_SAMPLES} requests"
		qwn_cmd benchmark run \
			-E "$QWN_MACHINE_URL" \
			-n "$QWN_BENCH_SAMPLES" \
			--max-tokens "$QWN_BENCH_MAX_TOKENS" \
			--no-ttft \
			-o "$qwn_output"

		wait "$tei_pid"
		log "TEI benchmark result: $tei_output"
		log "TEI benchmark log: $tei_log"
		log "QWN benchmark result: $qwn_output"
		return
	fi

	if [[ "$TEI_BENCH_SAMPLES" -gt 0 ]]; then
		log "Running TEI benchmark: ${TEI_BENCH_SAMPLES} samples"
		tei_cmd benchmark run -E "$TEI_MACHINE_URL" -n "$TEI_BENCH_SAMPLES" -o "$tei_output"
		log "TEI benchmark result: $tei_output"
	fi

	if [[ "$QWN_BENCH_SAMPLES" -gt 0 ]]; then
		log "Running QWN benchmark: ${QWN_BENCH_SAMPLES} requests"
		qwn_cmd benchmark run \
			-E "$QWN_MACHINE_URL" \
			-n "$QWN_BENCH_SAMPLES" \
			--max-tokens "$QWN_BENCH_MAX_TOKENS" \
			--no-ttft \
			-o "$qwn_output"
		log "QWN benchmark result: $qwn_output"
	fi
}

require_cmd python3
require_cmd curl
require_cmd docker

VISIBLE_GPU_CSV="$(detect_visible_gpu_csv)"
if [[ -z "$VISIBLE_GPU_CSV" ]]; then
	printf '[recovery] no visible GPUs were detected in this VM\n' >&2
	exit 1
fi

if [[ -z "$TEI_GPUS" ]]; then
	TEI_GPUS="$VISIBLE_GPU_CSV"
fi
if [[ -z "$QWN_GPUS" ]]; then
	QWN_GPUS="$VISIBLE_GPU_CSV"
fi

log "Resolved TEI GPUs: $TEI_GPUS"
log "Resolved QWN GPUs: $QWN_GPUS"

TEI_EXPECTED_TOTAL="$(count_csv_items "$TEI_GPUS")"
QWN_EXPECTED_TOTAL="$(count_csv_items "$QWN_GPUS")"
FIRST_TEI_GPU="${TEI_GPUS%%,*}"

preflight_guest_gpu_state "$FIRST_TEI_GPU"

log "Stopping existing QWN machine and compose stacks (best effort)"
qwn_cmd machine stop >/dev/null 2>&1 || true
qwn_cmd compose down -j "$QWN_PROJECT_NAME" >/dev/null 2>&1 || true
qwn_cmd compose down >/dev/null 2>&1 || true

log "Stopping existing TEI compose stack (best effort)"
tei_cmd compose down >/dev/null 2>&1 || true

log "Starting TEI backends on GPUs: $TEI_GPUS"
tei_cmd compose up -g "$TEI_GPUS"
TEI_STACK_STARTED=1
wait_backend_health "TEI" "$TEI_BACKEND_BASE_PORT" "$TEI_GPUS" "$WAIT_TIMEOUT_SEC"

start_tei_machine
wait_machine_health "TEI" "$TEI_MACHINE_URL/health" "$TEI_EXPECTED_TOTAL" "$WAIT_TIMEOUT_SEC"
validate_tei_lsh "$TEI_MACHINE_URL" "$TEI_LSH_INPUTS" "$TEI_LSH_ROUNDS"

log "Starting QWN backends on GPUs: $QWN_GPUS"
qwn_cmd compose up --gpu-layout uniform-awq -g "$QWN_GPUS"
QWN_STACK_STARTED=1
wait_backend_health "QWN" "$QWN_BACKEND_BASE_PORT" "$QWN_GPUS" "$WAIT_TIMEOUT_SEC"

log "Starting QWN machine on port ${QWN_MACHINE_PORT}"
qwn_cmd machine run -b --on-conflict replace
wait_machine_health "QWN" "$QWN_MACHINE_URL/health" "$QWN_EXPECTED_TOTAL" "$WAIT_TIMEOUT_SEC"
validate_qwn_chat "$QWN_MACHINE_URL"

run_optional_benchmarks

wait_machine_health "TEI" "$TEI_MACHINE_URL/health" "$TEI_EXPECTED_TOTAL" 30
wait_machine_health "QWN" "$QWN_MACHINE_URL/health" "$QWN_EXPECTED_TOTAL" 30

RECOVERY_SUCCESS=1
log "Recovery complete"
log "TEI machine log: $TEI_MACHINE_LOG"
log "TEI machine launcher pid file: $TEI_MACHINE_PID_FILE"
log "Recovery artifacts directory: $RESULTS_DIR"