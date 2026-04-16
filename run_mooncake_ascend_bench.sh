#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/mooncake_ascend_bench.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-10000}"
export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-10000}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

MODE=""
WORLD_SIZE="${WORLD_SIZE:-1}"
LOCAL_HOST="${LOCAL_HOST:-127.0.0.1}"
PEER_HOST="${PEER_HOST:-}"
BASE_PORT="${BASE_PORT:-12345}"
CONTROL_PORT="${CONTROL_PORT:-29600}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-15}"
LOCAL_HOSTNAME="${LOCAL_HOSTNAME:-${LOCAL_HOST}}"
MASTER_SERVER="${MASTER_SERVER:-127.0.0.1:50051}"
METADATA_SERVER="${MC_METADATA_SERVER:-P2PHANDSHAKE}"
PACKET_SIZES="${PACKET_SIZES:-1M,4M,16M}"
ITERATIONS="${ITERATIONS:-50}"
WARMUP="${WARMUP:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PIPELINE_DEPTH="${PIPELINE_DEPTH:-8}"
REPORT_UNIT="${REPORT_UNIT:-GB}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
PEER_RANK="${PEER_RANK:-}"
STREAM_LOGS="${STREAM_LOGS:-1}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --mode p2p [options]
  $(basename "$0") --mode store [options]
  $(basename "$0") --mode all [options]
  $(basename "$0") --mode all-single-node [options]
  $(basename "$0") --mode all-multi-node [options]

Common options:
  --mode MODE                 p2p | store | all | all-single-node | all-multi-node
  --local-world-size N        number of local ranks/processes to launch
  --log-dir DIR               log output directory
  --stream-logs 0|1           stream per-rank logs to terminal, default: ${STREAM_LOGS}
  --metadata-server ADDR      metadata server, default: ${METADATA_SERVER}
  --packet-sizes LIST         e.g. 1M,4M,16M
  --iterations N
  --warmup N
  --report-unit UNIT
  --startup-wait-seconds N    worker readiness wait, default: ${STARTUP_WAIT_SECONDS}

Ascend communication environment defaults:
  HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE}
  HCCL_RDMA_TIMEOUT=${HCCL_RDMA_TIMEOUT}
  ASCEND_CONNECT_TIMEOUT=${ASCEND_CONNECT_TIMEOUT}
  ASCEND_TRANSFER_TIMEOUT=${ASCEND_TRANSFER_TIMEOUT}
  ACL_OP_INIT_MODE=${ACL_OP_INIT_MODE}
  PYTHONHASHSEED=${PYTHONHASHSEED}

Transfer options:
  --local-host HOST
  --peer-host HOST            required for p2p, all-single-node, and all-multi-node
  --base-port PORT
  --batch-size N
  --pipeline-depth N
  --peer-rank N               limit to one peer rank
  --control-port PORT         control port for all-multi-node, default: ${CONTROL_PORT}

Store options:
  --local-host HOST            store 默认也使用该地址作为本地节点名
  --master-server ADDR

Extra args:
  Any trailing args after '--' are passed through to mooncake_ascend_bench.py.

Examples:
  $(basename "$0") --mode p2p --local-world-size 8 --local-host 10.20.130.155 --peer-host 10.20.130.154
  $(basename "$0") --mode store --local-world-size 8 --local-host 10.20.130.155 --master-server 127.0.0.1:50051
  $(basename "$0") --mode all-single-node --local-world-size 8 --local-host 10.20.130.155 --peer-host 10.20.130.154 --master-server 127.0.0.1:50051
  $(basename "$0") --mode all-multi-node --local-world-size 8 --local-host 10.20.130.155 --peer-host 10.20.130.154 --master-server 127.0.0.1:50051
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --local-world-size|--world-size) WORLD_SIZE="$2"; shift 2 ;;
    --local-host) LOCAL_HOST="$2"; shift 2 ;;
    --peer-host) PEER_HOST="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --peer-base-port) shift 2 ;;
    --control-port) CONTROL_PORT="$2"; shift 2 ;;
    --startup-wait-seconds) STARTUP_WAIT_SECONDS="$2"; shift 2 ;;
    --local-hostname) LOCAL_HOSTNAME="$2"; shift 2 ;;
    --master-server) MASTER_SERVER="$2"; shift 2 ;;
    --metadata-server) METADATA_SERVER="$2"; shift 2 ;;
    --packet-sizes) PACKET_SIZES="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --pipeline-depth) PIPELINE_DEPTH="$2"; shift 2 ;;
    --report-unit) REPORT_UNIT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --stream-logs) STREAM_LOGS="$2"; shift 2 ;;
    --peer-rank) PEER_RANK="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  echo "error: --mode is required" >&2
  usage
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "error: benchmark script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

if [[ ( "${MODE}" == "p2p" || "${MODE}" == "all" || "${MODE}" == "all-single-node" || "${MODE}" == "all-multi-node" ) && -z "${PEER_HOST}" ]]; then
  echo "error: --peer-host is required for ${MODE} mode" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

declare -a PIDS=()
declare -a BACKGROUND_PIDS=()
declare -a STREAM_PIDS=()
declare -A PID_TO_RANK=()
declare -A PID_TO_LOG=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${BACKGROUND_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${STREAM_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

trap cleanup INT TERM

summarize_transfer_logs() {
  local log_dir="$1"
  local world_size="$2"

  "${PYTHON_BIN}" - "$log_dir" "$world_size" <<'PY'
import csv
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
world_size = int(sys.argv[2])

ok_pattern = re.compile(
    r'^(transfer_(?:sync|async)_write)\[src=(\d+),dst=(\d+)\]\s+size=\s*(\d+)KB.*throughput=([0-9.]+\s+\w+/s)$'
)
fail_pattern = re.compile(
    r'^(transfer_(?:sync|async)_write)\[src=(\d+),dst=(\d+)\]\s+size=\s*(\d+)KB.*status=([A-Z_]+).*$'
)

tables = {}
has_fail = False
paths = sorted(log_dir.glob("transfer_rank*.log")) + sorted(log_dir.glob("worker_rank*.log"))
for path in paths:
    try:
        text = path.read_text()
    except OSError:
        continue
    for line in text.splitlines():
        line = line.strip()
        match = ok_pattern.match(line)
        if match:
            op, src, dst, size, cell = match.groups()
        else:
            match = fail_pattern.match(line)
            if not match:
                continue
            op, src, dst, size, cell = match.groups()
        key = (op, int(size))
        table = tables.setdefault(
            key, [["-" for _ in range(world_size)] for _ in range(world_size)]
        )
        table[int(src)][int(dst)] = cell
        if cell.endswith("FAIL"):
            has_fail = True

if not tables:
    sys.exit(0)

def print_table(title, table):
    print(title)
    header = ["src\\dst"] + [str(i) for i in range(len(table[0]))]
    widths = [max(len(header[i]), 10) for i in range(len(header))]
    for row_idx, row in enumerate(table):
        widths[0] = max(widths[0], len(str(row_idx)))
        for col_idx, cell in enumerate(row, start=1):
            widths[col_idx] = max(widths[col_idx], len(cell))
    print("  ".join(header[i].rjust(widths[i]) for i in range(len(header))))
    for row_idx, row in enumerate(table):
        values = [str(row_idx)] + row
        print("  ".join(values[i].rjust(widths[i]) for i in range(len(values))))

def write_csv(path, table):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src\\dst", *range(len(table[0]))])
        for row_idx, row in enumerate(table):
            writer.writerow([row_idx, *row])

for key in sorted(tables):
    op, size_kb = key
    print()
    table = tables[key]
    print_table(f"[summary] {op} size={size_kb}KB", table)
    csv_name = f"{op}_size_{size_kb}KB.csv"
    csv_path = log_dir / csv_name
    write_csv(csv_path, table)
    print(f"[summary] csv={csv_path}")
if has_fail:
    sys.exit(2)
PY
}

summarize_store_logs() {
  local log_dir="$1"
  local world_size="$2"

  "${PYTHON_BIN}" - "$log_dir" "$world_size" <<'PY'
import csv
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
world_size = int(sys.argv[2])

rank_pattern = re.compile(r"^store_rank=(\d+)$")
ok_pattern = re.compile(
    r"^(pool_(?:put_from|get_into))\s+size=\s*(\d+)KB.*throughput=([0-9.]+\s+\w+/s)$"
)
fail_pattern = re.compile(
    r"^(pool_(?:put_from|get_into))\s+size=\s*(\d+)KB.*status=([A-Z_]+).*$"
)

tables = {}
has_fail = False
paths = sorted(log_dir.glob("store_rank*.log"))
for path in paths:
    try:
        text = path.read_text()
    except OSError:
        continue
    rank = None
    for line in text.splitlines():
        line = line.strip()
        match = rank_pattern.match(line)
        if match:
            rank = int(match.group(1))
            continue
        match = ok_pattern.match(line)
        if match:
            op, size, cell = match.groups()
        else:
            match = fail_pattern.match(line)
            if not match:
                continue
            op, size, cell = match.groups()
        if rank is None:
            continue
        key = (int(size), op)
        table = tables.setdefault(key, ["-" for _ in range(world_size)])
        table[rank] = cell
        if cell.endswith("FAIL"):
            has_fail = True

if not tables:
    sys.exit(0)

def print_table(title, values):
    header_rank = "rank"
    header_value = "result"
    width_rank = max(len(header_rank), max(len(str(i)) for i in range(len(values))))
    width_value = max(len(header_value), max(len(v) for v in values))
    print(title)
    print(f"{header_rank.rjust(width_rank)}  {header_value.rjust(width_value)}")
    for rank, value in enumerate(values):
        print(f"{str(rank).rjust(width_rank)}  {value.rjust(width_value)}")

def write_csv(path, values):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "result"])
        for rank, value in enumerate(values):
            writer.writerow([rank, value])

for key in sorted(tables):
    size_kb, op = key
    values = tables[key]
    print()
    print_table(f"[summary] {op} size={size_kb}KB", values)
    csv_name = f"{op}_size_{size_kb}KB.csv"
    csv_path = log_dir / csv_name
    write_csv(csv_path, values)
    print(f"[summary] csv={csv_path}")

if has_fail:
    sys.exit(2)
PY
}

start_log_stream() {
  local label="$1"
  local log_file="$2"
  if [[ "${STREAM_LOGS}" != "1" ]]; then
    return
  fi
  (
    tail -n 0 -F "${log_file}" 2>/dev/null | while IFS= read -r line; do
      printf '[%s] %s\n' "${label}" "${line}"
    done
  ) &
  STREAM_PIDS+=("$!")
}

stop_log_streams() {
  local pid
  for pid in "${STREAM_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  STREAM_PIDS=()
}

launch_rank() {
  local run_mode="$1"
  local rank="$2"
  local log_prefix="${3:-${run_mode}}"
  local peer_host_override="${4:-__GLOBAL__}"
  local log_file="${LOG_DIR}/${log_prefix}_rank${rank}.log"
  local -a cmd
  local peer_host_value

  cmd=(
    "${PYTHON_BIN}" "${PY_SCRIPT}"
    "--rank" "${rank}"
    "--world-size" "${WORLD_SIZE}"
  )

  if [[ "${peer_host_override}" == "__GLOBAL__" ]]; then
    peer_host_value="${PEER_HOST}"
  else
    peer_host_value="${peer_host_override}"
  fi

  case "${run_mode}" in
    worker)
      cmd+=(
        "worker"
        "--local-host" "${LOCAL_HOST}"
        "--base-port" "${BASE_PORT}"
        "--metadata-server" "${METADATA_SERVER}"
        "--batch-size" "${BATCH_SIZE}"
        "--pipeline-depth" "${PIPELINE_DEPTH}"
        "--packet-sizes" "${PACKET_SIZES}"
        "--iterations" "${ITERATIONS}"
        "--warmup" "${WARMUP}"
        "--report-unit" "${REPORT_UNIT}"
        "--startup-wait-seconds" "${STARTUP_WAIT_SECONDS}"
      )
      if [[ -n "${peer_host_value}" ]]; then
        cmd+=(
          "--peer-host" "${peer_host_value}"
        )
      fi
      if [[ -n "${PEER_RANK}" ]]; then
        cmd+=("--peer-rank" "${PEER_RANK}")
      fi
      ;;
    transfer)
      cmd+=(
        "transfer"
        "--local-host" "${LOCAL_HOST}"
        "--peer-host" "${PEER_HOST}"
        "--base-port" "${BASE_PORT}"
        "--metadata-server" "${METADATA_SERVER}"
        "--batch-size" "${BATCH_SIZE}"
        "--pipeline-depth" "${PIPELINE_DEPTH}"
        "--packet-sizes" "${PACKET_SIZES}"
        "--iterations" "${ITERATIONS}"
        "--warmup" "${WARMUP}"
        "--report-unit" "${REPORT_UNIT}"
      )
      if [[ -n "${PEER_RANK}" ]]; then
        cmd+=("--peer-rank" "${PEER_RANK}")
      fi
      ;;
    store)
      cmd+=(
        "store"
        "--local-host" "${LOCAL_HOST}"
        "--metadata-server" "${METADATA_SERVER}"
        "--master-server" "${MASTER_SERVER}"
        "--packet-sizes" "${PACKET_SIZES}"
        "--iterations" "${ITERATIONS}"
        "--warmup" "${WARMUP}"
        "--report-unit" "${REPORT_UNIT}"
      )
      ;;
    *)
      echo "error: unsupported mode '${run_mode}'" >&2
      exit 1
      ;;
  esac

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "[launch] rank=${rank} log=${log_file}"
  "${cmd[@]}" >"${log_file}" 2>&1 &
  local pid="$!"
  start_log_stream "${log_prefix}:rank${rank}" "${log_file}"
  PIDS+=("${pid}")
  PID_TO_RANK["${pid}"]="${rank}"
  PID_TO_LOG["${pid}"]="${log_file}"
}

start_passive_workers() {
  local excluded_rank="${1:-}"
  local rank
  BACKGROUND_PIDS=()
  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    if [[ -n "${excluded_rank}" && "${rank}" == "${excluded_rank}" ]]; then
      continue
    fi
    local log_file="${LOG_DIR}/worker_passive_rank${rank}.log"
    local -a cmd=(
      "${PYTHON_BIN}" "${PY_SCRIPT}"
      "--rank" "${rank}"
      "--world-size" "${WORLD_SIZE}"
      "worker"
      "--local-host" "${LOCAL_HOST}"
      "--base-port" "${BASE_PORT}"
      "--metadata-server" "${METADATA_SERVER}"
      "--batch-size" "${BATCH_SIZE}"
      "--pipeline-depth" "${PIPELINE_DEPTH}"
      "--packet-sizes" "${PACKET_SIZES}"
      "--iterations" "${ITERATIONS}"
      "--warmup" "${WARMUP}"
      "--report-unit" "${REPORT_UNIT}"
      "--startup-wait-seconds" "${STARTUP_WAIT_SECONDS}"
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_ARGS[@]}")
    fi
    echo "[launch] passive-worker rank=${rank} log=${log_file}"
    "${cmd[@]}" >"${log_file}" 2>&1 &
    local pid="$!"
    start_log_stream "passive-worker:rank${rank}" "${log_file}"
    BACKGROUND_PIDS+=("${pid}")
  done
}

stop_passive_workers() {
  local pid
  for pid in "${BACKGROUND_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${BACKGROUND_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  BACKGROUND_PIDS=()
  stop_log_streams
}

control_send() {
  local message="$1"
  "${PYTHON_BIN}" - "${PEER_HOST}" "${CONTROL_PORT}" "${message}" <<'PY'
import socket
import sys
host = sys.argv[1]
port = int(sys.argv[2])
message = sys.argv[3].encode("utf-8")
with socket.create_connection((host, port), timeout=300) as sock:
    sock.sendall(message)
PY
}

control_wait() {
  local expected="$1"
  echo "[sync] waiting for ${expected} on port ${CONTROL_PORT}"
  "${PYTHON_BIN}" - "${CONTROL_PORT}" "${expected}" <<'PY'
import socket
import sys
port = int(sys.argv[1])
expected = sys.argv[2].encode("utf-8")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", port))
    server.listen(1)
    conn, _ = server.accept()
    with conn:
        data = conn.recv(1024)
if data != expected:
    raise SystemExit(f"unexpected control message: {data!r}, expected {expected!r}")
PY
}

control_send_file() {
  local label="$1"
  local file_path="$2"
  "${PYTHON_BIN}" - "${PEER_HOST}" "${CONTROL_PORT}" "${label}" "${file_path}" <<'PY'
import pathlib
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
label = sys.argv[3].encode("utf-8")
payload = pathlib.Path(sys.argv[4]).read_bytes()
with socket.create_connection((host, port), timeout=300) as sock:
    sock.sendall(label + b"\n" + payload)
PY
}

control_wait_file() {
  local expected="$1"
  local output_path="$2"
  echo "[sync] waiting for file ${expected} on port ${CONTROL_PORT}"
  "${PYTHON_BIN}" - "${CONTROL_PORT}" "${expected}" "${output_path}" <<'PY'
import pathlib
import socket
import sys

port = int(sys.argv[1])
expected = sys.argv[2].encode("utf-8")
output_path = pathlib.Path(sys.argv[3])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", port))
    server.listen(1)
    conn, _ = server.accept()
    with conn:
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
data = b"".join(chunks)
label, sep, payload = data.partition(b"\n")
if sep == b"":
    raise SystemExit("invalid file payload without header separator")
if label != expected:
    raise SystemExit(f"unexpected file label: {label!r}, expected {expected!r}")
output_path.write_bytes(payload)
PY
}

infer_node_role() {
  "${PYTHON_BIN}" - "${LOCAL_HOST}" "${PEER_HOST}" <<'PY'
import ipaddress
import socket
import sys

local_host = sys.argv[1]
peer_host = sys.argv[2]

def normalize(host: str):
    try:
        return (0, int(ipaddress.ip_address(host)))
    except ValueError:
        try:
            resolved = socket.gethostbyname(host)
            return (0, int(ipaddress.ip_address(resolved)))
        except OSError:
            return (1, host)

local_key = normalize(local_host)
peer_key = normalize(peer_host)
if local_key == peer_key:
    raise SystemExit("local-host and peer-host resolve to the same endpoint; cannot infer multi-node order")
print("primary" if local_key < peer_key else "secondary")
PY
}

hosts_same() {
  "${PYTHON_BIN}" - "${LOCAL_HOST}" "${PEER_HOST}" <<'PY'
import socket
import sys

def normalize(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host

print("1" if normalize(sys.argv[1]) == normalize(sys.argv[2]) else "0")
PY
}

run_nested_p2p() {
  local phase_log_dir="$1"
  local phase_local_host="$2"
  local phase_peer_host="$3"

  mkdir -p "${phase_log_dir}"
  local -a cmd=(
    bash "${BASH_SOURCE[0]}"
    --mode p2p
    --local-world-size "${WORLD_SIZE}"
    --local-host "${phase_local_host}"
    --peer-host "${phase_peer_host}"
    --base-port "${BASE_PORT}"
    --metadata-server "${METADATA_SERVER}"
    --packet-sizes "${PACKET_SIZES}"
    --iterations "${ITERATIONS}"
    --warmup "${WARMUP}"
    --batch-size "${BATCH_SIZE}"
    --pipeline-depth "${PIPELINE_DEPTH}"
    --report-unit "${REPORT_UNIT}"
    --startup-wait-seconds "${STARTUP_WAIT_SECONDS}"
    --stream-logs "${STREAM_LOGS}"
    --log-dir "${phase_log_dir}"
  )
  if [[ -n "${PEER_RANK}" ]]; then
    cmd+=(--peer-rank "${PEER_RANK}")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi
  "${cmd[@]}"
}

build_transfer_bundle_json() {
  local output_json="$1"
  shift
  "${PYTHON_BIN}" - "${output_json}" "$@" <<'PY'
import csv
import json
import pathlib
import re
import sys

output = pathlib.Path(sys.argv[1])
args = sys.argv[2:]
pattern = re.compile(r"^(transfer_(?:sync|async)_write)_size_(\d+)KB\.csv$")
bundle = {}

for idx in range(0, len(args), 2):
    phase = args[idx]
    phase_dir = pathlib.Path(args[idx + 1])
    phase_data = {}
    if phase_dir.exists():
        for path in sorted(phase_dir.glob("transfer_*_size_*KB.csv")):
            match = pattern.match(path.name)
            if not match:
                continue
            op, size = match.groups()
            matrix = []
            with path.open(newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    matrix.append(row[1:])
            phase_data.setdefault(op, {})[size] = matrix
    bundle[phase] = phase_data

output.write_text(json.dumps(bundle, sort_keys=True))
PY
}

merge_transfer_bundle_json() {
  local local_bundle="$1"
  local remote_bundle="$2"
  local local_world_size="$3"
  local output_dir="$4"
  "${PYTHON_BIN}" - "${local_bundle}" "${remote_bundle}" "${local_world_size}" "${output_dir}" <<'PY'
import csv
import json
import pathlib
import sys

local_bundle = json.loads(pathlib.Path(sys.argv[1]).read_text())
remote_bundle = json.loads(pathlib.Path(sys.argv[2]).read_text())
local_world_size = int(sys.argv[3])
output_dir = pathlib.Path(sys.argv[4])
output_dir.mkdir(parents=True, exist_ok=True)
global_world_size = local_world_size * 2

phase_to_offset = {
    "aa": (0, 0),
    "ab": (0, local_world_size),
    "ba": (local_world_size, 0),
    "bb": (local_world_size, local_world_size),
}

tables = {}
for bundle in (local_bundle, remote_bundle):
    for phase, phase_data in bundle.items():
        if phase not in phase_to_offset:
            continue
        src_offset, dst_offset = phase_to_offset[phase]
        for op, sizes in phase_data.items():
            for size, matrix in sizes.items():
                table = tables.setdefault(
                    (op, int(size)),
                    [["-" for _ in range(global_world_size)] for _ in range(global_world_size)],
                )
                for src_idx, row in enumerate(matrix):
                    for dst_idx, cell in enumerate(row):
                        table[src_offset + src_idx][dst_offset + dst_idx] = cell

def print_table(title, table):
    print(title)
    header = ["src\\dst"] + [str(i) for i in range(len(table[0]))]
    widths = [max(len(header[i]), 10) for i in range(len(header))]
    for row_idx, row in enumerate(table):
        widths[0] = max(widths[0], len(str(row_idx)))
        for col_idx, cell in enumerate(row, start=1):
            widths[col_idx] = max(widths[col_idx], len(cell))
    print("  ".join(header[i].rjust(widths[i]) for i in range(len(header))))
    for row_idx, row in enumerate(table):
        values = [str(row_idx)] + row
        print("  ".join(values[i].rjust(widths[i]) for i in range(len(values))))

def write_csv(path, table):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src\\dst", *range(len(table[0]))])
        for row_idx, row in enumerate(table):
            writer.writerow([row_idx, *row])

has_fail = False
for key in sorted(tables):
    op, size_kb = key
    table = tables[key]
    print()
    print_table(f"[global-summary] {op} size={size_kb}KB", table)
    csv_path = output_dir / f"global_{op}_size_{size_kb}KB.csv"
    write_csv(csv_path, table)
    print(f"[global-summary] csv={csv_path}")
    for row in table:
        for cell in row:
            if cell.endswith("FAIL"):
                has_fail = True

if has_fail:
    sys.exit(2)
PY
}

run_p2p_round_robin() {
  local same_host="$1"
  local active_base_port="${BASE_PORT}"
  local rc=0
  local rank

  if [[ "${same_host}" == "1" ]]; then
    active_base_port=$((BASE_PORT + 1000))
    echo "[info] same-host p2p detected; using per-src passive workers on base-port=${BASE_PORT} and sequential transfers on base-port=${active_base_port}"
  else
    echo "[info] p2p will run in round-robin mode; active transfers execute one src rank at a time"
  fi

  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    if [[ "${same_host}" == "1" ]]; then
      start_passive_workers "${rank}"
      sleep "${STARTUP_WAIT_SECONDS}"
    fi
    local log_file="${LOG_DIR}/transfer_rank${rank}.log"
    local -a cmd=(
      "${PYTHON_BIN}" "${PY_SCRIPT}"
      "--rank" "${rank}"
      "--world-size" "${WORLD_SIZE}"
      "transfer"
      "--local-host" "${LOCAL_HOST}"
      "--peer-host" "${PEER_HOST}"
      "--base-port" "${active_base_port}"
      "--peer-base-port" "${BASE_PORT}"
      "--metadata-server" "${METADATA_SERVER}"
      "--batch-size" "${BATCH_SIZE}"
      "--pipeline-depth" "${PIPELINE_DEPTH}"
      "--packet-sizes" "${PACKET_SIZES}"
      "--iterations" "${ITERATIONS}"
      "--warmup" "${WARMUP}"
      "--report-unit" "${REPORT_UNIT}"
    )
    if [[ -n "${PEER_RANK}" ]]; then
      cmd+=("--peer-rank" "${PEER_RANK}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_ARGS[@]}")
    fi
    echo "[launch] transfer rank=${rank} log=${log_file}"
    "${cmd[@]}" >"${log_file}" 2>&1 &
    local pid="$!"
    start_log_stream "transfer:rank${rank}" "${log_file}"
    PID_TO_RANK["${pid}"]="${rank}"
    PID_TO_LOG["${pid}"]="${log_file}"
    if wait "${pid}"; then
      :
    else
      local status=$?
      rc=1
      local rank="${PID_TO_RANK[${pid}]:-unknown}"
      local log_file="${PID_TO_LOG[${pid}]:-unknown}"
      if [[ "${status}" -eq 139 ]]; then
        echo "[warn] p2p rank=${rank} exited with SIGSEGV (139). log=${log_file}" >&2
      else
        echo "[warn] p2p rank=${rank} exited with status=${status}. log=${log_file}" >&2
      fi
    fi
    stop_log_streams
    if [[ "${same_host}" == "1" ]]; then
      stop_passive_workers
    fi
  done
  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] at least one p2p rank failed. logs are under ${LOG_DIR}" >&2
    exit "${rc}"
  fi

  if summarize_transfer_logs "${LOG_DIR}" "${WORLD_SIZE}"; then
    :
  else
    rc=$?
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] p2p summary contains failed transfers. logs are under ${LOG_DIR}" >&2
    exit "${rc}"
  fi
  echo "[done] all p2p ranks finished successfully. logs: ${LOG_DIR}"
}

run_store_round_robin() {
  local rc=0
  local rank

  echo "[info] store will run in round-robin mode; one rank executes at a time"

  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    local log_file="${LOG_DIR}/store_rank${rank}.log"
    local -a cmd=(
      "${PYTHON_BIN}" "${PY_SCRIPT}"
      "--rank" "${rank}"
      "--world-size" "${WORLD_SIZE}"
      "store"
      "--local-host" "${LOCAL_HOST}"
      "--metadata-server" "${METADATA_SERVER}"
      "--master-server" "${MASTER_SERVER}"
      "--packet-sizes" "${PACKET_SIZES}"
      "--iterations" "${ITERATIONS}"
      "--warmup" "${WARMUP}"
      "--report-unit" "${REPORT_UNIT}"
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_ARGS[@]}")
    fi
    echo "[launch] store rank=${rank} log=${log_file}"
    "${cmd[@]}" >"${log_file}" 2>&1 &
    local pid="$!"
    start_log_stream "store:rank${rank}" "${log_file}"
    PID_TO_RANK["${pid}"]="${rank}"
    PID_TO_LOG["${pid}"]="${log_file}"
    if wait "${pid}"; then
      :
    else
      local status=$?
      rc=1
      local failed_rank="${PID_TO_RANK[${pid}]:-unknown}"
      local failed_log="${PID_TO_LOG[${pid}]:-unknown}"
      if [[ "${status}" -eq 139 ]]; then
        echo "[warn] store rank=${failed_rank} exited with SIGSEGV (139). log=${failed_log}" >&2
      else
        echo "[warn] store rank=${failed_rank} exited with status=${status}. log=${failed_log}" >&2
      fi
    fi
    stop_log_streams
  done

  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] at least one store rank failed. logs are under ${LOG_DIR}" >&2
    exit "${rc}"
  fi

  if summarize_store_logs "${LOG_DIR}" "${WORLD_SIZE}"; then
    :
  else
    rc=$?
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] store summary contains failed transfers. logs are under ${LOG_DIR}" >&2
    exit "${rc}"
  fi

  echo "[done] all store ranks finished successfully. logs: ${LOG_DIR}"
}

run_mode() {
  local run_mode="$1"
  local display_mode="${run_mode}"
  local rc=0

  if [[ "${run_mode}" == "worker" && -n "${PEER_HOST}" ]]; then
    display_mode="p2p"
  fi

  if [[ "${run_mode}" == "worker" && -n "${PEER_HOST}" ]]; then
    run_p2p_round_robin "$(hosts_same)"
    return
  fi

  if [[ "${run_mode}" == "store" ]]; then
    run_store_round_robin
    return
  fi

  PIDS=()
  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    launch_rank "${run_mode}" "${rank}"
  done

  if [[ "${run_mode}" == "worker" ]]; then
    if [[ -n "${PEER_HOST}" ]]; then
      echo "[info] p2p ranks started with peer-host=${PEER_HOST}; each worker will run transfer after startup wait."
    else
      echo "[info] worker ranks started. press Ctrl-C to stop."
    fi
  fi

  for pid in "${PIDS[@]}"; do
    if wait "${pid}"; then
      :
    else
      local status=$?
      rc=1
      local rank="${PID_TO_RANK[${pid}]:-unknown}"
      local log_file="${PID_TO_LOG[${pid}]:-unknown}"
      if [[ "${status}" -eq 139 ]]; then
        echo "[warn] ${display_mode} rank=${rank} exited with SIGSEGV (139). log=${log_file}" >&2
      else
        echo "[warn] ${display_mode} rank=${rank} exited with status=${status}. log=${log_file}" >&2
      fi
    fi
  done
  stop_log_streams

  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] at least one ${display_mode} rank failed. logs are under ${LOG_DIR}" >&2
    exit "${rc}"
  fi

  if [[ "${run_mode}" == "transfer" || ( "${run_mode}" == "worker" && -n "${PEER_HOST}" ) ]]; then
    summarize_transfer_logs "${LOG_DIR}" "${WORLD_SIZE}"
  fi

  echo "[done] all ${display_mode} ranks finished successfully. logs: ${LOG_DIR}"
}

run_all_single_node() {
  echo "[stage] running p2p benchmark through worker mode"
  run_mode "worker"
  echo
  echo "[stage] running store benchmark"
  run_mode "store"
  echo
  echo "[done] all benchmarks finished successfully. logs: ${LOG_DIR}"
}

run_all_multi_node() {
  local node_role
  node_role="$(infer_node_role)"
  echo "[info] auto-selected multi-node role=${node_role} from local-host=${LOCAL_HOST} peer-host=${PEER_HOST}"

  local p2p_root="${LOG_DIR}/all_multi_node_p2p"
  local aa_dir="${p2p_root}/aa"
  local ab_dir="${p2p_root}/ab"
  local ba_dir="${p2p_root}/ba"
  local bb_dir="${p2p_root}/bb"
  local local_bundle="${p2p_root}/local_bundle.json"
  local remote_bundle="${p2p_root}/remote_bundle.json"
  mkdir -p "${p2p_root}"

  if [[ "${node_role}" == "primary" ]]; then
    echo "[stage] running primary intra-node p2p benchmark"
    run_nested_p2p "${aa_dir}" "${LOCAL_HOST}" "${LOCAL_HOST}"
    control_send "phase_aa_done"

    echo
    echo "[stage] waiting for secondary passive workers for primary -> secondary p2p"
    control_wait "ab_workers_ready"
    echo "[stage] running primary -> secondary p2p benchmark"
    run_nested_p2p "${ab_dir}" "${LOCAL_HOST}" "${PEER_HOST}"
    control_send "phase_ab_done"

    echo
    control_wait "phase_bb_done"
    echo "[stage] starting passive workers for secondary -> primary p2p"
    start_passive_workers
    sleep "${STARTUP_WAIT_SECONDS}"
    control_send "ba_workers_ready"
    control_wait "phase_ba_done"
    stop_passive_workers

    echo
    build_transfer_bundle_json "${local_bundle}" aa "${aa_dir}" ab "${ab_dir}"
    control_wait_file "secondary_p2p_bundle" "${remote_bundle}"
    echo "[stage] merging full multi-node p2p matrix"
    if merge_transfer_bundle_json "${local_bundle}" "${remote_bundle}" "${WORLD_SIZE}" "${p2p_root}"; then
      :
    else
      local rc=$?
      echo "[error] merged multi-node p2p summary contains failed transfers. logs are under ${p2p_root}" >&2
      exit "${rc}"
    fi

    echo
    echo "[stage] running primary store benchmark"
    run_mode "store"
    control_send "primary_store_done"
    control_wait "secondary_store_done"
  else
    control_wait "phase_aa_done"
    echo "[stage] starting passive workers for primary -> secondary p2p"
    start_passive_workers
    sleep "${STARTUP_WAIT_SECONDS}"
    control_send "ab_workers_ready"
    control_wait "phase_ab_done"
    stop_passive_workers

    echo
    echo "[stage] running secondary intra-node p2p benchmark"
    run_nested_p2p "${bb_dir}" "${LOCAL_HOST}" "${LOCAL_HOST}"
    control_send "phase_bb_done"

    control_wait "ba_workers_ready"
    echo
    echo "[stage] running secondary -> primary p2p benchmark"
    run_nested_p2p "${ba_dir}" "${LOCAL_HOST}" "${PEER_HOST}"
    control_send "phase_ba_done"

    echo
    build_transfer_bundle_json "${local_bundle}" bb "${bb_dir}" ba "${ba_dir}"
    control_send_file "secondary_p2p_bundle" "${local_bundle}"

    control_wait "primary_store_done"
    echo
    echo "[stage] running secondary store benchmark"
    run_mode "store"
    control_send "secondary_store_done"
  fi

  echo
  echo "[done] all multi-node benchmarks finished successfully. logs: ${LOG_DIR}"
}

if [[ "${MODE}" == "all" || "${MODE}" == "all-single-node" ]]; then
  run_all_single_node
elif [[ "${MODE}" == "all-multi-node" ]]; then
  run_all_multi_node
else
  case "${MODE}" in
    p2p) run_mode "worker" ;;
    store) run_mode "store" ;;
    *)
      echo "error: unsupported mode '${MODE}', expected one of: p2p, store, all, all-single-node, all-multi-node" >&2
      exit 1
      ;;
  esac
fi
