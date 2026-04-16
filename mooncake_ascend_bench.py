#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 KVCache.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend performance benchmark helpers for Mooncake.

This script focuses on two commonly used Ascend paths:
1. P2P transfer performance for transfer-engine:
   - transfer sync write
   - async write
2. Store zero-copy buffer performance:
   - put_from
   - get_into

Typical usage:

  # Server A/B: start one worker process per rank
  python scripts/ascend/perf/mooncake_ascend_bench.py worker \
      --local-host 10.20.130.154 \
      --base-port 12345 \
      --rank 0 \
      --world-size 8

  # Server B: run transfer from this machine to the peer machine
  python scripts/ascend/perf/mooncake_ascend_bench.py transfer \
      --local-host 10.20.130.155 \
      --peer-host 10.20.130.154 \
      --base-port 12345 \
      --peer-base-port 12345 \
      --rank 0 \
      --world-size 8 \
      --packet-sizes 1M,4M,16M \
      --iterations 100 \
      --batch-size 4 \
      --pipeline-depth 8

  # Server A: run the reverse direction too
  python scripts/ascend/perf/mooncake_ascend_bench.py transfer \
      --local-host 10.20.130.154 \
      --peer-host 10.20.130.155 \
      --base-port 12345 \
      --peer-base-port 12345 \
      --rank 0 \
      --world-size 8

  # Run store put/get benchmark against an existing master
  python scripts/ascend/perf/mooncake_ascend_bench.py store \
      --local-host 10.20.130.155 \
      --metadata-server P2PHANDSHAKE \
      --master-server 127.0.0.1:50051 \
      --rank 0 \
      --world-size 8 \
      --packet-sizes 1M,4M,16M
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Iterable


LOG = logging.getLogger("mooncake_ascend_bench")
RATE_UNITS = {
    "GB": 1000**3,
    "GiB": 1024**3,
    "MB": 1000**2,
    "MiB": 1024**2,
}
REGISTERED_BUFFER_TENSORS: dict[int, object] = {}


@dataclass
class WorkerMetadata:
    rank: int
    local_server_name: str
    endpoint_name: str
    buffer_addr: int
    buffer_capacity: int
    te_rpc_port: int

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "rank": self.rank,
                "local_server_name": self.local_server_name,
                "endpoint_name": self.endpoint_name,
                "buffer_addr": self.buffer_addr,
                "buffer_capacity": self.buffer_capacity,
                "te_rpc_port": self.te_rpc_port,
            },
            sort_keys=True,
        ).encode("utf-8")


class WorkerMetadataServer:
    def __init__(self, host: str, port: int, metadata: WorkerMetadata):
        self.host = host
        self.port = port
        self.metadata = metadata
        self._stop = Event()
        self._thread = Thread(target=self._serve, name=f"metadata-{port}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                pass
        except OSError:
            pass
        self._thread.join(timeout=2)

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            server.settimeout(1.0)
            LOG.info(
                "metadata_server listening rank=%s addr=%s:%s endpoint_name=%s",
                self.metadata.rank,
                self.host,
                self.port,
                self.metadata.endpoint_name,
            )
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with conn:
                    try:
                        _ = conn.recv(1024)
                        conn.sendall(self.metadata.to_json())
                    except OSError:
                        continue


def fetch_worker_metadata(host: str, port: int, timeout_seconds: int = 30) -> WorkerMetadata:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3) as sock:
                sock.sendall(b"GET")
                payload = sock.recv(65536)
            if not payload:
                raise RuntimeError("empty metadata response")
            data = json.loads(payload.decode("utf-8"))
            return WorkerMetadata(
                rank=int(data["rank"]),
                local_server_name=str(data["local_server_name"]),
                endpoint_name=str(data["endpoint_name"]),
                buffer_addr=int(data["buffer_addr"]),
                buffer_capacity=int(data["buffer_capacity"]),
                te_rpc_port=int(data["te_rpc_port"]),
            )
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(
        f"failed to fetch worker metadata from {host}:{port} within {timeout_seconds}s: {last_error}"
    )


def format_size_kb(num_bytes: int) -> str:
    return f"{num_bytes / 1024:.0f}KB"


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_size_token(token: str) -> int:
    token = token.strip()
    if not token:
        raise ValueError("empty size token")
    lower = token.lower()
    suffixes = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    for suffix, scale in suffixes.items():
        if lower.endswith(suffix):
            return int(float(lower[: -len(suffix)]) * scale)
    return int(lower)


def parse_packet_sizes(value: str) -> list[int]:
    sizes = [parse_size_token(token) for token in value.split(",") if token.strip()]
    if not sizes:
        raise ValueError("packet_sizes must not be empty")
    return sizes


def format_rate(num_bytes: int, seconds: float, unit: str) -> str:
    if seconds <= 0:
        return "inf"
    scale = RATE_UNITS[unit]
    return f"{num_bytes / seconds / scale:.2f} {unit}/s"


def infer_endpoint_name(server_name: str, rpc_port: int, metadata_server: str) -> str:
    if metadata_server != "P2PHANDSHAKE":
        return server_name
    host, _, _ = server_name.rpartition(":")
    return f"{host}:{rpc_port}" if host else server_name


def get_local_ip() -> str:
    env_ip = os.getenv("LOCAL_IP") or os.getenv("LOCAL_HOST_IP")
    if env_ip:
        return env_ip
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def normalize_host(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


def maybe_set_ascend_device(rank: int | None) -> None:
    if rank is None:
        return
    try:
        import torch  # type: ignore
        import torch_npu  # noqa: F401  # type: ignore
    except Exception:
        LOG.warning("torch/torch_npu not available, skip set_device(rank=%s)", rank)
        return

    if hasattr(torch, "npu"):
        torch.npu.set_device(rank)
        LOG.info("set Ascend device by rank: rank=%s", rank)


def server_name_from_rank(host: str, base_port: int, rank: int) -> str:
    return f"{host}:{base_port + rank}"


def resolve_local_server_name(args: argparse.Namespace, default_port: int) -> str:
    if getattr(args, "local_server_name", None):
        return args.local_server_name
    base_port = getattr(args, "base_port", default_port)
    rank = 0 if args.rank is None else args.rank
    return server_name_from_rank(args.local_host, base_port, rank)


def resolve_peer_server_names(args: argparse.Namespace) -> list[tuple[int, str]]:
    if getattr(args, "peer_server_name", None):
        peer_rank = args.peer_rank if args.peer_rank is not None else 0
        return [(peer_rank, args.peer_server_name)]
    peer_base_port = (
        args.peer_base_port if args.peer_base_port is not None else args.base_port
    )
    ranks = (
        [args.peer_rank]
        if args.peer_rank is not None
        else list(range(args.world_size))
    )
    return [
        (peer_rank, server_name_from_rank(args.peer_host, peer_base_port, peer_rank))
        for peer_rank in ranks
    ]


def resolve_peer_metadata(args: argparse.Namespace) -> list[tuple[int, WorkerMetadata]]:
    same_host = normalize_host(args.local_host) == normalize_host(args.peer_host)
    peer_base_port = (
        args.peer_base_port if getattr(args, "peer_base_port", None) is not None else args.base_port
    )
    peers: list[tuple[int, WorkerMetadata]] = []
    for peer_rank, peer_server_name in resolve_peer_server_names(args):
        if same_host and peer_rank == args.rank:
            LOG.info(
                "skip self peer for same-host transfer: rank=%s peer_rank=%s",
                args.rank,
                peer_rank,
            )
            continue
        if getattr(args, "peer_server_name", None):
            peers.append(
                (
                    peer_rank,
                    WorkerMetadata(
                        rank=peer_rank,
                        local_server_name=peer_server_name,
                        endpoint_name=peer_server_name,
                        buffer_addr=0,
                        buffer_capacity=0,
                        te_rpc_port=int(peer_server_name.rsplit(":", 1)[1]),
                    ),
                )
            )
            continue
        metadata = fetch_worker_metadata(
            args.peer_host,
            peer_base_port + peer_rank,
            timeout_seconds=max(getattr(args, "peer-connect-timeout", 30), 1),
        )
        peers.append((peer_rank, metadata))
    return peers


@dataclass
class BenchResult:
    name: str
    packet_size: int
    total_bytes: int
    seconds: float
    iterations: int
    success: bool = True
    status: str = "OK"
    error: str = ""

    def pretty(self, unit: str) -> str:
        if not self.success:
            return (
                f"{self.name:20s} size={format_size_kb(self.packet_size):>10s} "
                f"iters={self.iterations:6d} status={self.status} error={self.error}"
            )
        avg_ms = self.seconds * 1000.0 / max(self.iterations, 1)
        return (
            f"{self.name:20s} size={format_size_kb(self.packet_size):>10s} "
            f"iters={self.iterations:6d} avg={avg_ms:8.3f} ms "
            f"throughput={format_rate(self.total_bytes, self.seconds, unit)}"
        )


def failure_result(
    name: str,
    packet_size: int,
    iterations: int,
    error: Exception | str,
    status: str,
) -> BenchResult:
    return BenchResult(
        name=name,
        packet_size=packet_size,
        total_bytes=0,
        seconds=0.0,
        iterations=iterations,
        success=False,
        status=status,
        error=str(error),
    )

def init_worker_engine(args: argparse.Namespace) -> tuple[TransferEngine, str, int, int]:
    from mooncake.engine import TransferEngine

    maybe_set_ascend_device(args.rank)
    packet_sizes = parse_packet_sizes(args.packet_sizes)
    max_packet = max(packet_sizes)
    slot_count = max(args.batch_size, args.pipeline_depth, 1)
    buffer_capacity = max_packet * slot_count
    local_server_name = resolve_local_server_name(args, 12345)

    LOG.info("worker_init phase=set_device_done rank=%s local_server_name=%s", args.rank, local_server_name)
    engine = TransferEngine()
    LOG.info("worker_init phase=engine_initialize_start rank=%s", args.rank)
    ret = engine.initialize(
        local_server_name,
        args.metadata_server,
        "ascend",
        "",
    )
    if ret != 0:
        raise RuntimeError(f"worker initialize failed: {ret}")
    LOG.info("worker_init phase=engine_initialize_done rank=%s", args.rank)

    endpoint_name = infer_endpoint_name(
        local_server_name, engine.get_rpc_port(), args.metadata_server
    )
    LOG.info(
        "worker_init phase=resolve_endpoint_done rank=%s endpoint_name=%s",
        args.rank,
        endpoint_name,
    )
    try:
        import torch  # type: ignore

        LOG.info(
            "worker_init phase=allocate_tensor_buffer_start rank=%s buffer_capacity=%s",
            args.rank,
            buffer_capacity,
        )
        buffer_tensor = torch.empty(buffer_capacity, dtype=torch.uint8, device="npu")
        buffer_addr = buffer_tensor.data_ptr()
        LOG.info(
            "worker_init phase=allocate_tensor_buffer_done rank=%s buffer_addr=%s",
            args.rank,
            buffer_addr,
        )
        LOG.info(
            "worker_init phase=register_memory_start rank=%s buffer_addr=%s buffer_capacity=%s",
            args.rank,
            buffer_addr,
            buffer_capacity,
        )
        ret = engine.register_memory(buffer_addr, buffer_capacity)
        if ret != 0:
            raise RuntimeError(f"register_memory failed: {ret}")
        LOG.info(
            "worker_init phase=register_memory_done rank=%s buffer_addr=%s",
            args.rank,
            buffer_addr,
        )
        REGISTERED_BUFFER_TENSORS[args.rank] = buffer_tensor
    except Exception as exc:
        LOG.warning(
            "worker_init phase=register_memory_fallback rank=%s error=%s",
            args.rank,
            exc,
        )
        LOG.info(
            "worker_init phase=allocate_managed_buffer_start rank=%s buffer_capacity=%s",
            args.rank,
            buffer_capacity,
        )
        buffer_addr = engine.allocate_managed_buffer(buffer_capacity)
        if buffer_addr == 0:
            raise RuntimeError("failed to allocate worker managed buffer")
        LOG.info(
            "worker_init phase=allocate_managed_buffer_done rank=%s buffer_addr=%s",
            args.rank,
            buffer_addr,
        )

    return engine, endpoint_name, buffer_addr, buffer_capacity


def run_transfer_with_engine(
    args: argparse.Namespace,
    engine: TransferEngine,
    local_name: str,
    local_buffer_addr: int,
) -> list[BenchResult]:
    packet_sizes = parse_packet_sizes(args.packet_sizes)
    max_packet = max(packet_sizes)
    results: list[BenchResult] = []
    for peer_rank, peer_metadata in resolve_peer_metadata(args):
        peer_name = peer_metadata.endpoint_name
        peer_buffer_addr = peer_metadata.buffer_addr or engine.get_first_buffer_address(peer_name)
        peer_error: Exception | None = None
        if peer_buffer_addr == 0:
            peer_error = RuntimeError(
                f"peer rank={peer_rank} endpoint={peer_name} has no registered buffer"
            )
        for packet_size in packet_sizes:
            sync_name = f"transfer_sync_write[src={args.rank},dst={peer_rank}]"
            async_name = f"transfer_async_write[src={args.rank},dst={peer_rank}]"
            if peer_error is not None:
                results.append(
                    failure_result(
                        sync_name, packet_size, args.iterations, peer_error, "INIT_FAIL"
                    )
                )
                results.append(
                    failure_result(
                        async_name, packet_size, args.iterations, peer_error, "INIT_FAIL"
                    )
                )
                continue
            stride = max_packet
            src_addrs = build_address_list(local_buffer_addr, stride, args.batch_size)
            dst_addrs = build_address_list(peer_buffer_addr, stride, args.batch_size)
            try:
                sync_result = bench_transfer_sync(
                    engine,
                    peer_name,
                    src_addrs,
                    dst_addrs,
                    packet_size,
                    args.batch_size,
                    args.iterations,
                    args.warmup,
                )
                sync_result.name = sync_name
                results.append(sync_result)
            except Exception as exc:
                results.append(
                    failure_result(
                        sync_name,
                        packet_size,
                        args.iterations,
                        exc,
                        "TRANSFER_FAIL",
                    )
                )
            try:
                async_result = bench_transfer_async(
                    engine,
                    peer_name,
                    src_addrs,
                    dst_addrs,
                    packet_size,
                    args.batch_size,
                    args.iterations,
                    args.warmup,
                    args.pipeline_depth,
                )
                async_result.name = async_name
                results.append(async_result)
            except Exception as exc:
                results.append(
                    failure_result(
                        async_name,
                        packet_size,
                        args.iterations,
                        exc,
                        "TRANSFER_FAIL",
                    )
                )
    print(f"transfer_server_name={local_name}")
    print(f"transfer_rank={args.rank}")
    for result in results:
        print(result.pretty(args.report_unit))
    sys.stdout.flush()
    return results


def maybe_fast_exit_after_p2p(args: argparse.Namespace) -> None:
    if getattr(args, "skip_p2p_teardown", False):
        LOG.warning(
            "p2p phase=teardown_skipped rank=%s reason=avoid_ascend_native_teardown_crash",
            args.rank,
        )
        logging.shutdown()
        os._exit(0)


def run_worker(args: argparse.Namespace) -> int:
    try:
        engine, endpoint_name, buffer_addr, buffer_capacity = init_worker_engine(args)
    except Exception as exc:
        print(f"worker_init rank={args.rank} status=INIT_FAIL error={exc}")
        sys.stdout.flush()
        return 1

    local_server_name = resolve_local_server_name(args, 12345)
    metadata_host, _, metadata_port_str = local_server_name.rpartition(":")
    metadata_port = int(metadata_port_str)
    metadata = WorkerMetadata(
        rank=args.rank,
        local_server_name=local_server_name,
        endpoint_name=endpoint_name,
        buffer_addr=buffer_addr,
        buffer_capacity=buffer_capacity,
        te_rpc_port=int(endpoint_name.rsplit(":", 1)[1]),
    )
    metadata_server = WorkerMetadataServer(metadata_host, metadata_port, metadata)
    metadata_server.start()

    print(f"worker_server_name={endpoint_name}")
    print(f"worker_rank={args.rank}")
    print(f"worker_buffer_addr={buffer_addr}")
    print(f"worker_buffer_capacity={buffer_capacity}")
    print(f"worker_metadata_addr={metadata_host}:{metadata_port}")
    print("worker_ready=true")
    sys.stdout.flush()

    stop = False

    def _handle_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        if getattr(args, "peer_host", None) or getattr(args, "peer_server_name", None):
            wait_seconds = max(0, args.startup_wait_seconds)
            if wait_seconds:
                LOG.info("worker rank=%s waiting %ss before transfer", args.rank, wait_seconds)
                time.sleep(wait_seconds)
            run_transfer_with_engine(args, engine, endpoint_name, buffer_addr)
            linger_seconds = max(0, args.linger_seconds)
            if linger_seconds:
                LOG.info("worker rank=%s lingering %ss after transfer", args.rank, linger_seconds)
                time.sleep(linger_seconds)
            metadata_server.stop()
            maybe_fast_exit_after_p2p(args)
            return 0

        while not stop:
            time.sleep(1)
        maybe_fast_exit_after_p2p(args)
        return 0
    finally:
        metadata_server.stop()


def prepare_transfer_engine(args: argparse.Namespace) -> tuple[TransferEngine, str, int]:
    engine, endpoint_name, buffer_addr, _ = init_worker_engine(args)
    return engine, endpoint_name, buffer_addr


def build_address_list(base_addr: int, stride: int, batch_size: int) -> list[int]:
    return [base_addr + idx * stride for idx in range(batch_size)]


def bench_transfer_sync(
    engine: TransferEngine,
    peer_name: str,
    src_addrs: list[int],
    dst_addrs: list[int],
    packet_size: int,
    batch_size: int,
    iterations: int,
    warmup: int,
) -> BenchResult:
    lengths = [packet_size] * batch_size
    warmup_src_addrs = src_addrs[:1]
    warmup_dst_addrs = dst_addrs[:1]
    warmup_lengths = [packet_size]
    for _ in range(warmup):
        ret = engine.batch_transfer_sync_write(
            peer_name, warmup_src_addrs, warmup_dst_addrs, warmup_lengths
        )
        if ret != 0:
            raise RuntimeError(f"batch_transfer_sync_write failed during warmup: {ret}")
    start = time.perf_counter()
    for _ in range(iterations):
        ret = engine.batch_transfer_sync_write(peer_name, src_addrs, dst_addrs, lengths)
        if ret != 0:
            raise RuntimeError(f"batch_transfer_sync_write failed: {ret}")
    seconds = time.perf_counter() - start
    return BenchResult(
        name="transfer_sync_write",
        packet_size=packet_size,
        total_bytes=packet_size * batch_size * iterations,
        seconds=seconds,
        iterations=iterations,
    )


def bench_transfer_async(
    engine: TransferEngine,
    peer_name: str,
    src_addrs: list[int],
    dst_addrs: list[int],
    packet_size: int,
    batch_size: int,
    iterations: int,
    warmup: int,
    pipeline_depth: int,
) -> BenchResult:
    lengths = [packet_size] * batch_size
    warmup_src_addrs = src_addrs[:1]
    warmup_dst_addrs = dst_addrs[:1]
    warmup_lengths = [packet_size]
    for _ in range(warmup):
        batch_id = engine.batch_transfer_async_write(
            peer_name, warmup_src_addrs, warmup_dst_addrs, warmup_lengths
        )
        if batch_id == 0:
            raise RuntimeError("batch_transfer_async_write failed during warmup")
        ret = engine.get_batch_transfer_status([batch_id])
        if ret != 0:
            raise RuntimeError(f"get_batch_transfer_status failed during warmup: {ret}")

    inflight: list[int] = []
    start = time.perf_counter()
    for _ in range(iterations):
        batch_id = engine.batch_transfer_async_write(peer_name, src_addrs, dst_addrs, lengths)
        if batch_id == 0:
            raise RuntimeError("batch_transfer_async_write failed")
        inflight.append(batch_id)
        if len(inflight) >= pipeline_depth:
            ret = engine.get_batch_transfer_status([inflight.pop(0)])
            if ret != 0:
                raise RuntimeError(f"get_batch_transfer_status failed: {ret}")
    if inflight:
        ret = engine.get_batch_transfer_status(inflight)
        if ret != 0:
            raise RuntimeError(f"get_batch_transfer_status final wait failed: {ret}")
    seconds = time.perf_counter() - start
    return BenchResult(
        name="transfer_async_write",
        packet_size=packet_size,
        total_bytes=packet_size * batch_size * iterations,
        seconds=seconds,
        iterations=iterations,
    )


def run_transfer(args: argparse.Namespace) -> int:
    try:
        engine, local_name, local_buffer_addr = prepare_transfer_engine(args)
    except Exception as exc:
        print(f"transfer_init rank={args.rank} status=INIT_FAIL error={exc}")
        sys.stdout.flush()
        return 1
    run_transfer_with_engine(args, engine, local_name, local_buffer_addr)
    maybe_fast_exit_after_p2p(args)
    return 0


def setup_store(args: argparse.Namespace) -> MooncakeDistributedStore:
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    LOG.info(
        "store_init phase=setup_start rank=%s local_host=%s master_server=%s",
        args.rank,
        args.local_host,
        args.master_server,
    )
    ret = store.setup(
        args.local_host,
        args.metadata_server,
        args.global_segment_size,
        args.local_buffer_size,
        "ascend",
        "",
        args.master_server,
    )
    if ret != 0:
        raise RuntimeError(f"store.setup failed: {ret}")
    LOG.info("store_init phase=setup_done rank=%s", args.rank)
    return store


def allocate_store_buffers(size: int) -> tuple[object | None, object | None, int, int]:
    try:
        import torch  # type: ignore

        src_tensor = torch.empty(size, dtype=torch.uint8, device="npu")
        dst_tensor = torch.empty(size, dtype=torch.uint8, device="npu")
        return src_tensor, dst_tensor, src_tensor.data_ptr(), dst_tensor.data_ptr()
    except Exception:
        src_buffer = (ctypes.c_ubyte * size)()
        dst_buffer = (ctypes.c_ubyte * size)()
        return src_buffer, dst_buffer, ctypes.addressof(src_buffer), ctypes.addressof(dst_buffer)


def fill_store_buffer(buffer_obj: object, buffer_ptr: int, packet_size: int, seed: int) -> int:
    pattern_value = seed % 251
    try:
        import torch  # type: ignore

        if isinstance(buffer_obj, torch.Tensor):
            buffer_obj[:packet_size].fill_(pattern_value)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            return pattern_value
    except Exception:
        pass

    pattern = bytes([pattern_value]) * packet_size
    ctypes.memmove(ctypes.c_void_p(buffer_ptr), pattern, packet_size)
    return pattern_value


def verify_store_buffer(buffer_obj: object, packet_size: int, expected_value: int) -> None:
    try:
        import torch  # type: ignore

        if isinstance(buffer_obj, torch.Tensor):
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            expected = torch.full(
                (packet_size,),
                expected_value,
                dtype=torch.uint8,
                device=buffer_obj.device,
            )
            if not torch.equal(buffer_obj[:packet_size], expected):
                raise RuntimeError(
                    f"store get_into data verification failed for packet_size={packet_size}"
                )
            return
    except Exception:
        pass

    actual = bytes(buffer_obj[:packet_size])
    expected = bytes([expected_value]) * packet_size
    if actual != expected:
        raise RuntimeError(
            f"store get_into data verification failed for packet_size={packet_size}"
        )


def bench_store_put(
    store: MooncakeDistributedStore,
    src_ptr: int,
    packet_size: int,
    iterations: int,
    warmup: int,
) -> BenchResult:
    for idx in range(warmup):
        key = f"warmup_put_{packet_size}_{idx}"
        LOG.info("store_put phase=warmup rank_key=%s size=%s", key, packet_size)
        ret = store.put_from(key, src_ptr, packet_size)
        if ret != 0:
            raise RuntimeError(f"put_from warmup failed for key={key}: {ret}")
    start = time.perf_counter()
    for idx in range(iterations):
        key = f"bench_put_{packet_size}_{idx}"
        if idx == 0:
            LOG.info("store_put phase=bench_start first_key=%s size=%s", key, packet_size)
        ret = store.put_from(key, src_ptr, packet_size)
        if ret != 0:
            raise RuntimeError(f"put_from failed for key={key}: {ret}")
    seconds = time.perf_counter() - start
    return BenchResult(
        name="pool_put_from",
        packet_size=packet_size,
        total_bytes=packet_size * iterations,
        seconds=seconds,
        iterations=iterations,
    )


def bench_store_get(
    store: MooncakeDistributedStore,
    src_ptr: int,
    dst_buffer: object,
    dst_ptr: int,
    packet_size: int,
    iterations: int,
    warmup: int,
    expected_value: int,
) -> BenchResult:
    keys = [f"bench_get_{packet_size}_{idx}" for idx in range(iterations)]
    for key in keys:
        if key.endswith("_0"):
            LOG.info("store_get phase=prepare_put_start first_key=%s size=%s", key, packet_size)
        ret = store.put_from(key, src_ptr, packet_size)
        if ret != 0:
            raise RuntimeError(f"prepare put_from failed for key={key}: {ret}")

    for idx in range(min(warmup, len(keys))):
        if idx == 0:
            LOG.info("store_get phase=warmup_start first_key=%s size=%s", keys[idx], packet_size)
        bytes_read = store.get_into(keys[idx], dst_ptr, packet_size)
        if bytes_read != packet_size:
            raise RuntimeError(f"get_into warmup failed for key={keys[idx]}: {bytes_read}")

    start = time.perf_counter()
    for key in keys:
        if key.endswith("_0"):
            LOG.info("store_get phase=bench_start first_key=%s size=%s", key, packet_size)
        bytes_read = store.get_into(key, dst_ptr, packet_size)
        if bytes_read != packet_size:
            raise RuntimeError(f"get_into failed for key={key}: {bytes_read}")
    seconds = time.perf_counter() - start

    verify_store_buffer(dst_buffer, packet_size, expected_value)

    return BenchResult(
        name="pool_get_into",
        packet_size=packet_size,
        total_bytes=packet_size * iterations,
        seconds=seconds,
        iterations=iterations,
    )


def run_store_bench(args: argparse.Namespace) -> int:
    maybe_set_ascend_device(args.rank)
    packet_sizes = parse_packet_sizes(args.packet_sizes)
    max_packet = max(packet_sizes)
    try:
        store = setup_store(args)
    except Exception as exc:
        print(f"store_init rank={args.rank} status=INIT_FAIL error={exc}")
        sys.stdout.flush()
        return 1
    src_buffer, dst_buffer, src_ptr, dst_ptr = allocate_store_buffers(max_packet)

    try:
        LOG.info("store_init phase=register_src_start rank=%s ptr=%s size=%s", args.rank, src_ptr, max_packet)
        ret = store.register_buffer(src_ptr, max_packet)
        if ret != 0:
            raise RuntimeError(f"register_buffer(src) failed: {ret}")
        LOG.info("store_init phase=register_src_done rank=%s ptr=%s", args.rank, src_ptr)
        LOG.info("store_init phase=register_dst_start rank=%s ptr=%s size=%s", args.rank, dst_ptr, max_packet)
        ret = store.register_buffer(dst_ptr, max_packet)
        if ret != 0:
            raise RuntimeError(f"register_buffer(dst) failed: {ret}")
        LOG.info("store_init phase=register_dst_done rank=%s ptr=%s", args.rank, dst_ptr)
    except Exception as exc:
        try:
            store.close()
        finally:
            print(f"store_init rank={args.rank} status=INIT_FAIL error={exc}")
            sys.stdout.flush()
        return 1

    results: list[BenchResult] = []
    try:
        for packet_size in packet_sizes:
            LOG.info("store_bench phase=packet_start rank=%s size=%s", args.rank, packet_size)
            expected_value = fill_store_buffer(
                src_buffer, src_ptr, packet_size, seed=17 + packet_size
            )
            try:
                results.append(
                    bench_store_put(store, src_ptr, packet_size, args.iterations, args.warmup)
                )
            except Exception as exc:
                results.append(
                    failure_result(
                        "pool_put_from",
                        packet_size,
                        args.iterations,
                        exc,
                        "STORE_FAIL",
                    )
                )
            try:
                results.append(
                    bench_store_get(
                        store,
                        src_ptr,
                        dst_buffer,
                        dst_ptr,
                        packet_size,
                        args.iterations,
                        args.warmup,
                        expected_value,
                    )
                )
            except Exception as exc:
                results.append(
                    failure_result(
                        "pool_get_into",
                        packet_size,
                        args.iterations,
                        exc,
                        "STORE_FAIL",
                    )
                )
            try:
                LOG.info("store_bench phase=remove_all_start rank=%s size=%s", args.rank, packet_size)
                store.remove_all()
                LOG.info("store_bench phase=remove_all_done rank=%s size=%s", args.rank, packet_size)
            except Exception as exc:
                LOG.warning("store.remove_all failed for packet_size=%s: %s", packet_size, exc)
    finally:
        if args.skip_store_teardown:
            LOG.warning(
                "store_init phase=teardown_skipped rank=%s reason=avoid_ascend_native_teardown_crash",
                args.rank,
            )
        else:
            try:
                LOG.info("store_init phase=unregister_src_start rank=%s ptr=%s", args.rank, src_ptr)
                store.unregister_buffer(src_ptr)
                LOG.info("store_init phase=unregister_src_done rank=%s ptr=%s", args.rank, src_ptr)
            finally:
                LOG.info("store_init phase=unregister_dst_start rank=%s ptr=%s", args.rank, dst_ptr)
                store.unregister_buffer(dst_ptr)
                LOG.info("store_init phase=unregister_dst_done rank=%s ptr=%s", args.rank, dst_ptr)
                LOG.info("store_init phase=close_start rank=%s", args.rank)
                store.close()
                LOG.info("store_init phase=close_done rank=%s", args.rank)

    print(f"store_hostname={store.get_hostname()}")
    print(f"store_rank={args.rank}")
    for result in results:
        print(result.pretty(args.report_unit))
    return 0


def add_common_perf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet-sizes", default="1M,4M,16M", help="Comma separated packet sizes, e.g. 1M,4M,16M")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations per packet size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per packet size")
    parser.add_argument("--report-unit", choices=sorted(RATE_UNITS), default="GB", help="Throughput display unit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mooncake Ascend benchmark helper")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", "0")))
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "1")),
        help="Number of ranks/processes per server",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", help="Start transfer-engine worker endpoint")
    worker.add_argument("--local-server-name")
    worker.add_argument("--local-host", default=get_local_ip())
    worker.add_argument("--base-port", type=int, default=12345)
    worker.add_argument("--peer-server-name")
    worker.add_argument("--peer-host")
    worker.add_argument("--peer-base-port", type=int, help=argparse.SUPPRESS)
    worker.add_argument("--peer-rank", type=int, help="Only benchmark one peer rank; default sweeps all ranks")
    worker.add_argument("--peer-connect-timeout", type=int, default=30, help="Seconds to wait for peer metadata")
    worker.add_argument("--metadata-server", default=os.getenv("MC_METADATA_SERVER", "P2PHANDSHAKE"))
    worker.add_argument("--batch-size", type=int, default=4)
    worker.add_argument("--pipeline-depth", type=int, default=8)
    worker.add_argument(
        "--skip-p2p-teardown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip explicit process teardown after active p2p benchmark; enabled by default for Ascend stability",
    )
    worker.add_argument("--startup-wait-seconds", type=int, default=15, help="Wait before active transfer when peer host is set")
    worker.add_argument("--linger-seconds", type=int, default=0, help="Keep worker alive for a bit after transfer completes")
    add_common_perf_args(worker)
    worker.set_defaults(func=run_worker)

    transfer = subparsers.add_parser("transfer", help="Run transfer benchmark from this worker set to peer workers")
    transfer.add_argument("--local-server-name")
    transfer.add_argument("--peer-server-name")
    transfer.add_argument("--local-host", default=get_local_ip())
    transfer.add_argument("--peer-host", default=os.getenv("PEER_HOST", get_local_ip()))
    transfer.add_argument("--base-port", type=int, default=12345)
    transfer.add_argument("--peer-base-port", type=int, help=argparse.SUPPRESS)
    transfer.add_argument("--peer-rank", type=int, help="Only benchmark one peer rank; default sweeps all ranks")
    transfer.add_argument("--peer-connect-timeout", type=int, default=30, help="Seconds to wait for peer metadata")
    transfer.add_argument("--metadata-server", default=os.getenv("MC_METADATA_SERVER", "P2PHANDSHAKE"))
    transfer.add_argument("--batch-size", type=int, default=4)
    transfer.add_argument("--pipeline-depth", type=int, default=8)
    transfer.add_argument(
        "--skip-p2p-teardown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip explicit process teardown after p2p benchmark; enabled by default for Ascend stability",
    )
    add_common_perf_args(transfer)
    transfer.set_defaults(func=run_transfer)

    store = subparsers.add_parser("store", help="Run store put/get benchmark")
    store.add_argument(
        "--local-host",
        dest="local_host",
        default=os.getenv("LOCAL_HOSTNAME", get_local_ip()),
    )
    store.add_argument(
        "--local-hostname",
        dest="local_host",
        help=argparse.SUPPRESS,
    )
    store.add_argument("--metadata-server", default=os.getenv("MC_METADATA_SERVER", "P2PHANDSHAKE"))
    store.add_argument("--master-server", default=os.getenv("MASTER_SERVER", "127.0.0.1:50051"))
    store.add_argument("--global-segment-size", type=parse_size_token, default=parse_size_token("4G"))
    store.add_argument("--local-buffer-size", type=parse_size_token, default=parse_size_token("2G"))
    store.add_argument(
        "--skip-store-teardown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip explicit unregister/close at process end; enabled by default for Ascend stability",
    )
    add_common_perf_args(store)
    store.set_defaults(func=run_store_bench)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
