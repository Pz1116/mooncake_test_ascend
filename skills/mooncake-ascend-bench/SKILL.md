---
name: mooncake-ascend-bench
description: Use when running or debugging Mooncake Ascend P2P/store benchmarks, especially on single-host multi-card setups where TransferEngine/store teardown or highly concurrent connect patterns can cause misleading failures.
---

# Mooncake Ascend Bench

Use this skill when working with Mooncake Ascend benchmark scripts such as `mooncake_ascend_bench.py` and `run_mooncake_ascend_bench.sh`.

## Scope

Apply this skill for:
- Mooncake Ascend `p2p` benchmark runs
- Mooncake Ascend `store` benchmark runs
- Single-host multi-card benchmark instability analysis
- Double-checking whether failures come from the data path or teardown/launch strategy

## Key constraints

- `protocol=ascend` uses real TE RPC ports chosen at runtime under `P2PHANDSHAKE`; do not assume `base_port + rank` is the final TE RPC port.
- Use `base_port + rank` only as the peer discovery/metadata address.
- On a single host, do not benchmark `src_rank == dst_rank`; skip self-pairs by default.
- On a single host, do not launch all active sources concurrently. Prefer round-robin execution by `src rank`.
- On a single host, do not run a passive worker and an active transfer process on the same rank/device at the same time.
- For same-host runs, start passive workers only for destination ranks of the current source rank.
- Multi-node config should use a shared `hosts` list plus shared benchmark parameters; host IPs are the primary machine identity.
- Current `all-multi-node` support is limited to exactly 2 machines. Do not assume 3+ hosts are supported.

## P2P guidance

- Keep passive workers alive until the whole P2P round finishes; do not tear them down mid-round.
- For same-host runs, execute active transfers sequentially, one source rank at a time.
- Warmup should be lighter than measured iterations. Prefer a minimal warmup footprint before the real timed batch.
- If a result line is printed and failures appear only during native cleanup, treat it as a teardown problem, not a transfer-path failure.
- If many pairs fail with `Failed to connect to target` only under concurrent same-host runs, suspect connect pressure or device/process conflict before suspecting metadata parsing.
- For 2-host `all-multi-node`, the P2P execution order is:
  1. `A -> A`
  2. `A -> B`
  3. `B -> B`
  4. `B -> A`
- For 2-host `all-multi-node`, the primary host merges the four `8x8` blocks into one global `16x16` summary/CSV when `local_world_size=8`.
- Multi-node P2P should be treated as `(host, local_rank)` addressing, not a single global rank space, unless a dedicated global-rank mapping layer is added.

## Store guidance

- For Ascend store runs, use `--local-host` with a real reachable IP, not `localhost`.
- Register NPU/device buffers for `put_from/get_into`; host-side ctypes buffers can fail on Ascend.
- If `put/get/remove_all` succeed and the process crashes only during unregister/close, treat it as a teardown issue.
- Default store runs may skip explicit teardown when Ascend native cleanup is unstable.
- Store shell runs currently use round-robin execution by local rank and produce per-packet-size, per-operation rank summaries plus CSV files.

## How to interpret failures

- `INIT_FAIL`: setup or registration problem.
- `TRANSFER_FAIL`: the data path failed during warmup or measured transfer.
- `STORE_FAIL`: the store data path failed during `put_from/get_into`.
- Teardown-only native crashes after results are printed should be separated from benchmark-path failures.

## Recommended execution model

- Same-host P2P:
  1. Start passive workers for destination ranks only.
  2. Run one active source rank.
  3. Stop passive workers.
  4. Move to the next source rank.
- Two-host P2P:
  1. Keep the passive side stable for each current stage.
  2. Run active source ranks in round-robin order rather than all at once when stability matters.
  3. For full-matrix runs, execute `A->A`, `A->B`, `B->B`, `B->A` in that order and merge the blocks afterwards.
- Store:
  1. Validate a single-rank minimal case first.
  2. Expand to multi-rank after buffer registration and teardown behavior are understood.

## Current multi-node limits

- `all-multi-node` is implemented only for 2 hosts today.
- The current control-plane logic uses point-to-point synchronization between two machines only.
- The current global matrix merge logic assumes exactly four blocks and does not generalize to 3+ machines.

## TODO

- Extend `all-multi-node` to 3, 4, and more hosts.
- Add a general multi-host scheduler/barrier instead of pairwise control messages.
- Add a generalized `N * local_world_size` global matrix merge path.
