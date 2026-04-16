# Mooncake Ascend Test

这个目录整理了 Mooncake Ascend 性能测试相关脚本，方便在独立环境里直接使用。

## 文件

- `mooncake_ascend_bench.py`
  Python 测试脚本，负责执行 P2P 和 store 压测。
- `run_mooncake_ascend_bench.sh`
  Shell 启动脚本，负责按本机 `local-world-size` 拉起多进程并汇总结果。

## 前置条件

- 已安装 `mooncake`
- Ascend 运行环境已就绪
- 如果需要自动 `set_device(rank)`，环境里应可导入 `torch` 和 `torch_npu`
- 双机 P2P 场景下，两台机器网络互通，`local-host` 和 `peer-host` 可互相访问

## 环境变量

这些环境变量会被当前脚本直接读取，可作为命令行参数的默认值：

- `PYTHON_BIN`
  shell 脚本使用的 Python 解释器，默认值为 `python3`
- `MC_METADATA_SERVER`
  默认 metadata server，默认值为 `P2PHANDSHAKE`
- `MASTER_SERVER`
  Python `store` 子命令默认 master 服务地址，默认值为 `127.0.0.1:50051`
- `RANK`
  Python 脚本默认 rank
- `WORLD_SIZE`
  Python 脚本默认本机 `local-world-size`
- `PEER_HOST`
  Python `transfer` 子命令默认对端地址
- `LOCAL_IP` / `LOCAL_HOST_IP`
  Python 脚本推导本机地址时优先使用

另外，shell 脚本会为以下 Ascend 通信环境变量提供默认值，并在执行时导出：

- `HCCL_INTRA_ROCE_ENABLE=1`
- `HCCL_RDMA_TIMEOUT=17`
- `ASCEND_CONNECT_TIMEOUT=10000`
- `ASCEND_TRANSFER_TIMEOUT=10000`
- `ACL_OP_INIT_MODE=1`
- `PYTHONHASHSEED=0`

如需使用其他配置，可在执行脚本前显式覆盖。

## 任意目录运行

这两个脚本都可以在任意目录运行，但建议用绝对路径调用，例如：

```bash
bash "path/to/your/mooncake test/run_mooncake_ascend_bench.sh" --help
python3 "path/to/your/mooncake test/mooncake_ascend_bench.py" --help
```

原因是：

- shell 脚本会自动定位同目录下的 Python 脚本
- Python 脚本本身不依赖当前工作目录

## Shell 脚本用法

### 1. P2P 模式

用于测试 Ascend `device -> remote device` 的传输性能：

- `transfer_sync_write`
- `transfer_async_write`

示例：

```bash
bash "path/to/your/mooncake test/run_mooncake_ascend_bench.sh" \
  --mode p2p \
  --local-world-size 8 \
  --local-host 10.20.130.155 \
  --peer-host 10.20.130.154 \
  --base-port 12345 \
  --packet-sizes 1M,4M,16M \
  --iterations 100 \
  --warmup 10
```

说明：

- 本机会按 `rank=0..local-world-size-1` 拉起多个进程
- 每个进程默认执行 `set_device(rank)`
- 每个 `worker` 会在 `base_port + rank` 发布 metadata，主动端据此自动发现对端真实 `TransferEngine` endpoint
- 每个本地 rank 会遍历对端 rank，并在测试结束后输出带宽矩阵汇总
- shell 的 `p2p` 调度默认按 `src rank` 轮巡执行，不并发发起多组测量
- 单机同 host 的被动 `worker` 会在整轮 `p2p` summary 和 CSV 输出完成后再统一释放
- 当 `local-host` 和 `peer-host` 解析到同一台机器时，默认跳过 `src_rank == dst_rank` 的自连路径
- 当 `local-host` 和 `peer-host` 为同一台机器时，shell 会自动采用“被动 worker + 独立 transfer 进程”的方式避免 rank 提前退出
- 单向 P2P 测试可采用“对端启动 worker，本端执行 `--mode p2p`”的方式

对端机器示例：

```bash
python3 "path/to/your/mooncake test/mooncake_ascend_bench.py" \
  --rank 0 \
  --local-world-size 8 \
  worker \
  --local-host 10.20.130.154 \
  --base-port 12345 \
  --packet-sizes 1M,4M,16M
```

其余 `rank` 可按相同方式分别启动。

本机示例：

```bash
bash "path/to/your/mooncake test/run_mooncake_ascend_bench.sh" \
  --mode p2p \
  --local-world-size 8 \
  --local-host 10.20.130.155 \
  --peer-host 10.20.130.154
```

### 2. Store 模式

用于测试 Ascend store 路径：

- `put_from`
- `get_into`

示例：

```bash
bash "path/to/your/mooncake test/run_mooncake_ascend_bench.sh" \
  --mode store \
  --local-world-size 8 \
  --local-host 10.20.130.155 \
  --master-server 127.0.0.1:50051 \
  --packet-sizes 1M,4M,16M \
  --iterations 100 \
  --warmup 10
```

说明：

- `--local-host` 用于指定 `store.setup(...)` 的本地节点名
- shell 脚本会将 `--local-host` 原样透传给 Python `store` 子命令
- Ascend 场景建议使用实际可达 IP，不建议使用 `localhost`
- shell 的 `store` 调度默认按 `rank` 轮巡执行，每次只运行一个 rank
- `store` 结果会在终端打印按 `packet_size` 和操作类型拆分的 rank 汇总表，并在 `--log-dir` 下写出对应 CSV
- 如环境要求使用特定节点名，可显式传入 `--local-host your-hostname`
- `store` 模式默认跳过显式 `unregister_buffer/close`，用于规避 Ascend 环境下进程退出阶段的 native teardown crash
- `p2p` 模式默认在结果输出后快速退出，用于规避 Ascend 环境下进程退出阶段的 native teardown crash

### 3. All Single Node 模式

`all` 和 `all-single-node` 等价。

它会在本机按顺序先跑 P2P，再跑 store。

示例：

```bash
bash "path/to/your/mooncake test/run_mooncake_ascend_bench.sh" \
  --mode all-single-node \
  --local-world-size 8 \
  --local-host 10.20.130.155 \
  --peer-host 10.20.130.154 \
  --base-port 12345 \
  --master-server 127.0.0.1:50051 \
  --packet-sizes 1M,4M,16M \
  --iterations 100 \
  --warmup 10
```

### 4. All Multi Node 模式

用于双机顺序联动测试，覆盖：

1. 主机 A 机内 P2P
2. 主机 A 到主机 B 的 P2P
3. 主机 B 机内 P2P
4. 主机 B 到主机 A 的 P2P
5. 主机 A store
6. 主机 B store

该模式需要两台机器分别执行脚本，命令形式保持一致。

脚本会根据 `local-host` 和 `peer-host` 自动确定执行顺序，并通过 `--control-port` 进行阶段同步。

机器 A：

```bash
bash "path/to/your/mooncake test/run_mooncake_ascend_bench.sh" \
  --mode all-multi-node \
  --local-world-size 8 \
  --local-host 10.20.130.155 \
  --peer-host 10.20.130.154 \
  --base-port 12345 \
  --control-port 29600 \
  --startup-wait-seconds 15 \
  --master-server 127.0.0.1:50051 \
  --packet-sizes 1M,4M,16M \
  --iterations 100 \
  --warmup 10
```

机器 B：

```bash
bash "path/to/your/mooncake test/run_mooncake_ascend_bench.sh" \
  --mode all-multi-node \
  --local-world-size 8 \
  --local-host 10.20.130.154 \
  --peer-host 10.20.130.155 \
  --base-port 12345 \
  --control-port 29600 \
  --startup-wait-seconds 15 \
  --master-server 127.0.0.1:50051 \
  --packet-sizes 1M,4M,16M \
  --iterations 100 \
  --warmup 10
```

说明：

- 两边的 `peer-host` 需要互相指向对端机器
- `control-port` 需要在两端之间可访问
- 脚本会自动确定哪一侧作为全局矩阵的前半区块
- P2P 阶段会按 `A->A`、`A->B`、`B->B`、`B->A` 的顺序执行
- 主机 A 侧会在 `all_multi_node_p2p` 目录下汇总输出完整 `16x16` 矩阵和对应 CSV
- 当前使用方式面向双机场景

## 日志和结果

- 默认日志目录：`path/to/your/mooncake test/logs`
- shell 脚本默认会实时打印各 rank 的日志；如需仅写文件，可传 `--stream-logs 0`
- P2P 模式结束后会自动输出 `src_rank x dst_rank` 的矩阵汇总
- P2P 模式结束后会在日志目录额外生成 CSV 文件，例如 `transfer_sync_write_size_1024KB.csv`
- `all-multi-node` 模式的主机 A 侧会额外输出完整多机矩阵，例如 `global_transfer_sync_write_size_1024KB.csv`
- 单条结果中的 `size` 以 `KB` 显示

## Python 脚本直接用法

Python 脚本提供以下常用子命令：

- `worker`
- `store`

说明：

- 常用场景建议通过 shell 脚本使用 `p2p`、`store`、`all-single-node` 和 `all-multi-node`
- 直接调用 Python 脚本时，可优先使用 `worker` 和 `store`

查看帮助：

```bash
python3 "path/to/your/mooncake test/mooncake_ascend_bench.py" --help
python3 "path/to/your/mooncake test/mooncake_ascend_bench.py" worker --help
python3 "path/to/your/mooncake test/mooncake_ascend_bench.py" store --help
```
