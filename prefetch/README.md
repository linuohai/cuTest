# cuTest/prefetch

目标：在尽量关闭延迟隐藏（单 SM/单 warp）的前提下，观察是否存在“通用硬件自动 prefetch”签名，并用显式 `prefetch.global.L2` / `cp.async` 做上界对照。

## 运行方式（只在 `fa` 容器内执行）

在宿主机运行（仅负责 `docker cp/exec`，不会在宿主机直接跑 GPU 代码）：

```bash
bash cuTest/prefetch/run_fa.sh
```

输出会写入 `cuTest/prefetch/out/<timestamp>/`，包含：
- `exp1_stride.csv` / `stride_latency.png` / `stride_l2hit.png` / `workingset_boundary.png`
- `exp2_pointer_chase.csv` / `pointer_chase_latency.png`
- `exp3_gather.csv` / `gather_latency.png` / `gather_l2hit.png`
- `exp4_explicit.csv` / `explicit_vs_baseline.png`
- `env.json`（环境信息）与 `ncu/`（每个配置的原始 ncu csv）

## 微基准（单 SM/单 warp）

所有 kernel 都以 `grid=(1,1,1)`、`block=(32,1,1)` 启动，只有 lane0 执行访存循环，尽量避免 occupancy/并发带来的延迟隐藏。

- **Stream/Stride**：`stride_baseline_kernel`（显式对照：`stride_prefetch_l2_kernel` / `stride_cp_async_kernel`）
- **Pointer-chase**：`pointer_chase_kernel`（LCG 置换链，作为不可预取反例）
- **Gather/IMA**：`gather_ima_kernel`（`j = indices[i]` 后 `data[j]` 的间接访存；支持 hotset 分布）

## ncu 指标口径（表中字段）

- `L1 hit(%)`：`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum / (hit+miss)`
- `L2 hit(%)`：`lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum / (hit+miss)`
- `DRAM read`：`dram__sectors_read.sum`（单位为 32B sector）
- `stall_long_scoreboard_pct`：`smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`

补充：为避免“power-of-two 工作集 + power-of-two stride”导致短周期循环（提前回绕带来假复用），`stride_*` 内部对工作集长度取了一个奇数 modulus（`N` 为偶数时用 `N-1`）。
