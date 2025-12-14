# GPU L1/L2：VIVT vs VIPT/PIPT 的微基准（Pointer-Chase + TLB Sweep）

本仓库的 `TLB_test/VIVT/4090` 目录实现了一个**指针追逐（pointer-chase）**微基准，用于复现你图中提到的思路：通过对 **TLB 覆盖范围**做 sweep，并对比 **L1 启用/禁用**两种访问路径，推断 GPU 的 L1/L2 data cache 更接近 **VIVT** 还是 **VIPT/PIPT**。

---

## 目录结构

- `TLB_test/VIVT/4090/src/main.cu`：命令行 + 内存分配 + 统计输出
- `TLB_test/VIVT/4090/src/tlb_kernels.cuh`：`init_ring` + `chase` kernel
- `TLB_test/VIVT/scripts/sweep.py`：按 `stride/bytes/policy` 扫描并写 CSV（所有 GPU 共用）
- `TLB_test/VIVT/scripts/plot_latency.py`：读取 CSV 画图（输出 `*.cg.png` / `*.ca.png`，所有 GPU 共用）

---

## 测量方法（对应你图中的“TLB warm-up scan + pointer chase”）

### 1) 构造 “每个 stride 一个节点” 的环

对长度为 `bytes` 的 device buffer，以间隔 `stride` 放置节点：

- 节点数：`nodes = bytes / stride`
- 每个节点只存一个 64-bit “下一跳地址”（device VA）
- `--step` 用于把访问顺序“打散”（`node[i] -> node[(i+step)%nodes]`），减弱顺序预取/投机

### 2) 两遍扫描：warm-up + timed

kernel 内部做两遍（或多遍）pointer-chase：

1. **warm-up**：访问完整一圈，用于 page-in / 填充 cache / 形成稳定的 TLB 状态
2. **timed**：再访问若干圈，使用 `clock64` 计时，输出平均 `cycles/access`

> 只用单线程做依赖链 load，避免并行隐藏延迟。

### 3) 两种 cache policy = L1 启用/禁用（对应你图的对比思路）

微基准用内联 PTX 强制 load 策略：

- `--policy ca`：`ld.global.ca`（缓存到 L1+L2）≈ **L1 enabled**
- `--policy cg`：`ld.global.cg`（绕过 L1，仅缓存到 L2）≈ **L1 disabled / L2 path**

---

## 为什么能用它判断 VIVT / VIPT / PIPT？

关键是选 **足够大的 stride**（典型：`2MiB`、`32MiB`）：

- 每次只触碰一个 cache line（每页仅 8B 指针，但会带入一条 line）
- 因此即便 `bytes` 很大、触及很多虚拟页，**cache footprint** 仍可能很小（容易一直命中 L1 或 L2）
- 一旦 `bytes` 变大到超过某级 TLB 的 coverage，就会出现 **TLB miss 台阶**（如果该访问路径依赖地址翻译）

### 推断规则（实操）

1. **看 `policy=ca`（L1 enabled）**
   - 在 “明显是 L1 命中” 的区间（延迟显著低于 `cg`，通常几十 cycles/access）
   - 如果随着 `bytes` 增大、跨过 TLB coverage **没有出现台阶**：
     - 说明访问 **L1 命中**时不需要 TLB 翻译进入关键路径  
     - 结论：L1 更符合 **VIVT（或至少 VA-indexed 且翻译不在关键路径）**  
   - 如果出现台阶：说明 L1 命中仍受翻译影响，更接近 **VIPT/PIPT**（仅靠此实验一般难再区分 VIPT vs PIPT）

2. **看 `policy=cg`（L1 bypass，走 L2）**
   - 若随着 `bytes` 增大出现明显台阶（尤其在更大 `bytes` 时出现更大幅度的台阶）：
     - 说明访问 **L2 命中**时仍需要地址翻译
     - 结论：L2 更符合 **PIPT（physical cache）**
   - 若没有台阶：通常是 **还没扫到足够大**（例如 L2-TLB coverage 很大），把 `--max_mem_frac` 调大或直接指定更大的 `--max_bytes`（允许每个 stride 都扫到同一个最大 array size）

---

## 一键流程图（原理）

```mermaid
flowchart TD
  A[Choose stride(s)\n(2MiB/32MiB ...)] --> B[Choose bytes sweep\n(bytes = stride * pages)]
  B --> C[init_ring\n(one node per stride)]
  C --> D[warm-up scan\n(fill cache/TLB)]
  D --> E[timed pointer-chase\n(dependent loads)]
  E --> F[cycles/access vs bytes]
  F --> G{policy}
  G -->|ca (L1 enabled)| H[Focus on L1-hit region\n(low latency)]
  H --> I{Latency steps when\npages > TLB coverage?}
  I -->|No| J[L1 not translation-bound\n=> consistent with VIVT]
  I -->|Yes| K[L1 translation-bound\n=> VIPT/PIPT-like]
  G -->|cg (bypass L1)| L[Access served from L2]
  L --> M{Latency steps when\npages > TLB coverage?}
  M -->|Yes| N[L2 involves TLB\n=> PIPT-like]
  M -->|No| O[Increase sweep range\n(maybe not exceeding coverage)]
```

---

## 怎么跑（4090 目录）

编译：

```bash
make -C TLB_test/VIVT/4090
TLB_test/VIVT/4090/bin/tlbbench --info
```

扫描并生成 `results/results.csv`（默认会覆盖；需要追加用 `--append`）。默认 stride 以大页为主（2MiB/8MiB/32MiB/64MiB），方便直接复现论文式的 TLB 台阶：

 ```bash
 python3 TLB_test/VIVT/scripts/sweep.py \
   --exe TLB_test/VIVT/4090/bin/tlbbench \
   --out TLB_test/VIVT/4090/results/results.csv
 ```

只跑指定 stride（支持 `MiB/GiB` 等后缀）：

```bash
python3 TLB_test/VIVT/scripts/sweep.py \
  --exe TLB_test/VIVT/4090/bin/tlbbench \
  --out TLB_test/VIVT/4090/results/results.csv \
  --strides 2MiB 32MiB
```

用 stride “范围”生成（等比序列，适合扫 512KiB/1MiB/2MiB/... 这种）：

```bash
python3 TLB_test/VIVT/scripts/sweep.py \
  --exe TLB_test/VIVT/4090/bin/tlbbench \
  --out TLB_test/VIVT/4090/results/results.csv \
  --stride_min 512KiB --stride_max 64MiB --stride_factor 2
```

画图：

```bash
python3 TLB_test/VIVT/scripts/plot_latency.py \
  TLB_test/VIVT/4090/results/results.csv \
  --out TLB_test/VIVT/4090/results/figs/latency.png
```

横轴改成 `nodes(=bytes/stride)`（不同 stride 的台阶会对齐，便于看“TLB entry 数”而不是“覆盖的字节数”）：

```bash
python3 TLB_test/VIVT/scripts/plot_latency.py \
  TLB_test/VIVT/4090/results/results.csv \
  --out TLB_test/VIVT/4090/results/figs/latency.nodes.png \
  --x nodes
```

输出文件：

- `TLB_test/VIVT/4090/results/figs/latency.ca.png`（L1 enabled）
- `TLB_test/VIVT/4090/results/figs/latency.cg.png`（L1 bypass / L2）

> 画图脚本的横轴是 MiB 为单位的人类可读刻度（仍用 log2 间距，便于跨多个数量级对齐台阶）。
