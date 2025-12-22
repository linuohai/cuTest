# L1 store 策略微基准：allocate / evict / write-through vs write-back

这个目录提供一个可用 `nsight-compute (ncu)` 直接 profile 的小程序，用来验证你提出的 3 个判断思路，并且支持切换不同的 PTX store cache op（`st.global.{wb,wt,cg,cs}`）做对照。

> 重点：命中率的“50% / 66% / 33%”取决于你看的是不是“load+store 合并”指标；如果只看 load 指标，比例会不一样（见下文）。

---

## 文件组织（按测试目的拆分）

- `cache/strategy/main.cu`：命令行解析 + host 侧分配/launch
- `cache/strategy/ptx_cache_ops.cuh`：`ld.global.{ca,cg}` / `st.global.{wb,wt,cg,cs}` 封装
- `cache/strategy/alloc_cross_thread_kernels.cuh`：跨线程（store→load）排除 forwarding 的 alloc 对照
- `cache/strategy/alloc_write_read_kernels.cuh`：写后读（allocate vs non-allocate）相关 kernel
- `cache/strategy/store_only_kernels.cuh`：纯 store-only（stst）对照 kernel
- `cache/strategy/evict_read_write_read_kernels.cuh`：读-写-读（evict vs not-evict）相关 kernel
- `cache/strategy/wb_wt_visibility_kernels.cuh`：跨 block 的 L2 可见性（write-back vs write-through）相关 kernel

---

## 你提出的 3 条逻辑是否正确？

### 1) allocate vs non-allocate（写后读）

思路整体正确，但要注意 Nsight 里到底统计了哪些 op：

- 如果你看的是 **L1 lookup 的 load+store 合并 hit-rate**：  
  - allocate：`st`(miss) + `ld`(hit) → ≈ `1/2 = 50%`  
  - non-allocate：`st` 不进入 L1 或 miss + `ld`(miss) → ≈ `0%`
- 如果你看的是 **只统计 load 的 hit-rate**：  
  - allocate：`ld` 应该 hit → `100%`  
  - non-allocate：`ld` miss → `0%`

本仓库的 kernel 会对 **不同 cache op 的 store**分别生成独立 kernel，便于你在 Nsight 里对比。

#### allocate 测试详解（写后读）

**测试逻辑**
- 单 block / 单 thread，循环 `iters` 次。
- 每次访问地址：`p = base + i * stride_bytes`（保证是不同 cache line）。
- 对 `p` 先执行 `st.global.{default|wb|wt|cg|cs}`，再对同一 cache line 的相邻 word 执行 `ld.global.ca`（避免 store-to-load forwarding 优化）。
- 如果 store 会在 L1 分配（allocate），紧随的 load 应该 L1 hit；如果不分配（non-allocate），该 load 应该 L1 miss。
- 访问总范围：`working_set = iters * stride_bytes`（默认 `1 * 128B = 128B`，仅用于最小化测试；想要更稳定的命中率请把 `iters` 调大）。

**示意图（ASCII）**
```
for i in [0..iters):
  p = base + i * stride
  st.global.*  p   ──► L1 ? (allocate or bypass)
  ld.global.ca p   ──► L1 hit / miss
```

**示意图提示词（你来绘制）**
```
画一张 GPU cache 测试流程图：单线程循环访问；每次对新地址执行 store（可选 wb/wt/cg/cs），
随后立刻执行 ld.ca 读同一地址；标出 L1/TEX、L2、DRAM 三层，
并用两种颜色区分 allocate vs non-allocate 路径，标注预期 L1 hit/miss。
```

**输入参数（alloc 相关）**
| 参数 | 默认 | 说明 | 影响/建议 |
| --- | --- | --- | --- |
| `--test alloc` | - | 选择 allocate 测试 | 必选 |
| `--store <default|wb|wt|cg|cs>` | `default` | 选择 store cache op | 用于对照 allocate vs non-allocate |
| `--iters N` | `1` | 访问的 unique line 数 | `working_set = iters * stride_bytes`，建议大于 L1 |
| `--stride_bytes B` | `128` | 相邻访问间隔 | 建议 ≥ L1 line size（常见为 128B） |
| `--info` | false | 打印 GPU 信息 | 可用于记录测试环境 |

**预期结果（以 L1/TEX 指标为准）**
- **allocate 组（通常是 `default/wb/wt`）**：紧随其后的 `ld.ca` 应该 L1 hit。
- **non-allocate 组（通常是 `cg/cs`）**：`ld.ca` 应该 L1 miss。
- 若看 **load-only 命中率**：allocate ≈ `100%`，non-allocate ≈ `0%`。  
- 若看 **load+store 合并命中率**：allocate ≈ `50%`，non-allocate ≈ `0%`。  
  （因为每轮是 1 次 store + 1 次 load，store 通常是冷 miss。）

**如何观测结果**
- **终端输出**：只会打印 `ok: out=... (ignore; use Nsight for hit-rate)`，用于防止编译器优化，不代表 cache 结果。
- **Nsight Compute（推荐）**：
  1) 直接抓完整指标：
     ```bash
     ncu --set full --kernel-name k_alloc_wb cache/strategy/bin/strategybench --test alloc --store wb
     ```
  2) 打开报告（UI）：
     ```bash
     ncu -o alloc_wb --set full cache/strategy/bin/strategybench --test alloc --store wb
     ncu-ui alloc_wb.ncu-rep
     ```
     在 UI 中搜索 `L1/TEX` 或 `lookup`，关注 **L1/TEX Hit Rate** 与 **mem_global_op_{ld,st}_lookup_{hit,miss}**。
  3) 只取需要的 L1 lookup 指标（不同版本名字略有差异）：
     ```bash
     ncu --query-metrics | rg "mem_global_op_(ld|st)_lookup"
     ```
     把匹配到的具体指标名填入 `ncu --metrics ...` 再跑一次。

**示例指令（alloc）**
```bash
# 1) 编译
make -C cache/strategy SM=89

# 2) 直接跑（终端只用于确认能跑通）
cache/strategy/bin/strategybench --test alloc --store wb
cache/strategy/bin/strategybench --test alloc --store cg

# 3) 调整访问范围（更大工作集更稳）
cache/strategy/bin/strategybench --test alloc --store wb --iters 524288 --stride_bytes 128

# 4) 用 ncu 抓 L1/TEX 指标（推荐）
ncu --set full --kernel-name k_alloc_wb cache/strategy/bin/strategybench --test alloc --store wb
ncu --set full --kernel-name k_alloc_cg cache/strategy/bin/strategybench --test alloc --store cg

# 5) 先找指标名，再指定 metrics 精确采样
ncu --query-metrics | rg "mem_global_op_(ld|st)_lookup"
# 例如匹配到:
#   l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
#   l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum
#   l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum
#   l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum \
cache/strategy/bin/strategybench --test alloc --store wb
```

**常见偏差/排查**
- `stride_bytes` 太小或 `working_set` 太小，会导致跨迭代复用，命中率偏高。
- 不同架构/驱动/版本对 L1 统计口径略有差异；建议固定 `--store` 只改一个变量来做对比。
- 如果 `k_alloc_*` 内核名没匹配到，确认 `--store` 对应的 kernel（`k_alloc_default/wb/wt/cg/cs`）。
- 若怀疑同线程的 store->load forwarding 干扰，可用 `--test alloc2`（thread0 store、thread1 load **同一地址**）做对照；`alloc2` 使用 `ld.cg`，为保持“单次写入”的 profile 口径，当前不再输出 mismatch 次数。
- 如需“纯 store-only”的 stst 对照，可用 `--test stst`，配合 `--delay_cycles` 控制两次 store 的间隔。

### 2) evict or not（读-写-读）

同样依赖你看的是否是“合并指标”：

- 合并（ld+st）：
  - not-evict：`ld`(miss) + `st`(hit) + `ld`(hit) → `2/3 ≈ 66%`
  - evict：`ld`(miss) + `st`(hit) + `ld`(miss) → `1/3 ≈ 33%`
- 只看 load：
  - not-evict：两次 load 一次 miss 一次 hit → `50%`
  - evict：两次 load 都 miss → `0%`

### 3) write-back vs write-through（跨 SM 单地址可见性）

当前 `vis` 测试已简化为 **单地址 + 单次读**：

- consumer 先用 `ld.cg` 读一次初始值，然后通知 producer。
- producer 做 `ld.ca` + `st.global.{default|wb|wt|cg|cs}` 写入新值，延迟后通知 consumer。
- consumer 再用 `ld.cg` 读一次并输出 before/after。

为了避免把 write-back 强制推到 L2，这里仍然不使用 `__threadfence()`/`membar.gl`；因此存在“flag 先可见、数据后可见”的可能，结果可能抖动。需要更稳时可增大 `--delay_cycles` 或多跑几次。

#### write-back vs write-through 测试详解（vis）

**测试逻辑**
- 2 个 block / 每 block 1 个线程：block0 作为 producer，block1 作为 consumer。
- consumer 先读一次 `ld.cg`，记录 `before` 并置 `flags[0]=1`。
- producer 等待 `flags[0]=1` 后，`ld.ca` 触发该 cache line 进入 L1，再用 `st.global.{default|wb|wt|cg|cs}` 写入新值。
- producer 延迟 `--delay_cycles` 后置 `flags[1]=1`，作为 “store 完成” 的信号。
- consumer 等待 `flags[1]=1` 后，再读一次 `ld.cg`，记录 `after`。

**示意图（ASCII）**
```
Block1 (consumer, ld.cg)                 Block0 (producer)
before = ld.cg(data)  -----------------> wait flag0
flag0=1  ------------------------------> ld.ca data -> L1
                                        st.global.* data = new
                                        delay + flag1=1
after = ld.cg(data)  <------------------ wait flag1
```

**示意图提示词（你来绘制）**
```
画一张 GPU cache 可见性测试流程图：两个 block，consumer 先 ld.cg 读旧值并发旗标；
producer 做 ld.ca + store 写新值，延迟后发旗标；consumer 再 ld.cg 读新值；
标出 L1/L2/DRAM 层级与写回/直写的可见性差异。
```

**输入参数（vis 相关）**
| 参数 | 默认 | 说明 | 影响/建议 |
| --- | --- | --- | --- |
| `--test vis` | - | 选择可见性测试 | 必选 |
| `--store <default|wb|wt|cg|cs>` | `default` | producer 的 store cache op | 重点对比 `wb` vs `wt` |
| `--delay_cycles N` | `10000` | store 后延迟 | 避免“太快触发 flag” |
| `--info` | false | 打印 GPU 信息 | 记录测试环境 |

**预期结果（以 `after` 是否为新值为主）**
- **write-through（`--store wt`）**：`after` 更容易读到新值。
- **write-back（`--store wb`）**：`after` 往往仍是旧值（新值可能只在 producer 的 L1）。
- `default` 在多数架构上接近 `wb`，但请以实测为准。
- 若打印 `WARNING: same SM`，consumer 可能从共享 L1 读到新值，结论会不可靠。

**如何观测结果**
- 终端输出包含 `before/after: value/seen_new`，重点看 `after` 是否等于新值。
- 如果结果不稳定，增大 `--delay_cycles`，并多跑几次确保 producer/consumer 落在不同 SM。

**示例指令（vis）**
```bash
# 1) 编译
make -C cache/strategy SM=89

# 2) write-back vs write-through
cache/strategy/bin/strategybench --test vis --store wb
cache/strategy/bin/strategybench --test vis --store wt

# 3) 需要更稳时可加大延迟
cache/strategy/bin/strategybench --test vis --store wb --delay_cycles 50000
```

---

## 编译

```bash
make -C cache/strategy
```

可选：指定架构（默认 `SM=89`）

```bash
make -C cache/strategy SM=89
```

---

## 运行（用于 Nsight profile）

### allocate vs non-allocate

```bash
cache/strategy/bin/strategybench --test alloc --store default
cache/strategy/bin/strategybench --test alloc --store wb
cache/strategy/bin/strategybench --test alloc --store wt
cache/strategy/bin/strategybench --test alloc --store cg
cache/strategy/bin/strategybench --test alloc --store cs
```

### store-only (stst)

```bash
cache/strategy/bin/strategybench --test stst --store wb --delay_cycles 10000
cache/strategy/bin/strategybench --test stst --store cg --delay_cycles 10000
```

说明：`stst` 中**第 1 次 store 使用 `--store` 指定的 cache op**，**第 2 次 store 使用默认弱存储**（`st.global.u32`），便于观察“是否真的 allocate 到 L1”。

### evict or not

```bash
cache/strategy/bin/strategybench --test evict --store default
cache/strategy/bin/strategybench --test evict --store wb
```

### write-back vs write-through（L2 可见性）

```bash
cache/strategy/bin/strategybench --test vis --store wb
cache/strategy/bin/strategybench --test vis --store wt
```

输出示例（`seen_new=1` 表示该次 `ld.cg` 读到的新值等于写入值）：

```
smid: producer=... consumer=...
data init=0x... new=0x...
before: value=... seen_new=...
after:  value=... seen_new=...
```

如果打印 `WARNING: same SM`，说明两个 block 可能落在同一个 SM 上（共享 L1），`vis` 的结论会更不可信；建议多跑几次或调整实现（比如把 block 数增大、或用更“占资源”的配置强制分散到不同 SM）。

---

## Nsight Compute 指标建议

你可以直接用 `ncu --set full` 看 “L1/TEX Hit Rate”，也可以选更细的 lookup 计数（不同 Nsight/CUDA 版本名字可能略有差异）：

- load lookup hit/miss：`...mem_global_op_ld_lookup_{hit,miss}...`
- store lookup hit/miss：`...mem_global_op_st_lookup_{hit,miss}...`

这样能明确区分你要的 “load-only” 与 “load+store 合并” 两种 hit rate 口径。
