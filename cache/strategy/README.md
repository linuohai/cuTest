# L1 store 策略微基准：allocate / evict / write-through vs write-back

这个目录提供一个可用 `nsight-compute (ncu)` 直接 profile 的小程序，用来验证你提出的 3 个判断思路，并且支持切换不同的 PTX store cache op（`st.global.{wb,wt,cg,cs}`）做对照。

> 重点：命中率的“50% / 66% / 33%”取决于你看的是不是“load+store 合并”指标；如果只看 load 指标，比例会不一样（见下文）。

---

## 文件组织（按测试目的拆分）

- `cache/strategy/main.cu`：命令行解析 + host 侧分配/launch
- `cache/strategy/ptx_cache_ops.cuh`：`ld.global.{ca,cg}` / `st.global.{wb,wt,cg,cs}` 封装
- `cache/strategy/alloc_write_read_kernels.cuh`：写后读（allocate vs non-allocate）相关 kernel
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

### 2) evict or not（读-写-读）

同样依赖你看的是否是“合并指标”：

- 合并（ld+st）：
  - not-evict：`ld`(miss) + `st`(hit) + `ld`(hit) → `2/3 ≈ 66%`
  - evict：`ld`(miss) + `st`(hit) + `ld`(miss) → `1/3 ≈ 33%`
- 只看 load：
  - not-evict：两次 load 一次 miss 一次 hit → `50%`
  - evict：两次 load 都 miss → `0%`

### 3) write-back vs write-through（跨 SM 读 L2 观察可见性）

“让另一个 block 用 `ld.cg` 从 L2 读”这个方向是对的，但**同步方式要小心**：

- 如果你用 `__threadfence()`/`membar.gl` 去保证可见性，会把写强制推到 L2，从而把 write-back 也“变成可见”，会误判成 write-through。
- 如果你完全不用 fence，只用 flag/atomic 来同步顺序，则可能遇到“flag 先可见、数据后可见”的乱序现象（这是 CUDA 里经典的 producer/consumer 问题），会导致偶发误判。

所以更稳的做法是：**读两次 L2：一次在主动 evict 前，一次在主动 evict 后**。如果是 write-back，你会看到“evict 前读不到新值，evict 后读到新值”；如果是 write-through，两次都能读到新值。

本目录的 `vis` 测试就是这么做的，并输出 producer/consumer 的 `smid`（若相同 SM，会提示 warning）。

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

输出示例（`seen_new=1` 表示在该阶段的 `ld.cg` 轮询中曾读到新值）：

```
smid: producer=... consumer=...
before: first=... last=... seen_new=...
after:  first=... last=... seen_new=...
```

如果打印 `WARNING: same SM`，说明两个 block 可能落在同一个 SM 上（共享 L1），`vis` 的结论会更不可信；建议多跑几次或调整实现（比如把 block 数增大、或用更“占资源”的配置强制分散到不同 SM）。

---

## Nsight Compute 指标建议

你可以直接用 `ncu --set full` 看 “L1/TEX Hit Rate”，也可以选更细的 lookup 计数（不同 Nsight/CUDA 版本名字可能略有差异）：

- load lookup hit/miss：`...mem_global_op_ld_lookup_{hit,miss}...`
- store lookup hit/miss：`...mem_global_op_st_lookup_{hit,miss}...`

这样能明确区分你要的 “load-only” 与 “load+store 合并” 两种 hit rate 口径。
