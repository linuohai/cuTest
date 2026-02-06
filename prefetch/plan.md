## 实验方案（可执行、可复核）】
 目标：在“尽量关闭延迟隐藏”的前提下，检测是否存在“通用硬件自动 prefetch”签名，并给出对照实验（显式 prefetch 作为上界对照）。
 必须包含：
 1.1 实验设计原则（如何控制 occupancy/ILP、如何构造 dependent chain、如何设置工作集跨 L1/L2/DRAM 边界）
 1.2 微基准清单（至少 4 类）：
- Stream/Stride（最易触发自动预取）
- Pointer-chase（作为不可预取反例）
- Gather/IMA（贴近真实痛点）
- 至少一个“显式 prefetch/显式 async 搬运”对照版本（用于证明测量链路敏感）
 对每个微基准：给出伪代码、关键参数 sweep 范围、预期现象（若存在自动预取应出现什么拐点/签名）、以及可能的失败原因与排查步骤。
 1.3 指标与采集清单：
- 必须给出“通用指标”（时间、带宽、L2 hit、L1 hit、cache miss、mem transactions、stall reason 等）
- 若我提供具体 profiler 工具：再给“工具特定指标映射表”（例如哪些 counters 对应 L2 hit、dram read）
 1.4 图片与图表要求（我回传给你的材料格式）：
- 每张图要指定：横轴/纵轴、是否 log、数据点数量、要画的曲线组
- 图像格式：PNG；命名规则；每图配一段 2-3 句解释模板

## 实验平台
所有的实验都在 fa 这个 container 中进行操作，已经在宿主机上启动，禁止在宿主机上直接运行实验代码
**禁止运行**rm 类指令，以免误删宿主机文件

## 结果要求
【环境信息】
- GPU 型号/架构代际：
- 显存与频率（如可得）：
- 驱动版本：
- CUDA/ROCm/oneAPI 版本（如适用）：
- 编译器与编译参数：
- profiler 工具与版本：
- 运行方式（独占/共享；固定 clocks 与否）：
- 关键运行约束（如限制 SM、限制 occupancy 的方法）：
【实验 1：Stream/Stride】
- Kernel 版本：Baseline（无显式 prefetch）
- 参数 sweep：
  - stride 列表：
  - 工作集大小列表：
  - blocks/threads 设置：
  - 目标 occupancy：
- 数据表（每行一个配置）：
 | stride | 工作集 | blocks x threads | 时间(ns/iter) | 带宽(GB/s) | L2 hit(%) | L1 hit(%) | DRAM read txn | 备注 |
 |---|---:|---:|---:|---:|---:|---:|---:|---|
- 图片：
  - IMG:stride_latency.png（横轴 stride，纵轴 时间/延迟）
  - IMG:stride_l2hit.png（横轴 stride，纵轴 L2 hit）
  - IMG:workingset_boundary.png（横轴 工作集，纵轴 延迟/带宽）
【实验 2：Pointer-chase（反例）】
- 同上结构（表 + 至少 1 图）：
  - IMG:pointer_chase_latency.png
【实验 3：Gather / IMA】
- 访问模式描述（如何生成索引、随机度/相关度）：
- 表 + 图：
  - IMG:gather_latency.png
  - IMG:gather_l2hit.png
【实验 4：显式对照】
- 对照 A：显式 prefetch（或 cache hint / pragma）
- 对照 B：显式 async 搬运到 shared（如适用）
- 表 + 图：
  - IMG:explicit_vs_baseline.png
【汇总对比表（至少一张总表）】
暂时无法在飞书文档外展示此内容
【原始数据附件（可选但推荐）】
- CSV/JSON 链接或粘贴摘要：
- profiler 截图（可选）：
  - IMG:profiler_counter_screenshot.png
