# V100 pointer-chase TLB sweep

与 `../4090` 相同的微基准，只是：
- `Makefile` 目标架构改为 `sm_70`（V100）
- 画图需要的话可用 `--title` 指定 GPU 名称

使用方法：
```bash
make                 # 生成 bin/tlbbench
python3 ../scripts/sweep.py --exe ./bin/tlbbench --out results/results.csv
python3 ../scripts/plot_latency.py results/results.csv --out results/figs/latency.png --title "Pointer-chase TLB sweep (NVIDIA V100)"
```
