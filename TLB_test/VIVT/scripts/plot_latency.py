import argparse, pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

def fmt_bytes(n: int) -> str:
    if n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}MiB"
    if n % 1024 == 0:
        return f"{n // 1024}KiB"
    return f"{n}B"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="results/results.csv")
    ap.add_argument("--out", default="results/figs/latency.png")
    ap.add_argument("--title", default="Pointer-chase TLB sweep",
                    help="title prefix (e.g., 'Pointer-chase TLB sweep (RTX 4090)')")
    ap.add_argument("--x", default="bytes", choices=["bytes", "nodes"],
                    help="x axis: bytes (MiB) or nodes (=bytes/stride)")
    ap.add_argument("--stat", default="median", choices=["median", "mean", "min", "max"],
                    help="which statistic to plot when multiple are available")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 兼容旧格式（cycles_per_access）与新格式（*_cpa）
    stat_to_col = {
        "median": "median_cpa",
        "mean": "mean_cpa",
        "min": "min_cpa",
        "max": "max_cpa",
    }
    ycol = stat_to_col[args.stat]
    if ycol not in df.columns:
        if "cycles_per_access" in df.columns:
            ycol = "cycles_per_access"
        else:
            raise SystemExit(f"Cannot find y column: {stat_to_col[args.stat]} (or legacy cycles_per_access).")

    # 分组画：每个 policy 一张图（你也可以按 stride 分图）
    for policy in ["cg", "ca"]:
        sub = df[df["policy"] == policy].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots()
        # 每个 stride 一条线
        for stride, g in sub.groupby("stride"):
            g = g.sort_values("bytes")
            if args.x == "nodes":
                x = g["nodes"] if "nodes" in g.columns else (g["bytes"] // g["stride"])
                xlabel = "nodes (=bytes/stride)"
            else:
                x = g["bytes"] / (1024 * 1024)
                xlabel = "array size (MiB)"

            ax.plot(x, g[ycol], marker="o", linestyle="-", label=f"stride={fmt_bytes(int(stride))}")

        ax.set_xscale("log", base=2)
        ax.set_yscale("linear")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):d}" if x >= 1 else f"{x:.2g}"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{args.stat} latency (cycles/access)")
        subtitle = "L1 bypass (cache in L2)" if policy == "cg" else "L1 enabled (cache in L1+L2)"
        ax.set_title(f"{args.title}: {subtitle}")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        out = args.out.replace(".png", f".{policy}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)
