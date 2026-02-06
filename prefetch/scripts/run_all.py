#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "prefetch_bench"


NCU_METRICS = ",".join(
    [
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
        "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum",
        "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum",
        "dram__sectors_read.sum",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    ]
)


@dataclasses.dataclass(frozen=True)
class BenchCfg:
    benchmark: str
    variant: str
    workset_bytes: int
    stride_bytes: int = 0
    iters: int = 1000000
    repeats: int = 5
    warmup: int = 0
    flush_bytes: int = 128 * 1024 * 1024
    hot_frac: float = 1.0
    hot_prob: float = 0.9
    seed: int = 1234


def sh(cmd: List[str], *, cwd: Optional[pathlib.Path] = None, quiet: bool = False) -> str:
    if not quiet:
        print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def ensure_built() -> None:
    if BIN.exists():
        return
    sh(["make", "-j"], cwd=ROOT)
    if not BIN.exists():
        raise RuntimeError(f"Build did not produce {BIN}")


def run_bench_json(device: int, cfg: BenchCfg) -> Dict:
    cmd = [
        str(BIN),
        "--device",
        str(device),
        "--benchmark",
        cfg.benchmark,
        "--variant",
        cfg.variant,
        "--workset-bytes",
        str(cfg.workset_bytes),
        "--stride-bytes",
        str(cfg.stride_bytes),
        "--iters",
        str(cfg.iters),
        "--repeats",
        str(cfg.repeats),
        "--warmup",
        str(cfg.warmup),
        "--flush-bytes",
        str(cfg.flush_bytes),
        "--hot-frac",
        f"{cfg.hot_frac:.6f}",
        "--hot-prob",
        f"{cfg.hot_prob:.6f}",
        "--seed",
        str(cfg.seed),
        "--format",
        "json",
    ]
    out = sh(cmd, cwd=ROOT, quiet=True).strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON output: {out}") from e


def parse_ncu_csv(path: pathlib.Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("==")]
    reader = csv.DictReader(lines)
    for row in reader:
        name = row.get("Metric Name", "").strip()
        val = row.get("Metric Value", "").strip().strip('"')
        if not name:
            continue
        try:
            metrics[name] = float(val)
        except ValueError:
            continue
    return metrics


def run_ncu_metrics(device: int, cfg: BenchCfg, kernel_substr: str, out_csv: pathlib.Path) -> Dict[str, float]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ncu",
        "--metrics",
        NCU_METRICS,
        "--kernel-name",
        kernel_substr,
        "--csv",
        "--log-file",
        str(out_csv),
        str(BIN),
        "--device",
        str(device),
        "--benchmark",
        cfg.benchmark,
        "--variant",
        cfg.variant,
        "--workset-bytes",
        str(cfg.workset_bytes),
        "--stride-bytes",
        str(cfg.stride_bytes),
        "--iters",
        str(cfg.iters),
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--flush-bytes",
        str(cfg.flush_bytes),
        "--hot-frac",
        f"{cfg.hot_frac:.6f}",
        "--hot-prob",
        f"{cfg.hot_prob:.6f}",
        "--seed",
        str(cfg.seed),
        "--format",
        "csv",
    ]
    sh(cmd, cwd=ROOT, quiet=True)
    return parse_ncu_csv(out_csv)


def hit_pct(hit: float, miss: float) -> float:
    denom = hit + miss
    if denom <= 0:
        return 0.0
    return 100.0 * hit / denom


def query_gpu_mem(device: int) -> Tuple[int, int]:
    out = sh(
        [
            "nvidia-smi",
            "-i",
            str(device),
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        quiet=True,
    ).strip()
    # Example: "81194, 78000"
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != 2:
        raise RuntimeError(f"Unexpected nvidia-smi output: {out}")
    total_mib = int(parts[0])
    free_mib = int(parts[1])
    return total_mib, free_mib


def align_down(x: int, align: int) -> int:
    return x - (x % align)


def effective_stride_n_elems(workset_bytes: int) -> int:
    # Mirrors prefetch_bench: stride uses odd modulus to avoid short cycles.
    n = workset_bytes // 4
    if n % 2 == 0:
        n -= 1
    return max(2, int(n))


def cycle_len_stride(workset_bytes: int, stride_bytes: int) -> int:
    n = effective_stride_n_elems(workset_bytes)
    s = max(1, stride_bytes // 4)
    return n // math.gcd(n, s)

def max_iters_no_mod_wrap_stride(workset_bytes: int, stride_bytes: int, *, extra_iters: int = 0) -> int:
    # Ensure stride kernel's `idx += stride; if (idx >= n) idx -= n;` never takes the wrap branch.
    # Avoids wrap-around artifacts where subtracting an odd modulus shifts cache-line offsets and can
    # create unintended cache-line reuse for large strides.
    n = effective_stride_n_elems(workset_bytes)
    s = max(1, stride_bytes // 4)
    max_iters = (n - 1) // s
    if extra_iters > 0:
        max_iters = max(1, max_iters - extra_iters)
    return max(1, int(max_iters))


def clamp_iters_no_wrap(cfg: BenchCfg) -> BenchCfg:
    if cfg.benchmark == "stride":
        cycle = cycle_len_stride(cfg.workset_bytes, cfg.stride_bytes)
        iters = min(cfg.iters, max(1, cycle - 1))
        if iters != cfg.iters:
            return dataclasses.replace(cfg, iters=iters)
        return cfg

    if cfg.benchmark == "pointer_chase":
        # lcg: full-period permutation for power-of-two n; stride: modulus may be odd-adjusted.
        if cfg.variant in ("stride",):
            cycle = cycle_len_stride(cfg.workset_bytes, cfg.stride_bytes)
        else:
            n = cfg.workset_bytes // 4
            cycle = max(2, int(n))
        iters = min(cfg.iters, max(1, cycle - 1))
        if iters != cfg.iters:
            return dataclasses.replace(cfg, iters=iters)
        return cfg

    return cfg


def clamp_iters_no_mod_wrap(cfg: BenchCfg) -> BenchCfg:
    if cfg.benchmark != "stride":
        return cfg

    # Conservative lookahead for kernels that touch future iterations.
    extra = 0
    if cfg.variant == "prefetch_l2":
        extra = 8
    elif cfg.variant == "cp_async":
        extra = 4

    cycle = cycle_len_stride(cfg.workset_bytes, cfg.stride_bytes)
    max_no_mod = max_iters_no_mod_wrap_stride(cfg.workset_bytes, cfg.stride_bytes, extra_iters=extra)
    iters = min(cfg.iters, max(1, cycle - 1), max_no_mod)
    if iters != cfg.iters:
        return dataclasses.replace(cfg, iters=iters)
    return cfg


def write_csv(path: pathlib.Path, rows: List[Dict]) -> None:
    if not rows:
        raise RuntimeError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: pathlib.Path) -> List[Dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(x: str) -> int:
    return int(float(x))


def to_float(x: str) -> float:
    return float(x)


def plot_stride_latency(rows: List[Dict], out_png: pathlib.Path, workset_bytes: int) -> None:
    import matplotlib.pyplot as plt

    data = [r for r in rows if to_int(r["workset_bytes"]) == workset_bytes and r["variant"] == "baseline"]
    data.sort(key=lambda r: to_int(r["stride_bytes"]))
    x = [to_int(r["stride_bytes"]) for r in data]
    y = [to_float(r["time_ns_per_iter"]) for r in data]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Stride (bytes)")
    plt.ylabel("Time (ns/iter)")
    plt.title(f"Stride sweep @ workset={workset_bytes/1024/1024:.0f} MiB (baseline)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_stride_l2hit(rows: List[Dict], out_png: pathlib.Path, workset_bytes: int) -> None:
    import matplotlib.pyplot as plt

    data = [r for r in rows if to_int(r["workset_bytes"]) == workset_bytes and r["variant"] == "baseline"]
    data.sort(key=lambda r: to_int(r["stride_bytes"]))
    x = [to_int(r["stride_bytes"]) for r in data]
    y = [to_float(r["l2_hit_pct"]) for r in data]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Stride (bytes)")
    plt.ylabel("L2 hit (%)")
    plt.ylim(0, 100)
    plt.title(f"L2 hit vs stride @ workset={workset_bytes/1024/1024:.0f} MiB (baseline)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_workset_boundary(rows: List[Dict], out_png: pathlib.Path, stride_bytes: int) -> None:
    import matplotlib.pyplot as plt

    data = [r for r in rows if to_int(r["stride_bytes"]) == stride_bytes and r["variant"] == "baseline"]
    data.sort(key=lambda r: to_int(r["workset_bytes"]))
    x = [to_int(r["workset_bytes"]) / (1024 * 1024) for r in data]
    y = [to_float(r["time_ns_per_iter"]) for r in data]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Working set (MiB)")
    plt.ylabel("Time (ns/iter)")
    plt.title(f"Working set boundary @ stride={stride_bytes}B (baseline)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_workset_boundary_multi(rows: List[Dict], out_png: pathlib.Path, stride_list: List[int]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for st in stride_list:
        data = [r for r in rows if to_int(r["stride_bytes"]) == st and r["variant"] == "baseline"]
        data.sort(key=lambda r: to_int(r["workset_bytes"]))
        x = [to_int(r["workset_bytes"]) / (1024 * 1024) for r in data]
        y = [to_float(r["time_ns_per_iter"]) for r in data]
        plt.plot(x, y, marker="o", label=f"stride={st}B")

    plt.xscale("log", base=2)
    plt.xlabel("Working set (MiB)")
    plt.ylabel("Time (ns/iter)")
    plt.title("Working set boundary (baseline)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_pointer_chase(rows: List[Dict], out_png: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    data = rows[:]
    data.sort(key=lambda r: to_int(r["workset_bytes"]))
    x = [to_int(r["workset_bytes"]) / (1024 * 1024) for r in data]
    y = [to_float(r["time_ns_per_iter"]) for r in data]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Working set (MiB)")
    plt.ylabel("Time (ns/iter)")
    plt.title("Pointer-chase latency vs working set")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_gather(rows: List[Dict], out_png_latency: pathlib.Path, out_png_l2: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    for metric, out_png, ylabel in [
        ("time_ns_per_iter", out_png_latency, "Time (ns/iter)"),
        ("l2_hit_pct", out_png_l2, "L2 hit (%)"),
    ]:
        plt.figure(figsize=(7, 4))
        for hot_frac in sorted({r["hot_frac"] for r in rows}):
            data = [r for r in rows if r["hot_frac"] == hot_frac]
            data.sort(key=lambda r: to_int(r["workset_bytes"]))
            x = [to_int(r["workset_bytes"]) / (1024 * 1024) for r in data]
            y = [to_float(r[metric]) for r in data]
            plt.plot(x, y, marker="o", label=f"hot_frac={hot_frac}")
        plt.xscale("log", base=2)
        plt.xlabel("Working set (MiB)")
        plt.ylabel(ylabel)
        if metric == "l2_hit_pct":
            plt.ylim(0, 100)
        plt.title("Gather/IMA vs working set")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.4)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()


def plot_explicit(rows: List[Dict], out_png: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    order = ["baseline", "prefetch_l2", "cp_async"]
    strides = sorted({to_int(r["stride_bytes"]) for r in rows})

    # Grouped bars: x=stride, hue=variant
    width = 0.25
    x0 = list(range(len(strides)))
    plt.figure(figsize=(8, 4))
    for i, v in enumerate(order):
        ys = []
        for st in strides:
            match = [r for r in rows if r["variant"] == v and to_int(r["stride_bytes"]) == st]
            ys.append(to_float(match[0]["time_ns_per_iter"]) if match else float("nan"))
        xs = [x + (i - 1) * width for x in x0]
        plt.bar(xs, ys, width=width, label=v)

    plt.xticks(x0, [str(st) for st in strides], rotation=0)
    plt.xlabel("Stride (bytes)")
    plt.ylabel("Time (ns/iter)")
    plt.title("Explicit control vs stride")
    plt.legend()
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def collect_env(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "cwd": str(ROOT),
    }
    env["nvidia_smi"] = sh(["nvidia-smi"], quiet=True)
    env["nvcc_version"] = sh(["nvcc", "--version"], quiet=True)
    env["ncu_version"] = sh(["ncu", "--version"], quiet=True)
    env["nsys_version"] = sh(["nsys", "--version"], quiet=True)
    (out_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--fast", action="store_true", help="Reduce sweep sizes for quick sanity runs.")
    args = ap.parse_args()

    ensure_built()

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (ROOT / "out" / dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_env(out_dir)

    total_mib, free_mib = query_gpu_mem(args.device)
    free_bytes = free_mib * 1024 * 1024
    reserve_bytes = 10 * 1024 * 1024 * 1024  # leave headroom for other users + profiler overhead
    max_workset = max(256 * 1024 * 1024, free_bytes - reserve_bytes)
    max_workset = align_down(max_workset, 256 * 1024 * 1024)

    # Sweep design (prefetch exploration first).
    if args.fast:
        stride_sweep = [4096, 8192, 16384, 65536, 262144, 1048576]
        workset_large = min(max_workset, 4 * 1024 * 1024 * 1024)
        workset_boundary = [256 * 1024 * 1024, workset_large]
        explicit_strides = [8192, 65536]
        pc_worksets = [256 * 1024 * 1024, workset_large]
        dep_stride_sweep = [4096, 8192, 16384, 65536, 262144, 1048576]
        gather_worksets = [256 * 1024 * 1024, workset_large]
    else:
        stride_sweep = [
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            6144,
            8192,
            10240,
            12288,
            16384,
            24576,
            32768,
            49152,
            65536,
            98304,
            131072,
            196608,
            262144,
            393216,
            524288,
            786432,
            1048576,
            2097152,
            4194304,
        ]
        # Cap to avoid extremely large per-run cudaMalloc overhead while still being far beyond cache.
        workset_large = min(max_workset, 64 * 1024 * 1024 * 1024)
        workset_boundary = [
            256 * 1024,
            1 * 1024 * 1024,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            256 * 1024 * 1024,
            1 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
            workset_large,
        ]
        explicit_strides = [128, 8192, 65536, 1048576]
        pc_worksets = [
            1 * 1024 * 1024,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            256 * 1024 * 1024,
            1 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            min(workset_large, 16 * 1024 * 1024 * 1024),
        ]
        dep_stride_sweep = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576, 4194304]
        gather_worksets = [
            256 * 1024 * 1024,
            1 * 1024 * 1024 * 1024,
            min(workset_large, 4 * 1024 * 1024 * 1024),
        ]

    # ===== Experiment 1: Stream/Stride (baseline) =====
    stride_rows: List[Dict] = []
    ncu_dir = out_dir / "ncu"
    print(f"[Env] GPU mem: total={total_mib}MiB free={free_mib}MiB; workset_large={workset_large} bytes", flush=True)

    # Exp1a: stride sweep on very large working set.
    print(f"[Exp1a] stride sweep: workset={workset_large}, {len(stride_sweep)} strides", flush=True)
    for st in stride_sweep:
        cfg0 = BenchCfg(benchmark="stride", variant="baseline", workset_bytes=workset_large, stride_bytes=st)
        cfg = clamp_iters_no_mod_wrap(cfg0)
        print(f"[Exp1a] stride={st} iters={cfg.iters}", flush=True)
        timing = run_bench_json(args.device, cfg)
        ncu_csv = ncu_dir / f"stride_ws{workset_large}_st{st}.csv"
        m = run_ncu_metrics(args.device, cfg, "stride_baseline_kernel", ncu_csv)

        l1_hit = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum", 0.0)
        l1_miss = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum", 0.0)
        l2_hit = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum", 0.0)
        l2_miss = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum", 0.0)

        stride_rows.append(
            {
                "benchmark": "stride",
                "variant": "baseline",
                "workset_bytes": workset_large,
                "stride_bytes": st,
                "iters": cfg.iters,
                "blocks_x_threads": "1x32",
                "time_ns_per_iter": f"{timing['time_ns_per_iter']:.3f}",
                "bandwidth_gbs": f"{timing['bandwidth_gbs']:.6f}",
                "l2_hit_pct": f"{hit_pct(l2_hit, l2_miss):.2f}",
                "l1_hit_pct": f"{hit_pct(l1_hit, l1_miss):.2f}",
                "dram_read_sectors": f"{m.get('dram__sectors_read.sum', 0.0):.0f}",
                "stall_long_scoreboard_pct": f"{m.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct', 0.0):.2f}",
                "note": "",
            }
        )

    # Exp1b: working-set boundary sweep for a couple of representative strides.
    boundary_strides = [128, 8192]
    print(f"[Exp1b] workset boundary: strides={boundary_strides}, {len(workset_boundary)} worksets", flush=True)
    for st in boundary_strides:
        for ws in workset_boundary:
            cfg0 = BenchCfg(benchmark="stride", variant="baseline", workset_bytes=ws, stride_bytes=st)
            cfg = clamp_iters_no_wrap(cfg0)
            print(f"[Exp1b] stride={st} workset={ws} iters={cfg.iters}", flush=True)
            timing = run_bench_json(args.device, cfg)
            ncu_csv = ncu_dir / f"stride_ws{ws}_st{st}.csv"
            m = run_ncu_metrics(args.device, cfg, "stride_baseline_kernel", ncu_csv)

            l1_hit = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum", 0.0)
            l1_miss = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum", 0.0)
            l2_hit = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum", 0.0)
            l2_miss = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum", 0.0)

            stride_rows.append(
                {
                    "benchmark": "stride",
                    "variant": "baseline",
                    "workset_bytes": ws,
                    "stride_bytes": st,
                    "iters": cfg.iters,
                    "blocks_x_threads": "1x32",
                    "time_ns_per_iter": f"{timing['time_ns_per_iter']:.3f}",
                    "bandwidth_gbs": f"{timing['bandwidth_gbs']:.6f}",
                    "l2_hit_pct": f"{hit_pct(l2_hit, l2_miss):.2f}",
                    "l1_hit_pct": f"{hit_pct(l1_hit, l1_miss):.2f}",
                    "dram_read_sectors": f"{m.get('dram__sectors_read.sum', 0.0):.0f}",
                    "stall_long_scoreboard_pct": f"{m.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct', 0.0):.2f}",
                    "note": "boundary_sweep",
                }
            )

    stride_csv = out_dir / "exp1_stride.csv"
    write_csv(stride_csv, stride_rows)

    # Plots for Exp1.
    plot_stride_latency(stride_rows, out_dir / "stride_latency.png", workset_bytes=workset_large)
    plot_stride_l2hit(stride_rows, out_dir / "stride_l2hit.png", workset_bytes=workset_large)
    plot_workset_boundary_multi(stride_rows, out_dir / "workingset_boundary.png", stride_list=boundary_strides)

    # ===== Experiment 2: Pointer-chase (negative control) =====
    pc_rows: List[Dict] = []
    print(f"[Exp2] pointer-chase(lcg): {len(pc_worksets)} configs", flush=True)
    for ws in pc_worksets:
        print(f"[Exp2] workset={ws}", flush=True)
        cfg0 = BenchCfg(benchmark="pointer_chase", variant="lcg", workset_bytes=ws, stride_bytes=0)
        cfg = clamp_iters_no_wrap(cfg0)
        timing = run_bench_json(args.device, cfg)
        ncu_csv = ncu_dir / f"pointer_chase_ws{ws}.csv"
        m = run_ncu_metrics(args.device, cfg, "pointer_chase_kernel", ncu_csv)

        l1_hit = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum", 0.0)
        l1_miss = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum", 0.0)
        l2_hit = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum", 0.0)
        l2_miss = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum", 0.0)

        pc_rows.append(
            {
                "benchmark": "pointer_chase",
                "variant": cfg.variant,
                "workset_bytes": ws,
                "stride_bytes": 0,
                "iters": cfg.iters,
                "blocks_x_threads": "1x32",
                "time_ns_per_iter": f"{timing['time_ns_per_iter']:.3f}",
                "bandwidth_gbs": f"{timing['bandwidth_gbs']:.6f}",
                "l2_hit_pct": f"{hit_pct(l2_hit, l2_miss):.2f}",
                "l1_hit_pct": f"{hit_pct(l1_hit, l1_miss):.2f}",
                "dram_read_sectors": f"{m.get('dram__sectors_read.sum', 0.0):.0f}",
                "stall_long_scoreboard_pct": f"{m.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct', 0.0):.2f}",
                "note": "LCG permutation",
            }
        )

    pc_csv = out_dir / "exp2_pointer_chase.csv"
    write_csv(pc_csv, pc_rows)
    plot_pointer_chase(pc_rows, out_dir / "pointer_chase_latency.png")

    # ===== Experiment 2b: Dependent stride (pointer-chase w/ stride mapping) =====
    dep_rows: List[Dict] = []
    dep_workset = min(workset_large, 16 * 1024 * 1024 * 1024)
    print(f"[Exp2b] pointer_chase(stride): workset={dep_workset}, {len(dep_stride_sweep)} strides", flush=True)
    for st in dep_stride_sweep:
        cfg0 = BenchCfg(benchmark="pointer_chase", variant="stride", workset_bytes=dep_workset, stride_bytes=st)
        cfg = clamp_iters_no_wrap(cfg0)
        print(f"[Exp2b] stride={st} iters={cfg.iters}", flush=True)
        timing = run_bench_json(args.device, cfg)
        ncu_csv = ncu_dir / f"dep_stride_ws{dep_workset}_st{st}.csv"
        m = run_ncu_metrics(args.device, cfg, "pointer_chase_kernel", ncu_csv)

        l1_hit = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum", 0.0)
        l1_miss = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum", 0.0)
        l2_hit = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum", 0.0)
        l2_miss = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum", 0.0)

        dep_rows.append(
            {
                "benchmark": "dep_stride",
                "variant": cfg.variant,
                "workset_bytes": dep_workset,
                "stride_bytes": st,
                "iters": cfg.iters,
                "blocks_x_threads": "1x32",
                "time_ns_per_iter": f"{timing['time_ns_per_iter']:.3f}",
                "bandwidth_gbs": f"{timing['bandwidth_gbs']:.6f}",
                "l2_hit_pct": f"{hit_pct(l2_hit, l2_miss):.2f}",
                "l1_hit_pct": f"{hit_pct(l1_hit, l1_miss):.2f}",
                "dram_read_sectors": f"{m.get('dram__sectors_read.sum', 0.0):.0f}",
                "stall_long_scoreboard_pct": f"{m.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct', 0.0):.2f}",
                "note": "",
            }
        )

    dep_csv = out_dir / "exp2b_dep_stride.csv"
    write_csv(dep_csv, dep_rows)

    # ===== Experiment 3: Gather / IMA =====
    gather_rows: List[Dict] = []
    total = len(gather_worksets) * 2
    done = 0
    print(f"[Exp3] gather/IMA: {total} configs (hot_frac in [1.0, 0.1])", flush=True)
    for hot_frac in [1.0, 0.1]:
        for ws in gather_worksets:
            done += 1
            if done == 1 or done % 10 == 0 or done == total:
                print(f"[Exp3] {done}/{total} hot_frac={hot_frac} workset={ws}", flush=True)
            cfg = BenchCfg(
                benchmark="gather",
                variant="baseline",
                workset_bytes=ws,
                stride_bytes=0,
                hot_frac=hot_frac,
                hot_prob=0.9,
                seed=1234,
            )
            timing = run_bench_json(args.device, cfg)
            ncu_csv = ncu_dir / f"gather_hot{hot_frac}_ws{ws}.csv"
            m = run_ncu_metrics(args.device, cfg, "gather_ima_kernel", ncu_csv)

            l1_hit = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum", 0.0)
            l1_miss = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum", 0.0)
            l2_hit = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum", 0.0)
            l2_miss = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum", 0.0)

            gather_rows.append(
                {
                    "benchmark": "gather",
                    "variant": "baseline",
                    "workset_bytes": ws,
                    "stride_bytes": 0,
                    "iters": cfg.iters,
                    "blocks_x_threads": "1x32",
                    "hot_frac": f"{hot_frac:.1f}",
                    "hot_prob": "0.9",
                    "time_ns_per_iter": f"{timing['time_ns_per_iter']:.3f}",
                    "bandwidth_gbs": f"{timing['bandwidth_gbs']:.6f}",
                    "l2_hit_pct": f"{hit_pct(l2_hit, l2_miss):.2f}",
                    "l1_hit_pct": f"{hit_pct(l1_hit, l1_miss):.2f}",
                    "dram_read_sectors": f"{m.get('dram__sectors_read.sum', 0.0):.0f}",
                    "stall_long_scoreboard_pct": f"{m.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct', 0.0):.2f}",
                    "note": "indices_len=iters",
                }
            )

    gather_csv = out_dir / "exp3_gather.csv"
    write_csv(gather_csv, gather_rows)
    plot_gather(gather_rows, out_dir / "gather_latency.png", out_dir / "gather_l2hit.png")

    # ===== Experiment 4: Explicit controls =====
    explicit_rows: List[Dict] = []
    variants = ["baseline", "prefetch_l2", "cp_async"]
    print(f"[Exp4] explicit controls: workset={workset_large}, strides={explicit_strides}, variants={variants}", flush=True)
    for st in explicit_strides:
        for v in variants:
            cfg0 = BenchCfg(benchmark="stride", variant=v, workset_bytes=workset_large, stride_bytes=st)
            cfg = clamp_iters_no_mod_wrap(cfg0)
            print(f"[Exp4] stride={st} variant={v} iters={cfg.iters}", flush=True)
            timing = run_bench_json(args.device, cfg)
            kernel = {
                "baseline": "stride_baseline_kernel",
                "prefetch_l2": "stride_prefetch_l2_kernel",
                "cp_async": "stride_cp_async_kernel",
            }[cfg.variant]
            ncu_csv = ncu_dir / f"explicit_ws{workset_large}_st{st}_{cfg.variant}.csv"
            m = run_ncu_metrics(args.device, cfg, kernel, ncu_csv)

            l1_hit = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum", 0.0)
            l1_miss = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum", 0.0)
            l2_hit = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum", 0.0)
            l2_miss = m.get("lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum", 0.0)

            explicit_rows.append(
                {
                    "benchmark": "explicit",
                    "variant": cfg.variant,
                    "workset_bytes": workset_large,
                    "stride_bytes": st,
                    "iters": cfg.iters,
                    "blocks_x_threads": "1x32",
                    "time_ns_per_iter": f"{timing['time_ns_per_iter']:.3f}",
                    "bandwidth_gbs": f"{timing['bandwidth_gbs']:.6f}",
                    "l2_hit_pct": f"{hit_pct(l2_hit, l2_miss):.2f}",
                    "l1_hit_pct": f"{hit_pct(l1_hit, l1_miss):.2f}",
                    "dram_read_sectors": f"{m.get('dram__sectors_read.sum', 0.0):.0f}",
                    "stall_long_scoreboard_pct": f"{m.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct', 0.0):.2f}",
                    "note": "",
                }
            )

    explicit_csv = out_dir / "exp4_explicit.csv"
    write_csv(explicit_csv, explicit_rows)
    plot_explicit(explicit_rows, out_dir / "explicit_vs_baseline.png")

    # Minimal summary JSON for quick inspection.
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "files": [p.name for p in sorted(out_dir.iterdir())],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Done. Results in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
