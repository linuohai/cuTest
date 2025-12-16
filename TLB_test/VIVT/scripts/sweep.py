import argparse, csv, subprocess, os, math

def parse_bytes(s: str) -> int:
    s = (s or "").strip().replace("_", "")
    if not s:
        raise ValueError("empty size")

    sl = s.lower()
    suffix_to_mul = {
        "gib": 1024**3,
        "gb": 1024**3,
        "g": 1024**3,
        "mib": 1024**2,
        "mb": 1024**2,
        "m": 1024**2,
        "kib": 1024,
        "kb": 1024,
        "k": 1024,
        "b": 1,
    }

    num = sl
    mul = 1
    for suf in sorted(suffix_to_mul.keys(), key=len, reverse=True):
        if sl.endswith(suf) and len(sl) > len(suf):
            num = sl[: -len(suf)].strip()
            mul = suffix_to_mul[suf]
            break

    if not num:
        raise ValueError(f"missing number in size: {s!r}")
    return int(float(num) * mul)

def parse_stride_list(strides):
    out = []
    for raw in strides:
        for tok in str(raw).split(","):
            tok = tok.strip()
            if tok:
                out.append(parse_bytes(tok))
    return out

def gen_strides_geo(min_stride: int, max_stride: int, factor: float):
    if min_stride <= 0 or max_stride <= 0:
        raise ValueError("stride bounds must be > 0")
    if max_stride < min_stride:
        raise ValueError("max_stride must be >= min_stride")
    if factor <= 1.0:
        raise ValueError("factor must be > 1")

    strides = []
    v = float(min_stride)
    while int(v) <= max_stride:
        s = int(v)
        # tlbbench requires stride >= 8 and 8B aligned
        s = max(8, (s // 8) * 8)
        if not strides or s != strides[-1]:
            strides.append(s)
        v *= factor
    return strides

def get_total_mem_bytes(exe):
    # 复用 tlbbench 的 --info 输出：GPU: ..., totalGlobalMem=XX.YY GiB
    try:
        p = subprocess.run([exe, "--info"], capture_output=True, text=True, check=True)
        s = (p.stdout or "").strip()
        key = "totalGlobalMem="
        if key not in s:
            return None
        tail = s.split(key, 1)[1].strip()
        # tail like: "23.99 GiB"
        num = tail.split()[0]
        gib = float(num)
        return int(gib * 1024 * 1024 * 1024)
    except Exception:
        return None

def run_one(exe, bytes_, stride, step, policy, repeats, warmup, rounds):
    cmd = [
        exe,
        "--bytes", str(bytes_),
        "--stride", str(stride),
        "--step", str(step),
        "--policy", policy,
        "--repeats", str(repeats),
        "--warmup", str(warmup),
        "--rounds", str(rounds),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        # 约定：tlbbench 返回码 2 表示 OOM
        if p.returncode == 2 or "out of memory" in err.lower() or "oom" in err.lower():
            raise MemoryError(err or "CUDA OOM")
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)
    out = (p.stdout or "").strip()
    # 输出一行 CSV：bytes,stride,policy,...
    parts = out.split(",")
    return {
        "bytes": int(parts[0]),
        "stride": int(parts[1]),
        "policy": parts[2],
        "nodes": int(parts[3]),
        "step": int(parts[4]),
        "warmup_rounds": int(parts[5]),
        "timed_rounds": int(parts[6]),
        "repeats": int(parts[7]),
        "median_cpa": float(parts[8]),
        "mean_cpa": float(parts[9]),
        "min_cpa": float(parts[10]),
        "max_cpa": float(parts[11]),
        "std_cpa": float(parts[12]),
    }

def gen_sizes_pow2(min_bytes, max_bytes):
    sizes = []
    v = min_bytes
    while v <= max_bytes:
        sizes.append(v)
        v *= 2
    return sizes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="./bin/tlbbench")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--append", action="store_true", help="append to existing CSV (requires matching header)")
    ap.add_argument("--step", type=int, default=131, help="ring step (must be coprime with nodes; odd works for pow2 nodes)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max_mem_frac", type=float, default=0.9,
                    help="fraction of total GPU mem used as safety cap; set 0 to disable")
    ap.add_argument("--max_bytes", type=str, default=None,
                    help="absolute cap (e.g., 20GiB); overrides max_mem_frac if set")

    # 关键：用多组 stride 做“TLB sweep”
    # 默认从大页开始，优先覆盖 L2 路径（参考论文中 2MiB / 32MiB）
    ap.add_argument(
        "--strides",
        nargs="+",
        default=[
            "1MiB",   # 对比
            "2MiB",   # 大页起点
            "8MiB",   # 拉开 L2 台阶间距
            "32MiB",  # 论文对比常用
            "64MiB",  # 进一步拉大 L2 访问 footprint
        ],
        help="one or more strides (e.g., 2MiB 32MiB; also supports comma list: 2MiB,32MiB)",
    )
    ap.add_argument("--stride_min", type=str, default=None,
                    help="generate strides in a geometric range (e.g., 512KiB)")
    ap.add_argument("--stride_max", type=str, default=None,
                    help="generate strides in a geometric range (e.g., 64MiB; default: 64MiB when --stride_min is set)")
    ap.add_argument("--stride_factor", type=float, default=2.0,
                    help="geometric factor when using --stride_min/--stride_max (default: 2)")

    ap.add_argument("--min_pages", type=int, default=8)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.stride_min is None and args.stride_max is not None:
        raise SystemExit("Use --stride_min when providing --stride_max.")

    if args.stride_min is not None:
        min_stride = parse_bytes(args.stride_min)
        max_stride_s = args.stride_max if args.stride_max is not None else "64MiB"
        max_stride = parse_bytes(max_stride_s)
        strides = gen_strides_geo(min_stride, max_stride, args.stride_factor)
    else:
        strides = parse_stride_list(args.strides)

    if not strides:
        raise SystemExit("No valid strides.")

    total_mem = get_total_mem_bytes(args.exe)
    if args.max_bytes:
        safe_max_bytes_global = parse_bytes(args.max_bytes)
    elif total_mem and args.max_mem_frac > 0:
        # 给显存留余量（runtime/其他进程/额外 malloc）
        safe_max_bytes_global = int(total_mem * args.max_mem_frac)
    else:
        # 不裁剪，靠 OOM 捕获来 break
        safe_max_bytes_global = None

    fields = [
        "bytes","stride","policy","nodes","step",
        "warmup_rounds","timed_rounds","repeats",
        "median_cpa","mean_cpa","min_cpa","max_cpa","std_cpa",
    ]

    expected_header = ",".join(fields)
    file_exists = os.path.exists(args.out)
    if args.append and file_exists:
        with open(args.out, "r", newline="") as rf:
            first = (rf.readline() or "").strip()
        if first and first != expected_header:
            raise SystemExit(
                f"CSV header mismatch for --append.\n"
                f"expected: {expected_header}\n"
                f"found:    {first}\n"
                f"Use a different --out, or omit --append to overwrite."
            )

    mode = "a" if args.append else "w"
    with open(args.out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if (not args.append) or (not file_exists):
            w.writeheader()

        for stride in strides:
            min_bytes = stride * args.min_pages
            if safe_max_bytes_global is not None and safe_max_bytes_global < min_bytes:
                print(f"SKIP stride={stride}: min_bytes={min_bytes} > max_bytes_cap={safe_max_bytes_global}")
                continue

            # 取消“每个 stride 不同 max_pages”的限制：所有 stride 都扫到同一个最大 array_size。
            # 若 safe_max_bytes_global=None（例如关闭 max_mem_frac 且未指定 --max_bytes），则一直翻倍直到 OOM。
            if safe_max_bytes_global is None:
                sizes = []
                v = min_bytes
                while True:
                    sizes.append(v)
                    v *= 2
                    if v > (1 << 60):
                        break
            else:
                sizes = gen_sizes_pow2(min_bytes, safe_max_bytes_global)

            for b in sizes:
                oom = False
                for policy in ["cg", "ca"]:
                    try:
                        row = run_one(args.exe, b, stride, args.step, policy, args.repeats, args.warmup, args.rounds)
                        w.writerow(row)
                        print("OK", row)
                    except MemoryError as e:
                        print(f"OOM stride={stride} bytes={b}: {e}")
                        oom = True
                        break
                if oom:
                    break
