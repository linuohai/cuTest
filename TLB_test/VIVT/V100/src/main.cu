#include "cuda_utils.cuh"
#include "tlb_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// 简单参数解析（够用即可）
static uint64_t parse_u64(const char* s) {
  // 支持 K/M/G 后缀（KiB/MiB/GiB 用 1024）
  char* end = nullptr;
  uint64_t v = strtoull(s, &end, 10);
  if (!end || *end == 0) return v;
  if (*end == 'K' || *end == 'k') return v * 1024ull;
  if (*end == 'M' || *end == 'm') return v * 1024ull * 1024ull;
  if (*end == 'G' || *end == 'g') return v * 1024ull * 1024ull * 1024ull;
  return v;
}

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
  while (b) {
    uint64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

int main(int argc, char** argv) {
  // 默认参数
  uint64_t bytes  = 128ull * 1024ull * 1024ull; // 128MiB
  uint64_t stride = 2ull   * 1024ull * 1024ull; // 2MiB
  uint64_t ring_step = 131; // 访问顺序“打散”，减弱顺序预取/投机
  int warmup_rounds = 1;
  int timed_rounds  = 128; // 用更多步数均摊循环/计时开销
  int repeats       = 9;   // 多次重复取中位数，抗噪声
  int use_ca        = 1; // ca
  int print_header  = 0;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--bytes") && i+1 < argc) bytes = parse_u64(argv[++i]);
    else if (!strcmp(argv[i], "--stride") && i+1 < argc) stride = parse_u64(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i+1 < argc) warmup_rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rounds") && i+1 < argc) timed_rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--repeats") && i+1 < argc) repeats = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--step") && i+1 < argc) ring_step = parse_u64(argv[++i]);
    else if (!strcmp(argv[i], "--policy") && i+1 < argc) {
      const char* p = argv[++i];
      if (!strcmp(p, "ca")) use_ca = 1;
      else if (!strcmp(p, "cg")) use_ca = 0;
      else { fprintf(stderr, "Unknown --policy %s (use ca|cg)\n", p); return 1; }
    } else if (!strcmp(argv[i], "--header")) print_header = 1;
    else if (!strcmp(argv[i], "--info")) { print_device_info(); return 0; }
    else {
      fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      return 1;
    }
  }

  if (stride < 8 || (stride % 8) != 0) {
    fprintf(stderr, "stride must be >=8 and 8-byte aligned.\n");
    return 1;
  }
  if (bytes < stride * 2) {
    fprintf(stderr, "bytes must be >= 2*stride.\n");
    return 1;
  }
  if (timed_rounds <= 0 || warmup_rounds < 0 || repeats <= 0) {
    fprintf(stderr, "warmup must be >=0, rounds must be >0, repeats must be >0.\n");
    return 1;
  }
  bytes = (bytes / stride) * stride; // 对齐到 stride
  uint64_t n = bytes / stride;

  uint64_t step_mod = ring_step % n;
  if (step_mod == 0 || gcd_u64(step_mod, n) != 1) {
    fprintf(stderr,
            "--step must be coprime with nodes=%" PRIu64 " (got step=%" PRIu64 ", step%%nodes=%" PRIu64 ")\n",
            n, ring_step, step_mod);
    return 1;
  }

  uint8_t* d_buf = nullptr;
  uint64_t *d_cycles = nullptr, *d_steps = nullptr, *d_sink = nullptr;

  auto oom_fail = [&](const char* what, uint64_t req_bytes) -> int {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr,
            "CUDA OOM while allocating %s (requested=%" PRIu64 " bytes, free=%zu bytes, total=%zu bytes)\n",
            what, req_bytes, free_b, total_b);
    return 2; // 约定：2 表示 OOM（便于 sweep.py 判断/跳过）
  };

  {
    cudaError_t e = cudaMalloc(&d_buf, bytes);
    if (e != cudaSuccess) return oom_fail("d_buf", bytes);
  }
  {
    cudaError_t e = cudaMalloc(&d_cycles, sizeof(uint64_t));
    if (e != cudaSuccess) { cudaFree(d_buf); return oom_fail("d_cycles", sizeof(uint64_t)); }
  }
  {
    cudaError_t e = cudaMalloc(&d_steps, sizeof(uint64_t));
    if (e != cudaSuccess) { cudaFree(d_cycles); cudaFree(d_buf); return oom_fail("d_steps", sizeof(uint64_t)); }
  }
  {
    cudaError_t e = cudaMalloc(&d_sink, sizeof(uint64_t));
    if (e != cudaSuccess) { cudaFree(d_steps); cudaFree(d_cycles); cudaFree(d_buf); return oom_fail("d_sink", sizeof(uint64_t)); }
  }

  // 初始化环形指针（在 GPU 上写入 device VA）
  {
    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    init_ring<<<blocks, threads>>>(d_buf, bytes, stride, step_mod);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // 多次重复，统计 cycles/access（减少噪声）
  std::vector<double> cpas;
  cpas.reserve((size_t)repeats);

  for (int r = 0; r < repeats; r++) {
    chase<<<1, 32>>>(d_buf, bytes, stride, warmup_rounds, timed_rounds, use_ca,
                     d_cycles, d_steps, d_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t h_cycles=0, h_steps=0, h_sink=0;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_steps,  d_steps,  sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sink,   d_sink,   sizeof(uint64_t), cudaMemcpyDeviceToHost));
    (void)h_sink;

    if (h_steps == 0) continue;
    double cpa = (double)h_cycles / (double)h_steps;
    cpas.push_back(cpa);
  }

  if (cpas.empty()) {
    fprintf(stderr, "No valid measurements.\n");
    return 3;
  }

  std::vector<double> sorted = cpas;
  std::sort(sorted.begin(), sorted.end());
  double min_cpa = sorted.front();
  double max_cpa = sorted.back();
  double median_cpa = 0.0;
  if (sorted.size() % 2 == 1) {
    median_cpa = sorted[sorted.size() / 2];
  } else {
    size_t hi = sorted.size() / 2;
    median_cpa = 0.5 * (sorted[hi - 1] + sorted[hi]);
  }
  double mean_cpa = std::accumulate(cpas.begin(), cpas.end(), 0.0) / (double)cpas.size();
  double var = 0.0;
  for (double x : cpas) var += (x - mean_cpa) * (x - mean_cpa);
  var /= (double)cpas.size();
  double std_cpa = std::sqrt(var);

  if (print_header) {
    printf("bytes,stride,policy,nodes,step,warmup_rounds,timed_rounds,repeats,median_cpa,mean_cpa,min_cpa,max_cpa,std_cpa\n");
  }
  printf("%" PRIu64 ",%" PRIu64 ",%s,%" PRIu64 ",%" PRIu64 ",%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
         bytes, stride, use_ca ? "ca" : "cg", n,
         step_mod,
         warmup_rounds, timed_rounds, repeats,
         median_cpa, mean_cpa, min_cpa, max_cpa, std_cpa);

  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_steps));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_buf));
  return 0;
}
