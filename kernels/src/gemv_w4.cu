// Original work: Copyright 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// Licensed under the Apache License, Version 2.0. Modified for RDNA2 fat-binary
// (gfx1030+gfx1031+gfx1032), Windows build and RDNA2 tuning by sebastianmechno-sys.
// See THIRD_PARTY_NOTICES.md and LICENSE.
//
// M=1 W4A16 dequant-GEMV for RDNA3 (gfx1100), hand-written HIP to hit DRAM bandwidth.
// Layout (conch-normalized): wq [K//8, N] uint32 packed-along-K straight order (value for global
// row k is (wq[k/8, n] >> (k%8)*4) & 0xF); s [K//G, N] half; z [K//G, N] uint8; G=128.
// c[n] = sum_k a[k] * (q(k,n) - z(g,n)) * s(g,n).
//
// Bandwidth strategy (research §5): lane -> N, each thread owns COLS=4 consecutive columns = one
// 16-byte global_load_dwordx4 per k-row (32 lanes x 16B = 512B coalesced); __builtin_nontemporal_
// load (glc dlc, streaming/one-shot); split-K via gridDim.y + atomicAdd fp32 partials. U-unroll
// the k-row loads to keep multiple dwordx4 in flight (cover the ~230 ns DRAM latency / in-order vmcnt).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;
#define COLS 4
#define UNROLL 8

__global__ void __launch_bounds__(256)
gemv_w4_splitk(const __half* __restrict__ a, const unsigned int* __restrict__ wq,
               const __half* __restrict__ s, const unsigned char* __restrict__ z,
               float* __restrict__ c_acc, int K, int N, int Gsz, int split_groups) {
  const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * COLS;
  if (n0 >= N) return;
  const int num_groups = K / Gsz;
  const int g0 = blockIdx.y * split_groups;
  int g1 = g0 + split_groups; if (g1 > num_groups) g1 = num_groups;
  const int ROWS = Gsz / 8;  // int32 rows per group (16 for G=128)
  const bool full = (n0 + COLS <= N);

  float acc[COLS]; for (int c = 0; c < COLS; ++c) acc[c] = 0.f;

  for (int g = g0; g < g1; ++g) {
    const int k0 = g * Gsz;
    float sc[COLS], zr[COLS];
    for (int c = 0; c < COLS; ++c) {
      int n = n0 + c; if (n >= N) n = N - 1;
      sc[c] = __half2float(s[(size_t)g * N + n]);
      zr[c] = (float)z[(size_t)g * N + n];
    }
    float gacc[COLS]; for (int c = 0; c < COLS; ++c) gacc[c] = 0.f;
    float asum = 0.f;

    for (int rb = 0; rb < ROWS; rb += UNROLL) {
      u32x4 buf[UNROLL];
      // issue all UNROLL streaming dwordx4 loads before consuming -> latency hiding
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        int r = rb + u;
        const u32x4* wptr = (const u32x4*)(wq + (size_t)((k0 >> 3) + r) * N + n0);
        buf[u] = full ? __builtin_nontemporal_load(wptr) : *wptr;
      }
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        int r = rb + u;
        const __half* arow = a + k0 + r * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          float av = __half2float(arow[j]);
          asum += av;
          #pragma unroll
          for (int c = 0; c < COLS; ++c)
            gacc[c] += av * (float)((buf[u][c] >> (j * 4)) & 0xF);
        }
      }
    }
    for (int c = 0; c < COLS; ++c) acc[c] += (gacc[c] - zr[c] * asum) * sc[c];
  }

  for (int c = 0; c < COLS; ++c)
    if (n0 + c < N) atomicAdd(&c_acc[n0 + c], acc[c]);
}

// Symmetric variant: no per-group zero-point table, constant BIAS (uint4b8 -> 8).
// U = int32 rows consumed per chunk (latency cover). Rows-per-group ROWS = Gsz/8 must be a
// multiple of U; dispatch picks U = min(8, ROWS).
// direct=1: write fp16 result straight into c_out (no zero-init / atomicAdd / cast). Only valid
// when gridDim.y == 1.
template <int U>
__global__ void __launch_bounds__(256)
gemv_w4_sym_splitk(const __half* __restrict__ a, const unsigned int* __restrict__ wq,
                   const __half* __restrict__ s, float* __restrict__ c_acc,
                   __half* __restrict__ c_out, int direct,
                   int K, int N, int Gsz, int split_groups, const float bias) {
  const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * COLS;
  if (n0 >= N) return;
  const int num_groups = K / Gsz;
  const int g0 = blockIdx.y * split_groups;
  int g1 = g0 + split_groups; if (g1 > num_groups) g1 = num_groups;
  const int ROWS = Gsz / 8;
  const bool full = (n0 + COLS <= N);

  float acc[COLS]; for (int c = 0; c < COLS; ++c) acc[c] = 0.f;

  for (int g = g0; g < g1; ++g) {
    const int k0 = g * Gsz;
    float sc[COLS];
    for (int c = 0; c < COLS; ++c) {
      int n = n0 + c; if (n >= N) n = N - 1;
      sc[c] = __half2float(s[(size_t)g * N + n]);
    }
    float gacc[COLS]; for (int c = 0; c < COLS; ++c) gacc[c] = 0.f;
    float asum = 0.f;

    for (int rb = 0; rb < ROWS; rb += U) {
      u32x4 buf[U];
      #pragma unroll
      for (int u = 0; u < U; ++u) {
        int r = rb + u;
        const u32x4* wptr = (const u32x4*)(wq + (size_t)((k0 >> 3) + r) * N + n0);
        buf[u] = full ? __builtin_nontemporal_load(wptr) : *wptr;
      }
      #pragma unroll
      for (int u = 0; u < U; ++u) {
        int r = rb + u;
        const __half* arow = a + k0 + r * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          float av = __half2float(arow[j]);
          asum += av;
          #pragma unroll
          for (int c = 0; c < COLS; ++c)
            gacc[c] += av * (float)((buf[u][c] >> (j * 4)) & 0xF);
        }
      }
    }
    for (int c = 0; c < COLS; ++c) acc[c] += (gacc[c] - bias * asum) * sc[c];
  }

  if (direct) {
    for (int c = 0; c < COLS; ++c)
      if (n0 + c < N) c_out[n0 + c] = (__half)acc[c];
  } else {
    for (int c = 0; c < COLS; ++c)
      if (n0 + c < N) atomicAdd(&c_acc[n0 + c], acc[c]);
  }
}

torch::Tensor gemv_w4_sym(torch::Tensor a, torch::Tensor wq, torch::Tensor s,
                          int64_t group_size, int64_t split_groups, double bias) {
  int K = a.size(-1);
  int N = s.size(1);
  int num_groups = K / (int)group_size;
  int BT = 256;
  dim3 block(BT);
  int grid_y = (num_groups + (int)split_groups - 1) / (int)split_groups;
  dim3 grid((N + BT * COLS - 1) / (BT * COLS), grid_y);
  // No split-K needed (single k-range) -> write fp16 directly: skip zeros/atomicAdd/cast.
  const int direct = grid_y <= 1;
  torch::Tensor c_acc, c_out_t;
  float* cp = nullptr;
  __half* op = nullptr;
  if (!direct) {
    c_acc = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));
    cp = c_acc.data_ptr<float>();
  } else {
    c_out_t = torch::empty({N}, a.options());
    op = (__half*)c_out_t.data_ptr<at::Half>();
  }
  auto stream = c10::hip::getCurrentHIPStream();
  const __half* ap = (const __half*)a.data_ptr<at::Half>();
  const unsigned int* wqp = (const unsigned int*)wq.data_ptr<int>();
  const __half* sp = (const __half*)s.data_ptr<at::Half>();
  int rows = (int)group_size / 8;
  if (rows % 8 == 0)
    gemv_w4_sym_splitk<8><<<grid, block, 0, stream>>>(
        ap, wqp, sp, cp, op, direct, K, N, (int)group_size, (int)split_groups, (float)bias);
  else if (rows % 4 == 0)
    gemv_w4_sym_splitk<4><<<grid, block, 0, stream>>>(
        ap, wqp, sp, cp, op, direct, K, N, (int)group_size, (int)split_groups, (float)bias);
  else if (rows % 2 == 0)
    gemv_w4_sym_splitk<2><<<grid, block, 0, stream>>>(
        ap, wqp, sp, cp, op, direct, K, N, (int)group_size, (int)split_groups, (float)bias);
  else
    gemv_w4_sym_splitk<1><<<grid, block, 0, stream>>>(
        ap, wqp, sp, cp, op, direct, K, N, (int)group_size, (int)split_groups, (float)bias);
  if (!direct) c_out_t = c_acc.to(a.dtype());
  return c_out_t.view({1, N});
}

torch::Tensor gemv_w4_sym_meta(torch::Tensor a, torch::Tensor wq, torch::Tensor s,
                               int64_t group_size, int64_t split_groups, double bias) {
  return torch::empty({1, s.size(1)}, a.options());
}

torch::Tensor gemv_w4(torch::Tensor a, torch::Tensor wq, torch::Tensor s, torch::Tensor z,
                      int64_t group_size, int64_t split_groups) {
  int K = a.size(-1);
  int N = s.size(1);
  auto c_acc = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));
  int num_groups = K / group_size;
  int BT = 256;
  dim3 block(BT);
  dim3 grid((N + BT * COLS - 1) / (BT * COLS),
            (num_groups + split_groups - 1) / split_groups);
  auto stream = c10::hip::getCurrentHIPStream();
  gemv_w4_splitk<<<grid, block, 0, stream>>>(
      (const __half*)a.data_ptr<at::Half>(), (const unsigned int*)wq.data_ptr<int>(),
      (const __half*)s.data_ptr<at::Half>(), (const unsigned char*)z.data_ptr<uint8_t>(),
      c_acc.data_ptr<float>(), K, N, (int)group_size, (int)split_groups);
  return c_acc.to(a.dtype()).view({1, N});
}

// Meta (fake) impl for torch.compile tracing: shape only, no compute.
torch::Tensor gemv_w4_meta(torch::Tensor a, torch::Tensor wq, torch::Tensor s, torch::Tensor z,
                           int64_t group_size, int64_t split_groups) {
  return torch::empty({1, s.size(1)}, a.options());
}

// Register as a real torch op (like the _C kernels): dynamo-traceable AND cudagraph-safe,
// unlike a raw pybind call (dynamo error) or torch.library.custom_op (disables FULL_DECODE_ONLY).
TORCH_LIBRARY(vllm_win_hip, m) {
  m.def("gemv_w4(Tensor a, Tensor wq, Tensor s, Tensor z, int group_size, int split_groups) -> Tensor");
  m.def("gemv_w4_sym(Tensor a, Tensor wq, Tensor s, int group_size, int split_groups, float bias) -> Tensor");
  m.impl("gemv_w4", torch::kCUDA, &gemv_w4);
  m.impl("gemv_w4", torch::kMeta, &gemv_w4_meta);
  m.impl("gemv_w4_sym", torch::kCUDA, &gemv_w4_sym);
  m.impl("gemv_w4_sym", torch::kMeta, &gemv_w4_sym_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}  // empty: import triggers the TORCH_LIBRARY static init

