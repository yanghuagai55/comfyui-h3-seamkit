// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 comfyui-h3-seamkit contributors
//
// Hadamard Rotated-Domain INT8 x BF16 Fused LoRA Down-Projection & Radix-4 ConvRot-256 Kernel
// Architecture Spec: docs/H3_ANIME_ENGINE_V3_ARCHITECTURE.md (§3.2)
//
// Mathematical Invariant:
//   In MiniMax-H3 INT8 ConvRot checkpoints (e.g. DaSiWa v5xs), activations h [M, K] are rotated
//   online by the symmetric orthogonal group-256 Regular Hadamard matrix H_256 = (H_4)^{\otimes 4} / 16
//   (H_256^T = H_256, H_256^2 = I) and quantized row-wise into (X_q [M, K] int8, s_x [M, 1] fp32)
//   such that h @ H ~= diag(s_x) * X_q.
//   Pre-rotating LoRA down-projection weights once at load time (A_rot = A @ H) allows computing:
//       Z = h @ A^T = (h @ H) @ (A @ H)^T ~= diag(s_x) * (X_q @ A_rot^T)
//   directly from resident INT8 X_q (1 byte/elem, 50% HBM traffic, hot in 24MB L2 cache on SM89)
//   without ever materializing or re-reading unrotated BF16 activations (saving 316 GB/step!).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

#ifdef _WIN32
#define H3_ROT_EXPORT extern "C" __declspec(dllexport)
#else
#define H3_ROT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

// ---------------------------------------------------------------------------
// 1. Fast Radix-4 Regular Hadamard Transform (ConvRot-256) on BF16/FP32 rows
//    H_4 = [[ 1,  1,  1, -1],
//           [ 1,  1, -1,  1],
//           [ 1, -1,  1,  1],
//           [-1,  1,  1,  1]] * (1/2)
//    y_j = (x_0 + x_1 + x_2 + x_3) - 2 * x_{3 - j}
// ---------------------------------------------------------------------------

__global__ void convrot_hadamard256_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int rows,
    int cols
) {
    const int group_idx = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) return;

    const int tid = threadIdx.x;  // 0..63
    const int64_t base_offset = static_cast<int64_t>(row) * cols + static_cast<int64_t>(group_idx) * 256;

    __shared__ float smem[256];

    // Stage 0 (stride = 1): load 4 contiguous BF16 values per thread and apply H_4 in registers
    int i0 = tid * 4;
    float x0 = __bfloat162float(src[base_offset + i0 + 0]);
    float x1 = __bfloat162float(src[base_offset + i0 + 1]);
    float x2 = __bfloat162float(src[base_offset + i0 + 2]);
    float x3 = __bfloat162float(src[base_offset + i0 + 3]);
    float s0 = x0 + x1 + x2 + x3;
    smem[i0 + 0] = s0 - 2.0f * x3;
    smem[i0 + 1] = s0 - 2.0f * x2;
    smem[i0 + 2] = s0 - 2.0f * x1;
    smem[i0 + 3] = s0 - 2.0f * x0;
    __syncthreads();

    // Stage 1 (stride = 4, log2 = 2)
    {
        int inner = tid & 3;
        int outer = tid >> 2;
        int b = (outer << 4) + inner;
        float v0 = smem[b + 0];
        float v1 = smem[b + 4];
        float v2 = smem[b + 8];
        float v3 = smem[b + 12];
        float s = v0 + v1 + v2 + v3;
        smem[b + 0]  = s - 2.0f * v3;
        smem[b + 4]  = s - 2.0f * v2;
        smem[b + 8]  = s - 2.0f * v1;
        smem[b + 12] = s - 2.0f * v0;
    }
    __syncthreads();

    // Stage 2 (stride = 16, log2 = 4)
    {
        int inner = tid & 15;
        int outer = tid >> 4;
        int b = (outer << 6) + inner;
        float v0 = smem[b + 0];
        float v1 = smem[b + 16];
        float v2 = smem[b + 32];
        float v3 = smem[b + 48];
        float s = v0 + v1 + v2 + v3;
        smem[b + 0]  = s - 2.0f * v3;
        smem[b + 16] = s - 2.0f * v2;
        smem[b + 32] = s - 2.0f * v1;
        smem[b + 48] = s - 2.0f * v0;
    }
    __syncthreads();

    // Stage 3 (stride = 64, log2 = 6) + 1/16 normalization + store BF16
    {
        int b = tid;
        float v0 = smem[b + 0];
        float v1 = smem[b + 64];
        float v2 = smem[b + 128];
        float v3 = smem[b + 192];
        float s = v0 + v1 + v2 + v3;
        constexpr float inv16 = 0.0625f;
        dst[base_offset + b + 0]   = __float2bfloat16((s - 2.0f * v3) * inv16);
        dst[base_offset + b + 64]  = __float2bfloat16((s - 2.0f * v2) * inv16);
        dst[base_offset + b + 128] = __float2bfloat16((s - 2.0f * v1) * inv16);
        dst[base_offset + b + 192] = __float2bfloat16((s - 2.0f * v0) * inv16);
    }
}

// ---------------------------------------------------------------------------
// 2. Fused INT8 Activation x BF16 Rotated LoRA Tensor Core Down-Projection
//    Computes Z[M, R] = diag(s_x[M]) * (X_q[M, K] @ A_rot[R, K]^T)
//    - Tile: BM = 64, BN = 64, BK = 64, 4 warps (128 threads/CTA)
//    - Shared memory: 18.4 KB/CTA (high SM occupancy, zero bank conflicts via +8 skew)
//    - Uses SM80+ BF16 Tensor Core WMMA (m16n16k16, FP32 accumulation)
// ---------------------------------------------------------------------------

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int SMEM_STRIDE = 72;  // 64 + 8 (multiple of 8 for WMMA & 16B aligned, bank-conflict free)

__global__ __launch_bounds__(128, 2)
void rot_int8_lora_down_bf16_kernel(
    const int8_t* __restrict__ x_q,          // [M, K] row-major int8
    const float* __restrict__ x_scale,       // [M] fp32
    const __nv_bfloat16* __restrict__ a_rot, // [R, K] row-major bf16
    __nv_bfloat16* __restrict__ out_z,       // [M, R] row-major bf16
    int M,
    int R,
    int K
) {
    const int block_m = blockIdx.x * BM;
    const int block_n = blockIdx.y * BN;
    const int tid = threadIdx.x;       // 0..127
    const int warp_id = tid >> 5;      // 0..3
    const int lane_id = tid & 31;      // 0..31

    __shared__ __align__(16) __nv_bfloat16 smem_x[BM][SMEM_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_a[BN][SMEM_STRIDE];

#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    const int warp_m = (warp_id >> 1) * 32;  // 0 or 32
    const int warp_n = (warp_id & 1) * 32;   // 0 or 32

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    const bool fast_k16 = ((K & 15) == 0);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 1. Cooperative INT8 -> BF16 load into smem_x [64][72] (4096 elems / 128 threads = 32 elems/thread)
        if (fast_k16 && (k0 + BK <= K)) {
            #pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                int vec_idx = tid + iter * 128;      // 0..255 (each vec is 16 int8 bytes)
                int r_local = vec_idx >> 2;          // 0..63
                int c_local = (vec_idx & 3) << 4;    // 0, 16, 32, 48
                int gm = block_m + r_local;
                if (gm < M) {
                    int4 raw = *reinterpret_cast<const int4*>(
                        x_q + static_cast<int64_t>(gm) * K + k0 + c_local
                    );
                    const int8_t* bytes = reinterpret_cast<const int8_t*>(&raw);
                    #pragma unroll
                    for (int j = 0; j < 16; j += 2) {
                        __nv_bfloat162 pair = __floats2bfloat162_rn(
                            static_cast<float>(bytes[j]),
                            static_cast<float>(bytes[j + 1])
                        );
                        *reinterpret_cast<__nv_bfloat162*>(&smem_x[r_local][c_local + j]) = pair;
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; j += 2) {
                        *reinterpret_cast<uint32_t*>(&smem_x[r_local][c_local + j]) = 0u;
                    }
                }
            }
        } else {
            for (int idx = tid; idx < BM * BK; idx += 128) {
                int r_local = idx >> 6;
                int c_local = idx & 63;
                int gm = block_m + r_local;
                int gk = k0 + c_local;
                float val = (gm < M && gk < K)
                    ? static_cast<float>(x_q[static_cast<int64_t>(gm) * K + gk])
                    : 0.0f;
                smem_x[r_local][c_local] = __float2bfloat16(val);
            }
        }

        // 2. Cooperative BF16 load of A_rot into smem_a [64][72] (32 bf16 elems = 4 int4 per thread)
        if (fast_k16 && (k0 + BK <= K)) {
            #pragma unroll
            for (int iter = 0; iter < 4; ++iter) {
                int vec_idx = tid + iter * 128;      // 0..511 (each int4 is 8 bf16 = 16 bytes)
                int r_local = vec_idx >> 3;          // 0..63
                int c_local = (vec_idx & 7) << 3;    // 0, 8, ..., 56
                int gn = block_n + r_local;
                if (gn < R) {
                    int4 raw = *reinterpret_cast<const int4*>(
                        a_rot + static_cast<int64_t>(gn) * K + k0 + c_local
                    );
                    *reinterpret_cast<int4*>(&smem_a[r_local][c_local]) = raw;
                } else {
                    *reinterpret_cast<int4*>(&smem_a[r_local][c_local]) = make_int4(0, 0, 0, 0);
                }
            }
        } else {
            for (int idx = tid; idx < BN * BK; idx += 128) {
                int r_local = idx >> 6;
                int c_local = idx & 63;
                int gn = block_n + r_local;
                int gk = k0 + c_local;
                smem_a[r_local][c_local] = (gn < R && gk < K)
                    ? a_rot[static_cast<int64_t>(gn) * K + gk]
                    : __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // 3. Tensor Core WMMA accumulation over BK = 64 (4 steps of k=16)
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag[2];

            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                wmma::load_matrix_sync(
                    a_frag[mi],
                    &smem_x[warp_m + mi * 16][kk * 16],
                    SMEM_STRIDE
                );
            }
            #pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                // smem_a[n][k] row-major with stride SMEM_STRIDE == col-major [K, N] with ldm=SMEM_STRIDE!
                wmma::load_matrix_sync(
                    b_frag[ni],
                    &smem_a[warp_n + ni * 16][kk * 16],
                    SMEM_STRIDE
                );
            }
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    // 4. Fused row-scale epilogue using shared memory staging (reusing smem_x storage)
    float* warp_stage = reinterpret_cast<float*>(&smem_x[0][0]) + warp_id * 256;
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::store_matrix_sync(warp_stage, acc[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = lane_id; e < 256; e += 32) {
                int rm = e >> 4;
                int cn = e & 15;
                int gm = block_m + warp_m + mi * 16 + rm;
                int gn = block_n + warp_n + ni * 16 + cn;
                if (gm < M && gn < R) {
                    float scaled = warp_stage[e] * x_scale[gm];
                    out_z[static_cast<int64_t>(gm) * R + gn] = __float2bfloat16(scaled);
                }
            }
            __syncwarp();
        }
    }
#else
    // Pre-SM80 fallback (e.g. Turing SM75): tiled shared-memory FP32 FMA
    const int row_local = tid >> 1;          // 0..63
    const int col_half = (tid & 1) * 32;     // 0 or 32
    float acc_fallback[32];
    #pragma unroll
    for (int c = 0; c < 32; ++c) acc_fallback[c] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += 128) {
            int rl = idx >> 6, cl = idx & 63;
            int gm = block_m + rl, gk = k0 + cl;
            smem_x[rl][cl] = __float2bfloat16((gm < M && gk < K) ? (float)x_q[(int64_t)gm * K + gk] : 0.0f);
        }
        for (int idx = tid; idx < BN * BK; idx += 128) {
            int rl = idx >> 6, cl = idx & 63;
            int gn = block_n + rl, gk = k0 + cl;
            smem_a[rl][cl] = (gn < R && gk < K) ? a_rot[(int64_t)gn * K + gk] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        for (int kk = 0; kk < BK; ++kk) {
            float xv = __bfloat162float(smem_x[row_local][kk]);
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                acc_fallback[c] += xv * __bfloat162float(smem_a[col_half + c][kk]);
            }
        }
        __syncthreads();
    }
    int gm = block_m + row_local;
    if (gm < M) {
        float sc = x_scale[gm];
        for (int c = 0; c < 32; ++c) {
            int gn = block_n + col_half + c;
            if (gn < R) {
                out_z[(int64_t)gm * R + gn] = __float2bfloat16(acc_fallback[c] * sc);
            }
        }
    }
#endif
}

}  // namespace

H3_ROT_EXPORT int h3_convrot_hadamard256_bf16(
    const void* src,
    void* dst,
    int rows,
    int cols,
    void* stream
) {
    if (rows <= 0 || cols <= 0) return 0;
    if ((cols & 255) != 0) return -1;
    dim3 grid(cols / 256, rows, 1);
    dim3 block(64, 1, 1);
    convrot_hadamard256_bf16_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(src),
        reinterpret_cast<__nv_bfloat16*>(dst),
        rows,
        cols
    );
    return static_cast<int>(cudaGetLastError());
}

H3_ROT_EXPORT int h3_rot_int8_lora_down_bf16(
    const void* x_q,
    const void* x_scale,
    const void* a_rot,
    void* out_z,
    int M,
    int R,
    int K,
    void* stream
) {
    if (M <= 0 || R <= 0 || K <= 0) return 0;
    dim3 grid((M + BM - 1) / BM, (R + BN - 1) / BN, 1);
    dim3 block(128, 1, 1);
    rot_int8_lora_down_bf16_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int8_t*>(x_q),
        reinterpret_cast<const float*>(x_scale),
        reinterpret_cast<const __nv_bfloat16*>(a_rot),
        reinterpret_cast<__nv_bfloat16*>(out_z),
        M,
        R,
        K
    );
    return static_cast<int>(cudaGetLastError());
}
