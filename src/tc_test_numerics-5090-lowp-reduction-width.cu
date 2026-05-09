/*
 * Copyright (c) 2026
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, version 2.
 */

#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "include/tcnb_cuda_check.cuh"
#include "include/tcnb_output.hpp"

static const int kInstructionK = 32;
static const int kLog2InstructionK = 5;
static const int kMinShift = 16;
static const int kMaxShift = 40;
static const int kExpectedBinary32Bits = std::numeric_limits<float>::digits;
enum { kObservedModelBits = 21 };

enum MmaVariant {
  kFp8E4m3E4m3,
  kFp8E5m2E5m2,
  kFp8E4m3E5m2,
  kFp8E5m2E4m3,
  kFp6E2m3E2m3,
  kFp6E3m2E3m2,
  kFp4E2m1E2m1,
};

struct WidthResult {
  bool executed;
  int max_changed_shift;
  int observed_bits;
  float sample_at_max;
  float sample_after_max;
};

static std::uint32_t repeat_byte(std::uint8_t value) {
  return static_cast<std::uint32_t>(value) * 0x01010101u;
}

static std::uint8_t encode_fp8_e4m3(float value) {
  return __nv_fp8_e4m3(value).__x;
}

static std::uint8_t encode_fp8_e5m2(float value) {
  return __nv_fp8_e5m2(value).__x;
}

static std::uint8_t encode_fp6_e2m3(float value) {
  return __nv_fp6_e2m3(value).__x;
}

static std::uint8_t encode_fp6_e3m2(float value) {
  return __nv_fp6_e3m2(value).__x;
}

static std::uint8_t encode_fp4_e2m1_padded(float value) {
  return static_cast<std::uint8_t>(__nv_fp4_e2m1(value).__x << 2);
}

static const char *variant_label(MmaVariant variant) {
  switch (variant) {
  case kFp8E4m3E4m3:
    return "FP8 E4M3 x E4M3";
  case kFp8E5m2E5m2:
    return "FP8 E5M2 x E5M2";
  case kFp8E4m3E5m2:
    return "FP8 E4M3 x E5M2";
  case kFp8E5m2E4m3:
    return "FP8 E5M2 x E4M3";
  case kFp6E2m3E2m3:
    return "FP6 E2M3 x E2M3";
  case kFp6E3m2E3m2:
    return "FP6 E3M2 x E3M2";
  case kFp4E2m1E2m1:
    return "FP4 E2M1 x E2M1";
  }

  return "unknown";
}

static const char *variant_instruction(MmaVariant variant) {
  switch (variant) {
  case kFp8E4m3E4m3:
    return "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32";
  case kFp8E5m2E5m2:
    return "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32";
  case kFp8E4m3E5m2:
    return "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32";
  case kFp8E5m2E4m3:
    return "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32";
  case kFp6E2m3E2m3:
    return "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32";
  case kFp6E3m2E3m2:
    return "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32";
  case kFp4E2m1E2m1:
    return "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32";
  }

  return "unknown";
}

static void variant_operands(MmaVariant variant, std::uint32_t *a_pack,
                             std::uint32_t *b_pack) {
  switch (variant) {
  case kFp8E4m3E4m3:
    *a_pack = repeat_byte(encode_fp8_e4m3(1.0f));
    *b_pack = repeat_byte(encode_fp8_e4m3(1.0f));
    break;
  case kFp8E5m2E5m2:
    *a_pack = repeat_byte(encode_fp8_e5m2(1.0f));
    *b_pack = repeat_byte(encode_fp8_e5m2(1.0f));
    break;
  case kFp8E4m3E5m2:
    *a_pack = repeat_byte(encode_fp8_e4m3(1.0f));
    *b_pack = repeat_byte(encode_fp8_e5m2(1.0f));
    break;
  case kFp8E5m2E4m3:
    *a_pack = repeat_byte(encode_fp8_e5m2(1.0f));
    *b_pack = repeat_byte(encode_fp8_e4m3(1.0f));
    break;
  case kFp6E2m3E2m3:
    *a_pack = repeat_byte(encode_fp6_e2m3(1.0f));
    *b_pack = repeat_byte(encode_fp6_e2m3(1.0f));
    break;
  case kFp6E3m2E3m2:
    *a_pack = repeat_byte(encode_fp6_e3m2(1.0f));
    *b_pack = repeat_byte(encode_fp6_e3m2(1.0f));
    break;
  case kFp4E2m1E2m1:
    *a_pack = repeat_byte(encode_fp4_e2m1_padded(1.0f));
    *b_pack = repeat_byte(encode_fp4_e2m1_padded(1.0f));
    break;
  }
}

#define TCNB_MMA_ASM(instruction)                                               \
  asm volatile(instruction " {%0, %1, %2, %3}, "                                \
                           "{%4, %5, %6, %7}, {%8, %9}, "                      \
                           "{%10, %11, %12, %13};\n"                           \
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                       \
               : "r"(a_pack), "r"(a_pack), "r"(a_pack), "r"(a_pack),         \
                 "r"(b_pack), "r"(b_pack), "f"(c0), "f"(c1),                 \
                 "f"(c2), "f"(c3))

template <MmaVariant variant>
__device__ __forceinline__ void run_direct_mma_fragment(
    std::uint32_t a_pack, std::uint32_t b_pack, float c0, float c1, float c2,
    float c3, float *out0, float *out1, float *out2, float *out3) {
  float d0;
  float d1;
  float d2;
  float d3;

  if (variant == kFp8E4m3E4m3) {
    TCNB_MMA_ASM(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32");
  } else if (variant == kFp8E5m2E5m2) {
    TCNB_MMA_ASM(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32");
  } else if (variant == kFp8E4m3E5m2) {
    TCNB_MMA_ASM(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32");
  } else if (variant == kFp8E5m2E4m3) {
    TCNB_MMA_ASM(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32");
  } else if (variant == kFp6E2m3E2m3) {
    TCNB_MMA_ASM(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32");
  } else if (variant == kFp6E3m2E3m2) {
    TCNB_MMA_ASM(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32");
  } else {
    TCNB_MMA_ASM(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32");
  }

  *out0 = d0;
  *out1 = d1;
  *out2 = d2;
  *out3 = d3;
}

template <MmaVariant variant, int repeat_count>
__device__ __forceinline__ void run_direct_mma_chain(std::uint32_t a_pack,
                                                     std::uint32_t b_pack,
                                                     float base, float *out) {
  float c0 = base;
  float c1 = base;
  float c2 = base;
  float c3 = base;

#pragma unroll
  for (int repeat = 0; repeat < repeat_count; ++repeat) {
    float d0;
    float d1;
    float d2;
    float d3;
    run_direct_mma_fragment<variant>(a_pack, b_pack, c0, c1, c2, c3, &d0, &d1,
                                     &d2, &d3);
    c0 = d0;
    c1 = d1;
    c2 = d2;
    c3 = d3;
  }

  if ((threadIdx.x & 31) == 0) {
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
  }
}

#undef TCNB_MMA_ASM

template <MmaVariant variant, int repeat_count>
__global__ void reduction_width_kernel(std::uint32_t a_pack,
                                       std::uint32_t b_pack, float base,
                                       float *out) {
  run_direct_mma_chain<variant, repeat_count>(a_pack, b_pack, base, out);
}

template <MmaVariant variant, int repeat_count>
static float run_once(std::uint32_t a_pack, std::uint32_t b_pack, int shift,
                      float *device_out) {
  float host_out[4] = {};
  float base = std::ldexp(1.0f, shift);

  reduction_width_kernel<variant, repeat_count><<<1, 32>>>(a_pack, b_pack,
                                                           base, device_out);
  TCNB_CUDA_CHECK(cudaGetLastError());
  TCNB_CUDA_CHECK(cudaMemcpy(host_out, device_out, sizeof(host_out),
                             cudaMemcpyDeviceToHost));

  return host_out[0];
}

template <MmaVariant variant, int repeat_count> static WidthResult probe_variant() {
  WidthResult result = {};
  result.executed = false;
  result.max_changed_shift = -1;
  result.observed_bits = -1;
  result.sample_at_max = 0.0f;
  result.sample_after_max = 0.0f;

  std::uint32_t a_pack = 0;
  std::uint32_t b_pack = 0;
  variant_operands(variant, &a_pack, &b_pack);

  float *device_out = nullptr;
  TCNB_CUDA_CHECK(cudaMalloc(&device_out, 4 * sizeof(float)));

  for (int shift = kMinShift; shift <= kMaxShift; ++shift) {
    float base = std::ldexp(1.0f, shift);
    float sample = run_once<variant, repeat_count>(a_pack, b_pack, shift,
                                                   device_out);
    result.executed = true;

    if (sample > base) {
      result.max_changed_shift = shift;
      result.sample_at_max = sample;
    } else if (result.max_changed_shift >= 0 &&
               result.sample_after_max == 0.0f) {
      result.sample_after_max = sample;
    }
  }

  if (result.max_changed_shift >= 0) {
    result.observed_bits =
        result.max_changed_shift - kLog2InstructionK + 1;
  }

  TCNB_CUDA_CHECK(cudaFree(device_out));
  return result;
}

static int max_shift_for_width(int log2_contribution, int width_bits) {
  return log2_contribution + width_bits - 1;
}

static bool print_result(FILE *outfile, MmaVariant variant,
                         const WidthResult &result, int repeat_count,
                         int log2_contribution) {
  int total_contribution = kInstructionK * repeat_count;
  int binary32_per_mma_reference_max_shift =
      max_shift_for_width(kLog2InstructionK, kExpectedBinary32Bits);
  int single_round_21_bit_max_shift =
      max_shift_for_width(log2_contribution, kObservedModelBits);
  int single_round_binary32_max_shift =
      max_shift_for_width(log2_contribution, kExpectedBinary32Bits);

  fprintf(outfile, "  | %-58s |\n", variant_label(variant));
  fprintf(outfile, "  |   PTX: %-51s |\n", variant_instruction(variant));
  fprintf(outfile, "  |   K per instruction: %-35d |\n", kInstructionK);
  fprintf(outfile, "  |   repeated MMA instructions: %-24d |\n",
          repeat_count);
  fprintf(outfile, "  |   total contribution: %-32d |\n",
          total_contribution);
  fprintf(outfile, "  |   max shift with D > C: %-27d |\n",
          result.max_changed_shift);
  fprintf(outfile, "  |   per-MMA observed width bits: %-23d |\n",
          result.observed_bits);
  fprintf(outfile, "  |   binary32 per-MMA reference shift: %-17d |\n",
          binary32_per_mma_reference_max_shift);
  fprintf(outfile, "  |   binary32 reference bits: %-25d |\n",
          kExpectedBinary32Bits);
  if (repeat_count > 1) {
    fprintf(outfile, "  |   single-round 21-bit shift: %-23d |\n",
            single_round_21_bit_max_shift);
    fprintf(outfile, "  |   single-round binary32 shift: %-22d |\n",
            single_round_binary32_max_shift);
  }
  fprintf(outfile, "  |   sample at max shift: %-28.9g |\n",
          result.sample_at_max);
  fprintf(outfile, "  |   sample after max shift: %-26.9g |\n",
          result.sample_after_max);

  bool passed = result.executed && result.observed_bits > 0;
  printitem(outfile, "*) Direct PTX mma probe executed");
  printpass(outfile, passed);
  return passed;
}

template <MmaVariant variant>
static bool print_single_result(FILE *outfile) {
  return print_result(outfile, variant, probe_variant<variant, 1>(), 1,
                      kLog2InstructionK);
}

template <MmaVariant variant, int repeat_count, int log2_repeat_count>
static bool print_repeat_result(FILE *outfile) {
  const int log2_contribution = kLog2InstructionK + log2_repeat_count;
  WidthResult result = probe_variant<variant, repeat_count>();
  bool passed =
      print_result(outfile, variant, result, repeat_count, log2_contribution);
  int expected_model_max_shift =
      max_shift_for_width(kLog2InstructionK, kObservedModelBits);
  fprintf(outfile, "  |   per-MMA 21-bit model shift: %-20d |\n",
          expected_model_max_shift);
  printitem(outfile, "*) Matches per-MMA 21-bit model");
  bool model_pass = result.max_changed_shift == expected_model_max_shift &&
                    result.observed_bits == kObservedModelBits;
  printpass(outfile, model_pass);
  return passed && model_pass;
}

template <MmaVariant variant>
static bool print_repeat_results(FILE *outfile) {
  bool pass = true;
  pass = print_repeat_result<variant, 2, 1>(outfile) && pass;
  pass = print_repeat_result<variant, 4, 2>(outfile) && pass;
  pass = print_repeat_result<variant, 8, 3>(outfile) && pass;
  pass = print_repeat_result<variant, 16, 4>(outfile) && pass;
  return pass;
}

int main(int argc, char **argv) {
  FILE *outfile = stdout;
  bool pass = true;

#if defined(TCNB_REDUCTION_REPEAT)
  printheader(outfile,
              "A. 5090 repeated-MMA low-precision reduction-width probe");
  fprintf(outfile, "  | Probe: %-51s |\n", "D = 2^shift + repeat_count * K");
  fprintf(outfile, "  | Method: %-50s |\n", "chain 2/4/8/16 direct MMA ops");
  fprintf(outfile, "  | Notes: %-51s |\n", "same warp, accumulator fed forward");
#else
  printheader(outfile, "A. 5090 direct-PTX low-precision reduction-width probe");
  fprintf(outfile, "  | Probe: %-51s |\n", "D = 2^shift + K");
  fprintf(outfile, "  | Method: %-50s |\n", "scan largest shift where D changes");
  fprintf(outfile, "  | Notes: %-51s |\n", "single warp, direct mma.sync.aligned PTX");
#endif

#if defined(TCNB_REDUCTION_FP8)
#if defined(TCNB_REDUCTION_REPEAT)
  pass = print_repeat_results<kFp8E4m3E4m3>(outfile) && pass;
  pass = print_repeat_results<kFp8E5m2E5m2>(outfile) && pass;
  pass = print_repeat_results<kFp8E4m3E5m2>(outfile) && pass;
  pass = print_repeat_results<kFp8E5m2E4m3>(outfile) && pass;
#else
  pass = print_single_result<kFp8E4m3E4m3>(outfile) && pass;
  pass = print_single_result<kFp8E5m2E5m2>(outfile) && pass;
  pass = print_single_result<kFp8E4m3E5m2>(outfile) && pass;
  pass = print_single_result<kFp8E5m2E4m3>(outfile) && pass;
#endif
#elif defined(TCNB_REDUCTION_FP6)
#if defined(TCNB_REDUCTION_REPEAT)
  pass = print_repeat_results<kFp6E2m3E2m3>(outfile) && pass;
  pass = print_repeat_results<kFp6E3m2E3m2>(outfile) && pass;
#else
  pass = print_single_result<kFp6E2m3E2m3>(outfile) && pass;
  pass = print_single_result<kFp6E3m2E3m2>(outfile) && pass;
#endif
#elif defined(TCNB_REDUCTION_FP4)
#if defined(TCNB_REDUCTION_REPEAT)
  pass = print_repeat_results<kFp4E2m1E2m1>(outfile) && pass;
#else
  pass = print_single_result<kFp4E2m1E2m1>(outfile) && pass;
#endif
#else
#error "Define one TCNB_REDUCTION_* format macro"
#endif

  printfooter(outfile);
  return pass ? 0 : 1;
}
