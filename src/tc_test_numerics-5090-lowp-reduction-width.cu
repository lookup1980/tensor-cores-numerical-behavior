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
static const int kExpectedBinary32MaxShift =
    kLog2InstructionK + kExpectedBinary32Bits - 1;

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
                 "r"(b_pack), "r"(b_pack), "f"(base), "f"(base),             \
                 "f"(base), "f"(base))

template <MmaVariant variant>
__device__ __forceinline__ void run_direct_mma(std::uint32_t a_pack,
                                               std::uint32_t b_pack, float base,
                                               float *out) {
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

  if ((threadIdx.x & 31) == 0) {
    out[0] = d0;
    out[1] = d1;
    out[2] = d2;
    out[3] = d3;
  }
}

#undef TCNB_MMA_ASM

template <MmaVariant variant>
__global__ void reduction_width_kernel(std::uint32_t a_pack,
                                       std::uint32_t b_pack, float base,
                                       float *out) {
  run_direct_mma<variant>(a_pack, b_pack, base, out);
}

template <MmaVariant variant>
static float run_once(std::uint32_t a_pack, std::uint32_t b_pack, int shift,
                      float *device_out) {
  float host_out[4] = {};
  float base = std::ldexp(1.0f, shift);

  reduction_width_kernel<variant><<<1, 32>>>(a_pack, b_pack, base, device_out);
  TCNB_CUDA_CHECK(cudaGetLastError());
  TCNB_CUDA_CHECK(cudaMemcpy(host_out, device_out, sizeof(host_out),
                             cudaMemcpyDeviceToHost));

  return host_out[0];
}

template <MmaVariant variant> static WidthResult probe_variant() {
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
    float sample = run_once<variant>(a_pack, b_pack, shift, device_out);
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

static bool print_result(FILE *outfile, MmaVariant variant,
                         const WidthResult &result) {
  fprintf(outfile, "  | %-58s |\n", variant_label(variant));
  fprintf(outfile, "  |   PTX: %-51s |\n", variant_instruction(variant));
  fprintf(outfile, "  |   K per instruction: %-35d |\n", kInstructionK);
  fprintf(outfile, "  |   max shift with D > C: %-27d |\n",
          result.max_changed_shift);
  fprintf(outfile, "  |   observed width bits: %-31d |\n",
          result.observed_bits);
  fprintf(outfile, "  |   binary32 reference max shift: %-20d |\n",
          kExpectedBinary32MaxShift);
  fprintf(outfile, "  |   binary32 reference bits: %-25d |\n",
          kExpectedBinary32Bits);
  fprintf(outfile, "  |   sample at max shift: %-28.9g |\n",
          result.sample_at_max);
  fprintf(outfile, "  |   sample after max shift: %-26.9g |\n",
          result.sample_after_max);

  bool passed = result.executed && result.observed_bits > 0;
  printitem(outfile, "*) Direct PTX mma probe executed");
  printpass(outfile, passed);
  return passed;
}

int main(int argc, char **argv) {
  FILE *outfile = stdout;
  bool pass = true;

  printheader(outfile, "A. 5090 direct-PTX low-precision reduction-width probe");
  fprintf(outfile, "  | Probe: %-51s |\n", "D = 2^shift + K");
  fprintf(outfile, "  | Method: %-50s |\n", "scan largest shift where D changes");
  fprintf(outfile, "  | Notes: %-51s |\n", "single warp, direct mma.sync.aligned PTX");

#if defined(TCNB_REDUCTION_FP8)
  pass = print_result(outfile, kFp8E4m3E4m3,
                      probe_variant<kFp8E4m3E4m3>()) &&
         pass;
  pass = print_result(outfile, kFp8E5m2E5m2,
                      probe_variant<kFp8E5m2E5m2>()) &&
         pass;
  pass = print_result(outfile, kFp8E4m3E5m2,
                      probe_variant<kFp8E4m3E5m2>()) &&
         pass;
  pass = print_result(outfile, kFp8E5m2E4m3,
                      probe_variant<kFp8E5m2E4m3>()) &&
         pass;
#elif defined(TCNB_REDUCTION_FP6)
  pass = print_result(outfile, kFp6E2m3E2m3,
                      probe_variant<kFp6E2m3E2m3>()) &&
         pass;
  pass = print_result(outfile, kFp6E3m2E3m2,
                      probe_variant<kFp6E3m2E3m2>()) &&
         pass;
#elif defined(TCNB_REDUCTION_FP4)
  pass = print_result(outfile, kFp4E2m1E2m1,
                      probe_variant<kFp4E2m1E2m1>()) &&
         pass;
#else
#error "Define one TCNB_REDUCTION_* format macro"
#endif

  printfooter(outfile);
  return pass ? 0 : 1;
}
