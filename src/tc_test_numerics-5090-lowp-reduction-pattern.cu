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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "include/tcnb_cuda_check.cuh"
#include "include/tcnb_output.hpp"

static const int kInstructionK = 32;
static const int kABytes = 16;
static const int kBBytes = 8;
static const int kMinProductExponent = -96;
static const int kMaxProductExponent = 96;
static const int kProductExponentCount =
    kMaxProductExponent - kMinProductExponent + 1;

enum MmaVariant {
  kFp8E4m3E4m3,
  kFp8E5m2E5m2,
  kFp8E4m3E5m2,
  kFp8E5m2E4m3,
  kFp6E2m3E2m3,
  kFp6E3m2E3m2,
  kFp4E2m1E2m1,
};

enum LowpFormat {
  kE4m3,
  kE5m2,
  kE2m3,
  kE3m2,
  kE2m1,
};

struct FormatValue {
  std::uint8_t raw;
  std::uint8_t ptx_bits;
  float value;
};

struct ProductChoice {
  bool valid;
  std::uint8_t a_bits;
  std::uint8_t b_bits;
  float a_value;
  float b_value;
  float product;
  int exponent;
};

struct ProductTerm {
  int a_index;
  int b_index;
  float weight;
};

struct PatternResult {
  bool executed;
  bool mapped;
  int tested_cases;
  int survived_cases;
  int max_tested_width;
  int max_effective_width;
  int max_width_large_exp;
  int max_width_epsilon_exp;
  bool extreme_survived;
  int extreme_width;
  int extreme_large_exp;
  int extreme_epsilon_exp;
  float extreme_observed;
  float extreme_expected;
  float max_product;
  float min_product;
  float observed_at_max;
  float expected_at_max;
  ProductTerm selected[3];
};

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

static LowpFormat a_format(MmaVariant variant) {
  switch (variant) {
  case kFp8E4m3E4m3:
  case kFp8E4m3E5m2:
    return kE4m3;
  case kFp8E5m2E5m2:
  case kFp8E5m2E4m3:
    return kE5m2;
  case kFp6E2m3E2m3:
    return kE2m3;
  case kFp6E3m2E3m2:
    return kE3m2;
  case kFp4E2m1E2m1:
    return kE2m1;
  }

  return kE4m3;
}

static LowpFormat b_format(MmaVariant variant) {
  switch (variant) {
  case kFp8E4m3E4m3:
  case kFp8E5m2E4m3:
    return kE4m3;
  case kFp8E5m2E5m2:
  case kFp8E4m3E5m2:
    return kE5m2;
  case kFp6E2m3E2m3:
    return kE2m3;
  case kFp6E3m2E3m2:
    return kE3m2;
  case kFp4E2m1E2m1:
    return kE2m1;
  }

  return kE4m3;
}

static int raw_count(LowpFormat format) {
  switch (format) {
  case kE4m3:
  case kE5m2:
    return 256;
  case kE2m3:
  case kE3m2:
    return 64;
  case kE2m1:
    return 16;
  }

  return 0;
}

static float decode_raw_value(LowpFormat format, int raw) {
  switch (format) {
  case kE4m3: {
    __nv_fp8_e4m3 value;
    value.__x = static_cast<std::uint8_t>(raw);
    return static_cast<float>(value);
  }
  case kE5m2: {
    __nv_fp8_e5m2 value;
    value.__x = static_cast<std::uint8_t>(raw);
    return static_cast<float>(value);
  }
  case kE2m3: {
    __nv_fp6_e2m3 value;
    value.__x = static_cast<std::uint8_t>(raw);
    return static_cast<float>(value);
  }
  case kE3m2: {
    __nv_fp6_e3m2 value;
    value.__x = static_cast<std::uint8_t>(raw);
    return static_cast<float>(value);
  }
  case kE2m1: {
    __nv_fp4_e2m1 value;
    value.__x = static_cast<std::uint8_t>(raw);
    return static_cast<float>(value);
  }
  }

  return 0.0f;
}

static std::uint8_t encode_value(LowpFormat format, float value) {
  switch (format) {
  case kE4m3:
    return __nv_fp8_e4m3(value).__x;
  case kE5m2:
    return __nv_fp8_e5m2(value).__x;
  case kE2m3:
    return __nv_fp6_e2m3(value).__x;
  case kE3m2:
    return __nv_fp6_e3m2(value).__x;
  case kE2m1:
    return static_cast<std::uint8_t>(__nv_fp4_e2m1(value).__x << 2);
  }

  return 0;
}

static std::uint8_t ptx_bits_from_raw(LowpFormat format, int raw) {
  if (format == kE2m1) {
    return static_cast<std::uint8_t>(raw << 2);
  }
  return static_cast<std::uint8_t>(raw);
}

static int collect_positive_values(LowpFormat format, FormatValue values[256]) {
  int count = 0;
  for (int raw = 0; raw < raw_count(format); ++raw) {
    float value = decode_raw_value(format, raw);
    if (value > 0.0f && std::isfinite(value)) {
      values[count].raw = static_cast<std::uint8_t>(raw);
      values[count].ptx_bits = ptx_bits_from_raw(format, raw);
      values[count].value = value;
      ++count;
    }
  }
  return count;
}

static int product_index(int exponent) {
  return exponent - kMinProductExponent;
}

static bool product_exponent_in_range(int exponent) {
  return exponent >= kMinProductExponent && exponent <= kMaxProductExponent;
}

static void update_max_product(ProductChoice *slot,
                               const ProductChoice &candidate) {
  if (!slot->valid || candidate.product > slot->product) {
    *slot = candidate;
  }
}

static void update_min_product(ProductChoice *slot,
                               const ProductChoice &candidate) {
  if (!slot->valid || candidate.product < slot->product) {
    *slot = candidate;
  }
}

static void build_product_tables(
    LowpFormat a_fmt, LowpFormat b_fmt,
    ProductChoice large_by_exp[kProductExponentCount],
    ProductChoice epsilon_by_exp[kProductExponentCount],
    ProductChoice *max_product, ProductChoice *min_product,
    int *product_count) {
  FormatValue a_values[256] = {};
  FormatValue b_values[256] = {};
  int a_count = collect_positive_values(a_fmt, a_values);
  int b_count = collect_positive_values(b_fmt, b_values);

  *product_count = 0;
  for (int a = 0; a < a_count; ++a) {
    for (int b = 0; b < b_count; ++b) {
      float product = a_values[a].value * b_values[b].value;
      if (!(product > 0.0f) || !std::isfinite(product)) {
        continue;
      }

      int exponent = std::ilogb(product);
      if (!product_exponent_in_range(exponent)) {
        continue;
      }

      ProductChoice choice = {};
      choice.valid = true;
      choice.a_bits = a_values[a].ptx_bits;
      choice.b_bits = b_values[b].ptx_bits;
      choice.a_value = a_values[a].value;
      choice.b_value = b_values[b].value;
      choice.product = product;
      choice.exponent = exponent;

      int index = product_index(exponent);
      update_max_product(&large_by_exp[index], choice);
      update_max_product(&epsilon_by_exp[index], choice);
      update_max_product(max_product, choice);
      update_min_product(min_product, choice);
      ++*product_count;
    }
  }
}

static std::uint32_t set_pack_byte(std::uint32_t pack, int byte_index,
                                   std::uint8_t value) {
  return (pack & ~(0xffu << (byte_index * 8))) |
         (static_cast<std::uint32_t>(value) << (byte_index * 8));
}

static void clear_regs(std::uint32_t *regs, int count) {
  for (int i = 0; i < count; ++i) {
    regs[i] = 0;
  }
}

static void set_a_position(std::uint32_t a_regs[4], int position,
                           std::uint8_t value) {
  int reg = position / 4;
  int byte = position % 4;
  a_regs[reg] = set_pack_byte(a_regs[reg], byte, value);
}

static void set_b_position(std::uint32_t b_regs[2], int position,
                           std::uint8_t value) {
  int reg = position / 4;
  int byte = position % 4;
  b_regs[reg] = set_pack_byte(b_regs[reg], byte, value);
}

#define TCNB_PATTERN_MMA(instruction)                                           \
  asm volatile(instruction " {%0, %1, %2, %3}, "                                \
                           "{%4, %5, %6, %7}, {%8, %9}, "                      \
                           "{%10, %11, %12, %13};\n"                           \
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                       \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),       \
                 "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f))

template <MmaVariant variant>
__device__ __forceinline__ void direct_mma_pattern(
    std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
    std::uint32_t b0, std::uint32_t b1, float *out0, float *out1, float *out2,
    float *out3) {
  float d0;
  float d1;
  float d2;
  float d3;

  if (variant == kFp8E4m3E4m3) {
    TCNB_PATTERN_MMA(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32");
  } else if (variant == kFp8E5m2E5m2) {
    TCNB_PATTERN_MMA(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32");
  } else if (variant == kFp8E4m3E5m2) {
    TCNB_PATTERN_MMA(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32");
  } else if (variant == kFp8E5m2E4m3) {
    TCNB_PATTERN_MMA(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32");
  } else if (variant == kFp6E2m3E2m3) {
    TCNB_PATTERN_MMA(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32");
  } else if (variant == kFp6E3m2E3m2) {
    TCNB_PATTERN_MMA(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32");
  } else {
    TCNB_PATTERN_MMA(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32");
  }

  *out0 = d0;
  *out1 = d1;
  *out2 = d2;
  *out3 = d3;
}

#undef TCNB_PATTERN_MMA

template <MmaVariant variant>
__global__ void pattern_kernel(std::uint32_t a0, std::uint32_t a1,
                               std::uint32_t a2, std::uint32_t a3,
                               std::uint32_t b0, std::uint32_t b1,
                               float *out) {
  float d0;
  float d1;
  float d2;
  float d3;
  direct_mma_pattern<variant>(a0, a1, a2, a3, b0, b1, &d0, &d1, &d2, &d3);

  if ((threadIdx.x & 31) == 0) {
    out[0] = d0;
    out[1] = d1;
    out[2] = d2;
    out[3] = d3;
  }
}

template <MmaVariant variant>
static void run_pattern(const std::uint32_t a_regs[4],
                        const std::uint32_t b_regs[2], float *device_out,
                        float host_out[4]) {
  pattern_kernel<variant><<<1, 32>>>(a_regs[0], a_regs[1], a_regs[2],
                                     a_regs[3], b_regs[0], b_regs[1],
                                     device_out);
  TCNB_CUDA_CHECK(cudaGetLastError());
  TCNB_CUDA_CHECK(cudaMemcpy(host_out, device_out, 4 * sizeof(float),
                             cudaMemcpyDeviceToHost));
}

template <MmaVariant variant>
static void map_pair_weights(float *device_out,
                             float weights[kABytes][kBBytes]) {
  std::uint8_t a_one = encode_value(a_format(variant), 1.0f);
  std::uint8_t b_one = encode_value(b_format(variant), 1.0f);

  for (int a = 0; a < kABytes; ++a) {
    for (int b = 0; b < kBBytes; ++b) {
      std::uint32_t a_regs[4] = {};
      std::uint32_t b_regs[2] = {};
      float host_out[4] = {};
      clear_regs(a_regs, 4);
      clear_regs(b_regs, 2);
      set_a_position(a_regs, a, a_one);
      set_b_position(b_regs, b, b_one);
      run_pattern<variant>(a_regs, b_regs, device_out, host_out);
      weights[a][b] = host_out[0];
    }
  }
}

static bool terms_are_independent(const ProductTerm selected[3],
                                  float weights[kABytes][kBBytes]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        continue;
      }
      float cross = weights[selected[i].a_index][selected[j].b_index];
      if (std::fabs(cross) > 0.125f) {
        return false;
      }
    }
  }
  return true;
}

static bool choose_independent_triplet(float weights[kABytes][kBBytes],
                                       ProductTerm selected[3]) {
  ProductTerm candidates[kABytes * kBBytes] = {};
  int count = 0;
  for (int a = 0; a < kABytes; ++a) {
    for (int b = 0; b < kBBytes; ++b) {
      if (weights[a][b] > 0.0f) {
        candidates[count].a_index = a;
        candidates[count].b_index = b;
        candidates[count].weight = weights[a][b];
        ++count;
      }
    }
  }

  for (int i = 0; i < count; ++i) {
    for (int j = i + 1; j < count; ++j) {
      if (std::fabs(candidates[i].weight - candidates[j].weight) > 0.125f) {
        continue;
      }
      for (int k = j + 1; k < count; ++k) {
        if (std::fabs(candidates[i].weight - candidates[k].weight) > 0.125f) {
          continue;
        }

        selected[0] = candidates[i];
        selected[1] = candidates[j];
        selected[2] = candidates[k];
        if (terms_are_independent(selected, weights)) {
          return true;
        }
      }
    }
  }

  return false;
}

template <MmaVariant variant>
static bool run_triplet(const ProductTerm selected[3],
                        const ProductChoice &large,
                        const ProductChoice &epsilon, float *device_out,
                        float *observed, float *expected) {
  std::uint32_t a_regs[4] = {};
  std::uint32_t b_regs[2] = {};
  float host_out[4] = {};
  LowpFormat a_fmt = a_format(variant);

  clear_regs(a_regs, 4);
  clear_regs(b_regs, 2);
  set_a_position(a_regs, selected[0].a_index, large.a_bits);
  set_b_position(b_regs, selected[0].b_index, large.b_bits);
  set_a_position(a_regs, selected[1].a_index,
                 encode_value(a_fmt, -large.a_value));
  set_b_position(b_regs, selected[1].b_index, large.b_bits);
  set_a_position(a_regs, selected[2].a_index, epsilon.a_bits);
  set_b_position(b_regs, selected[2].b_index, epsilon.b_bits);
  run_pattern<variant>(a_regs, b_regs, device_out, host_out);

  *observed = host_out[0];
  *expected = selected[2].weight * epsilon.product;
  float error = std::fabs(*observed - *expected);
  float tolerance = std::fabs(*expected) * 0.5f;
  return *observed > 0.0f && error <= tolerance;
}

template <MmaVariant variant>
static PatternResult probe_variant() {
  PatternResult result = {};
  result.executed = false;
  result.mapped = false;
  result.tested_cases = 0;
  result.survived_cases = 0;
  result.max_tested_width = -1;
  result.max_effective_width = -1;
  result.max_width_large_exp = 0;
  result.max_width_epsilon_exp = 0;
  result.extreme_survived = false;
  result.extreme_width = -1;
  result.extreme_large_exp = 0;
  result.extreme_epsilon_exp = 0;
  result.extreme_observed = 0.0f;
  result.extreme_expected = 0.0f;
  result.max_product = 0.0f;
  result.min_product = 0.0f;
  result.observed_at_max = 0.0f;
  result.expected_at_max = 0.0f;

  float *device_out = nullptr;
  TCNB_CUDA_CHECK(cudaMalloc(&device_out, 4 * sizeof(float)));

  float weights[kABytes][kBBytes] = {};
  map_pair_weights<variant>(device_out, weights);
  result.mapped = choose_independent_triplet(weights, result.selected);
  if (!result.mapped) {
    TCNB_CUDA_CHECK(cudaFree(device_out));
    return result;
  }

  ProductChoice large_by_exp[kProductExponentCount] = {};
  ProductChoice epsilon_by_exp[kProductExponentCount] = {};
  ProductChoice max_product = {};
  ProductChoice min_product = {};
  int product_count = 0;
  build_product_tables(a_format(variant), b_format(variant), large_by_exp,
                       epsilon_by_exp, &max_product, &min_product,
                       &product_count);
  if (!max_product.valid || !min_product.valid || product_count == 0) {
    TCNB_CUDA_CHECK(cudaFree(device_out));
    return result;
  }

  result.max_product = max_product.product;
  result.min_product = min_product.product;
  result.extreme_large_exp = max_product.exponent;
  result.extreme_epsilon_exp = min_product.exponent;
  result.extreme_width =
      result.extreme_large_exp - result.extreme_epsilon_exp + 1;
  result.extreme_survived =
      run_triplet<variant>(result.selected, max_product, min_product,
                           device_out, &result.extreme_observed,
                           &result.extreme_expected);

  for (int large_index = 0; large_index < kProductExponentCount;
       ++large_index) {
    if (!large_by_exp[large_index].valid) {
      continue;
    }

    int large_exp = large_index + kMinProductExponent;
    for (int epsilon_index = 0; epsilon_index <= large_index;
         ++epsilon_index) {
      if (!epsilon_by_exp[epsilon_index].valid) {
        continue;
      }

      int epsilon_exp = epsilon_index + kMinProductExponent;
      int width = large_exp - epsilon_exp + 1;
      float observed = 0.0f;
      float expected = 0.0f;
      bool survived =
          run_triplet<variant>(result.selected, large_by_exp[large_index],
                               epsilon_by_exp[epsilon_index], device_out,
                               &observed, &expected);

      result.executed = true;
      ++result.tested_cases;
      result.max_tested_width = std::max(result.max_tested_width, width);
      if (survived) {
        ++result.survived_cases;
        if (width > result.max_effective_width) {
          result.max_effective_width = width;
          result.max_width_large_exp = large_exp;
          result.max_width_epsilon_exp = epsilon_exp;
          result.observed_at_max = observed;
          result.expected_at_max = expected;
        }
      }
    }
  }

  TCNB_CUDA_CHECK(cudaFree(device_out));
  return result;
}

static void print_term(FILE *outfile, const char *label,
                       const ProductTerm &term) {
  fprintf(outfile, "  |   %-13s A%d.byte%d * B%d.byte%d, weight %-12.9g |\n",
          label, term.a_index / 4, term.a_index % 4, term.b_index / 4,
          term.b_index % 4, term.weight);
}

template <MmaVariant variant>
static bool print_variant(FILE *outfile) {
  PatternResult result = probe_variant<variant>();
  bool format_limited =
      result.max_effective_width == result.max_tested_width;

  fprintf(outfile, "  | %-58s |\n", variant_label(variant));
  fprintf(outfile, "  |   PTX: %-51s |\n", variant_instruction(variant));
  fprintf(outfile, "  |   K per instruction: %-35d |\n", kInstructionK);
  fprintf(outfile, "  |   Pattern: %-47s |\n",
          "A*B product M - M + epsilon");

  if (result.mapped) {
    print_term(outfile, "+M", result.selected[0]);
    print_term(outfile, "-M", result.selected[1]);
    print_term(outfile, "epsilon", result.selected[2]);
  } else {
    fprintf(outfile, "  |   mapping status: %-38s |\n",
            "failed to find independent product terms");
  }

  fprintf(outfile, "  |   tested product-exponent pairs: %-18d |\n",
          result.tested_cases);
  fprintf(outfile, "  |   survived pairs: %-35d |\n",
          result.survived_cases);
  fprintf(outfile, "  |   max product: %-38.9g |\n", result.max_product);
  fprintf(outfile, "  |   min product: %-38.9g |\n", result.min_product);
  fprintf(outfile, "  |   extreme product width: %-25d |\n",
          result.extreme_width);
  fprintf(outfile, "  |   extreme survived: %-30s |\n",
          result.extreme_survived ? "yes" : "no");
  fprintf(outfile, "  |   extreme observed: %-30.9g |\n",
          result.extreme_observed);
  fprintf(outfile, "  |   extreme expected: %-30.9g |\n",
          result.extreme_expected);
  fprintf(outfile, "  |   max tested effective width: %-22d |\n",
          result.max_tested_width);
  fprintf(outfile, "  |   max effective width: %-29d |\n",
          result.max_effective_width);
  fprintf(outfile, "  |   width is format-limited: %-24s |\n",
          format_limited ? "yes" : "no");
  fprintf(outfile, "  |   best product M exponent: %-25d |\n",
          result.max_width_large_exp);
  fprintf(outfile, "  |   best product eps exponent: %-23d |\n",
          result.max_width_epsilon_exp);
  fprintf(outfile, "  |   observed at max width: %-27.9g |\n",
          result.observed_at_max);
  fprintf(outfile, "  |   expected at max width: %-27.9g |\n",
          result.expected_at_max);

  bool passed = result.executed && result.mapped && result.tested_cases > 0 &&
                result.max_effective_width > 0;
  printitem(outfile, "*) Direct PTX product-pattern probe executed");
  printpass(outfile, passed);
  return passed;
}

int main(int argc, char **argv) {
  FILE *outfile = stdout;
  bool pass = true;

  (void)argc;
  (void)argv;

  printheader(outfile,
              "A. 5090 direct-PTX low-precision product-pattern probe");
  fprintf(outfile, "  | Probe: %-51s |\n", "A*B M - M + epsilon");
  fprintf(outfile, "  | Method: %-50s |\n", "independent product terms");
  fprintf(outfile, "  | Notes: %-51s |\n", "A and B both provide range");

#if defined(TCNB_PATTERN_FP8)
  pass = print_variant<kFp8E4m3E4m3>(outfile) && pass;
  pass = print_variant<kFp8E5m2E5m2>(outfile) && pass;
  pass = print_variant<kFp8E4m3E5m2>(outfile) && pass;
  pass = print_variant<kFp8E5m2E4m3>(outfile) && pass;
#elif defined(TCNB_PATTERN_FP6)
  pass = print_variant<kFp6E2m3E2m3>(outfile) && pass;
  pass = print_variant<kFp6E3m2E3m2>(outfile) && pass;
#elif defined(TCNB_PATTERN_FP4)
  pass = print_variant<kFp4E2m1E2m1>(outfile) && pass;
#else
#error "Define TCNB_PATTERN_FP8, TCNB_PATTERN_FP6, or TCNB_PATTERN_FP4"
#endif

  printfooter(outfile);
  return pass ? 0 : 1;
}
