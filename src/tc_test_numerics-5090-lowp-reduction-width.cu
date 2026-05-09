/*
 * Copyright (c) 2026
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, version 2.
 */

#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "include/tcnb_cuda_check.cuh"
#include "include/tcnb_output.hpp"

#define TCNB_CUBLAS_CHECK(expr) tcnb_check_cublas((expr), #expr, __FILE__, __LINE__)

static const int kM = 16;
static const int kN = 16;
static const int kK = 1024;
static const int kMinShift = 20;
static const int kMaxShift = 48;

static const char *cublas_status_name(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "CUBLAS_STATUS_UNKNOWN";
  }
}

static void tcnb_check_cublas(cublasStatus_t status, const char *expr,
                              const char *file, int line) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return;

  fprintf(stderr, "cuBLASLt error: %s failed with %s at %s:%d\n", expr,
          cublas_status_name(status), file, line);
  std::exit(static_cast<int>(status));
}

template <typename value_type> struct DeviceBuffer {
  value_type *ptr;

  DeviceBuffer() : ptr(nullptr) {}
  ~DeviceBuffer() {
    if (ptr != nullptr)
      cudaFree(ptr);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void allocate(size_t count) {
    TCNB_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(value_type)));
  }

  void copy_from_host(const std::vector<value_type> &host) {
    allocate(host.size());
    TCNB_CUDA_CHECK(cudaMemcpy(ptr, host.data(),
                               host.size() * sizeof(value_type),
                               cudaMemcpyHostToDevice));
  }
};

struct LtHandle {
  cublasLtHandle_t value;

  LtHandle() : value(nullptr) {}
  ~LtHandle() {
    if (value != nullptr)
      cublasLtDestroy(value);
  }

  LtHandle(const LtHandle &) = delete;
  LtHandle &operator=(const LtHandle &) = delete;
};

struct MatmulDesc {
  cublasLtMatmulDesc_t value;

  MatmulDesc() : value(nullptr) {}
  ~MatmulDesc() {
    if (value != nullptr)
      cublasLtMatmulDescDestroy(value);
  }

  MatmulDesc(const MatmulDesc &) = delete;
  MatmulDesc &operator=(const MatmulDesc &) = delete;
};

struct MatrixLayout {
  cublasLtMatrixLayout_t value;

  MatrixLayout() : value(nullptr) {}
  ~MatrixLayout() {
    if (value != nullptr)
      cublasLtMatrixLayoutDestroy(value);
  }

  MatrixLayout(const MatrixLayout &) = delete;
  MatrixLayout &operator=(const MatrixLayout &) = delete;
};

struct MatmulPreference {
  cublasLtMatmulPreference_t value;

  MatmulPreference() : value(nullptr) {}
  ~MatmulPreference() {
    if (value != nullptr)
      cublasLtMatmulPreferenceDestroy(value);
  }

  MatmulPreference(const MatmulPreference &) = delete;
  MatmulPreference &operator=(const MatmulPreference &) = delete;
};

static bool output_changed(float value) {
  return value > 1.0f;
}

struct WidthResult {
  bool supported;
  bool tensor_op;
  cublasStatus_t status;
  int max_changed_shift;
  int exact_tree_expected_shift;
  uint64_t numerical_flags;
  float sample_value;
};

template <typename value_type>
static cublasStatus_t run_once(cublasLtHandle_t handle, cudaDataType_t data_type,
                               bool fast_accum, int shift, float *device_d,
                               float *device_b_scale, float *host_sample,
                               cublasLtNumericalImplFlags_t *flags_out) {
  MatmulDesc op_desc;
  MatrixLayout a_desc;
  MatrixLayout b_desc;
  MatrixLayout c_desc;
  MatrixLayout d_desc;
  MatmulPreference pref;

  cublasStatus_t status =
      cublasLtMatmulDescCreate(&op_desc.value, CUBLAS_COMPUTE_32F,
                               CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  int8_t fast_accum_value = fast_accum ? 1 : 0;
  status = cublasLtMatmulDescSetAttribute(
      op_desc.value, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum_value,
      sizeof(fast_accum_value));
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  status = cublasLtMatmulDescSetAttribute(
      op_desc.value, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &device_b_scale,
      sizeof(device_b_scale));
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  status = cublasLtMatrixLayoutCreate(&a_desc.value, data_type, kM, kK, kM);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  status = cublasLtMatrixLayoutCreate(&b_desc.value, data_type, kK, kN, kK);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  status = cublasLtMatrixLayoutCreate(&c_desc.value, CUDA_R_32F, kM, kN, kM);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  status = cublasLtMatrixLayoutCreate(&d_desc.value, CUDA_R_32F, kM, kN, kM);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  status = cublasLtMatmulPreferenceCreate(&pref.value);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  size_t max_workspace = 64 * 1024 * 1024;
  status = cublasLtMatmulPreferenceSetAttribute(
      pref.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
      sizeof(max_workspace));
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  std::vector<value_type> host_a(kM * kK, value_type(1.0f));
  std::vector<value_type> host_b(kK * kN, value_type(1.0f));
  std::vector<float> host_c(kM * kN, 1.0f);
  DeviceBuffer<value_type> device_a;
  DeviceBuffer<value_type> device_b;
  DeviceBuffer<float> device_c;
  device_a.copy_from_host(host_a);
  device_b.copy_from_host(host_b);
  device_c.copy_from_host(host_c);

  float scale = std::ldexp(1.0f, -shift);
  TCNB_CUDA_CHECK(cudaMemcpy(device_b_scale, &scale, sizeof(scale),
                             cudaMemcpyHostToDevice));

  cublasLtMatmulHeuristicResult_t heuristic;
  int returned_results = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      handle, op_desc.value, a_desc.value, b_desc.value, c_desc.value,
      d_desc.value, pref.value, 1, &heuristic, &returned_results);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  if (returned_results == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
    return returned_results == 0 ? CUBLAS_STATUS_NOT_SUPPORTED
                                 : heuristic.state;
  }

  DeviceBuffer<unsigned char> workspace;
  if (heuristic.workspaceSize > 0)
    workspace.allocate(heuristic.workspaceSize);

  uint64_t numerical_flags = 0;
  size_t written = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(
          &heuristic.algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
          &numerical_flags, sizeof(numerical_flags), &written) ==
      CUBLAS_STATUS_SUCCESS) {
    *flags_out = numerical_flags;
  }

  const float alpha = 1.0f;
  const float beta = 1.0f;
  status = cublasLtMatmul(handle, op_desc.value, &alpha, device_a.ptr,
                          a_desc.value, device_b.ptr, b_desc.value, &beta,
                          device_c.ptr, c_desc.value, device_d, d_desc.value,
                          &heuristic.algo, workspace.ptr, heuristic.workspaceSize,
                          0);
  if (status == CUBLAS_STATUS_SUCCESS) {
    TCNB_CUDA_CHECK(cudaMemcpy(host_sample, device_d, sizeof(float),
                               cudaMemcpyDeviceToHost));
  }

  return status;
}

template <typename value_type>
static WidthResult probe_width(cudaDataType_t data_type, bool fast_accum) {
  WidthResult result;
  result.supported = false;
  result.tensor_op = false;
  result.status = CUBLAS_STATUS_SUCCESS;
  result.max_changed_shift = -1;
  result.exact_tree_expected_shift = 33;
  result.numerical_flags = 0;
  result.sample_value = 0.0f;

  LtHandle handle;
  TCNB_CUBLAS_CHECK(cublasLtCreate(&handle.value));

  DeviceBuffer<float> device_d;
  DeviceBuffer<float> device_b_scale;
  device_d.allocate(kM * kN);
  device_b_scale.allocate(1);

  for (int shift = kMinShift; shift <= kMaxShift; ++shift) {
    float sample = 0.0f;
    cublasLtNumericalImplFlags_t flags = 0;
    cublasStatus_t status = run_once<value_type>(
        handle.value, data_type, fast_accum, shift, device_d.ptr,
        device_b_scale.ptr,
        &sample, &flags);

    if (status != CUBLAS_STATUS_SUCCESS) {
      result.status = status;
      break;
    }

    result.supported = true;
    result.numerical_flags = flags;
    result.tensor_op =
        (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK) != 0;
    result.sample_value = sample;
    if (output_changed(sample))
      result.max_changed_shift = shift;
  }

  return result;
}

static void print_width_result(FILE *outfile, const char *label,
                               bool fast_accum, const WidthResult &result) {
  fprintf(outfile, "  | %-58s |\n", label);
  fprintf(outfile, "  |   fast_accum: %-43s |\n",
          fast_accum ? "enabled" : "disabled");
  if (!result.supported) {
    fprintf(outfile, "  |   cuBLASLt status: %-37s |\n",
            cublas_status_name(result.status));
    fprintf(outfile, "  |   reduction-width result: %-27s |\n",
            "unsupported");
    return;
  }

  fprintf(outfile, "  |   tensor-op algorithm: %-33s |\n",
          result.tensor_op ? "yes" : "no");
  fprintf(outfile, "  |   numerical flags: 0x%-34llx |\n",
          static_cast<unsigned long long>(result.numerical_flags));
  fprintf(outfile, "  |   max shift with D > 1: %-29d |\n",
          result.max_changed_shift);
  fprintf(outfile, "  |   exact-tree expected max shift: %-18d |\n",
          result.exact_tree_expected_shift);
  fprintf(outfile, "  |   last sample value: %-31.9g |\n",
          result.sample_value);
}

template <typename value_type>
static void run_case(FILE *outfile, const char *label, cudaDataType_t data_type) {
  print_width_result(outfile, label, false,
                     probe_width<value_type>(data_type, false));
  print_width_result(outfile, label, true,
                     probe_width<value_type>(data_type, true));
}

int main(int argc, char **argv) {
  FILE *outfile = stdout;

  printheader(outfile, "A. 5090 low-precision reduction-width probe");
  fprintf(outfile, "  | M,N,K: %-48s |\n", "16,16,1024");
  fprintf(outfile, "  | Probe: %-51s |\n", "D = 1 + K * 2^-shift");
  fprintf(outfile, "  | Exact-tree FP32 output changes through shift 33.       |\n");

#if defined(TCNB_REDUCTION_FP8)
  run_case<__nv_fp8_e4m3>(outfile, "FP8 E4M3", CUDA_R_8F_E4M3);
  run_case<__nv_fp8_e5m2>(outfile, "FP8 E5M2", CUDA_R_8F_E5M2);
#elif defined(TCNB_REDUCTION_FP6)
  run_case<__nv_fp6_e2m3>(outfile, "FP6 E2M3", CUDA_R_6F_E2M3);
  run_case<__nv_fp6_e3m2>(outfile, "FP6 E3M2", CUDA_R_6F_E3M2);
#elif defined(TCNB_REDUCTION_FP4)
  run_case<__nv_fp4_e2m1>(outfile, "FP4 E2M1", CUDA_R_4F_E2M1);
#else
#error "Define one TCNB_REDUCTION_* format macro"
#endif

  printfooter(outfile);

  return 0;
}
