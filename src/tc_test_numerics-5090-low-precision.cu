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
#include <cstdio>

#include "include/tcnb_cuda_check.cuh"
#include "include/tcnb_output.hpp"

template <typename scalartype, typename vector4type>
__global__ void conversion_probe_kernel(float *out) {
  scalartype one(1.0f);
  scalartype minusone(-1.0f);
  scalartype half(0.5f);
  scalartype two(2.0f);

  out[0] = static_cast<float>(one);
  out[1] = static_cast<float>(minusone);
  out[2] = static_cast<float>(half);
  out[3] = static_cast<float>(two);

  float4 input = make_float4(1.0f, -1.0f, 0.5f, 2.0f);
  vector4type packed(input);
  float4 unpacked = static_cast<float4>(packed);

  out[4] = unpacked.x;
  out[5] = unpacked.y;
  out[6] = unpacked.z;
  out[7] = unpacked.w;
}

static bool check_probe_values(const float *values) {
  return values[0] == 1.0f && values[1] == -1.0f &&
         values[2] == 0.5f && values[3] == 2.0f &&
         values[4] == 1.0f && values[5] == -1.0f &&
         values[6] == 0.5f && values[7] == 2.0f;
}

template <typename scalartype, typename vector4type>
static bool host_conversion_probe() {
  scalartype one(1.0f);
  scalartype minusone(-1.0f);
  scalartype half(0.5f);
  scalartype two(2.0f);

  float4 input = make_float4(1.0f, -1.0f, 0.5f, 2.0f);
  vector4type packed(input);
  float4 unpacked = static_cast<float4>(packed);

  float values[8] = {
      static_cast<float>(one),
      static_cast<float>(minusone),
      static_cast<float>(half),
      static_cast<float>(two),
      unpacked.x,
      unpacked.y,
      unpacked.z,
      unpacked.w,
  };

  return check_probe_values(values);
}

template <typename scalartype, typename vector4type>
static bool device_conversion_probe() {
  float host_values[8] = {};
  float *device_values = nullptr;

  TCNB_CUDA_CHECK(cudaMalloc(&device_values, sizeof(host_values)));
  conversion_probe_kernel<scalartype, vector4type><<<1, 1>>>(device_values);
  TCNB_CUDA_CHECK(cudaGetLastError());
  TCNB_CUDA_CHECK(cudaMemcpy(host_values, device_values, sizeof(host_values),
                             cudaMemcpyDeviceToHost));
  TCNB_CUDA_CHECK(cudaFree(device_values));

  return check_probe_values(host_values);
}

template <typename scalartype, typename vector4type>
static bool run_format_case(FILE *outfile, const char *label) {
  char item[96];
  std::snprintf(item, sizeof(item), "*) %s host scalar/vector conversions",
                label);
  printitem(outfile, item);
  bool host_pass = host_conversion_probe<scalartype, vector4type>();
  printpass(outfile, host_pass);

  std::snprintf(item, sizeof(item), "*) %s device scalar/vector conversions",
                label);
  printitem(outfile, item);
  bool device_pass = device_conversion_probe<scalartype, vector4type>();
  printpass(outfile, device_pass);

  return host_pass && device_pass;
}

int main(int argc, char **argv) {
  cudaDeviceProp prop;
  TCNB_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  FILE *outfile = stdout;
  bool pass = true;

  printheader(outfile, "A. 5090 low-precision format support");

  printitem(outfile, "*) Device compute capability is at least 12.0");
  bool sm120_or_newer = (prop.major > 12) || (prop.major == 12 && prop.minor >= 0);
  printpass(outfile, sm120_or_newer);
  pass = pass && sm120_or_newer;

#if defined(TCNB_LOW_PRECISION_FP8)
  pass = run_format_case<__nv_fp8_e4m3, __nv_fp8x4_e4m3>(
             outfile, "FP8 E4M3") && pass;
  pass = run_format_case<__nv_fp8_e5m2, __nv_fp8x4_e5m2>(
             outfile, "FP8 E5M2") && pass;
#elif defined(TCNB_LOW_PRECISION_FP6)
  pass = run_format_case<__nv_fp6_e2m3, __nv_fp6x4_e2m3>(
             outfile, "FP6 E2M3") && pass;
  pass = run_format_case<__nv_fp6_e3m2, __nv_fp6x4_e3m2>(
             outfile, "FP6 E3M2") && pass;
#elif defined(TCNB_LOW_PRECISION_FP4)
  pass = run_format_case<__nv_fp4_e2m1, __nv_fp4x4_e2m1>(
             outfile, "FP4 E2M1") && pass;
#else
#error "Define one TCNB_LOW_PRECISION_* format macro"
#endif

  printfooter(outfile);

  return pass ? 0 : 1;
}
