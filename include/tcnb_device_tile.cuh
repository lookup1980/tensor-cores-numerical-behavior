#ifndef TCNB_DEVICE_TILE_CUH
#define TCNB_DEVICE_TILE_CUH

#include "tcnb_cuda_check.cuh"
#include "tcnb_matrix.cuh"

template <typename value_type>
inline value_type *device_tile_alloc() {
  value_type *ptr = nullptr;
  TCNB_CUDA_CHECK(cudaMalloc(&ptr, TCNB_TILE_ELEMENTS * sizeof(value_type)));
  return ptr;
}

template <typename value_type>
inline void copy_tile_to_device(value_type *dst, const value_type *src) {
  TCNB_CUDA_CHECK(cudaMemcpy(dst, src,
                             TCNB_TILE_ELEMENTS * sizeof(value_type),
                             cudaMemcpyHostToDevice));
}

template <typename value_type>
inline void copy_tile_to_host(value_type *dst, const value_type *src) {
  TCNB_CUDA_CHECK(cudaMemcpy(dst, src,
                             TCNB_TILE_ELEMENTS * sizeof(value_type),
                             cudaMemcpyDeviceToHost));
}

#endif
