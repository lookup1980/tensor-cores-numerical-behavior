#ifndef TCNB_CUDA_CHECK_CUH
#define TCNB_CUDA_CHECK_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void tcnb_check_cuda(cudaError_t code, const char *expr,
                            const char *file, int line) {
  if (code == cudaSuccess)
    return;

  fprintf(stderr, "CUDA error: %s failed with %s at %s:%d\n",
          expr, cudaGetErrorString(code), file, line);
  std::exit(static_cast<int>(code));
}

#define TCNB_CUDA_CHECK(expr) \
  tcnb_check_cuda((expr), #expr, __FILE__, __LINE__)

#endif
