#ifndef TCNB_MATRIX_CUH
#define TCNB_MATRIX_CUH

#include <cstddef>
#include <cstring>

static constexpr int TCNB_TILE_DIM = 16;
static constexpr std::size_t TCNB_TILE_ELEMENTS =
    TCNB_TILE_DIM * TCNB_TILE_DIM;

template <typename inputtype, typename returntype>
inline void host_reset(inputtype *a, inputtype *b, returntype *c) {
  memset(a, 0, TCNB_TILE_ELEMENTS * sizeof(inputtype));
  memset(b, 0, TCNB_TILE_ELEMENTS * sizeof(inputtype));
  memset(c, 0, TCNB_TILE_ELEMENTS * sizeof(returntype));
}

#endif
