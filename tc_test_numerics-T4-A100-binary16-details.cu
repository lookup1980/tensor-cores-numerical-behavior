/*
 * Copyright (c) 2020, Massimiliano Fasi and Mantas Mikaitis
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 *  You should have received a copy of the GNU General Public License along with
 *  this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>
#include <unistd.h>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <mma.h>
#include <iomanip>

using namespace nvcuda;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*******************
 * Debug functions *
 *******************/
/* Print the elements of the m x n matrix A. The elements are assumed to be
   stored by columns if `bycols` is `true` and by rows if `bycols` is false. */
template <typename floattype>
void print_matrix (half *a,
                   size_t m, size_t n,
                   bool bycols) {
  int i, j;
  if (bycols) {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++)
        std::cout << __half2float(a[j*n+i]) << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  } else {
    for (i=0; i<m; i++ ) {
      for (j=0; j<n; j++)
        std::cout << __half2float(a[i*m+j]) << " ";
      std::cout  << std::endl;
    }
    std::cout << std::endl;
   }
}


/****************************************************
 * Memory management and wmma::mma_sync() interface *
 ****************************************************/

/* Set the entries of host arrays to zero. */
template <typename returntype>
void host_reset(half *a, half *b, returntype *c) {
  memset(a, 0, 16*16*sizeof(half));
  memset(b, 0, 16*16*sizeof(half));
  memset(c, 0, 16*16*sizeof(returntype));
}

/* Compute C += A*B, where A, B, and C are 16x16x16 matrices.
   The matrix C is initialized to 0 when `init` is true. */
template <typename returntype>
__global__ void wmma_ker(half *a, half *b, returntype *c, bool init) {

  // Declare fragments.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, returntype> c_fragment;

  // Load input matrices and initialize output (if required).
  wmma::load_matrix_sync(a_fragment, a, 16);
  wmma::load_matrix_sync(b_fragment, b, 16);
  if (init)
    wmma::fill_fragment(c_fragment, 0.0f);
  else
    wmma::load_matrix_sync(c_fragment, c, 16, wmma::mem_col_major);

  // Multiply
  wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

  // Store the output
  wmma::store_matrix_sync(c, c_fragment, 16, wmma::mem_col_major);
}

/* Copy data from host to device, perform the operation, and copy result back to
   host. */
template <typename returntype>
void wmma_init_run (half *h_a, half *h_b, returntype *h_c,
                    half *d_a, half *d_b, returntype *d_c,
                    bool init) {

  // Copy input from host to device.
  cudaMemcpy(d_a, h_a, 16*16*sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, 16*16*sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, 16*16*sizeof(returntype), cudaMemcpyHostToDevice);

  // Perform matrix multiplication.
  wmma_ker<<<1,32>>>(d_a, d_b, d_c, init);

  // Copy result from device to host.
  cudaMemcpy(h_c, d_c, 16*16*sizeof(returntype), cudaMemcpyDeviceToHost);
}


// mgu
void printhalf(half h) {
  uint16_t *ph = (uint16_t*) &h;
  printf("0x%04hx ", *ph);
}
void printfloat(float h) {
  uint32_t *ph = (uint32_t*) &h;
  printf("0x%08x ", *ph);
}

void float_to_half(float *fa, float *fb, half *ha, half *hb) {
  for (size_t i = 0; i < 4; i++)
  {
    ha[i] = __float2half(fa[i]);
    hb[i] = __float2half(fb[i]);
  }
}

float get_float_reference(float *fa, float *fb, float c) {
  float cc = 0.0f;
  for (size_t i = 0; i < 4; i++)
  {
    cc += fa[i] * fb[i];
  }
  return cc + c;
}

void print_result(float *fa, float *fb, half *ha, half *hb, float c, float result) {
  printf("\nh_a: ");
  for (size_t i = 0; i < 4; i++)
  {
    printf("%04hx ", *(uint16_t*)&ha[i]);
  }
  printf("\nh_b: ");
  for (size_t i = 0; i < 4; i++)
  {
    printf("%04hx ", *(uint16_t*)&hb[i]);
  }
  printf("\nf_c: ");
  printf("%08x ", *(uint32_t*)&c);
  printf("\nr  : ");
  printf("%08x ", *(uint32_t*)&result);

  float ref = get_float_reference(fa, fb, c);
  printf("\nref: ");
  printf("%08x ", *(uint32_t*)&ref);

  printf("\n");
}

void reset_all(float *fa, float *fb, half *ha, half *hb, float *fc) {
  memset(fa, 0, 16*16*sizeof(float));
  memset(fb, 0, 16*16*sizeof(float));
  memset(ha, 0, 16*16*sizeof(half));
  memset(hb, 0, 16*16*sizeof(half));
  memset(fc, 0, 16*16*sizeof(float));
}

void my_test_addr() {

  // Declare pointers and allocate memory.
  half *h_a, *h_b, *h16_c, *d16_a, *d16_b, *d16_c;
  float *d_c, *h_c;

  h_a = new half[16*16];
  h_b = new half[16*16];
  h_c = new float[16*16];
  h16_c = new half[16*16];

  cudaMalloc(&d16_a, 16*16*sizeof(half));
  cudaMalloc(&d16_b, 16*16*sizeof(half));
  cudaMalloc(&d16_c, 16*16*sizeof(half));
  cudaMalloc(&d_c, 16*16*sizeof(float));

  // mgu
  float fa[16*16] = {};
  float fb[16*16] = {};
  float temp = 0;

  printf("\n");
  printf("---------------------------------------------------\n");
  printf("Tests for addr of DP\n");

  // case
  printf("\n");
  printf("case: 2^-23 + 1");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = ldexp(1.f, -23);
  fb[0] = 1.0f;
  fa[3] = 1.0f;
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^-24 + 1");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = ldexp(1.f, -24);
  fb[0] = 1.0f;
  fa[3] = 1.0f;
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];

  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^-12 * 2^-12 + 1");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = ldexp(1.f, -12);
  fb[0] = ldexp(1.f, -12);
  fa[3] = 1.0f;
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];

  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^-23 - 1 + 1");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = ldexp(1.f, -23);
  fb[0] = 1.0f;
  fa[1] = -1.0f;
  fb[1] = 1.0f;
  fa[3] = 1.0f;
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^-24 - 1 + 1");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = ldexp(1.f, -24);
  fb[0] = 1.0f;
  fa[1] = -1.0f;
  fb[1] = 1.0f;
  fa[3] = 1.0f;
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // // case
  // printf("\n");
  // printf("case: 2^-24 - 2 + 2");
  // reset_all(fa, fb, h_a, h_b, h_c);
  // fa[0] = ldexp(1.f, -24);
  // fb[0] = 1.0f;
  // fa[2] = -1.0f;
  // fb[2] = 2.0f;
  // fa[3] = 1.0f;
  // fb[3] = 2.0f;
  // h_c[0] = 0.0f;
  // temp = h_c[0];
  // float_to_half(fa, fb, h_a, h_b);
  // wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  // print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // // case
  // reset_all(fa, fb, h_a, h_b, h_c);
  // fa[0] = ldexp(1.f, -24);
  // fb[0] = 1.0f;
  // fa[1] = -1.0f;
  // fb[1] = 2.0f;
  // fa[3] = 1.0f;
  // fb[3] = 2.0f;
  // h_c[0] = 0.0f;
  // temp = h_c[0];
  // float_to_half(fa, fb, h_a, h_b);
  // wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  // print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2 - 2 + 2^-24");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = 2.0f;
  fa[2] = -1.0f;
  fb[2] = 2.0f;
  fa[3] = ldexp(1.f, -24);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // // case
  // printf("\n");
  // printf("case: 2^-24 - 4 + 4");
  // reset_all(fa, fb, h_a, h_b, h_c);
  // fa[0] = ldexp(1.f, -24);
  // fb[0] = 1.0f;
  // fa[2] = -1.0f;
  // fb[2] = 4.0f;
  // fa[3] = 1.0f;
  // fb[3] = 4.0f;
  // h_c[0] = 0.0f;
  // temp = h_c[0];
  // float_to_half(fa, fb, h_a, h_b);
  // wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  // print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // // case
  // reset_all(fa, fb, h_a, h_b, h_c);
  // fa[0] = ldexp(1.f, -24);
  // fb[0] = 1.0f;
  // fa[1] = -1.0f;
  // fb[1] = 4.0f;
  // fa[3] = 1.0f;
  // fb[3] = 4.0f;
  // h_c[0] = 0.0f;
  // temp = h_c[0];
  // float_to_half(fa, fb, h_a, h_b);
  // wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  // print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^4 - 2^4 + 2^-24");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = ldexp(1.f, 4);
  fa[2] = -1.0f;
  fb[2] = ldexp(1.f, 4);
  fa[3] = ldexp(1.f, -24);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^4 - 2^4 + 2^-23");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = ldexp(1.f, 4);
  fa[2] = -1.0f;
  fb[2] = ldexp(1.f, 4);
  fa[3] = ldexp(1.f, -23);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^4 - 2^4 + 2^-22");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = ldexp(1.f, 4);
  fa[2] = -1.0f;
  fb[2] = ldexp(1.f, 4);
  fa[3] = ldexp(1.f, -22);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^4 - 2^4 + 2^-21");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = ldexp(1.f, 4);
  fa[2] = -1.0f;
  fb[2] = ldexp(1.f, 4);
  fa[3] = ldexp(1.f, -21);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^4 - 2^4 + 2^-20");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = ldexp(1.f, 4);
  fa[2] = -1.0f;
  fb[2] = ldexp(1.f, 4);
  fa[3] = ldexp(1.f, -20);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^16 - 2^16 + 2^-24");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = ldexp(1.f, 8);
  fb[0] = ldexp(1.f, 8);
  fa[2] = -ldexp(1.f, 8);
  fb[2] = ldexp(1.f, 8);
  fa[3] = ldexp(1.f, -24);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^24 - 2^24 + 2^-24");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = ldexp(1.f, 12);
  fb[0] = ldexp(1.f, 12);
  fa[2] = -ldexp(1.f, 12);
  fb[2] = ldexp(1.f, 12);
  fa[3] = ldexp(1.f, -24);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 2^29 - 2^29 + 2^-24");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = ldexp(1.f, 14);
  fb[0] = ldexp(1.f, 15);
  fa[2] = -ldexp(1.f, 14);
  fb[2] = ldexp(1.f, 15);
  fa[3] = ldexp(1.f, -24);
  fb[3] = 1.0f;
  h_c[0] = 0.0f;
  temp = h_c[0];
  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);


  // Free dynamically allocated memory.
  free(h_a);
  free(h_b);
  free(h16_c);
  cudaFree(d16_a);
  cudaFree(d16_b);
  cudaFree(d16_c);
  cudaFree(d_c);
  free(h_c);
}

void my_test_normalize() {

  // Declare pointers and allocate memory.
  half *h_a, *h_b, *h16_c, *d16_a, *d16_b, *d16_c;
  float *d_c, *h_c;

  h_a = new half[16*16];
  h_b = new half[16*16];
  h_c = new float[16*16];
  h16_c = new half[16*16];

  cudaMalloc(&d16_a, 16*16*sizeof(half));
  cudaMalloc(&d16_b, 16*16*sizeof(half));
  cudaMalloc(&d16_c, 16*16*sizeof(half));
  cudaMalloc(&d_c, 16*16*sizeof(float));

  // mgu
  float fa[16*16] = {};
  float fb[16*16] = {};
  float temp = 0;

  printf("\n");
  printf("---------------------------------------------------\n");
  printf("Tests for normalization before add accumulator\n");

  // case
  printf("\n");
  printf("case: 1, 2^-23");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = 1.0f;
  fa[1] = -1.0f;
  fb[1] = 1.0f;
  h_c[0] = ldexp(1.f, -23);
  temp = h_c[0];

  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 1, 2^-24");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = 1.0f;
  fa[1] = -1.0f;
  fb[1] = 1.0f;
  h_c[0] = ldexp(1.f, -24);
  temp = h_c[0];

  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 1, 2^-25");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = 1.0f;
  fa[1] = -1.0f;
  fb[1] = 1.0f;
  h_c[0] = ldexp(1.f, -25);
  temp = h_c[0];

  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 1, 2^-26");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = 1.0f;
  fa[1] = -1.0f;
  fb[1] = 1.0f;
  h_c[0] = ldexp(1.f, -26);
  temp = h_c[0];

  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);

  // case
  printf("\n");
  printf("case: 1, 2^-40");
  reset_all(fa, fb, h_a, h_b, h_c);
  fa[0] = 1.0f;
  fb[0] = 1.0f;
  fa[1] = -1.0f;
  fb[1] = 1.0f;
  h_c[0] = ldexp(1.f, -40);
  temp = h_c[0];

  float_to_half(fa, fb, h_a, h_b);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  print_result(fa, fb, h_a, h_b, temp, h_c[0]);


  // Free dynamically allocated memory.
  free(h_a);
  free(h_b);
  free(h16_c);
  cudaFree(d16_a);
  cudaFree(d16_b);
  cudaFree(d16_c);
  cudaFree(d_c);
  free(h_c);
}

/***************
 * EXPERIMENTS *
 ***************/
int main(int argc, char** argv){
  my_test_addr();
  my_test_normalize();
}
