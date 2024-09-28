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
#include <cuda_bf16.h>

using namespace nvcuda;

/*******************
 * Debug functions *
 *******************/
/* Print the elements of the m x n matrix A. The elements are assumed to be
   stored by columns if `bycols` is `true` and by rows if `bycols` is false. */
template <typename floattype>
void print_matrix (nv_bfloat16 *a,
                   size_t m, size_t n,
                   bool bycols) {
  int i, j;
  if (bycols) {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++)
        std::cout << __bfloat162float(a[j*n+i]) << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  } else {
    for (i=0; i<m; i++ ) {
      for (j=0; j<n; j++)
        std::cout << __bfloat162float(a[i*m+j]) << " ";
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
void host_reset(nv_bfloat16 *a, nv_bfloat16 *b, returntype *c) {
  memset(a, 0, 16*16*sizeof(nv_bfloat16));
  memset(b, 0, 16*16*sizeof(nv_bfloat16));
  memset(c, 0, 16*16*sizeof(returntype));
}

/* Compute C += A*B, where A, B, and C are 16x16x16 matrices.
   The matrix C is initialized to 0 when `init` is true. */
template <typename returntype>
__global__ void wmma_ker(nv_bfloat16 *a, nv_bfloat16 *b,
                         returntype *c, bool init) {

  // Declare fragments.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, nv_bfloat16,
    wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, nv_bfloat16,
    wmma::col_major> b_fragment;
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Copy data from host to device, perform the operation, and copy result back to
   host. */
template <typename returntype>
void wmma_init_run (nv_bfloat16 *h_a, nv_bfloat16 *h_b, returntype *h_c,
                    nv_bfloat16 *d_a, nv_bfloat16 *d_b, returntype *d_c,
                    bool init) {
  
  gpuErrchk( (cudaGetLastError()) );

  // Copy input from host to device.
  cudaMemcpy(d_a, h_a, 16*16*sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, 16*16*sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, 16*16*sizeof(returntype), cudaMemcpyHostToDevice);

  gpuErrchk( (cudaGetLastError()) );

  // Perform matrix multiplication.
  wmma_ker<<<1,32>>>(d_a, d_b, d_c, init);

  gpuErrchk( (cudaGetLastError()) );

  // Copy result from device to host.
  cudaMemcpy(h_c, d_c, 16*16*sizeof(returntype), cudaMemcpyDeviceToHost);

  gpuErrchk( (cudaGetLastError()) );
}


/**********************
 * Printing functions *
 **********************/
void printheader(FILE *outfile, const char *string) {
  fprintf(outfile,
          "+--------------------------------------------------------------+\n");
  fprintf(outfile, "| %-60s |\n", string);
  fprintf(outfile,
          "+--------------------------------------------------------------+\n");
}
void printitem(FILE *outfile, const char *string) {
  fprintf(outfile, "  | %-49s", string);
}

void printpass(FILE *outfile, bool status) {
  if (status)
    fprintf(outfile, " [PASS] |\n");
  else
    fprintf(outfile, " [FAIL] |\n");
}
void printfooter(FILE *outfile) {
  fprintf(outfile,
          "  +----------------------------------------------------------+\n\n");
}


/***************
 * EXPERIMENTS *
 ***************/
int main(int argc, char** argv){

  // Declare pointers and allocate memory.
  nv_bfloat16 *h_a, *h_b, *h16_c, *d16_a, *d16_b, *d16_c,
    minsubnormal16 = __float2bfloat16(ldexp(1., -133)), // smallest subn. bf16
    belowone16 = __float2bfloat16(1. - ldexp(1, -8)),
    zero16 = __float2bfloat16(0.),
    one16 = __float2bfloat16(1.),
    minusone16 = __float2bfloat16(-1.),
    two16 = __float2bfloat16(2.),
    four16 = __float2bfloat16(4.);
  float *d_c, *h_c,
    minsubnormal32 = ldexp(1., -149), // smallest subnormal binary32
    belowone = nextafterf(1., 0.) ,   // largest float smaller than 1.0
    gapbelowone = 1. - belowone,
    aboveone = nextafterf(1., 2.),    // smallest float larger than 1.0
    belowtwo = 2. - ldexp(1., -23);   // largest float smaller than 2.0

  assert(belowone == 1. - ldexp(1., -24));
  assert(aboveone == 1. + ldexp(1., -23));

  h_a = new nv_bfloat16[16*16];
  h_b = new nv_bfloat16[16*16];
  h_c = new float[16*16];
  h16_c = new nv_bfloat16[16*16];

  cudaMalloc(&d16_a, 16*16*sizeof(nv_bfloat16));
  cudaMalloc(&d16_b, 16*16*sizeof(nv_bfloat16));
  cudaMalloc(&d16_c, 16*16*sizeof(nv_bfloat16));
  cudaMalloc(&d_c, 16*16*sizeof(float));

  FILE *outfile = stdout;
  bool pass;

  printheader(outfile, "A. Support for subnormal numbers");// ;

  printitem(outfile, "*) Bfloat16 subnormals in input (binary32 mode)");
  host_reset(h_a, h_b, h_c);
  h_a[0] = minsubnormal16;
  h_b[0] = __float2bfloat16(ldexp(1, 7));
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  printpass(outfile, h_c[0]==ldexp(1., -126));

  printitem(outfile, "*) Binary32 subnormals in input");
  host_reset(h_a, h_b, h_c);
  h_c[0] = minsubnormal32;
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  printpass(outfile, h_c[0] == minsubnormal32);

  printitem(outfile,
    "*) Bfloat16/binary32 subnormals in output (binary32 mode)");
  host_reset(h_a, h_b, h_c);
  h_a[0] = __float2bfloat16(ldexp(1., -126));
  h_b[0] = __float2bfloat16(ldexp(1., -1));
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = h_c[0] == ldexp(1, -127);
  h_a[0] = __float2bfloat16(ldexp(1., -126));
  h_b[0] = one16;
  h_c[0] = ldexp(-1., -127);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = pass && (h_c[0] == ldexp(1, -127));
  printpass(outfile, pass);

  printfooter(outfile);

  printheader(outfile, "B. Accuracy of the dot products ");// ;

  printitem(outfile, "*) Products are computed exactly ");
  host_reset(h_a, h_b, h_c);
  h_a[0] = belowone16;
  h_b[0] = belowone16;
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = (h_c[0] == 1 - ldexp(1, -7) + ldexp(1, -16));
  size_t i,j;
  for (i=0; i<4; i++) {
    h_a[i] = belowone16;
    h_b[i] = belowone16;
  }
  h_c[0] = zero16;
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = (h_c[0] == (4 * (1 - ldexp(1, -7) + ldexp(1, -16))));
  printpass(outfile, pass);

  printitem(outfile, "*) Products are accumulated in binary32 ");
  host_reset(h_a, h_b, h_c);
  pass = true;
  for (i=0; i<4; i++) {
    h_a[i] = 0.5;
    h_b[i] = __float2bfloat16(ldexp(1, -24));
  }
  h_c[0] = 1.;
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = pass && h_c[0] == 1;
  printpass(outfile, pass);

  printitem(outfile, "*) Sum starts from largest element");
  host_reset(h_a, h_b, h_c);
  pass = true;
  for (i=0; i<4; i++) {
    h_a[i] = 0.5;
    h_b[i] = __float2bfloat16(ldexp(1, -24));
  }
  for (j=0; j<4; j++) {
    h_c[0] = ldexp(1, -24);
    if (j>0)
      h_a[j-1] = 0.5;
    h_b[j-1] = __float2bfloat16(ldexp(1, -24));
    h_a[j] = one16;
    h_b[j] = one16;
    wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    pass = pass && h_c[0] == 1;
  }
  printpass(outfile, pass);

  printfooter(outfile);

  printheader(outfile, "C. Rounding modes in tensor core computations ");

  printitem(outfile, "*) Round-down for positive values ");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<4; i++) {
    h_a[i] = one16;
  }
  h_b[0] = __float2bfloat16(2.);
  h_b[1] = __float2bfloat16(ldexp(1., -23) + ldexp(1., -24));
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  printpass(outfile, h_c[0] == 2.);

  printitem(outfile, "*) Round-up for negative values ");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<4; i++) {
    h_a[i] = one16;
  }
  h_b[0] = __float2bfloat16(-2.);
  h_b[1] = __float2bfloat16(-ldexp(1., -23) - ldexp(1., -24));
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  printpass(outfile, h_c[0] == -2.);

  printfooter(outfile);

  printheader(outfile, "D. Features of the accumulator");

  printitem(outfile, "1) Extra bits in the significand alignment");
  host_reset(h_a, h_b, h_c);
  h_a[0] = one16;
  h_b[0] = one16;
  h_c[0] = -belowone;
  // h_c[0] = 1.0f;
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  assert(1 - belowone == ldexp(1., -24));
  assert(gapbelowone == ldexp(1., -24));
  printpass(outfile, h_c[0] == ldexp(1., -24));
  // fprintf(outfile, "%x \n", ((uint32_t*)h_c)[0]);

  printitem(outfile, "2) Normalization in addition");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<4; i++) {
    h_a[i] = one16;
    h_b[i] = __float2bfloat16(ldexp(1, -24));
  }
  h_c[0] = 1. - ldexp(1., -24);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = h_c[0] == 1. + ldexp(1., -23);
  printpass(outfile, pass);

  printitem(outfile, "3) Normalization in subtraction");
  host_reset(h_a, h_b, h_c);
  h_a[0] = one16;
  h_a[1] = one16;
  h_b[0] = one16;
  h_b[1] = __float2bfloat16(-ldexp(1., -24));
  h_c[0] = -1. + ldexp(1., -24);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = pass && h_c[0] == 0.0;
  printpass(outfile, pass);

  printitem(outfile, "4) Extra bits for carry out");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<4; i++) {
    h_a[i] = one16;
    h_b[i] = one16;
  }
  pass = true;
  for (i=0; i<4; i++) {
    if (i>0)
      h_b[i-1] = one16;
    h_b[i] = __float2bfloat16(ldexp(1., -23));
    h_c[0] = 1. + ldexp(1., -22) + ldexp(1., -23);
    wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    pass = pass && h_c[0] == 4. + ldexp(1., -21);
  }

  // Test for the third bit
  host_reset(h_a, h_b, h_c);
  for (i=0; i<4; i++) {
    h_a[i] = one16;
    h_b[i] = one16;
  }
  pass = true;
  h_b[0] = one16;
  h_b[1] = __float2bfloat16(1.5);
  h_b[2] = __float2bfloat16(1.75);
  h_b[3] = __float2bfloat16(1.875);
  h_c[0] = 1.875;
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = pass && h_c[0] == 8.;

  // Round-down in normalization of positive values.
  host_reset(h_a, h_b, h_c);
  for (i=0; i<4; i++) {
    h_a[i] = one16;
    h_b[i] = one16;
  }
  h_c[0] = 1. + ldexp(1., -22) + ldexp(1., -23);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = pass && (h_c[0] == 5.);

  // Round-up in normalization of negative values.
  host_reset(h_a, h_b, h_c);
  for (i=0; i<4; i++) {
    h_a[i] = one16;
    h_b[i] = minusone16;
  }
  h_c[0] = -1. - ldexp(1., -22) - ldexp(1., -23);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  pass = pass && (h_c[0] == -5.);

  printpass(outfile, pass);

  printitem(outfile, "5) Monotonicity of dot product");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<3; i++) {
    h_a[i] = 0.5;
    h_b[i] = ldexp(1., -24);
  }
  for (i=3; i<4; i++) {
    h_a[i] = 0.5;
    h_b[i] = ldexp(1., -23)+ldexp(1., -24);
  }
  h_c[0] = 1. - ldexp(1., -24);
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  float partial = h_c[0];
  h_c[0] = 1.0;
  wmma_init_run (h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  printpass(outfile, h_c[0] < partial);

  printfooter(outfile);

  // Free dynamically allocated memory.
  //  free(h_a);
  //  free(h_b);
  free(h16_c);
  cudaFree(d16_a);
  cudaFree(d16_b);
  cudaFree(d16_c);
  cudaFree(d_c);
  free(h_c);
}
