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
#include "include/tcnb_output.hpp"
#include "include/tcnb_cuda_check.cuh"
#include "include/tcnb_matrix.cuh"
#include "include/tcnb_device_tile.cuh"

using namespace nvcuda;

/*******************
 * Debug functions *
 *******************/
/* Print the elements of the m x n matrix A. The elements are assumed to be
   stored by columns if `bycols` is `true` and by rows if `bycols` is false. */
template <typename floattype>
void print_matrix (double *a,
                   size_t m, size_t n,
                   bool bycols) {
  int i, j;
  if (bycols) {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++)
        std::cout << a[j*n+i] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  } else {
    for (i=0; i<m; i++ ) {
      for (j=0; j<n; j++)
        std::cout << a[i*m+j] << " ";
      std::cout  << std::endl;
    }
    std::cout << std::endl;
   }
}


/****************************************************
 * Memory management and wmma::mma_sync() interface *
 ****************************************************/

/* Compute C += A*B, where A, B, and C are 16x16x16 matrices.
   The matrix C is initialized to 0 when `init` is true. */
__global__ void wmma_ker(double *a, double *b, double *c, bool init) {

  // Declare fragments.
  wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> b_fragment;
  wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_fragment;

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
void wmma_init_run (double *h_a, double *h_b, double *h_c,
                    double *d_a, double *d_b, double *d_c,
                    bool init) {

  // Copy input from host to device.
  copy_tile_to_device(d_a, h_a);
  copy_tile_to_device(d_b, h_b);
  copy_tile_to_device(d_c, h_c);

  // Perform matrix multiplication.
  wmma_ker<<<1,32>>>(d_a, d_b, d_c, init);
  TCNB_CUDA_CHECK(cudaGetLastError());

  // Copy result from device to host.
  copy_tile_to_host(h_c, d_c);
}


/***************
 * EXPERIMENTS *
 ***************/
int main(int argc, char** argv){

  // Declare pointers and allocate memory.
  double *h_a, *h_b, *h_c, *d_a, *d_b, *d_c,
    minsubnormal64 = ldexp(1., -1074), // smallest subnormal binary32
    belowone = nextafter(1., 0.) ,   // largest float smaller than 1.0
    gapbelowone = 1. - belowone,
    aboveone = nextafter(1., 2.),    // smallest float larger than 1.0
    belowtwo = 2. - ldexp(1., -52);   // largest float smaller than 2.0

  assert(belowone == 1. - ldexp(1., -53));
  assert(aboveone == 1. + ldexp(1., -52));

  h_a = new double[TCNB_TILE_ELEMENTS];
  h_b = new double[TCNB_TILE_ELEMENTS];
  h_c = new double[TCNB_TILE_ELEMENTS];
 
  d_a = device_tile_alloc<double>();
  d_b = device_tile_alloc<double>();
  d_c = device_tile_alloc<double>();

  FILE *outfile = stdout;
  bool pass;

  printheader(outfile, "A. Support for subnormal numbers");// ;

   printitem(outfile, "*) Binary64 subnormals in input");
  host_reset(h_a, h_b, h_c);
  h_a[0] = minsubnormal64;
  h_b[0] = ldexp(1, 52);
  wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
  printpass(outfile, h_c[0]==ldexp(1., -1022));

  printitem(outfile, "*) Binary64 subnormals in output");
  host_reset(h_a, h_b, h_c);
  h_a[0] = ldexp(1., -1022);
  h_b[0] = ldexp(1., -1);
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  pass = h_c[0] == ldexp(1, -1023);
  h_a[0] = ldexp(1., -1022);
  h_b[0] = 1.0;
  h_c[0] = ldexp(-1., -1023);
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  pass = pass && (h_c[0] == ldexp(1, -1023));
  printpass(outfile, pass);

  printfooter(outfile);

  printheader(outfile, "B. Accuracy of the dot products ");// ;

  int i;
  printitem(outfile, "*) Products are accumulated in binary64 ");
  host_reset(h_a, h_b, h_c);
  pass = true;
  for (i=0; i<2; i++) {
    h_a[i] = 0.5;
    h_b[i] = ldexp(1, -53);
  }
  h_c[0] = 1.;
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  pass = pass && h_c[0] == 1;
  printpass(outfile, pass);

  printfooter(outfile);

  printheader(outfile, "C. Rounding modes in tensor core computations ");

  printitem(outfile, "*) Round-to-nearest for positive values ");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<2; i++) {
    h_a[i] = 1.0;
  }
  h_b[0] = 2.;
  h_b[1] = ldexp(1., -52) + ldexp(1., -53);
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  pass = h_c[0] == 2. + ldexp(1, -51);
  h_b[1] = ldexp(1., -53);
  h_c[0] = 0.0;
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  printpass(outfile, pass && h_c[0] == 2.);

  printitem(outfile, "*) Round-to-nearest for negative values ");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<2; i++) {
    h_a[i] = 1.0;
  }
  h_b[0] = -2.;
  h_b[1] = -ldexp(1., -52) - ldexp(1., -53);
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  pass = h_c[0] == -2. - ldexp(1, -51);
  h_b[1] = -ldexp(1, -53);
  h_c[0] = 0.0;
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  printpass(outfile, pass && h_c[0] == -2.0);

  printitem(outfile, "*) Round-to-nearest ties broken to even ");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<4; i++) {
    h_a[i] = 1.0;
  }
  h_b[0] = 2.;
  h_b[1] = ldexp(1., -52);
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  pass = h_c[0] == 2.;
  h_b[0] = 2.+ldexp(1, -51);
  h_b[1] = ldexp(1, -52);
  h_c[0] = 0.0;
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  printpass(outfile, pass && h_c[0] == 2.0 + ldexp(1, -50));

  printfooter(outfile);

  printheader(outfile, "D. Features of the accumulator");

  printitem(outfile, "1) Extra bits in the significand alignment");
  host_reset(h_a, h_b, h_c);
  h_a[0] = 1.0;
  h_b[0] = 1.0;
  h_c[0] = -belowone;
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  assert(1 - belowone == ldexp(1., -53));
  assert(gapbelowone == ldexp(1., -53));
  printpass(outfile, h_c[0] == ldexp(1., -53));

  printitem(outfile, "2) Normalization in addition (after each add)");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<2; i++) {
    h_a[i] = 1.0;
    h_b[i] = ldexp(1, -53);
  }
  h_c[0] = 1. - ldexp(1., -53);
  assert(h_c[0] == belowone);
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  pass = h_c[0] == 1.;
  printpass(outfile, pass);

  printitem(outfile, "3) Normalization in subtraction");
  host_reset(h_a, h_b, h_c);
  h_a[0] = 1.0;
  h_a[1] = 1.0;
  h_b[0] = 1.0;
  h_b[1] = -ldexp(1., -53);
  h_c[0] = -1. + ldexp(1., -53);
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  pass = pass && h_c[0] == 0.0;
  printpass(outfile, pass);

  printitem(outfile, "4) No extra bits for carry out");
  host_reset(h_a, h_b, h_c);
  for (i=0; i<2; i++) {
    h_a[i] = 1.0;
    h_b[i] = 1.0;
  }
  pass = true;
  for (i=0; i<2; i++) {
    if (i>0)
      h_b[i-1] = 1.0;
    h_b[i] = ldexp(1., -52);
    h_c[0] = 1. + ldexp(1., -51) + ldexp(1., -52);
    wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
    pass = pass && (h_c[0] == 2. || (h_c[0] == (2. + ldexp(1, -50))));
  }

  printpass(outfile, pass);

  printitem(outfile, "5) Monotonicity of dot product");
  host_reset(h_a, h_b, h_c);
  h_a[0] = 1.0;
  h_b[0] = ldexp(1., -54);
  h_a[1] = 1.0;
  h_b[1] = ldexp(1., -53) + ldexp(1., -54);
  h_c[0] = 1. - ldexp(1., -53);
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  double partial = h_c[0];
  h_c[0] = 1.0;
  wmma_init_run (h_a, h_b, h_c, d_a, d_b, d_c, false);
  printpass(outfile, h_c[0] >= partial);

  printfooter(outfile);

  // Free dynamically allocated memory.
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
