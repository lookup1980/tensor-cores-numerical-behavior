# Numerical Behavior of the NVIDIA Tensor Cores

The aim of this test suite is to probe the numerical behavior of the tensor cores that equip some recent NVIDIA graphic cards. The tests in the suite are divided into 4 sections, and show the following features.

A. Support for subnormal numbers
* Tensor cores accept binary16 subnormals in input in binary16 mode
* Tensor cores accept binary16 subnormals in input in binary32 mode
* Tensor cores accept binary32 subnormals in input and return them
* Tensor cores can return binary16 subnormals in binary16 mode
* Tensor cores can return binary16 subnormals in binary32 mode

B. Accuracy of the dot products
* Tensor cores compute the products of two binary16 numbers exactly
* Tensor cores compute the products of two binary16 numbers exactly (binary16 mode)
* Tensor cores accumulate sums in binary32 arithmetic
* Tensor cores accumulate partial sums on the largest element in absolute value

C. Rounding modes in tensor core computations
* Tensor cores use round-down for positive values
* Tensor cores use round-up for negative values
* Tensor cores round the accumulator using round-to-nearest (binary16 mode)

D. Features of the accumulator
1) Tensor cores do not implement guard digits (extra bits on the right)
2) Tensor cores do not normalize by shifting right (in sums of elements with same sign)
3) Tensor cores do not normalize by shifting left (in sums of elements with opposite sign)
4) Tensor cores implement two carry-out digits (extra bits on the left)
5) The product of tensor cores is not monotonic

### Compiling and running the suite
The experiments can be compiled by issuing `make all`, which generates the
executable files supported by the installed CUDA compiler. Older CUDA releases
may support architectures that newer releases have dropped; explicit targets
remain available when the local `nvcc` supports their `sm_` architecture:
* `test-V100`, for testing Volta GPUs (requires version 9 or newer of the CUDA platform);
* `test-T4`, for testing Turing GPUs (requires version 10 or newer of the CUDA platform);
* `test-A100-binary16`, `test-A100-bf16`, `test-A100-tf32`, `test-A100-binary64`, for testing the four precision configurations available on Ampere GPUs (requires version 11 or newer of the CUDA platform).
* `test-H100-*`, `test-4090-*`, and `test-5090-*`, for testing the corresponding Hopper, Ada, and Blackwell-generation targets configured in the `Makefile`.
* `test-5090-fp8`, `test-5090-fp6`, and `test-5090-fp4`, for probing CUDA 13.2 low-precision type and conversion support on `sm_120`. These are format availability probes, not WMMA numerical-behavior tests.
* `test-5090-fp8-reduction-width`, `test-5090-fp6-reduction-width`, and `test-5090-fp4-reduction-width`, for probing RTX 5090 low-precision tensor-core reduction width with direct PTX `mma.sync.aligned` instructions. The matching `test-5090-fp8-reduction-repeat`, `test-5090-fp6-reduction-repeat`, and `test-5090-fp4-reduction-repeat` targets run a chained-MMA cross-check. The standalone `test-5090-fp8-reduction-pattern`, `test-5090-fp6-reduction-pattern`, and `test-5090-fp4-reduction-pattern` targets run product-pattern `M - M + epsilon` probes from a separate source file. These targets use `compute_120a`/`sm_120a` because FP6/FP4 `.kind::f8f6f4` PTX requires the SM120 accelerated feature target.

Result files can be generated with `run_tests.py`. With no selectors, it runs
all executable `test-*` binaries in the repository root. Selectors can limit the
run to one GPU target or one binary:

```
python3 run_tests.py 5090 -o 5090
python3 run_tests.py 5090-bf16 -o 5090
```

### Extending the suite
The build configuration is organized around target GPUs and data formats:

* Add a new target GPU by defining `GPU_SM_<name>` in the `Makefile` and adding
  the target name to `FORMAT_GPUS`.
* Add a common data format by adding its CUDA source under `src/`, defining a
  source variable, adding the format name to `BASE_FORMAT_TARGETS`, and adding
  a matching `test-%-<format>` rule.
* Add a GPU-specific data format by adding it to `EXTRA_FORMAT_TARGETS_<gpu>`
  and providing a matching target rule, as the 5090 fp8/fp6/fp4 probes do.
* Shared output, CUDA error handling, tile sizing, and device tile allocation
  live under `include/` so new format tests can reuse the same runtime helpers.

### Reference
Details about the code in this repository can be found in:

Massimiliano Fasi, Nicholas J. Higham, Mantas Mikaitis, and Srikara Pranesh. [Numerical Behavior of the NVIDIA Tensor Cores](https://doi.org/10.7717/peerj-cs.330). PeerJ Computer Science 7:e330, 2021.

### License
This software is distributed under the terms of the GNU GPL v.2 software license (see [LICENSE.md](./LICENSE.md)).
