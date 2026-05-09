# RTX 5090 Low-Precision Reduction Width Report

Date: 2026-05-09

## Environment

- GPU: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- CUDA toolkit: 13.2, `nvcc` V13.2.78
- Reduction-width build target: `compute_120a` / `sm_120a`

## Scope

These tests do not use cuBLASLt. They execute direct inline PTX
`mma.sync.aligned` instructions from a single warp.

The FP8 tests use regular unscaled FP8 MMA instructions. The FP6 and FP4 tests
use the PTX `.kind::f8f6f4` MMA form. In this form, FP6 values are stored in the
low 6 bits of each 8-bit container, and FP4 E2M1 values are stored in the
central 4 bits of each 8-bit container.

The compact block-scaled MXFP4/MXFP6 forms are not covered by these tests.

## Method

Each probe computes one `m16n8k32` MMA operation with all A and B elements set
to `1.0`, so the dot-product contribution is exactly `K = 32`.

The C accumulator is set to `2^shift`, and the test scans for the largest shift
where the output still changes:

```text
D = 2^shift + 32
```

For an accumulator with `p` effective significand bits, the largest observed
shift is:

```text
shift = log2(K) + p - 1
```

The observed width is therefore:

```text
p = max_changed_shift - log2(K) + 1
```

For binary32, `p = 24`, so the reference max shift for `K = 32` is `28`.

## Results

All tested direct PTX low-precision paths report 21 effective bits:

| Test | PTX instruction family | Max shift | Observed bits |
| --- | --- | ---: | ---: |
| FP8 E4M3 x E4M3 | `mma.sync.aligned.m16n8k32...f32.e4m3.e4m3.f32` | 25 | 21 |
| FP8 E5M2 x E5M2 | `mma.sync.aligned.m16n8k32...f32.e5m2.e5m2.f32` | 25 | 21 |
| FP8 E4M3 x E5M2 | `mma.sync.aligned.m16n8k32...f32.e4m3.e5m2.f32` | 25 | 21 |
| FP8 E5M2 x E4M3 | `mma.sync.aligned.m16n8k32...f32.e5m2.e4m3.f32` | 25 | 21 |
| FP6 E2M3 x E2M3 | `mma.sync.aligned.m16n8k32...kind::f8f6f4.f32.e2m3.e2m3.f32` | 25 | 21 |
| FP6 E3M2 x E3M2 | `mma.sync.aligned.m16n8k32...kind::f8f6f4.f32.e3m2.e3m2.f32` | 25 | 21 |
| FP4 E2M1 x E2M1 | `mma.sync.aligned.m16n8k32...kind::f8f6f4.f32.e2m1.e2m1.f32` | 25 | 21 |

The result files are:

- `5090/result-5090-fp8-reduction-width.txt`
- `5090/result-5090-fp6-reduction-width.txt`
- `5090/result-5090-fp4-reduction-width.txt`

## Instruction Verification

`cuobjdump --dump-sass` confirms that the binaries contain direct QMMA tensor
instructions rather than library calls:

```text
QMMA.16832.F32.E4M3.E4M3
QMMA.16832.F32.E5M2.E5M2
QMMA.16832.F32.E2M3.E2M3
QMMA.16832.F32.E3M2.E3M2
QMMA.16832.F32.E2M1.E2M1
```

## Root Cause

The direct SM120a low-precision QMMA path does not behave like a full binary32
accumulation path for this boundary probe. If it were binary32-equivalent, the
largest changed shift would be `28`. The observed boundary is `25`, which
corresponds to a 21-bit effective reduction/addition width.

This applies consistently across the tested FP8, FP6, and FP4 direct PTX MMA
forms.

## Build Note

FP6/FP4 `.kind::f8f6f4` PTX is rejected for plain `sm_120`. The Makefile uses
explicit accelerated-feature code generation:

```sh
nvcc ... -arch=compute_120a -code=sm_120a
```

Using only `-arch=sm_120a` was not sufficient with this CUDA 13.2 toolchain,
because the generated PTX target still lacked the accelerated feature suffix.
