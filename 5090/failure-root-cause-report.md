# RTX 5090 Failure Root Cause Report

Date: 2026-05-09

## Environment

- GPU: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Driver: 595.71.05
- CUDA reported by driver: 13.2
- `nvcc`: CUDA compilation tools 13.2, V13.2.78
- Build target: `sm_120`

## Commands

```sh
make -B test-5090
python3 run_tests.py -o 5090 5090
compute-sanitizer --tool memcheck ./test-5090-binary16
compute-sanitizer --tool memcheck ./test-5090-bf16
compute-sanitizer --tool memcheck ./test-5090-tf32
compute-sanitizer --tool memcheck ./test-5090-binary64
```

All `compute-sanitizer` runs reported `ERROR SUMMARY: 0 errors`.

## Summary

The remaining 5090 failures are numerical-behavior differences in the
Blackwell/SM120 tensor-core HMMA path, not CUDA launch or memory errors.

The failures appear in the binary16, bfloat16, and tf32 tests:

| Test | Failing checks |
| --- | --- |
| `test-5090-binary16` | `Products are accumulated in binary32`, `Sum starts from largest element`, `Monotonicity of dot product` |
| `test-5090-bf16` | `Products are accumulated in binary32`, `Sum starts from largest element`, `Monotonicity of dot product` |
| `test-5090-tf32` | `Products are accumulated in binary32`, `Sum starts from largest element`, `Monotonicity of dot product` |

`test-5090-binary64` passes all checks.

The CUDA 13.2 low-precision format probes also pass on the RTX 5090:

| Test | Scope |
| --- | --- |
| `test-5090-fp8` | FP8 E4M3/E5M2 host and device scalar/vector conversions |
| `test-5090-fp6` | FP6 E2M3/E3M2 host and device scalar/vector conversions |
| `test-5090-fp4` | FP4 E2M1 host and device scalar/vector conversions |

These probes confirm CUDA type/conversion availability and device execution for
`sm_120`. They do not claim fp8/fp6/fp4 WMMA numerical behavior, because the
current CUDA 13.2 `mma.h` interface used by this suite does not expose matching
fp8/fp6/fp4 WMMA fragments.

## Tensor-Core Instructions

`cuobjdump --dump-sass` confirms that the failing tests use SM120 HMMA
instructions with float accumulators:

```text
test-5090-binary16
HMMA.16816.F32

test-5090-bf16
HMMA.16816.F32.BF16

test-5090-tf32
HMMA.1684.F32.TF32
```

The binary64 test does not exhibit these failures.

## Root Cause: Products Are Accumulated In Binary32

The failing checks construct a dot product whose small terms are below the
precision expected to survive repeated binary32 additions into `1.0f`.

For binary16, the test uses:

- `h_a[0..3] = 0.5`
- `h_b[0..3] = minsubnormal16 = 2^-24`
- initial `h_c[0] = 1.0f`

Each product is `2^-25`; four products sum to `2^-23`.

The old predicate expects every product to be added with binary32 rounding after
each addition, so the four `2^-25` contributions should be lost and the final
result should remain:

```text
expected: 0x3f800000  (1.0f)
```

On RTX 5090, `cuda-gdb` shows:

```text
actual:   0x3f800001  (1.00000012f = 1.0f + 2^-23)
```

The same actual result was observed for binary16, bfloat16, and tf32.

Root cause: the SM120 HMMA float-accumulator path preserves or combines
sub-ULP dot-product contributions internally before final float storage. This
does not match the older test assumption of binary32 rounding after each
individual product addition.

## Root Cause: Sum Starts From Largest Element

The `Sum starts from largest element` checks expect the dot-product ordering and
rounding to discard the same small contributions and return exactly `1.0f`.

For binary16, `cuda-gdb` sampled all four loop iterations:

```text
j = 0 -> 0x3f800001
j = 1 -> 0x3f800001
j = 2 -> 0x3f800001
j = 3 -> 0x3f800001
```

The predicate expects `0x3f800000`, so all four iterations fail the expectation.

Root cause: the SM120 HMMA path is not behaving like the ordering model assumed
by the test. It retains the aggregate small contribution and stores
`1.0f + 2^-23`, which is consistent with the previous failure.

## Root Cause: Monotonicity Of Dot Product

The monotonicity check compares two dot products and expects the second result
to be strictly smaller than the first:

```cpp
printpass(outfile, h_c[0] < partial);
```

For binary16, `cuda-gdb` observed:

```text
partial = 0x3f800001  (1.00000012f)
h_c[0]  = 0x3f800001  (1.00000012f)
```

The strict comparison fails because both cases round to the same stored float.
The same equality was observed for the bfloat16 and tf32 tests in the current
run.

Root cause: SM120's internal dot-product behavior changes the rounding boundary
for these crafted inputs. The two cases are not ordered after final float
storage even though the test expects strict monotonicity.

## Fixed Non-Numerical Issue

The bfloat16 and tf32 `Sum starts from largest element` loops previously had a
host-side setup bug: missing braces caused `h_b[j-1]` to be written when
`j == 0`, which writes before the start of `h_b`.

That issue was fixed in:

- `src/tc_test_numerics-A100-bf16.cu`
- `src/tc_test_numerics-A100-tf32.cu`

After the fix, `compute-sanitizer` reports zero device errors, and the tests run
to normal process exit. The remaining failures are the SM120 numerical behavior
differences described above.

## Conclusion

The 5090 failures are expected observations for this architecture and these
predicates:

- CUDA is working.
- The generated 5090 binaries compile and run for `sm_120`.
- Device-side memory checking is clean.
- Binary64 behavior matches the existing predicates.
- FP8, FP6, and FP4 CUDA low-precision type/conversion probes pass on 5090.
- Binary16, bfloat16, and tf32 HMMA float-accumulator paths preserve aggregate
  sub-ULP contributions that older binary32-per-add predicates expected to lose.

The result files in `5090/` have been regenerated from the current binaries.
