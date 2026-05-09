# RTX 5090 FP8/FP6/FP4 Reduction Tree Width Report

Date: 2026-05-09

## Executive Summary

This report documents direct-PTX tensor-core probes for RTX 5090 low-precision
MMA reduction behavior. The tests cover FP8, FP6, and FP4 input formats and use
inline `mma.sync.aligned` PTX. They do not use cuBLASLt.

The measured value is the externally observable effective addition/reduction
width for one `m16n8k32` tensor-core instruction. It is a boundary probe of the
instruction result, not a direct map of the internal physical reduction tree.

All tested FP8, FP6, and FP4 direct-PTX QMMA paths report the same boundary:

| Metric | Value |
| --- | ---: |
| MMA shape | `m16n8k32` |
| Dot-product contribution per result | 32 |
| Largest shift where `D > C` | 25 |
| Observed effective width | 21 bits |
| Binary32 reference width | 24 bits |
| Difference from binary32 reference | -3 bits |

A second, intentionally different setting chains 2, 4, 8, and 16 dependent
MMA instructions with the accumulator fed forward. If the repeated additions
were combined and rounded only once, the max shift would move upward as the
total nominal contribution grows. Instead, every tested repeat count remains at
max shift 25. This confirms that the 21-bit boundary is applied per MMA update
for this path.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| Compute capability | 12.0 |
| CUDA toolkit | 13.2, `nvcc` V13.2.78 |
| Reduction-width build target | `compute_120a` / `sm_120a` |
| Test source | `src/tc_test_numerics-5090-lowp-reduction-width.cu` |
| Result files | `5090/result-5090-fp8-reduction-width.txt`, `5090/result-5090-fp6-reduction-width.txt`, `5090/result-5090-fp4-reduction-width.txt`, `5090/result-5090-fp8-reduction-repeat.txt`, `5090/result-5090-fp6-reduction-repeat.txt`, `5090/result-5090-fp4-reduction-repeat.txt` |

## What Was Measured

Each probe asks a narrow question:

If one direct tensor-core MMA computes

```text
D = C + sum(A[i] * B[i]), i = 0..31
```

with all A and B inputs exactly equal to `1.0`, how large can `C = 2^shift`
be before adding the exact contribution `32` no longer changes the result?

The largest shift that still changes the result is converted to an effective
significand width. This is the observable reduction/addition width of this
instruction path for this input pattern.

## Methodology

The test executes one warp and one direct MMA instruction per measurement. A
and B are packed as low-precision values representing `1.0`. The C accumulator
is a binary32 value set to `2^shift`.

For an instruction with `K = 32`, the exact contribution is:

```text
sum(A[i] * B[i]) = 32
```

The test scans `shift = 16..40` and records the largest shift where:

```text
sample > base
base = 2^shift
sample = MMA(A = 1, B = 1, C = base)
```

The conversion from boundary shift to effective width is:

```text
p = max_changed_shift - log2(K) + 1
```

For this probe:

```text
K = 32
log2(K) = 5
max_changed_shift = 25
p = 25 - 5 + 1 = 21
```

A full binary32 accumulator with 24 significand bits would remain sensitive to
the `+32` contribution until:

```text
max_changed_shift = log2(32) + 24 - 1 = 28
```

The binary32 reference is therefore 24 bits and max shift 28.

### Independent Cross-Check: Chained MMA Updates

To avoid relying only on the single-MMA setting, the second probe chains
multiple dependent MMA instructions in one warp:

```text
C0 = 2^shift
C1 = MMA(A = 1, B = 1, C = C0)
C2 = MMA(A = 1, B = 1, C = C1)
...
```

The repeat counts are 2, 4, 8, and 16. The nominal total contributions are
therefore 64, 128, 256, and 512.

This setting distinguishes two models:

| Model | Prediction |
| --- | --- |
| Single-round total model | The max shift moves upward with the total contribution. For a 21-bit path, repeat counts 2/4/8/16 would give shifts 26/27/28/29. |
| Per-MMA update model | Each MMA update independently sees only a `+32` contribution. The max shift stays at 25 for all repeat counts. |

The observed result matches the per-MMA update model. That is a different
setting from the original single-instruction probe and confirms that the
21-bit boundary is not an artifact of one isolated measurement.

## Key Code Snippets

The probe constants define the instruction K dimension, scan range, and
binary32 reference:

```cpp
static const int kInstructionK = 32;
static const int kLog2InstructionK = 5;
static const int kMinShift = 16;
static const int kMaxShift = 40;
static const int kExpectedBinary32Bits = std::numeric_limits<float>::digits;
enum { kObservedModelBits = 21 };
```

The inputs are encoded directly as low-precision values equal to `1.0`. FP4
E2M1 values are shifted into the central four bits required by the PTX
`kind::f8f6f4` form:

```cpp
static std::uint8_t encode_fp4_e2m1_padded(float value) {
  return static_cast<std::uint8_t>(__nv_fp4_e2m1(value).__x << 2);
}
```

The core path uses direct inline PTX. The output registers are binary32
accumulators, and the input packs are passed as raw 32-bit registers:

```cpp
#define TCNB_MMA_ASM(instruction)                                               \
  asm volatile(instruction " {%0, %1, %2, %3}, "                                \
                           "{%4, %5, %6, %7}, {%8, %9}, "                      \
                           "{%10, %11, %12, %13};\n"                           \
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                       \
               : "r"(a_pack), "r"(a_pack), "r"(a_pack), "r"(a_pack),         \
                 "r"(b_pack), "r"(b_pack), "f"(c0), "f"(c1),                 \
                 "f"(c2), "f"(c3))
```

The cross-check chains multiple direct MMA updates by feeding the accumulator
registers back into the next instruction:

```cpp
float c0 = base;
float c1 = base;
float c2 = base;
float c3 = base;

#pragma unroll
for (int repeat = 0; repeat < repeat_count; ++repeat) {
  float d0;
  float d1;
  float d2;
  float d3;
  run_direct_mma_fragment<variant>(a_pack, b_pack, c0, c1, c2, c3,
                                   &d0, &d1, &d2, &d3);
  c0 = d0;
  c1 = d1;
  c2 = d2;
  c3 = d3;
}
```

The scan records the last shift where the MMA result is larger than the base
accumulator, then converts that shift to bits:

```cpp
for (int shift = kMinShift; shift <= kMaxShift; ++shift) {
  float base = std::ldexp(1.0f, shift);
  float sample = run_once<variant>(a_pack, b_pack, shift, device_out);

  if (sample > base) {
    result.max_changed_shift = shift;
    result.sample_at_max = sample;
  } else if (result.max_changed_shift >= 0 &&
             result.sample_after_max == 0.0f) {
    result.sample_after_max = sample;
  }
}

result.observed_bits =
    result.max_changed_shift - kLog2InstructionK + 1;
```

The build rule targets the accelerated SM120a feature set explicitly:

```make
test-5090-fp8-reduction-width: $(SRC_5090_LOWP_REDUCTION_WIDTH)
	$(NVCC) $(NVCC_INCLUDES) -DTCNB_REDUCTION_FP8 -o $@ \
	  -arch=$(GPU_COMPUTE_5090_ACCEL) -code=$(GPU_SM_5090_ACCEL) \
	  $(NVCC_STD) $<
```

The same source is compiled with `TCNB_REDUCTION_FP6` and
`TCNB_REDUCTION_FP4` for the FP6 and FP4 targets.

## PTX Instructions Tested

| Family | Variant | PTX instruction |
| --- | --- | --- |
| FP8 | E4M3 x E4M3 | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` |
| FP8 | E5M2 x E5M2 | `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` |
| FP8 | E4M3 x E5M2 | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32` |
| FP8 | E5M2 x E4M3 | `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32` |
| FP6 | E2M3 x E2M3 | `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32` |
| FP6 | E3M2 x E3M2 | `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32` |
| FP4 | E2M1 x E2M1 | `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32` |

## Results

All tested variants produced the same effective width result.

| Family | Variant | Max changed shift | Sample at max shift | First unchanged sample | Observed bits | Binary32 ref bits | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | E4M3 x E4M3 | 25 | 33,554,464 | 67,108,864 | 21 | 24 | -3 |
| FP8 | E5M2 x E5M2 | 25 | 33,554,464 | 67,108,864 | 21 | 24 | -3 |
| FP8 | E4M3 x E5M2 | 25 | 33,554,464 | 67,108,864 | 21 | 24 | -3 |
| FP8 | E5M2 x E4M3 | 25 | 33,554,464 | 67,108,864 | 21 | 24 | -3 |
| FP6 | E2M3 x E2M3 | 25 | 33,554,464 | 67,108,864 | 21 | 24 | -3 |
| FP6 | E3M2 x E3M2 | 25 | 33,554,464 | 67,108,864 | 21 | 24 | -3 |
| FP4 | E2M1 x E2M1 | 25 | 33,554,464 | 67,108,864 | 21 | 24 | -3 |

At `shift = 25`, the base value is `33,554,432`, and the result is
`33,554,464`, so the `+32` contribution is still visible. At `shift = 26`,
the result is `67,108,864`, equal to the base, so the `+32` contribution is no
longer visible.

### Chained-MMA Cross-Check Results

The chained-MMA cross-check uses the same PTX variants but repeats the MMA
update 2, 4, 8, and 16 times with the accumulator fed forward. All tested FP8,
FP6, and FP4 variants produced the same pattern:

| Repeat count | Nominal total contribution | Single-round 21-bit predicted max shift | Single-round binary32 predicted max shift | Observed max shift | Per-MMA observed bits | Sample at max shift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 26 | 29 | 25 | 21 | 33,554,496 |
| 4 | 128 | 27 | 30 | 25 | 21 | 33,554,560 |
| 8 | 256 | 28 | 31 | 25 | 21 | 33,554,688 |
| 16 | 512 | 29 | 32 | 25 | 21 | 33,554,944 |

The key point is that the observed max shift does not follow the nominal total
contribution. It stays at 25. Therefore, once a single `+32` MMA update is too
small to affect `2^26`, repeating the same update does not accumulate hidden
sub-ULP contributions. Each MMA update is rounded or truncated at the same
effective 21-bit boundary.

## Facts Established

| Fact | Evidence |
| --- | --- |
| The probes execute direct PTX, not cuBLASLt | The test source uses inline `mma.sync.aligned` assembly and has no cuBLASLt dependency. |
| The instruction shape is `m16n8k32` | All tested PTX strings use `m16n8k32`. |
| The exact contribution is 32 | A and B operands are all encoded as `1.0`, so `K = 32` products each contribute `1`. |
| FP8, FP6, and FP4 have the same measured boundary | All result files report max changed shift 25 and observed width 21. |
| Chained repeated MMA confirms the boundary independently | Repeat counts 2/4/8/16 all keep max shift 25 and per-MMA observed width 21. |
| The binary32 reference boundary would be shift 28 | `log2(32) + 24 - 1 = 28`. |
| The tested low-precision QMMA paths are 3 bits below binary32 for this probe | Observed 21 bits versus binary32 reference 24 bits. |

## SASS Verification

`cuobjdump --dump-sass` confirms that the compiled binaries contain direct QMMA
tensor-core instructions:

| Target | Confirmed SASS mnemonic |
| --- | --- |
| `test-5090-fp8-reduction-width` | `QMMA.16832.F32.E4M3.E4M3` |
| `test-5090-fp8-reduction-width` | `QMMA.16832.F32.E5M2.E5M2` |
| `test-5090-fp8-reduction-width` | `QMMA.16832.F32.E4M3.E5M2` |
| `test-5090-fp8-reduction-width` | `QMMA.16832.F32.E5M2.E4M3` |
| `test-5090-fp6-reduction-width` | `QMMA.16832.F32.E2M3.E2M3` |
| `test-5090-fp6-reduction-width` | `QMMA.16832.F32.E3M2.E3M2` |
| `test-5090-fp4-reduction-width` | `QMMA.16832.F32.E2M1.E2M1` |

The repeated-MMA binaries are compiled from the same direct-PTX source with
`TCNB_REDUCTION_REPEAT` and contain the same QMMA instruction families.

## Verification Commands

The report data can be reproduced with:

```sh
make -B test-5090-fp8-reduction-width \
        test-5090-fp6-reduction-width \
        test-5090-fp4-reduction-width \
        test-5090-fp8-reduction-repeat \
        test-5090-fp6-reduction-repeat \
        test-5090-fp4-reduction-repeat

python3 run_tests.py -o 5090 \
        5090-fp8-reduction-width \
        5090-fp6-reduction-width \
        5090-fp4-reduction-width \
        5090-fp8-reduction-repeat \
        5090-fp6-reduction-repeat \
        5090-fp4-reduction-repeat
```

The instruction selection can be checked with:

```sh
cuobjdump --dump-sass test-5090-fp8-reduction-width
cuobjdump --dump-sass test-5090-fp6-reduction-width
cuobjdump --dump-sass test-5090-fp4-reduction-width
cuobjdump --dump-sass test-5090-fp8-reduction-repeat
cuobjdump --dump-sass test-5090-fp6-reduction-repeat
cuobjdump --dump-sass test-5090-fp4-reduction-repeat
```

Memory correctness can be checked with:

```sh
compute-sanitizer --tool memcheck ./test-5090-fp8-reduction-width
compute-sanitizer --tool memcheck ./test-5090-fp6-reduction-width
compute-sanitizer --tool memcheck ./test-5090-fp4-reduction-width
compute-sanitizer --tool memcheck ./test-5090-fp8-reduction-repeat
compute-sanitizer --tool memcheck ./test-5090-fp6-reduction-repeat
compute-sanitizer --tool memcheck ./test-5090-fp4-reduction-repeat
```

## Interpretation

For this direct-PTX boundary probe, RTX 5090 low-precision QMMA paths for FP8,
FP6, and FP4 expose a 21-bit effective addition/reduction width per MMA update.
That is consistent across all seven tested input type combinations.

The root cause of the observed binary32 mismatch is not input conversion error:
all inputs are exactly representable as `1.0`, the exact contribution is
`32`, and the SASS contains direct tensor-core QMMA instructions. The mismatch
comes from the low-precision QMMA accumulation/reduction path itself. It does
not preserve the `+32` contribution through the same shift range as a full
binary32 significand path.

The chained-MMA cross-check strengthens this interpretation. When the same MMA
is repeated 2/4/8/16 times, the boundary remains at shift 25 instead of moving
with the nominal total contribution. That means sub-threshold updates are not
being stored and later recovered by repeated accumulation. The visible boundary
is applied at each MMA update.

## Limitations

- This is a boundary probe for one `m16n8k32` instruction and for dependent
  chains of that same instruction, not a full GEMM accuracy characterization.
- The measurement reports an observable effective width, not the physical
  topology or exact internal staging of the hardware reduction tree.
- The tested FP6 and FP4 instructions use the PTX `kind::f8f6f4` form. Compact
  block-scaled MXFP4/MXFP6 instructions are not covered.
- The input pattern is intentionally simple: all A and B values are `1.0`.
  Other value distributions can exercise different rounding and normalization
  cases.

## Summary

The direct PTX probes show that RTX 5090 FP8, FP6, and FP4 tensor-core MMA
instructions tested here all have a measured 21-bit effective reduction/addition
boundary per `m16n8k32` MMA update. The repeated-MMA cross-check confirms the
same per-update boundary in a different setting. The binary32 reference for the
same boundary probe is 24 bits, so these low-precision QMMA paths are
consistently 3 effective bits narrower than binary32 for this measurement.
