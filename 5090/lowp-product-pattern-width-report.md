# RTX 5090 Low-Precision Product-Pattern Width Report

Date: 2026-05-09

## Summary

This report records the standalone direct-PTX product-pattern probe in
`src/tc_test_numerics-5090-lowp-reduction-pattern.cu`. It uses independent
`A*B` product terms to compute:

```text
M - M + epsilon
```

This is different from the accumulator-boundary probe. Both A and B provide
dynamic range, so FP8 E4M3 is no longer limited to the earlier A-only
exact-power range.

Headline results:

| Family | Variant | Extreme width | Extreme survived | Max effective width | Format-limited |
| --- | --- | ---: | --- | ---: | --- |
| FP8 | E4M3 x E4M3 | 36 | no | 27 | no |
| FP8 | E5M2 x E5M2 | 64 | no | 27 | no |
| FP8 | E4M3 x E5M2 | 50 | no | 27 | no |
| FP8 | E5M2 x E4M3 | 50 | no | 27 | no |
| FP6 | E2M3 x E2M3 | 12 | yes | 12 | yes |
| FP6 | E3M2 x E3M2 | 18 | yes | 18 | yes |
| FP4 | E2M1 x E2M1 | 8 | yes | 8 | yes |

## Method

The test first maps independent product positions for output register `D0`.
The selected product terms are:

| Role | Position | Weight in D0 |
| --- | --- | ---: |
| `+M` | `A0.byte0 * B0.byte0` | 4 |
| `-M` | `A0.byte1 * B0.byte1` | 4 |
| `epsilon` | `A0.byte2 * B0.byte2` | 4 |

The cross terms among those selected positions are checked to be absent for
`D0`, so the pattern isolates three product contributions:

```text
D0 = 4 * M - 4 * M + 4 * epsilon
```

The test enumerates every positive finite raw value in the A format and every
positive finite raw value in the B format. It forms all positive products,
groups them by product exponent, and scans exponent gaps:

```text
effective width = exponent(M) - exponent(epsilon) + 1
```

For each product exponent bucket, the test uses a large product candidate for
`M` and a product candidate for `epsilon`. A case counts as retained when:

```text
observed > 0
abs(observed - expected) <= 0.5 * abs(expected)
```

The 50% tolerance is intentional. This pattern is used to detect whether the
small term is still visible after cancellation, not to require exact arithmetic
on the retained term. Borderline FP8 cases are therefore reported explicitly.

## Full Results

| Variant | Tested exponent pairs | Survived pairs | Max product | Min product | Extreme width | Max effective width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 E4M3 x E4M3 | 666 | 612 | 200704 | 3.81469727e-06 | 36 | 27 |
| FP8 E5M2 x E5M2 | 2080 | 1332 | 3.28833434e+09 | 2.32830644e-10 | 64 | 27 |
| FP8 E4M3 x E5M2 | 1275 | 976 | 25690112 | 2.98023224e-08 | 50 | 27 |
| FP8 E5M2 x E4M3 | 1275 | 976 | 25690112 | 2.98023224e-08 | 50 | 27 |
| FP6 E2M3 x E2M3 | 78 | 78 | 56.25 | 0.015625 | 12 | 12 |
| FP6 E3M2 x E3M2 | 171 | 171 | 784 | 0.00390625 | 18 | 18 |
| FP4 E2M1 x E2M1 | 36 | 36 | 36 | 0.25 | 8 | 8 |

## Extreme Cases

The "extreme" case uses the largest finite product for `M` and the smallest
positive product for `epsilon`.

| Variant | Extreme width | Extreme observed | Extreme expected | Survived |
| --- | ---: | ---: | ---: | --- |
| FP8 E4M3 x E4M3 | 36 | 0 | 1.52587891e-05 | no |
| FP8 E5M2 x E5M2 | 64 | 0 | 9.31322575e-10 | no |
| FP8 E4M3 x E5M2 | 50 | 0 | 1.1920929e-07 | no |
| FP8 E5M2 x E4M3 | 50 | 0 | 1.1920929e-07 | no |
| FP6 E2M3 x E2M3 | 12 | 0.0625 | 0.0625 | yes |
| FP6 E3M2 x E3M2 | 18 | 0.015625 | 0.015625 | yes |
| FP4 E2M1 x E2M1 | 8 | 1 | 1 | yes |

## Max-Width Retained Cases

These are the widest retained cases found by the product-exponent scan.

| Variant | Width | M exponent | Epsilon exponent | Observed | Expected |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP8 E4M3 x E4M3 | 27 | 17 | -9 | 0.0078125 | 0.0153808594 |
| FP8 E5M2 x E5M2 | 27 | 31 | 5 | 128 | 240 |
| FP8 E4M3 x E5M2 | 27 | 24 | -2 | 1 | 1.96875 |
| FP8 E5M2 x E4M3 | 27 | 24 | -2 | 1 | 1.96875 |
| FP6 E2M3 x E2M3 | 12 | 5 | -6 | 0.0625 | 0.0625 |
| FP6 E3M2 x E3M2 | 18 | 9 | -8 | 0.015625 | 0.015625 |
| FP4 E2M1 x E2M1 | 8 | 5 | -2 | 1 | 1 |

The FP8 max-width retained cases are edge cases: the observed value is roughly
half of the expected epsilon contribution, but still positive and within the
retention tolerance. This means "27 bits" should be read as the widest
detectable retained epsilon under this criterion, not exact 27-bit arithmetic.

## Format Analysis

### FP8 E4M3 x E4M3

The earlier A-only pattern understated the available range. With A and B both
participating, the extreme range is:

```text
max product = 200704
min product = 3.81469727e-06
product exponent gap = 17 - (-18) + 1 = 36
```

The extreme case is lost (`observed = 0`). The widest retained case is 27 bits,
with `M` at exponent 17 and `epsilon` at exponent -9. This is not
format-limited.

### FP8 E5M2 x E5M2

E5M2 has a much wider exponent range than E4M3, so the extreme product width
reaches 64 bits. The extreme case is lost, and the widest retained case is
again 27 bits. Increasing the representable product range does not increase the
observed retained width for this pattern.

### FP8 Mixed E4M3/E5M2

Both mixed variants reach an extreme width of 50 bits and a max effective width
of 27 bits. The two directions match in this selected layout, but that should
not be generalized to every possible operand placement or output lane without
additional layout sweeps.

### FP6 E2M3 x E2M3

The entire tested product range is only 12 bits, and the extreme case survives
exactly. This result is format-limited: it proves the tested path preserves the
full E2M3 product range used here, but it does not prove the internal retention
width is only 12 bits.

### FP6 E3M2 x E3M2

The tested range expands to 18 bits because E3M2 has more exponent range than
E2M3. The extreme case survives exactly, so this is also format-limited. The
test does not find a failing FP6 E3M2 product-pattern case.

### FP4 E2M1 x E2M1

The tested product range is 8 bits. The extreme case survives exactly. This is
strong evidence that the tested path can retain every product gap expressible
by FP4 E2M1 in this layout, but it cannot bound a wider internal tree because
the format cannot generate a wider product gap.

## Edge Cases and Limitations

- The result is layout-sensitive. The selected product positions are
  independent for `D0`, but other positions, lanes, or output registers may
  have different association behavior.
- FP8's 27-bit retained cases are marginal by value: they pass because epsilon
  remains positive and within the 50% retention tolerance, not because the
  result equals the expected value.
- FP6 and FP4 are format-limited in this test. Their max effective width is the
  largest width the product format can generate here, not necessarily the
  physical internal reduction width.
- The grouping key is the product exponent. Products in the same exponent
  bucket can have different significands; the test intentionally reports an
  externally visible retention boundary, not a full exact arithmetic proof.
- This test uses direct `mma.sync.aligned.m16n8k32` PTX and one selected output
  register. It does not cover block-scaled MXFP instructions or full GEMM
  accuracy.

## Key Correction

The important correction is that FP8 E4M3 should be tested through products,
not through A-only exact powers with `B = 1`. Product terms make the E4M3 x
E4M3 extreme width 36 bits. The observed retained width under the current
product-pattern criterion is 27 bits.

## Verification

Commands run:

```sh
make -B test-5090-fp8-reduction-pattern \
        test-5090-fp6-reduction-pattern \
        test-5090-fp4-reduction-pattern

python3 run_tests.py -o 5090 \
        5090-fp8-reduction-pattern \
        5090-fp6-reduction-pattern \
        5090-fp4-reduction-pattern

compute-sanitizer --tool memcheck ./test-5090-fp8-reduction-pattern
compute-sanitizer --tool memcheck ./test-5090-fp6-reduction-pattern
compute-sanitizer --tool memcheck ./test-5090-fp4-reduction-pattern
```

SASS contains the expected direct tensor-core instructions:

| Target | Confirmed instruction family |
| --- | --- |
| `test-5090-fp8-reduction-pattern` | `QMMA.16832.F32.E4M3/E5M2` |
| `test-5090-fp6-reduction-pattern` | `QMMA.16832.F32.E2M3/E3M2` |
| `test-5090-fp4-reduction-pattern` | `QMMA.16832.F32.E2M1` |
