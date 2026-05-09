# Number-Pattern Method in `tc_test_numerics-T4-A100-binary16-details.cu`

Date: 2026-05-09

## Purpose

This note explains how `src/tc_test_numerics-T4-A100-binary16-details.cu`
uses carefully chosen number patterns to probe tensor-core dot-product
reduction behavior. The key idea is different from a simple boundary scan such
as `C = 2^shift + K`: it places tiny terms, large cancelling terms, and an
external accumulator in positions that turn hidden internal precision decisions
into visible output bit patterns.

The method is useful inspiration for reduction-width tests because it can
separate several effects:

- final binary32 rounding,
- whether a tiny product survives inside the dot-product reduction,
- whether cancellation happens before or after a tiny term is dropped,
- whether the dot-product result is normalized before adding the accumulator,
- whether different product positions appear to follow different reduction
tree paths.

The method still reports externally observable behavior. It does not directly
name the physical internal reduction tree topology.

## Execution Model Used by the File

The test runs a single WMMA operation:

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
wmma::fragment<wmma::accumulator, 16, 16, 16, returntype> c_fragment;
...
wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
```

Only the first few elements of A and B are nonzero. For output `c[0]`, this
creates a short controlled dot product inside the full `m16n16k16` WMMA tile:

```text
c[0] = C + a[0]b[0] + a[1]b[1] + a[2]b[2] + a[3]b[3]
```

The helper prints the actual half input encodings, the initial float
accumulator, the tensor-core result, and a simple host float reference:

```cpp
printf("\nh_a: ");  // half input bit patterns
printf("\nh_b: ");
printf("\nf_c: ");  // accumulator bit pattern
printf("\nr  : ");  // tensor-core result
printf("\nref: ");  // host float reference
```

This is important: the test is not only checking numerical equality. It checks
exact result bits such as `0x33800000` for `2^-24` or `0x00000000` for zero.

## Pattern Family 1: Final Rounding Calibration

The first pattern checks what happens when a tiny term is added to `1`:

```cpp
fa[0] = ldexp(1.f, -23);
fb[0] = 1.0f;
fa[3] = 1.0f;
fb[3] = 1.0f;
```

This computes:

```text
2^-23 + 1
```

The next case changes the tiny term to `2^-24`:

```text
2^-24 + 1
```

For binary32 around `1`, one ULP is `2^-23`. Therefore:

| Pattern | Expected binary32 result | Meaning |
| --- | --- | --- |
| `1 + 2^-23` | `0x3f800001` | One ULP above 1. |
| `1 + 2^-24` | `0x3f800000` | Half-ULP tie rounds back to 1 under round-to-nearest-even. |

This calibrates the final output rounding boundary. It tells us what output
bits to expect when the internal computation delivers an exactly representable
or halfway value to the final binary32 accumulator path.

## Pattern Family 2: Product Precision Calibration

The source also tests:

```cpp
fa[0] = ldexp(1.f, -12);
fb[0] = ldexp(1.f, -12);
fa[3] = 1.0f;
fb[3] = 1.0f;
```

This computes:

```text
(2^-12 * 2^-12) + 1 = 2^-24 + 1
```

This case is paired with the direct `2^-24 + 1` case. If both behave the same,
then the product `2^-12 * 2^-12` is effectively available as `2^-24` before
the final addition to `1`. If they differ, the product datapath is losing or
rounding information before the sum sees it.

This is a useful pattern-design rule:

```text
Test the same small value once as an input value and once as a product.
Differences isolate product-path precision from sum-path precision.
```

## Pattern Family 3: Cancellation as a Precision Amplifier

The most important pattern for reduction-width inspiration is:

```text
M - M + epsilon
```

For exact arithmetic, the result is `epsilon`. If the reduction path first
combines `epsilon` with `M` using limited internal precision, `epsilon` may be
dropped before the later `-M` cancellation. The final output then becomes zero.

The source uses examples such as:

```cpp
fa[0] = 1.0f;
fb[0] = ldexp(1.f, 4);    // +2^4
fa[2] = -1.0f;
fb[2] = ldexp(1.f, 4);    // -2^4
fa[3] = ldexp(1.f, -24);  // epsilon
fb[3] = 1.0f;
```

This computes:

```text
2^4 - 2^4 + 2^-24
```

Then the test increases `epsilon`:

```text
2^4 - 2^4 + 2^-23
2^4 - 2^4 + 2^-22
2^4 - 2^4 + 2^-21
2^4 - 2^4 + 2^-20
```

On the recorded RTX 5090 output, the first three are zero, and the last two
survive:

| Pattern | 5090 result bits | Interpretation |
| --- | --- | --- |
| `2^4 - 2^4 + 2^-24` | `0x00000000` | Tiny term lost before cancellation. |
| `2^4 - 2^4 + 2^-23` | `0x00000000` | Tiny term lost before cancellation. |
| `2^4 - 2^4 + 2^-22` | `0x00000000` | Tiny term lost before cancellation. |
| `2^4 - 2^4 + 2^-21` | `0x35000000` | Tiny term survives. |
| `2^4 - 2^4 + 2^-20` | `0x35800000` | Tiny term survives. |

The threshold gives an effective retention width for that specific reduction
path. If a term of magnitude `2^s` must survive while aligned with a term of
magnitude `2^E`, then the approximate width is:

```text
p = E - s + 1
```

For the RTX 5090 `2^4` sweep above, the smallest surviving term is `2^-21`:

```text
E = 4
s = -21
p = 4 - (-21) + 1 = 26
```

This should be read carefully. It is not a universal "tensor-core width" by
itself. It is the effective retention width for that particular input layout,
WMMA instruction, output element, and cancellation path. The value is useful
because it is obtained by a very different method from the `C = 2^shift + K`
boundary probe.

## Pattern Family 4: Sweeping the Large Cancelling Magnitude

The file also changes `M` while keeping `epsilon = 2^-24`:

```text
2 - 2 + 2^-24
2^4 - 2^4 + 2^-24
2^16 - 2^16 + 2^-24
2^24 - 2^24 + 2^-24
2^29 - 2^29 + 2^-24
```

This is a second axis of the same experiment. Instead of increasing the tiny
term, it increases the distance between the large cancelling terms and the
tiny term.

The interpretation is:

| Sweep axis | What it reveals |
| --- | --- |
| Increase `epsilon` at fixed `M` | Smallest low-order term retained for a fixed large term. |
| Increase `M` at fixed `epsilon` | Largest alignment gap tolerated before the small term is lost. |

Together, these two sweeps can locate the effective low-side retention boundary
more robustly than one isolated pattern.

## Pattern Family 5: Normalization Before Adding the Accumulator

The second function, `my_test_normalize()`, uses a different structure:

```cpp
fa[0] = 1.0f;
fb[0] = 1.0f;
fa[1] = -1.0f;
fb[1] = 1.0f;
h_c[0] = ldexp(1.f, -k);
```

The dot product is:

```text
1 - 1 = 0
```

The accumulator is:

```text
C = 2^-k
```

If the dot-product result is normalized to exact zero before adding `C`, the
output should be `C`. If the implementation keeps an internal representation
that cannot shift far enough left after cancellation, very small `C` values can
disappear.

Recorded RTX 5090 output:

| Accumulator pattern | 5090 result bits | Meaning |
| --- | --- | --- |
| `C = 2^-23` | `0x34000000` | Preserved. |
| `C = 2^-24` | `0x33800000` | Preserved. |
| `C = 2^-25` | `0x33000000` | Preserved on 5090. |
| `C = 2^-26` | `0x00000000` | Lost. |
| `C = 2^-40` | `0x00000000` | Lost. |

This pattern does not measure the same thing as `M - M + epsilon`. It asks
whether the cancelled dot product is normalized before the external accumulator
is added. That is another way to expose internal width and staging decisions.

## Why These Patterns Are Good for Reduction-Width Work

The key trick is cancellation:

```text
large + small - large
```

If the small term is retained internally, the final result is small and nonzero.
If the small term is dropped before cancellation, the final result is zero.

This turns a subtle internal precision question into a binary output question:

```text
nonzero epsilon  => enough internal retention on that path
zero             => epsilon was dropped before it could be recovered
```

Compared with a plain `2^shift + K` scan, these patterns can reveal more:

| Method | Main signal | Strength | Limitation |
| --- | --- | --- | --- |
| `C = 2^shift + K` boundary scan | When a known contribution stops changing C | Simple, stable, easy to convert to bits | Mostly measures one output boundary. |
| `M - M + epsilon` cancellation | Whether a tiny product survives before cancellation | Exposes internal retention and tree association | Requires knowing product placement and instruction layout. |
| `1 - 1 + C` normalization | Whether cancelled dot product is normalized before adding C | Probes staging between dot product and accumulator add | Measures a different stage from product reduction. |

## How This Can Inspire FP8/FP6/FP4 Tests

The current RTX 5090 FP8/FP6/FP4 report measures a 21-bit per-MMA update
boundary using direct PTX and accumulator scans. A pattern-based follow-up could
borrow the binary16-details structure:

1. Map product positions for one output lane.
   Toggle one packed low-precision product at a time and record which output
   register changes. This establishes where each `A[k] * B[k]` term lands.

2. Build cancellation triplets.
   Use patterns of the form:

   ```text
   2^E - 2^E + 2^s
   ```

   Sweep `s` at fixed `E`, then sweep `E` at fixed `s`.

3. Permute product positions.
   Move `+2^E`, `-2^E`, and `2^s` across different K positions. If the
   survival threshold changes, that is evidence that different terms enter
   different reduction-tree branches or stages.

4. Separate product precision from sum precision.
   Test the same tiny value as a direct representable input product and as a
   product of two smaller representable powers, when the format supports it.

5. Add accumulator normalization probes.
   Use:

   ```text
   dot product = 1 - 1
   C = 2^-k
   ```

   Sweep `k` to see whether a cancelled dot product is normalized before the
   external accumulator add.

For FP8 this is feasible, but a direct A-only power sweep can still be
misleading. E4M3 has much more useful range when the small and large terms are
formed as A/B products rather than by varying A while holding B at `1.0`. The
direct-PTX follow-up therefore uses product terms, not just input terms:

```text
M       = A_large * B_large
epsilon = A_small * B_small
```

For FP6 and especially FP4, the product range is still small enough that the
probe can become format-limited. In that case the result is a lower bound on
retention for the tested layout, not a measured failure boundary.

## Direct-PTX Follow-Up Outcome

The follow-up implementation is
`src/tc_test_numerics-5090-lowp-reduction-pattern.cu`, with detailed results in
`5090/lowp-product-pattern-width-report.md` and the matching TeX/PDF report.

It maps independent product positions for output `D0` and uses:

```text
D0 = 4 * M - 4 * M + 4 * epsilon
```

The important outcome is:

| Format pair | Extreme product width | Max effective width | Interpretation |
| --- | ---: | ---: | --- |
| FP8 E4M3 x E4M3 | 36 | 27 | Product range exposes a non-format-limited boundary. |
| FP8 E5M2 x E5M2 | 64 | 27 | Wider range, same retained-width boundary. |
| FP8 E4M3 x E5M2 | 50 | 27 | Mixed format, same retained-width boundary. |
| FP8 E5M2 x E4M3 | 50 | 27 | Mixed format, same retained-width boundary. |
| FP6 E2M3 x E2M3 | 12 | 12 | Format-limited; extreme case survives. |
| FP6 E3M2 x E3M2 | 18 | 18 | Format-limited; extreme case survives. |
| FP4 E2M1 x E2M1 | 8 | 8 | Format-limited; extreme case survives. |

This resolves the main pitfall from a direct translation of the binary16
pattern: FP8 E4M3 must be tested through product dynamic range. Holding B at
`1.0` understates the testable width.

## Cautions

- These patterns are layout-sensitive. The same mathematical expression can
  behave differently if the products occupy different K positions.
- WMMA does not expose the physical product ordering directly. A direct-PTX
  version should first map operand packing and output register placement.
- The method measures externally visible behavior, not a literal schematic of
  the hardware reduction tree.
- Input format limits matter. Binary16 can express subnormals down to `2^-24`.
  FP8/FP6/FP4 may not provide the same tiny values directly.
- A final zero does not always mean the whole tensor-core path lacks precision;
  it means the tested tiny term was not retained through the particular path
  exercised by that pattern.
- A retained nonzero value does not imply exact arithmetic. In the FP8
  follow-up, the widest retained 27-bit cases are near the 50% retention
  threshold and should be read as "epsilon remains visible", not as exact
  27-bit summation.

## Summary

`tc_test_numerics-T4-A100-binary16-details.cu` uses number patterns as precision
probes. The most important pattern is `M - M + epsilon`: exact arithmetic
returns `epsilon`, while a finite internal reduction path returns zero when
`epsilon` is dropped before cancellation. Sweeping `M`, sweeping `epsilon`, and
moving the terms across product positions can reveal an effective reduction
retention width and give clues about reduction-tree association.

This method is complementary to the existing RTX 5090 FP8/FP6/FP4
`C = 2^shift + K` probe. The boundary scan gives a clean per-MMA update width.
The pattern method can further test whether that same width appears inside the
dot-product reduction tree, whether it varies by product position, and whether
normalization before the accumulator add changes the visible result. The
direct-PTX product-pattern follow-up confirms that FP8 needs product-based
range construction and that FP6/FP4 are currently limited by the product ranges
the formats can express in this test.
