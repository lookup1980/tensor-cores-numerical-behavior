# Tensor Cores Numerical Behavior Skill

Use this skill when working in this repository on CUDA tensor-core numerical
behavior experiments, result interpretation, or reproducibility documentation.

## Core Context

The repository implements microbenchmarks for the paper "Numerical Behavior of
the NVIDIA Tensor Cores" (PeerJ Computer Science 7:e330, 2021,
DOI: 10.7717/peerj-cs.330). The paper uses carefully chosen WMMA matrix
multiply-accumulate inputs to infer undocumented tensor-core behavior:

- whether subnormal inputs and outputs are preserved,
- whether products are exact before accumulation,
- which precision is used for accumulation,
- how rounding behaves for positive and negative values,
- how partial sums are ordered and normalized,
- whether extra carry-out or guard digits exist,
- and whether observed tensor-core products are monotonic.

## Workflow

1. Read `README.md`, `Makefile`, and the relevant `tc_test_numerics-*.cu` file
   before changing behavior.
2. Identify the target architecture and numeric mode from file and executable
   names. Examples: `test-A100-tf32`, `test-H100-binary16`,
   `test-4090-bf16`, `test-5090-tf32`.
3. Keep generated artifacts separate from archived data:
   - root `test-*` files are build products,
   - root `result-*.txt` files are generated run output,
   - `3080/`, `4090/`, `5090/`, and `A10/` contain archived results.
4. When extending an experiment, preserve the existing output style:
   section headers, `printitem(...)` checks, and `[PASS]` or `[FAIL]` lines.
5. Prefer targeted Makefile additions over changing existing architecture
   targets. Add new targets only when the source file and compute capability are
   known.
6. Verify source changes with the narrowest relevant `make` target. Run
   `python3 run_tests.py` only when runtime results are required and the local
   GPU is appropriate for the target binaries.

## Interpretation Rules

- Do not assume IEEE 754 scalar behavior applies to tensor cores. The purpose of
  the tests is to expose differences in tensor-core multiply-accumulate units.
- Do not generalize a result from one GPU generation to another without a
  matching executable target and runtime output.
- If CUDA, driver, or GPU hardware differs from archived results, report that
  difference alongside any generated output.
- Treat discrepancies as experimental observations first. Check architecture,
  compute capability, CUDA version, source target, and archived output before
  editing predicates.

## Common Commands

```sh
make test-A100
make test-H100
make test-4090
make test-5090
python3 run_tests.py
make clean
git status --short
```
