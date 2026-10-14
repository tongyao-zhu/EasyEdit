# Ouro integration: current scope

This branch starts from EasyEdit `4c109870955a4522ac3d7cf10ad00f34de8e4f0d`.
The first acceptance run validates **FT integration**, not editing quality or
compatibility of all EasyEdit algorithms.

## Behavior

- Native Ouro forward and parameter identities are preserved. Edited shared
  weights apply on every invocation. There are no per-loop edit gates.
- `FixedDepth(model, K)` explicitly selects the final output of K = 2, 3, or 4
  loops and restores settings on exit. KV caching is disabled pending separate
  validation; gradient checkpointing is not yet supported.
- Native Ouro's `labels` path computes a weighted-exit objective. The context
  rejects that path. The upstream FT `target_new` implementation computes its
  original CE directly from final-depth logits without passing model labels.
- `SharedCallTrace` observes actual `current_ut` and output-gradient norms. It
  does not choose the representation or intervention semantics of ROME/MEMIT.
- `load_ft()` imports the exact upstream FT files under a private package name
  to avoid unrelated eager vision/API dependencies. This is not yet a full
  `BaseEditor` integration. The FT algorithm itself is unchanged.

## Verified environment and acceptance

Python 3.10.19, PyTorch 2.6.0+cu124, Transformers 4.57.0; isolated venv inheriting
the existing packages. These versions differ from current upstream's full
requirements; compatibility with that complete dependency stack is untested.

CPU tests:

```sh
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_ouro_integration.py -v
```

The GPU acceptance CLI consumes three external, local-only inputs:

- `--model`: pinned local Ouro snapshot directory.
- `--request`: JSON containing only case_id, family_id, prompt, target_new, subject.
- `--model-manifest`: filename-to-SHA256 mapping for that snapshot.

```sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m ouro_integration.validate \
  --model "$OURO_MODEL" --request "$ATOMIC_REQUEST" \
  --model-manifest "$MODEL_MANIFEST" --output "$NEW_OUTPUT_DIRECTORY"
```

Run through `ouro_integration.guard` on an explicitly allocated node for a
finite timeout and persistent GPU allocation ledger. It refuses an existing
run directory and a busy GPU; it does not retry automatically.

On 2026-10-14, three CPU tests and one real-model acceptance passed on A100 80GB.
The diagnostic used FP32, no cache, the first case of the previously frozen
screening order, and two upstream FT steps on two full output matrices. K2/3/4
logits matched independently configured native inference exactly; every loop
received gradients; only intended matrices changed; all model parameter hashes
matched after rollback; hooks were removed. Peak allocated memory was 5.959 GiB
and the complete one-GPU allocation consumed 0.014724 GPU-hours. These numbers
describe this short one-case diagnostic, not a complete benchmark budget.

## Not yet run

Scientific FT baseline settings and full-population evaluation remain to be
specified and frozen. ROME/MEMIT/AlphaEdit require an explicit audit of repeated
module captures, target optimization and moment statistics before adaptation
can be called faithful. MEND needs a compatible trained editor or separately
scoped editor training. No uniform one-matrix/rank-one restriction is imposed.
No evaluation questions or downstream answers were used as supervision.
