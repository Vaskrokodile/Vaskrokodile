# GPU.ai RLVR checkpoint — 2026-09-09

This folder is the continuation package for the pod `a100-80gb-1tb-aiaah`.

## Integrity

`rlvr_checkpoint_20260909.tar` is a verified copy of the pod snapshot.

SHA-256:

```text
e02cf2c1472c7dcb62d4e0d4708a81ff1ee17213ff06f7bfc11db2701e74ae84
```

The extracted copy is under `pod_snapshot/root/`. It contains the corrected sampler, GRPO trainer, verifiers, task manifests, all GRPO rollout and reward JSONL files, validation artifacts, and `grpo_pilot_adapter`, `grpo_v2_adapter`, and `grpo_v3_adapter`.

The base checkpoint is referenced by the adapters as `/root/models/AIAAH-1`. Only its small `config.json` and `tokenizer_config.json` are in the snapshot; the base weights should be restored from the original model source before loading an adapter.

## Current result

Raw AIAAH-1 remains the best policy. The strongest measured improvement is inference-time search: 4 raw rollouts on 500 untouched MBPP test tasks reached 72.2% pass@4 and 0.7893 mean best-of-four correctness. The trained GRPO adapters are preserved for analysis but were not promoted: on 90 MBPP validation tasks × 4, raw base scored 222/360 full passes and mean correctness 0.6704; the corrected v3 adapter scored 213/360 and 0.6491.

The v3 run is the only correctly aligned GRPO run. Earlier runs had a prompt-token BOS mismatch or too few informative groups. The report documents the diagnosis and the resume recipe in `pod_snapshot/root/rlvr_next_round_report.md`.

## Resume

1. Restore `/root/models/AIAAH-1`.
2. Copy the files from `pod_snapshot/root/` to `/root/`.
3. Run `python3 -m py_compile /root/rollout_policy.py /root/grpo_train.py`.
4. Use `grpo_var_tasks.txt` for MBPP train collection only.
5. Always run a raw-base control with identical seed, prompt, response cap, and sample count before evaluating an adapter.
6. Do not train on MBPP validation/test or benchmark artifacts.

## Publication

No Hugging Face or GitHub credentials were configured on the old PC, so external publication could not be completed from this session. The source/report copy is prepared in the local Git repository under `Vaskrokodile/rlvr_gpt_controller/`; authenticate on the next PC, run `hf upload`, and push the prepared commit.
