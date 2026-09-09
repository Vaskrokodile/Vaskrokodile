# RLVR Next Round Engineering Report

## Decision

The next round should keep the central architecture: one small policy model produces executable attempts at high throughput, an external GPT controller supplies reasoning and repair decisions, and a deterministic sandbox supplies the reward. The current DPO adapter path should be treated as an ablation, not the main trainer.

The highest-confidence improvement is to replace pairwise offline DPO with a single-policy, on-policy group update. Each group contains multiple attempts for one task, the verifier returns a graded executable score, and the policy receives a group-relative token-level advantage. The live system still holds one trainable model; rollout log-probabilities and reward metadata are carried in the data buffer rather than loading a second policy model.

## Evidence from the current pod

The raw AIAAH-1 checkpoint is currently the best policy under the corrected controller protocol.

| Policy | Pass@1 full solves | Mean per-test correctness |
|---|---:|---:|
| Raw AIAAH-1 | 302/500 = 60.4% | 0.6760 |
| Existing DPO parent | 298/500 = 59.6% | 0.6747 |
| Online DPO, raw prompt | 298/500 = 59.6% | 0.6720 |
| Online DPO, chat-aligned | 298/500 = 59.6% | 0.6760 |
| Two-epoch base DPO | 292/500 = 58.4% | 0.6613 |

Raw AIAAH-1 with four independent fast rollouts reaches 72.2% pass@4 and 0.7893 mean best-of-four test fraction. This is the strongest measured gain and confirms that the controller/search path is productive even before policy-weight improvement.

The 2,400-sample hard-task batch generated from raw AIAAH-1 produced 1,277 fully passing samples, mean correctness 0.6026, and 619 preference pairs over 113 tasks. The two-epoch DPO run fit those pairs well, but lost 4.1 points on held-out full solves. That is a generalization and objective failure, not a shortage of samples.

## What the literature changes

DeepSeekMath introduced GRPO as a memory-saving alternative to PPO for verifiable rewards; the key idea is to estimate advantages relative to other samples from the same prompt rather than training a separate critic. See [DeepSeekMath](https://arxiv.org/abs/2402.03300).

The analysis of GRPO shows that verifiable rewards can amplify success probability when the data is sampled on-policy and the update is KL-regularized. It also identifies the importance of the old-policy and reference-policy relationship. See [GRPO's Effective Loss, Dynamics, and Success Amplification](https://arxiv.org/abs/2503.06639).

DAPO adds the pieces our loop is missing: asymmetric clip bounds, dynamic sampling that filters groups with all-zero or all-one rewards, token-level loss aggregation, and soft penalties for overlong responses. The authors report that filtering homogeneous groups preserves usable gradient signal and that clip-higher improves entropy and diversity. See [DAPO](https://arxiv.org/abs/2503.14476) and the [official recipe](https://github.com/verl-project/verl-recipe/tree/main/dapo).

Dr. GRPO explains why naive GRPO can make incorrect outputs longer. This directly matches our earlier think-only truncation failure. We should measure response length and truncation as first-class metrics and avoid rewarding long unsuccessful traces. See [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783).

DRIVE recommends a two-stage code-RL schedule: broad uniform coverage with eight rollouts per prompt, then a hard-focused stage with a larger rollout budget and continuous retention of difficult instances. Its findings that difficult-case learning generalizes better than repeatedly training easy cases match our adaptive task selector. See [DRIVE](https://arxiv.org/abs/2511.06307).

For multi-turn code repair, μCode treats code generation as a recoverable one-step problem after execution feedback. ReVeal and R1-Code-Interpreter similarly show that explicit execution or self-verification turns code generation into a richer environment. This supports adding a repair branch after failed attempts while keeping the direct short-code path. See [μCode](https://arxiv.org/abs/2502.20380), [ReVeal](https://arxiv.org/abs/2506.11442), and [R1-Code-Interpreter](https://arxiv.org/abs/2505.21668).

Reasoning Arena is especially relevant to our DPO failure: when a group has equal executable scores, it routes traces to a small tournament rather than discarding them, uses anchor comparisons, and fits a Bradley-Terry ranking on the sparse comparison graph. That is a clean place for the external GPT controller to add signal without becoming a second policy model. See [Reasoning Arena](https://arxiv.org/abs/2606.09380).

SPADE extends the idea further by making environment generation adaptive. We should not begin there, but its lesson is useful: once fixed MBPP tasks saturate, generate new executable tasks near the policy's capability frontier instead of adding more easy examples. See the [SPADE repository](https://github.com/spade-rl/spade).

## Framework comparison

| System | Useful idea for this project | What to adopt now |
|---|---|---|
| slime | Explicit data buffer, custom generation and reward hooks, token-correct multi-turn trajectories, sandboxed coding-agent example | Copy the data contract and trajectory provenance; keep our lightweight single-GPU trainer first |
| PRIME-RL | Async rollout, first-class verifiers environments, modular algorithms, single-GPU debug path | Use its verifier/task schema and dynamic group semantics as the target interface |
| Meshy | Role-driven services, single-policy SPMD design, pluggable dataset/reward/advantage modules | Borrow its role separation concept without importing its distributed stack |
| verl DAPO | Dynamic sampling, token-level loss, clip-higher, overlong reward shaping | Implement the four mechanisms in the current custom trainer |

See [slime](https://github.com/THUDM/slime), [slime customization](https://github.com/THUDM/slime/blob/main/docs/en/get_started/customization.md), [slime coding-agent RL](https://github.com/THUDM/slime/blob/main/examples/coding_agent_rl/README.md), [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl), and [Meshy](https://github.com/OpenBMB/Meshy).

## Dataset decision

MBPP remains the calibration and regression set. It is too small and too narrow to be the main growth environment.

The primary next RL source should be a verifiable CodeForces subset. The [open-r1/codeforces dataset](https://huggingface.co/datasets/open-r1/codeforces) contains more than 10,000 problems, official tests, generated tests, executable metadata, and custom checkers for multi-solution tasks. It is a better environment than MBPP because correctness includes hidden large cases and algorithmic complexity.

For reasoning initialization and critique experiments, [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) provides about 2.5 million Python and C++ solution/critique samples over roughly 35,000 questions. The card says its question sources exclude the test splits of CodeContests and open-r1/codeforces, but we will still deduplicate by normalized statement and contest identifier before using it.

[open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) is useful for a small reasoning bridge, but its card says the traces were not correctness-filtered and only about 84% of Python samples pass public tests. It must be verified before training.

The [Kimi K3 coding and debugging traces](https://huggingface.co/datasets/greghavens/kimi-k3-coding-and-debugging-traces) are a useful small source of multi-turn repair and tool-use patterns. They are not a substitute for executable RL because the source corpus contains trajectories rather than a complete reward environment.

Fable-5 derivatives are useful for agent behavior research, but the dataset cards report AGPL-3.0 licensing and substantial context truncation. Community GPT-5.x and Kimi distillation mixtures have uneven provenance and licensing. They should be kept out of the first reproducible training run; use them only after a provenance and license review.

## Contamination firewall

The training and evaluation sets need separate manifests containing normalized task hashes, source identifiers, contest IDs, release dates, and prompt fingerprints.

1. Keep all MBPP test IDs outside every training artifact.
2. Reserve a fixed set of CodeForces contest IDs and all LiveCodeBench evaluation questions outside the pod's training cache.
3. Deduplicate across datasets after stripping markdown, whitespace, comments, and identifier names where safe.
4. Never use benchmark reference solutions, public hidden tests, or evaluation rollouts to build training pairs.
5. Use a fresh time-windowed evaluation set from [LiveCodeBench](https://github.com/livecodebench/LiveCodeBench), whose purpose is contamination-resistant evaluation over newly released coding problems.
6. Store a SHA-256 manifest before training and verify it after data preparation.

## Next training recipe

### Stage A: fast capability-frontier collection

Use raw AIAAH-1 as the parent and generate eight short direct-code attempts per task. Keep the forced output contract for the fast branch. Use a 4,096-token context so long statements never abort the batch, but cap the direct response at 512 or 768 tokens. Record completion token IDs, token log-probabilities, finish reason, and policy version.

Use dynamic sampling: retain a group only if its verifier rewards have nonzero variance and the group is not already saturated. Resample rejected groups until the batch contains the target number of informative groups. This avoids spending training steps on all-zero and all-one groups.

### Stage B: targeted repair branch

For groups with no full solution but at least one partial solution, expose failed test names and sanitized failure classes to the external GPT controller. The controller proposes a repair instruction or selects the most promising candidate. The policy then produces a new code attempt conditioned on that feedback. The sandbox runs the candidate in a fresh process with the same resource limits.

Store the full causal chain: original prompt, policy output tokens, verifier result, controller feedback, repaired output tokens, and final reward. Tool and controller text receives loss mask zero; policy-generated tokens receive loss mask one. This follows slime's token-provenance rule and keeps the external GPT role from becoming a hidden trainable model.

### Stage C: actual single-policy RL update

Implement GRPO or a Dr. GRPO-style token-level policy-gradient update over retained groups.

For each group, use the verifier score as the primary reward:

```text
reward = passed_tests / total_tests
          - truncation_penalty
          - compile_or_runtime_penalty
```

Normalize rewards within the group and use a mean-only or token-mean aggregation. Save old-policy log-probabilities at rollout time, so the trainer can use clipped importance ratios without loading a second live policy. Use a small adapter on the raw base, a low learning rate, and a KL or adapter-norm guardrail. The update should stop early if held-out MBPP validation falls or if pass@4 diversity collapses.

Initial ablation grid, one variable per run:

| Run | Groups | Samples/group | Response cap | Update | Purpose |
|---|---:|---:|---:|---|---|
| A | 128 | 8 | 768 | Dr. GRPO, 1 epoch | Baseline true on-policy update |
| B | 128 | 8 | 768 | DAPO filtering + clip-higher | Test diversity and usable gradients |
| C | 128 | 8 + repair | 768 | Same as B | Test controller-guided recovery |
| D | 256 | 8 + repair | 768 | Same as B | Test data/coverage scaling |

Do not choose a checkpoint from training reward. Choose by a fixed scorecard: MBPP validation pass@1, MBPP pass@4, unseen CodeForces pass@1, unseen CodeForces pass@4, mean verifier score, truncation rate, output length, and group reward entropy.

## Implementation order

1. Add a manifest builder and normalized deduplication before downloading any new training source.
2. Upgrade the rollout record to include token IDs, old-policy log-probabilities, policy version, and group ID.
3. Add dynamic group filtering and resampling.
4. Add a CodeForces environment adapter with isolated execution and generated-test support.
5. Implement the one-GPU GRPO update and run a 16-group smoke test where the loss direction is checked analytically.
6. Run Stage A on MBPP train only and compare against the raw-base control.
7. Add the repair branch and repeat the smoke test.
8. Run the CodeForces frontier batch, then evaluate on untouched MBPP test and a frozen LiveCodeBench window.

## Risks and stop conditions

The present checker uses process limits and fresh subprocesses but does not create a network namespace. CodeForces execution must add a stronger sandbox before untrusted submissions are run.

Stop a run if any of these occur: reward rises while held-out pass@1 falls by more than two points; response length rises without mean verifier improvement; pass@4 falls by more than two points; more than 5% of groups have malformed or truncated provenance; or a dataset hash overlaps the evaluation manifest.

The target for the next round is a real held-out bump from the raw-base control, with pass@4 retained or improved. A training run that only increases reward on the sampled hard tasks is not a win.

## Checkpoint from 2026-09-09

The working pod was `a100-80gb-1tb-aiaah` on `frp.gpu.ai:10101`. The base policy was `/root/models/AIAAH-1`, a roughly 2B Llama-compatible checkpoint. The pod was used for vLLM rollouts, isolated MBPP verification, and one-GPU LoRA experiments. The external GPT controller remained outside the loaded model and was not used as a second in-memory judge.

The first GRPO pilot exposed a token-context bug. vLLM completion prompts default to `add_special_tokens=True`, while the first trainer records used `False`. The rendered chat template already contains a BOS token, so the trainer saw one fewer BOS than the rollout policy. `rollout_policy.py` now records the exact vLLM-compatible prompt token IDs with `add_special_tokens=True`, and records the response token IDs and rendered prompt for provenance.

The implemented trainer is `/root/grpo_train.py`. It uses the raw base policy, zero-initialized LoRA on attention and MLP projections, mean-only group advantages, homogeneous-group filtering, asymmetric clipping (`0.20` lower and `0.28` upper), positive approximate KL to the rollout anchor, gradient clipping, and a 3e-7 learning rate in the corrected run. LoRA dropout is zero so sampling supplies the intended diversity and the update is reproducible.

The runs and validation results are:

| Run | Training data | Validation protocol | Result |
|---|---|---|---:|
| GRPO pilot | 32 MBPP train tasks, 256 rollouts, 10 informative groups | 90 MBPP validation tasks × 4 | 215/360 full, mean 0.6546 |
| GRPO v2 | 113 previously variable MBPP train tasks, 1,808 rollouts, 102 informative groups, misaligned prompt IDs | 90 × 4 | 217/360 full, mean 0.6583 |
| GRPO v3 | Same 113 tasks, 1,808 fresh rollouts, 109 informative groups, corrected prompt IDs | 90 × 4 | 213/360 full, mean 0.6491 |
| Raw base control | No adapter | 90 × 4 | 222/360 full, mean 0.6704 |

The trained v3 adapter is saved at `/root/grpo_v3_adapter`. Earlier quarantined adapters are `/root/grpo_pilot_adapter`, `/root/grpo_v2_adapter`, and `/root/grpo_v3_adapter`; v2 and v3 are not promoted because both lost to the raw validation control. The v3 temperature-zero validation job was interrupted during checkpointing and is not a valid result. The raw base remains the champion. The strongest positive result is still raw-base inference-time search: 500 MBPP test tasks with four fast rollouts reached 72.2% pass@4 and 0.7893 mean best-of-four correctness, compared with 60.4% pass@1 for one raw sample under the corrected protocol.

### Resume commands

After copying the checkpoint directory to a new pod, verify syntax and run a fresh training collection only from MBPP `train`:

```bash
python3 -m py_compile /root/rollout_policy.py /root/grpo_train.py
python3 /root/rollout_policy.py \
  --model /root/models/AIAAH-1 --split train \
  --task-ids-file /root/grpo_var_tasks.txt --n-samples 16 \
  --max-tokens 768 --temperature 0.8 --top-p 0.95 --seed 31415 \
  --output /root/rollouts_next.jsonl --gpu-mem-util 0.85 \
  --max-model-len 4096 --force-no-think
python3 /root/check_correctness_partial.py \
  --rollouts /root/rollouts_next.jsonl \
  --output /root/correctness_next.jsonl
python3 /root/grpo_train.py --model /root/models/AIAAH-1 \
  --rollouts /root/rollouts_next.jsonl \
  --correctness /root/correctness_next.jsonl \
  --output /root/grpo_next_adapter --epochs 1 --batch-size 2 \
  --grad-accum 8 --lr 3e-7 --clip-low 0.20 --clip-high 0.28 \
  --kl-beta 0.01 --min-group-std 0.05 --lora-r 16 \
  --lora-alpha 32 --seed 2718
```

Before any benchmark evaluation, run the same raw-base control with the same task list, seed, prompt, response cap, and number of samples. Do not train on MBPP `test`, MBPP `validation`, LiveCodeBench, or any benchmark reference/test artifact. Promote an adapter only after it wins on a fixed validation scorecard and then improves the untouched test score.

### Files in this checkpoint

`rollout_policy.py` is the fast vLLM sampler and exact prompt provenance writer. `check_correctness.py` and `check_correctness_partial.py` are the executable MBPP verifiers. `grpo_train.py` is the single-policy GRPO prototype. `align_preferences_chat.py`, `dpo_train.py`, and `merge_adapter_model.py` support the older DPO ablation. `select_tasks.py` and `build_preferences_v2.py` support task selection and preference construction. `grpo_var_tasks.txt` is the 113-task variable-reward training manifest. The JSONL rollout and correctness files preserve the experiments and their rewards.

### External publication status

No Hugging Face token was configured on the PC or cached on the pod, and GitHub had no authenticated CLI session or authorized SSH key. Therefore no external upload or GitHub push could be completed safely in this session. The local Git repository `C:\Users\kovax\Vaskrokodile` is the intended code publication target; the checkpoint folder includes a commit-ready copy. Authenticate on the next PC, then upload the adapter with `hf upload` and push the prepared commit. The report intentionally contains no API keys or private SSH material.
