# Forgetting-gated self-imitation (verl 0.6)

Dr. GRPO on the current policy, plus imitation of the policy's **own correct solutions from
`k` updates ago** — only on prompts it has since forgotten. Method and motivation:
[`RL_Idea_与过去的自己比较的策略优化.md`](../../RL_Idea_与过去的自己比较的策略优化.md).

Everything except the advantage is JustRL-II's 16k setup from [`recipe/justrl2`](../justrl2):
same data, prompt, 0/1 boxed reward, context budget, lr, batch, clip-higher, IcePop, Adam betas
and schedule. There is no critic: the method's baseline is the group pass rate `p_cur`.

Departures from the idea doc: the dual-clip constant is justrl2's `clip_ratio_c=10`, not the
doc's `c = 3` (§3.5).

```bash
bash recipe/justrl2/prepare_data.sh                     # once; this recipe only reads the data
bash recipe/self_imitation/run_self_imitation_16k.sh
```

## Per step

| | rows | advantage |
|---|---|---|
| current | n rollouts per prompt from π_prox | `r − p_cur` (Dr. GRPO, no std) |
| replay | π_lag's correct rollouts on prompts with `p_old − p_cur ≥ τ` | `λ · (r − p_cur)₊` |

π_lag samples are **pre-generated**: the rollout call at step `t` also samples `m` responses for
the prompts of step `t+k`, so they come from the weights `k` updates before they are trained on —
no second engine, no weight swapping. The first `k` steps train without replay.

| knob | env | default | |
|---|---|---|---|
| `self_imitation.lag_steps` | `LAG_STEPS` | 16 | `k` |
| `self_imitation.num_replay` | `NUM_REPLAY` | 8 | `m`; 0 = no replay generation |
| `self_imitation.replay_coef` | `REPLAY_COEF` | 1.0 | `λ`; 0 = generate and log, train on none |
| `self_imitation.gate_tau` | `GATE_TAU` | 0.375 | `τ` |

## Runs to compare

| run | command |
|---|---|
| method | `bash recipe/self_imitation/run_self_imitation_16k.sh` |
| Dr. GRPO-8 | `NUM_REPLAY=0 …` |
| **Dr. GRPO-16** (same rollout budget — the key control) | `NUM_REPLAY=0 N_RESP_PER_PROMPT=16 …` |
| null gate (optional: empirical check of `gate_null_frac`) | `LAG_STEPS=0 REPLAY_COEF=0 …` — a few dozen steps is enough |
| justrl2 (critic) | `bash recipe/justrl2/run_justrl2_16k.sh` |

**Look at `self_imitation/gate_excess` first.** It is the gate's open rate minus
`gate_null_frac`, the exact rate at which the same prompts would open on sampling noise alone
(idea doc §3.4.1), so no separate null run is needed. It is a *net* signal: zero in expectation
when `p_old = p_cur`, ≤ 0 in expectation when nothing was forgotten, and pulled down by prompts the
policy is learning.

Judge it over a window of steps with the **pooled z**, from the logged sums:

```
z_window = Σ (gate_open_count − gate_null_sum) / sqrt(Σ gate_null_var_sum)
```

(same with the `gate_learn_*` sums for the reverse gate). Steps draw different prompts, so the sums
add. Do not average the per-step `gate_excess_z` curve: the mean of W independent z has standard
deviation 1/√W, so a 36-step mean of 0.5 is already 3σ. "Clearly" below means pooled z ≳ 3.

| `gate_excess` | `gate_learn_excess` | reading |
|---|---|---|
| clearly > 0 | any | forgetting is real; replay has something to act on |
| ≤ 0 | clearly > 0 | inconclusive: no forgetting, or forgetting masked by learning |
| ≈ 0 | ≈ 0 | old and current policy barely differ per prompt: raise `k` |

Never lower τ to "find" forgetting: that only admits more noise.

## Metrics (`self_imitation/…`)

| metric | meaning |
|---|---|
| `gate_open_frac`, `delta_mean`, `delta_q95` | share of prompts replayed; distribution of `p_old − p_cur` |
| `gate_null_frac`, `gate_excess`, `gate_excess_z` | no-forgetting baseline of the open rate (hypergeometric, given each prompt's total correct count); open rate minus it; per-step z |
| `gate_learn_frac`, `gate_learn_excess`, `gate_learn_excess_z` | the same for the reverse gate `p_cur − p_old ≥ τ`, against its own null (equal to the forward one when n = m) |
| `gate_open_count`, `gate_null_sum`, `gate_null_var_sum` (and `gate_learn_count`, `gate_learn_null_sum`, `gate_learn_null_var_sum`) | raw per-step sums for the pooled z above |
| `cur_all_wrong_frac`, `rescued_all_wrong_frac` | zero-signal prompts for GRPO, and the share of them replay gave a positive signal |
| `replay_rows`, `replay_token_frac`, `pad_rows` | how much replay entered the batch |
| `replay_log_ratio_mean`, `replay_forgotten_token_frac` | `log π_prox/π_lag` on replayed tokens; share below the PPO lower clip (includes vLLM/FSDP mismatch) |
| `adv_scale` | the `N_all / N_cur` denominator fix below |

`actor/entropy`, `mismatch/*` and the `critic/score`, `response_length` data metrics are computed
on the current rows only, so they stay comparable to a plain GRPO run.

## Implementation notes

- **Replay is not importance-corrected.** IcePop masks/weights apply to current rows only; replay
  rows get weight 1. Forgotten tokens are exactly those with small `π_prox/π_lag`, so IS would
  zero them out. Both row kinds use the ordinary PPO ratio `π_θ/π_prox` (verl recomputes `π_prox`
  for the whole batch anyway), so replay is bounded by the `1 + clip_ratio_high` clip.
- **Denominator.** token-mean would divide the GRPO term by replay tokens too; all advantages are
  scaled by `N_all / N_cur` so the denominator is the current-token count and `λ` is a pure weight.
- **Mini-batching.** The batch grows by a variable number of rows. To keep exactly 4 optimizer
  steps per rollout the trainer passes `meta_info["num_mini_batches"]`, read by
  `verl/workers/actor/dp_actor.py` (the one core change; FSDP actor only). Rows are padded to a
  multiple of `world_size × num_mini_batches` with copies of the shortest current rows whose
  `response_mask` is zeroed (≤ 31 rows at 8 GPUs). With no replay rows the batch, mini-batching
  and loss are exactly Dr. GRPO's.
- **Cost.** Each step generates `64 × (n + m)` rollouts (2× rollout time at m = n). Only admitted
  replay rows are trained on. Correct replay rows wait `k` steps in the Ray object store: ~0.7 MB
  per row at 16k context, so the steady state is `k × 512 × train pass rate × 0.7 MB` — about
  2.9 GB at k = 16 and 11 GB at k = 64 for a pass rate of 0.5.
- **Resume.** The live dataloader runs `k` batches ahead of training, so checkpoints save the
  position right after the batch being trained on instead (`_save_checkpoint`): no prompts are
  skipped. Pending replay is not checkpointed, so the first `k` steps after a resume train without
  replay (a warning says so).
- **`rollout.calculate_log_probs=True` is required** whenever `num_replay > 0`: π_lag's log-probs
  only come from the rollout.
- **Rewards must be 0/1.** The gate, its noise baseline and replay's `r > 0` filter all assume it.
  The overlong penalty is rejected at startup while `replay_coef > 0`, and any non-0/1 reward fails
  at step 1 (`group_counts`). Other unsupported settings (critic, megatron, async rollout, in-reward
  KL, reward model, `rollout_data_dir`) fail at startup in `validate_self_imitation_config`.

## Files

| file | |
|---|---|
| `replay.py` | batch building, gate, advantages, IS split, metrics, config checks (pure, CPU-tested) |
| `self_imitation_trainer.py` | the training loop: lagged schedule, one rollout call per step |
| `main_self_imitation.py` | entry point (`main_dapo` minus critic / RM roles) |
| `config/self_imitation_trainer.yaml` | recipe defaults |
| `run_self_imitation_16k.sh` | the 16k run |

```bash
pytest tests/trainer/ppo/test_self_imitation_on_cpu.py \
       tests/trainer/ppo/test_self_imitation_trainer_on_cpu.py -q
# dp_actor num_mini_batches override (needs a process group):
MASTER_ADDR=127.0.0.1 MASTER_PORT=29533 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    pytest tests/workers/actor/test_special_dp_actor.py -q -k update_policy
```
