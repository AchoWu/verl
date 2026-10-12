# JustRL-II on verl 0.6: length-adaptive GAE with a value-head critic

> **This is the verl-0.6 port.** It targets CUDA 12.x / torch 2.8 / vllm 0.11 (the
> `verl0.6-cu128-torch2.8.0-fa2.7.4` image), because verl 0.10 requires cu130 and a driver
> >= 580. The algorithm is identical; what differs is where it hooks in:
>
> | | verl 0.10 | verl 0.6 (this branch) |
> |---|---|---|
> | trainer | v1 (`trainer_base.py`, TransferQueue) | `RayPPOTrainer` / `RayDAPOTrainer` |
> | entry point | `verl.trainer.main_ppo` | `recipe.dapo.main_dapo` |
> | reward config | `reward.reward_kwargs.overlong_buffer_cfg` | `reward_model.overlong_buffer` |
> | custom scorer | `reward.custom_reward_function` | `custom_reward_function` |
> | value head | `workers/engine/fsdp/transformer_impl.py` | `workers/fsdp_workers.py` |
> | critic reward plumbing | `critic_rm_scores` through TransferQueue | `attach_critic_token_level_rewards` |
>
> `recipe.dapo.main_dapo` is the entry point because on 0.6 only it wires
> `reward_model.overlong_buffer` into the DAPO reward manager. `filter_groups` stays off, so
> that trainer's extra sampling loop is a pass-through.

PPO with a **scalar value-head critic**, a **length-adaptive GAE λ**, and a **mean-reward-seeded
value head** — a port of [JustRL-II](https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic)
onto verl's PPO pipeline.

```bash
export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B
export RAY_DATA_HOME=/path/to/data-root

# 1. build JustRL-II's training set (UltraData-RL-2609, Math) and the AIME 2024/25/26 sets
bash recipe/justrl2/prepare_data.sh

# 2. measure the base model's mean reward -> the critic's prior (see mechanism 2)
MAX_PROMPTS=300 bash recipe/justrl2/run_measure_pass_rate.sh

# 3. train, with the number step 2 printed
bash recipe/justrl2/run_justrl2_16k.sh critic.model.value_head_bias_init=0.49
```

Step 3 runs in the foreground (`main_ppo` calls `ray.init()` itself). `SUBMIT=1` dispatches
through `ray job submit` against an existing cluster instead, which is what multi-node needs.

> **This 16k config is not a reproduction of the published recipe.** The core mechanism
> `λ_i = k^(1/L_i)` takes the response length as its input, so shortening the budget from 128k
> (or 30k) to 14k makes the same `k = 0.5` act on a completely different length distribution.
> None of the three core knobs changed, but what they act on did — curves from this config are
> not comparable to the 32k / 128k results. It exists to fit the recipe into a single-node
> memory budget.

## Why a critic at all

GRPO's baseline is the group mean, so **every token of a trajectory gets the same advantage**.
A response that sets up the problem correctly for 2000 tokens and then drops a sign gets those
2000 good tokens punished along with the bad one. A critic predicts "from here, will this end
up correct?" per token, which turns `this trajectory was wrong` into `this trajectory went
wrong at token 2001`.

That is the whole bet: spend GPUs on a second model to get per-token credit instead of
per-trajectory credit. JustRL-II's claim is that it pays off for long-context reasoning, where
the sequences are long enough that a uniform advantage wastes most of the signal.

## The four mechanisms

### 0. Data, prompt and reward: exactly JustRL-II's

The three have to match as a set, because the prompt decides the answer format and the reward
has to parse that format.

| | JustRL-II (miles) | this recipe |
|---|---|---|
| training set | UltraData-RL-2609, config `Math` (32,412 rows) | same, via `prepare_ultradata_math.py` |
| system prompt | none | none |
| user turn | raw `query`, which already ends with "Please reason step by step, and put your final answer within \boxed{}." | same |
| chat template | the model's own, `add_generation_prompt=True` | same (verl's `RLHFDataset`) |
| AIME eval prompt | `{problem}\n` + that same instruction; Qwen2.5-Math mirror / `math-ai/aime25` / `math-ai/aime26` | same, via `prepare_aime_data.py` |
| reward | `--rm-type math`: last `\boxed{}`, mathd + sympy, math-verify fallback, **0/1** | the same grader vendored unchanged (`math_utils.py`, `math_verify_fallback.py`), wrapped by `reward_score.py` |

verl's stock `math_dapo` scorer is not usable here: it parses an `Answer: ...` line and marks a
correct `\boxed{10}` wrong. The chat template is the one thing that differs in practice, simply
because the base model differs (R1-Distill-Qwen-1.5B here, MiniCPM5-2B upstream); neither side
adds a system prompt.

The 0/1 scale is what makes the critic's output meaningful:

| reward | what `V(s)` is |
|---|---|
| 0/1 | `P(this prefix ends up correct)` — at the prompt, a **difficulty estimate** |
| ±1 | `2·P(correct) − 1` — the same information, no longer a probability |

It also decides how much mechanism 2 buys. For a Bernoulli reward `E[r²] = E[r]`, so seeding
the head at the mean takes the opening value loss from `E[r]` to `Var(r) = p(1−p)` — a factor
of `1/(1−p)`. On ±1, `E[r²] = 1` no matter what, so seeding only removes the `mean²` term
(measured: 1.09× versus ~1.9×).

The DAPO overlong penalty is *subtracted* from the reward, so `penalty_factor=1.0` is 100% of
the 0/1 span: as in JustRL-II, a maximally-overlong correct answer scores 0.

### 1. Length-adaptive λ (the core)

In GAE, the first token receives the terminal reward with weight `λ^L`. With any fixed λ this
collapses as responses get longer:

| L | `λ = 0.999` → first-token credit | `λ_i = 0.5^(1/L)` → first-token credit |
|---|---|---|
| 16 | 0.985 | **0.50** |
| 256 | 0.775 | **0.50** |
| 2048 | 0.129 | **0.50** |
| 14336 | 5.9e-7 | **0.50** |

Solving `λ^L = k` for λ gives `λ_i = k^(1/L_i)`: a per-sample λ under which the first token
always sees fraction `k` of the terminal credit, regardless of length. Long chains-of-thought
stop losing their opening decisions — which is exactly where the hard reasoning choices are
made.

**Decoupled targets.** The advantage uses `λ_i`; the critic's regression target uses `λ = 1`
(the plain suffix reward sum). The critic regresses an unbiased return while the actor's
advantage gets the length-adaptive discount. This requires `γ = 1` and the estimator asserts it.

**fp32 is mandatory, and not just for λ.** At 14336 tokens `λ = 1 − 4.8e-5`; bf16 rounds that
to exactly `1.0` — as it does for *every* length ≥ 512 — and the mechanism silently degrades
into ordinary λ=1 GAE with no error and no metric. The critic's values arrive from an
`autocast(bf16)` forward, so the estimator upcasts its inputs and keeps λ, the GAE scan and the
returns cumsum in fp32 rather than inheriting the batch dtype. `TestDtypeHandling` in the test
file pins this down; it fails loudly if the cast creeps back in.

Code: `verl/trainer/ppo/core_algos.py::compute_length_adaptive_gae_advantage_return`.

### 2. Value-head seeding

The critic's scalar head is zero-initialized with its bias set to the expected mean reward, so
`V ≡ bias` at step 0 and **the bias is the critic's prior**.

HF's default `normal(0, 0.02)` on a wide hidden state yields `std(V) ≈ 0.02·√H·rms(h)` — order
10 against targets in `[0, 1]`. The critic then spends its first ~25 steps unlearning that
offset *while the actor already updates against it*. Seeded at the mean reward the value loss
opens at `Var(r)` instead of `E[r²]` and the first gradient norm is ~3× smaller. Gradients still
flow from step 0, since `dV/dw = h ≠ 0`.

Set it via `critic.model.value_head_bias_init`. Code: `verl/utils/model.py::seed_scalar_value_head`.

**Measure this number for your setup** — `0.52` is JustRL-II's value for *their* model and
data, not a universal constant:

```bash
MAX_PROMPTS=300 bash recipe/justrl2/run_measure_pass_rate.sh
```

It samples with the **training** sampling params (`temperature=1.0`, `n=8`, full response
length) and scores with the **same scorer training uses** (`reward_score.py`), so the
number is on the right scale. Both are load-bearing: a greedy decode measures something else
entirely, and a mismatched scorer puts the bias on the wrong scale. The overlong penalty is
deliberately not applied, because the critic's target excludes it.

Output:

```
  mean reward over all samples : 0.4896   <-- value_head_bias_init
  95% CI (Wilson)              : [0.4451, 0.5342]
  reward range seen            : [0, 1]

  initial value loss:
    bias = 0.0            -> E[r^2]  = 0.4896
    bias = 0.4896         -> Var(r)  = 0.2499   (1.96x lower)

  group diagnostics (relevant to algorithm.filter_groups):
    all-correct groups : 13.3%
    all-wrong groups   : 20.0%
    non-degenerate     : 66.7%
```

On the 0/1 scale the mean reward *is* the pass rate, so this number is directly readable.

The group diagnostics are a useful side-effect. Groups whose 8 samples all score the same carry
almost no advantage signal — DAPO drops them when `filter_groups` is on, and when it is off
they stay in the batch contributing nothing. Either way, a high fraction means the training set
is mismatched to the model's current ability, which is worth knowing *before* a long run.

Already have generations? `SKIP_GEN=1 GEN_OUTPUT=<parquet> bash recipe/justrl2/run_measure_pass_rate.sh`.
Train with a different scorer? Pass `CUSTOM_REWARD_PATH` so the measurement uses the same one.

### 3. Critic-only warmup

`trainer.critic_warmup: 31` freezes the actor for the first 30 steps so it never updates against
a critic that is still guessing. (verl's `global_steps` starts at 1 and the actor updates once
`critic_warmup <= global_steps`, so `31` gives exactly JustRL-II's 30 critic-only rollouts; `30`
would give 29.) Combined with (2), the value head is a usable baseline from step 0 and a
discriminating one by step 30. This is stock verl, no changes needed.

### 4. Overlong penalty kept out of the critic's target

DAPO's soft overlong punishment shapes the actor's reward (`r − penalty(L)` over the last 20% of
the length budget) to stop the policy from rambling. The critic's job is different: at token 500
it must predict "will this be correct", and it *cannot* predict how long the response will end up
being — that depends on tokens not yet generated. Forcing the length term into `V` injects noise
into the value signal.

With `reward_model.overlong_buffer.exclude_from_critic=True` the DAPO reward manager
emits a second pre-penalty score, which becomes the critic's regression target. The actor's
advantage is untouched. At 16k:

| response length | actor reward | critic target |
|---|---|---|
| 5000 | 1.000 | 1.000 |
| 11469 | 1.000 | 1.000 |
| 12000 | 0.815 | 1.000 |
| 14336 | 0.000 | 1.000 |

The critic also skips the in-reward KL penalty for the same reason.

## Configuration

| knob | value | what it is |
|---|---|---|
| `custom_reward_function.path` | `recipe/justrl2/reward_score.py` | mechanism 0 — JustRL-II's boxed 0/1 math reward |
| `algorithm.adv_estimator` | `gae_length_adaptive` | mechanism 1 |
| `algorithm.gamma` | `1.0` | **required**; asserted |
| `algorithm.gae_lambda_k` | `0.5` | terminal credit reaching token 0 |
| `critic.model.value_head_bias_init` | `0.52` | mechanism 2 — **measure yours** (see above) |
| `trainer.critic_warmup` | `31` | mechanism 3 — 30 critic-only rollouts (verl counts from 1) |
| `reward_model.overlong_buffer.exclude_from_critic` | `True` | mechanism 4 |
| `algorithm.rollout_is*` | `mask`, `token`, `[0.5, 5.0]`, veto off | IcePop: JustRL-II's `--use-tis` + `icepop_function` |
| `actor_rollout_ref.rollout.calculate_log_probs` | `True` | rollout log-probs for the IcePop ratio |
| `*.optim.betas` | `[0.9, 0.98]` | JustRL-II's Adam beta2, actor and critic (verl default 0.999) |
| `critic.optim.lr` | `5e-6` | 5× the actor: the critic chases a moving policy |
| `critic.cliprange_value` | `0.2` | PPO-style value clipping |
| `actor…clip_ratio_low/high` | `0.2` / `0.28` | clip-higher |
| `data.max_prompt_length` | `2048` | 16k budget |
| `data.max_response_length` | `14336` | 16k budget |
| `reward_model.overlong_buffer.len` | `2867` | 20% of the response budget |

### Deliberate departures from JustRL-II

- **No DAPO dynamic sampling / 2× over-sampling** (`filter_groups.enable=False`). JustRL-II drops
  zero-reward-variance groups. With critic-based advantages those groups still carry a non-zero
  advantage, so this changes the training signal rather than only the throughput. Turn it on with
  `algorithm.filter_groups.enable=True algorithm.filter_groups.metric=acc data.gen_batch_size=64`.
- **No partial rollout** — verl 0.6 has no equivalent.
- **Batch and schedule**: GBS 256 (JustRL-II 1-node 16k: 64). `trainer.total_training_steps=1000`
  (JustRL-II runs 500 rollouts) overrides `total_epochs=1`; checkpoints and AIME evals every 50
  steps. One epoch of UltraData-RL-Math is ⌊32,412 / 32⌋ = 1,012 steps (no prompt exceeds the
  2048-token budget, so none are filtered), so 1000 steps see 32,000 prompts — nearly the whole
  set once, each row at most once. JustRL-II's 1-node 16k run trains on 500 × 8 = 4,000
  prompts (more are generated, then dropped by dynamic sampling).

### Misconfigurations fail loudly

Each of this recipe's mechanisms can be *disabled* by a config mistake without anything
crashing — the run trains, the loss falls, and the mechanism simply is not there. So these are
hard errors rather than warnings:

| condition | where |
|---|---|
| `gae_lambda_k` set on an estimator that ignores it | `validate_length_adaptive_gae_config` |
| `gae_length_adaptive` without `gae_lambda_k`, a critic, or `gamma=1` | `validate_length_adaptive_gae_config` |
| `exclude_from_critic=True` with the penalty disabled | `validate_length_adaptive_gae_config` |
| `exclude_from_critic=True` with a reward manager other than `dapo` | `validate_length_adaptive_gae_config` |
| `exclude_from_critic=True` but `critic_reward_score` missing from a batch | `attach_critic_token_level_rewards` (every step) |
| `value_head_bias_init` on the megatron backend (unsupported) | `validate_length_adaptive_gae_config` |

`validate_length_adaptive_gae_config` runs from `validate_config` under `main_ppo`, and is
called directly from `main_dapo` (which does not call `validate_config`) before any worker starts.

## Backend support

**FSDP only** (`critic.strategy=fsdp`). The critic loads via HF `AutoModelForTokenClassification`, whose `score` layer is
not in the policy checkpoint — HF builds it fresh, so seeding at construction is sufficient and
there is nothing to re-apply after loading.

Megatron is deliberately not wired up, and setting `value_head_bias_init` with
`critic.strategy=megatron` raises `NotImplementedError` at config validation rather than being
ignored. Its critic loads through dist checkpoints,
where megatron's reader fills a `[1, H]` head from the overlapping region of the `[vocab, H]`
LM head instead of raising (`language_module.py` marks `output_layer.*` allow_shape_mismatch
for vocab padding). Construction-time seeding would be silently overwritten, so that path needs
a post-load re-zero plus `optimizer.reload_model_params()` — which is what JustRL-II does
upstream, and what would have to be ported and verified before trusting a megatron run here.

## Tests

```bash
pytest tests/trainer/ppo/test_length_adaptive_gae_on_cpu.py \
       tests/utils/test_value_head_seeding_on_cpu.py \
       tests/utils/test_config_length_adaptive_gae_on_cpu.py \
       tests/utils/test_justrl2_reward_score_on_cpu.py -q
```

Covers the `λ^L = k` invariant across lengths and `k` values; **bf16 inputs producing the same
result as fp32** (the regression that silently disabled mechanism 1); returns being independent
of `values` (i.e. genuinely λ=1 rather than `advantages + values`); the critic reward override
affecting only returns; advantage whitening; the γ≠1 and missing-`k` guards; both value-head
layouts (`score` and TRL's `v_head.summary`); bias creation when the head was built without one;
that gradients still flow through a zeroed weight; the boxed grader (equivalent forms, last box
wins, unboxed and truncated answers score 0, one rule for every `data_source`, the math-verify
fallback on Linux) and the prompt rows (raw query as one user turn, AIME suffix, stale-parquet
detection); every config guard in the table above; the critic config
instantiating through hydra with `value_head_bias_init` set or null; the recipe yaml clearing
validation; the DAPO trainer forwarding `config` to `compute_advantage`; and the critic target
landing on the shaped reward's token. (The multi-output `reward_extra_info` aliasing test is
v1-only and does not apply here.)

## Files

| file | change |
|---|---|
| `verl/trainer/ppo/core_algos.py` | `length_adaptive_lambda`, `compute_length_adaptive_gae_advantage_return`, enum entry |
| `verl/trainer/ppo/ray_trainer.py` | pass `values` / `gamma` / `lam` / `critic_token_level_rewards` to registry estimators; `attach_critic_token_level_rewards`, `critic_excludes_overlong_penalty` |
| `verl/trainer/ppo/utils.py` | `need_critic` recognizes the new estimator |
| `verl/utils/config.py` | `validate_length_adaptive_gae_config` guards |
| `verl/trainer/config/algorithm.py` | `AlgoConfig.gae_lambda_k` |
| `verl/trainer/config/config.py` | `BaseModelConfig.value_head_bias_init` (critic model dataclass) |
| `verl/trainer/config/critic/critic.yaml` | `model.value_head_bias_init: null` |
| `verl/utils/model.py` | `find_scalar_value_head`, `seed_scalar_value_head` |
| `verl/workers/fsdp_workers.py` | seed the head after `load_valuehead_model` |
| `verl/workers/reward_manager/dapo.py` | emit `critic_reward_score` |
| `recipe/dapo/dapo_ray_trainer.py` | call `attach_critic_token_level_rewards` in its own fit(); pass `config` to `compute_advantage` |
| `recipe/dapo/main_dapo.py` | run `validate_length_adaptive_gae_config` before workers start |

Recipe-local:

| file | what |
|---|---|
| `run_justrl2_16k.sh` | the 16k training run |
| `config/justrl2_trainer.yaml` | recipe defaults (estimator, `reward_model.overlong_buffer`) |
| `reward_score.py` | JustRL-II's `--rm-type math` reward on verl's custom-reward signature, 0/1 |
| `math_utils.py`, `math_verify_fallback.py` | that grader, vendored from JustRL-II's miles unchanged (one import line) |
| `prepare_data.sh` | build the UltraData-RL-Math training set and the AIME sets |
| `prepare_ultradata_math.py` | UltraData-RL-2609 (Math) → verl schema, raw query as one user turn |
| `prepare_aime_data.py` | AIME 2024/25/26 from JustRL-II's sources, its prompt, per-year `data_source` |
| `measure_pass_rate.py` | score generations → `value_head_bias_init` |
| `run_measure_pass_rate.sh` | generate + score in one go |

## Reference

- Blog: [JustRL-II: Scaling Small LLMs to 128k Reasoning with a Critic](https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic)
  ([中文](https://panhaoxuan.notion.site/justrl-ii-small-llms-to-128k-reasoning-with-a-critic-cn))
- DAPO soft overlong punishment: https://arxiv.org/abs/2503.14476
