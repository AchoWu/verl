# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config guards for the JustRL-II length-adaptive GAE estimator.

Every case here is a silent-failure mode -- the run would train happily while the mechanism is
off -- which is why they are hard errors rather than warnings.
"""

import os

import pytest
from omegaconf import OmegaConf

from verl.utils.config import validate_length_adaptive_gae_config

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _config(adv_estimator="gae_length_adaptive", gae_lambda_k=0.5, gamma=1.0):
    return OmegaConf.create(
        {"algorithm": {"adv_estimator": adv_estimator, "gae_lambda_k": gae_lambda_k, "gamma": gamma}}
    )


def test_accepts_a_valid_config():
    validate_length_adaptive_gae_config(_config(), use_critic=True)


def test_requires_a_critic():
    with pytest.raises(ValueError, match="needs a critic"):
        validate_length_adaptive_gae_config(_config(), use_critic=False)


def test_requires_gae_lambda_k():
    with pytest.raises(ValueError, match="requires algorithm.gae_lambda_k"):
        validate_length_adaptive_gae_config(_config(gae_lambda_k=None), use_critic=True)


@pytest.mark.parametrize("gamma", [0.99, 0.0, 1.01])
def test_requires_gamma_one(gamma):
    with pytest.raises(ValueError, match="requires algorithm.gamma=1.0"):
        validate_length_adaptive_gae_config(_config(gamma=gamma), use_critic=True)


def test_rejects_orphaned_gae_lambda_k():
    """Set on a non-GAE estimator it is silently ignored, which reads as "mechanism enabled"."""
    with pytest.raises(ValueError, match="ignores it"):
        validate_length_adaptive_gae_config(_config(adv_estimator="grpo"), use_critic=True)


@pytest.mark.parametrize("estimator", ["grpo", "gae", "reinforce_plus_plus"])
def test_leaves_other_estimators_alone(estimator):
    validate_length_adaptive_gae_config(_config(adv_estimator=estimator, gae_lambda_k=None), use_critic=True)


def _critic_side_config(**overrides):
    cfg = _config()
    cfg.merge_with(OmegaConf.create(overrides))
    return cfg


def test_rejects_value_head_seeding_on_megatron():
    cfg = _critic_side_config(critic={"strategy": "megatron", "model": {"value_head_bias_init": 0.5}})
    with pytest.raises(NotImplementedError, match="only wired into the FSDP critic"):
        validate_length_adaptive_gae_config(cfg, use_critic=True)


def test_allows_value_head_seeding_on_fsdp():
    cfg = _critic_side_config(critic={"strategy": "fsdp", "model": {"value_head_bias_init": 0.5}})
    validate_length_adaptive_gae_config(cfg, use_critic=True)


def test_rejects_exclude_from_critic_without_penalty():
    cfg = _critic_side_config(
        reward_model={"reward_manager": "dapo", "overlong_buffer": {"enable": False, "exclude_from_critic": True}}
    )
    with pytest.raises(ValueError, match="nothing to exclude"):
        validate_length_adaptive_gae_config(cfg, use_critic=True)


def test_rejects_exclude_from_critic_without_dapo_reward_manager():
    cfg = _critic_side_config(
        reward_model={"reward_manager": "naive", "overlong_buffer": {"enable": True, "exclude_from_critic": True}}
    )
    with pytest.raises(ValueError, match="reward_manager=dapo"):
        validate_length_adaptive_gae_config(cfg, use_critic=True)


def _compose(config_dir, config_name, overrides=()):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def test_critic_config_instantiates_with_value_head_bias_init():
    """critic.yaml carries the key for every critic, so the dataclass must declare it.

    Without the field, hydra's instantiate of critic.model raises TypeError at init_workers --
    for every critic run, including stock PPO with the key left at null.
    """
    from verl.utils.config import omega_conf_to_dataclass

    config_dir = os.path.join(_REPO_ROOT, "verl", "trainer", "config")
    # use_dynamic_bsz only satisfies the critic's unrelated micro-batch-size check.
    base = ["critic.use_dynamic_bsz=True"]
    for extra in ([], ["critic.model.value_head_bias_init=0.52"]):
        cfg = _compose(config_dir, "ppo_trainer", base + extra)
        critic_cfg = omega_conf_to_dataclass(cfg.critic)
        expected = 0.52 if extra else None
        assert critic_cfg.model.value_head_bias_init == expected


def test_justrl2_recipe_config_passes_the_guards():
    """The recipe's own yaml, with the overrides run_justrl2_16k.sh sets, must clear validation."""
    from verl.trainer.ppo.utils import need_critic

    cfg = _compose(
        os.path.join(_REPO_ROOT, "recipe", "justrl2", "config"),
        "justrl2_trainer",
        [
            "critic.model.value_head_bias_init=0.52",
            "reward_model.overlong_buffer.enable=True",
            "reward_model.overlong_buffer.len=2867",
            "reward_model.overlong_buffer.penalty_factor=1.0",
            "reward_model.overlong_buffer.exclude_from_critic=True",
        ],
    )
    validate_length_adaptive_gae_config(cfg, use_critic=need_critic(cfg))
