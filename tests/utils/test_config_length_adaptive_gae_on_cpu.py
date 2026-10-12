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

import pytest
from omegaconf import OmegaConf

from verl.utils.config import _validate_length_adaptive_gae_config


def _config(
    adv_estimator="gae_length_adaptive",
    gae_lambda_k=0.5,
    gamma=1.0,
    use_v1=True,
    exclude_from_critic=False,
):
    return OmegaConf.create(
        {
            "algorithm": {"adv_estimator": adv_estimator, "gae_lambda_k": gae_lambda_k, "gamma": gamma},
            "trainer": {"use_v1": use_v1},
            "reward": {
                "reward_kwargs": {"overlong_buffer_cfg": {"enable": True, "exclude_from_critic": exclude_from_critic}}
            },
        }
    )


def test_accepts_a_valid_config():
    _validate_length_adaptive_gae_config(_config(), use_critic=True)


def test_requires_a_critic():
    with pytest.raises(ValueError, match="needs a critic"):
        _validate_length_adaptive_gae_config(_config(), use_critic=False)


def test_requires_gae_lambda_k():
    with pytest.raises(ValueError, match="requires algorithm.gae_lambda_k"):
        _validate_length_adaptive_gae_config(_config(gae_lambda_k=None), use_critic=True)


@pytest.mark.parametrize("gamma", [0.99, 0.0, 1.01])
def test_requires_gamma_one(gamma):
    with pytest.raises(ValueError, match="requires algorithm.gamma=1.0"):
        _validate_length_adaptive_gae_config(_config(gamma=gamma), use_critic=True)


def test_rejects_orphaned_gae_lambda_k():
    """Set on a non-GAE estimator it is silently ignored, which reads as "mechanism enabled"."""
    with pytest.raises(ValueError, match="ignores it"):
        _validate_length_adaptive_gae_config(_config(adv_estimator="grpo"), use_critic=True)


@pytest.mark.parametrize("estimator", ["grpo", "gae", "reinforce_plus_plus"])
def test_leaves_other_estimators_alone(estimator):
    _validate_length_adaptive_gae_config(_config(adv_estimator=estimator, gae_lambda_k=None), use_critic=True)


def test_exclude_from_critic_requires_v1():
    """The legacy loop never writes critic_token_level_rewards, so mechanism 4 would be off."""
    with pytest.raises(ValueError, match="requires the v1 trainer"):
        _validate_length_adaptive_gae_config(_config(use_v1=False, exclude_from_critic=True), use_critic=True)


def test_v0_without_exclude_is_allowed():
    _validate_length_adaptive_gae_config(_config(use_v1=False, exclude_from_critic=False), use_critic=True)


def test_tolerates_a_missing_reward_section():
    cfg = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "gae_length_adaptive", "gae_lambda_k": 0.5, "gamma": 1.0},
            "trainer": {"use_v1": False},
        }
    )
    _validate_length_adaptive_gae_config(cfg, use_critic=True)
