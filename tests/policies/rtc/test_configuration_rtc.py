#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for RTC configuration module."""

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

# ====================== Initialization Tests ======================


def test_rtc_config_default_initialization():
    """Test RTCConfig initializes with default values."""
    config = RTCConfig()

    assert config.enabled is False
    assert config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR
    assert config.max_guidance_weight == 10.0
    assert config.execution_horizon == 10
    assert config.queue_blend_steps == 0
    assert config.startup_skip_replacements == 0
    assert config.startup_wait_for_first_queue_drain is False
    assert config.debug is False
    assert config.debug_maxlen == 100


def test_rtc_config_custom_initialization():
    """Test RTCConfig initializes with custom values."""
    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=5.0,
        execution_horizon=20,
        queue_blend_steps=6,
        startup_skip_replacements=2,
        startup_wait_for_first_queue_drain=True,
        debug=True,
        debug_maxlen=200,
    )

    assert config.enabled is True
    assert config.prefix_attention_schedule == RTCAttentionSchedule.EXP
    assert config.max_guidance_weight == 5.0
    assert config.execution_horizon == 20
    assert config.queue_blend_steps == 6
    assert config.startup_skip_replacements == 2
    assert config.startup_wait_for_first_queue_drain is True
    assert config.debug is True
    assert config.debug_maxlen == 200


def test_rtc_config_partial_initialization():
    """Test RTCConfig with partial custom values."""
    config = RTCConfig(enabled=True, max_guidance_weight=15.0)

    assert config.enabled is True
    assert config.max_guidance_weight == 15.0
    # Other values should be defaults
    assert config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR
    assert config.execution_horizon == 10
    assert config.queue_blend_steps == 0
    assert config.startup_skip_replacements == 0
    assert config.startup_wait_for_first_queue_drain is False
    assert config.debug is False


def test_rtc_config_rejects_negative_queue_blend_steps():
    """Test RTCConfig validates queue_blend_steps."""
    try:
        RTCConfig(queue_blend_steps=-1)
    except ValueError as exc:
        assert "queue_blend_steps" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative queue_blend_steps")


def test_rtc_config_rejects_negative_startup_skip_replacements():
    """Test RTCConfig validates startup_skip_replacements."""
    try:
        RTCConfig(startup_skip_replacements=-1)
    except ValueError as exc:
        assert "startup_skip_replacements" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative startup_skip_replacements")
