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

"""Action queue management for Real-Time Chunking (RTC).

This module provides ActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.
"""

import logging
from threading import Lock

import torch
from torch import Tensor

from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Args:
        cfg (RTCConfig): Configuration for Real-Time Chunking behavior.

    Attributes:
        queue (Tensor | None): Processed actions for robot rollout (time_steps, action_dim).
        original_queue (Tensor | None): Original actions for RTC computation (time_steps, action_dim).
        last_index (int): Current consumption index in the queue.
    """

    def __init__(self, cfg: RTCConfig):
        """Initialize the action queue.

        Args:
            cfg: RTC configuration controlling queue behavior.
        """
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        return self.last_index

    def get_left_over(self) -> Tensor | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            Tensor | None: Remaining original actions (remaining_steps, action_dim),
                          or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :]

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ):
        """Merge new actions into the queue.

        This method operates differently based on RTC mode:
        - RTC enabled: Replaces the queue, accounting for inference delay
        - RTC disabled: Appends to the queue, maintaining continuity

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            if self.cfg.enabled:
                effective_delay = self._resolve_rtc_delay(real_delay, action_index_before_inference)
                self._replace_actions_queue(original_actions, processed_actions, effective_delay)
                return

            self._check_delays(real_delay, action_index_before_inference)
            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.
        The remaining old queue and the newly predicted queue are aligned by
        future timestep before merging. This keeps in-flight actions stable and
        avoids overwriting the current chunk with misaligned early actions from
        the new prediction.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
        """
        new_original = original_actions[real_delay:].clone()
        new_processed = processed_actions[real_delay:].clone()

        if self.queue is None or self.original_queue is None:
            self.original_queue = new_original
            self.queue = new_processed
            self.last_index = 0
            return

        old_original_remaining = self.original_queue[self.last_index :].clone()
        old_processed_remaining = self.queue[self.last_index :].clone()

        overlap_steps = min(
            len(old_original_remaining),
            len(old_processed_remaining),
            len(new_original),
            len(new_processed),
        )
        blend_steps = min(self.cfg.queue_blend_steps, overlap_steps)

        merged_original_parts: list[Tensor] = []
        merged_processed_parts: list[Tensor] = []

        aligned_prefix_len = max(overlap_steps - blend_steps, 0)
        if aligned_prefix_len > 0:
            merged_original_parts.append(old_original_remaining[:aligned_prefix_len])
            merged_processed_parts.append(old_processed_remaining[:aligned_prefix_len])

        if blend_steps > 0:
            overlap_start = overlap_steps - blend_steps
            blended_original = self._blend_overlap(
                old_original_remaining[overlap_start:overlap_steps],
                new_original[overlap_start:overlap_steps],
            )
            blended_processed = self._blend_overlap(
                old_processed_remaining[overlap_start:overlap_steps],
                new_processed[overlap_start:overlap_steps],
            )
            merged_original_parts.append(blended_original)
            merged_processed_parts.append(blended_processed)
        elif overlap_steps > 0:
            merged_original_parts.append(old_original_remaining[:overlap_steps])
            merged_processed_parts.append(old_processed_remaining[:overlap_steps])

        if len(old_original_remaining) > overlap_steps:
            merged_original_parts.append(old_original_remaining[overlap_steps:])
            merged_processed_parts.append(old_processed_remaining[overlap_steps:])

        if len(new_original) > overlap_steps:
            merged_original_parts.append(new_original[overlap_steps:])
            merged_processed_parts.append(new_processed[overlap_steps:])

        if merged_original_parts:
            self.original_queue = torch.cat(merged_original_parts, dim=0)
            self.queue = torch.cat(merged_processed_parts, dim=0)
        else:
            self.original_queue = new_original
            self.queue = new_processed

        logger.debug(
            "Merged RTC queue with aligned overlap (overlap=%s, blend=%s, old_remaining=%s, new_remaining=%s)",
            overlap_steps,
            blend_steps,
            len(old_processed_remaining),
            len(new_processed),
        )

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}")

        self.last_index = 0

    def _blend_overlap(self, old_actions: Tensor, new_actions: Tensor) -> Tensor:
        """Linearly crossfade old queued actions into new actions."""
        if len(old_actions) != len(new_actions):
            raise ValueError("old_actions and new_actions must have the same overlap length")
        if len(old_actions) == 0:
            return new_actions
        if len(old_actions) == 1:
            return 0.5 * old_actions + 0.5 * new_actions

        blend = torch.linspace(
            0.0,
            1.0,
            steps=len(old_actions),
            device=old_actions.device,
            dtype=old_actions.dtype,
        ).unsqueeze(-1)
        return old_actions * (1.0 - blend) + new_actions * blend

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
        """
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_delays(self, real_delay: int, action_index_before_inference: int | None = None):
        """Validate that computed delays match expectations.

        Compares the delay computed from inference latency with the actual
        number of actions consumed during inference.

        Args:
            real_delay: Delay computed from inference latency.
            action_index_before_inference: Action index when inference started.
        """
        if action_index_before_inference is None:
            return

        indexes_diff = self.last_index - action_index_before_inference
        if indexes_diff != real_delay:
            # Let's check that action index difference (real delay calculated based on action queue)
            # is the same as delay calculated based on inference latency
            logger.warning(
                f"[ACTION_QUEUE] Indexes diff is not equal to real delay. "
                f"Indexes diff: {indexes_diff}, real delay: {real_delay}"
            )

    def _resolve_rtc_delay(self, real_delay: int, action_index_before_inference: int | None) -> int:
        """Resolve the effective delay used for RTC queue alignment.

        In RTC mode, the queue alignment should follow the number of actions that
        were actually consumed while inference was running. The latency-derived
        estimate is kept only as a fallback when the actual consumption count is
        unavailable.
        """
        if action_index_before_inference is None:
            return max(real_delay, 0)

        indexes_diff = self.last_index - action_index_before_inference
        effective_delay = max(indexes_diff, 0)

        if indexes_diff != real_delay:
            logger.warning(
                f"[ACTION_QUEUE] Indexes diff is not equal to real delay. "
                f"Indexes diff: {indexes_diff}, real delay: {real_delay}. "
                f"Using indexes diff for RTC queue alignment."
            )

        return effective_delay
