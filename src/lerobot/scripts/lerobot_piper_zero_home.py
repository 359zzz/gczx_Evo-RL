#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Capture a PiPER arm's current pose as a custom zero pose, or smoothly move back to it later.

Examples:

Capture the current pose as the custom zero pose:

```shell
python -m lerobot.scripts.lerobot_piper_zero_home \
  --mode=capture \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=my_piper
```

Smoothly return to the saved zero pose:

```shell
python -m lerobot.scripts.lerobot_piper_zero_home \
  --mode=home \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=my_piper \
  --duration_s=4.0 \
  --fps=50
```
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.piper_follower.config_piper_follower import (  # noqa: F401
    PiperFollowerConfig,
    PiperXFollowerConfig,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS, PIPER_JOINT_ACTION_KEYS
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


SUPPORTED_PIPER_TYPES = {"piper_follower", "piperx_follower"}


@dataclass
class PiperZeroHomeConfig:
    robot: RobotConfig
    mode: str = "home"
    zero_pose_path: str | None = None
    duration_s: float = 3.0
    fps: int = 50
    settle_time_s: float = 0.2
    include_gripper: bool = True
    connect_calibrate: bool = False


def _default_zero_pose_path(cfg: PiperZeroHomeConfig) -> Path:
    robot_id = cfg.robot.id if cfg.robot.id else "default"
    robot_type = cfg.robot.type if hasattr(cfg.robot, "type") else type(cfg.robot).__name__
    return HF_LEROBOT_HOME / "piper_zero_pose" / f"{robot_type}_{robot_id}.json"


def _resolve_zero_pose_path(cfg: PiperZeroHomeConfig) -> Path:
    if cfg.zero_pose_path is not None:
        return Path(cfg.zero_pose_path).expanduser()
    return _default_zero_pose_path(cfg)


def _active_pose_keys(include_gripper: bool) -> tuple[str, ...]:
    if include_gripper:
        return PIPER_ACTION_KEYS
    return PIPER_JOINT_ACTION_KEYS


def _validate_piper_zero_home_config(cfg: PiperZeroHomeConfig) -> None:
    if cfg.robot.type not in SUPPORTED_PIPER_TYPES:
        raise ValueError(
            f"`{cfg.mode}` mode currently supports only {sorted(SUPPORTED_PIPER_TYPES)}. "
            f"Got robot.type={cfg.robot.type!r}."
        )
    if cfg.mode not in {"capture", "home"}:
        raise ValueError("`mode` must be either 'capture' or 'home'.")
    if cfg.duration_s < 0:
        raise ValueError("`duration_s` must be >= 0.")
    if cfg.fps <= 0:
        raise ValueError("`fps` must be > 0.")
    if cfg.settle_time_s < 0:
        raise ValueError("`settle_time_s` must be >= 0.")


def _extract_zero_pose_from_observation(observation: dict[str, Any], include_gripper: bool) -> dict[str, float]:
    pose: dict[str, float] = {}
    for key in _active_pose_keys(include_gripper):
        if key not in observation:
            continue
        pose[key] = float(observation[key])

    missing_joint_keys = [key for key in PIPER_JOINT_ACTION_KEYS if key not in pose]
    if missing_joint_keys:
        raise ValueError(f"Missing PiPER joint positions in observation: {missing_joint_keys}")
    return pose


def _save_zero_pose(robot: Any, pose_path: Path, include_gripper: bool) -> dict[str, float]:
    observation = robot.get_observation()
    joint_pos = _extract_zero_pose_from_observation(observation, include_gripper=include_gripper)

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot_type": robot.robot_type,
        "robot_id": robot.id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "joint_pos": joint_pos,
    }
    with open(pose_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    logging.info("Saved PiPER zero pose to %s", pose_path)
    return joint_pos


def _load_zero_pose(pose_path: Path) -> dict[str, float]:
    with open(pose_path) as f:
        payload = json.load(f)

    joint_pos_raw = payload["joint_pos"] if isinstance(payload, dict) and "joint_pos" in payload else payload
    if not isinstance(joint_pos_raw, dict):
        raise ValueError(f"Invalid PiPER zero pose payload in {pose_path}: expected dict, got {type(joint_pos_raw)}")

    pose = {
        str(key): float(value)
        for key, value in joint_pos_raw.items()
        if str(key) in PIPER_ACTION_KEYS or str(key) in PIPER_JOINT_ACTION_KEYS
    }
    missing_joint_keys = [key for key in PIPER_JOINT_ACTION_KEYS if key not in pose]
    if missing_joint_keys:
        raise ValueError(f"Invalid PiPER zero pose payload in {pose_path}: missing keys {missing_joint_keys}")

    logging.info("Loaded PiPER zero pose from %s", pose_path)
    return pose


def _smoothstep(alpha: float) -> float:
    bounded = min(max(alpha, 0.0), 1.0)
    return bounded * bounded * (3.0 - 2.0 * bounded)


def _piper_absolute_pose_to_action(robot: Any, absolute_pose: dict[str, float]) -> dict[str, float]:
    if robot.is_calibrated:
        calibration_scale = float(robot.config.calibration_scale)
        action: dict[str, float] = {}
        for key, absolute_deg in absolute_pose.items():
            cal = robot.calibration[key]
            min_deg = float(cal.range_min) / calibration_scale
            max_deg = float(cal.range_max) / calibration_scale
            home_deg = float(cal.homing_offset) / calibration_scale
            bounded = min(max_deg, max(min_deg, float(absolute_deg)))
            centered = bounded - home_deg
            action[key] = -centered if cal.drive_mode else centered
        return action

    if getattr(robot.config, "require_calibration", True):
        raise RuntimeError(
            f"{robot} is not calibrated. Run `lerobot-calibrate --robot.type={robot.config.type} "
            f"--robot.id={robot.id}` before using `--mode=home`."
        )

    return {key: float(value) for key, value in absolute_pose.items()}


def _move_to_zero_pose(
    robot: Any,
    current_pose: dict[str, float],
    target_pose: dict[str, float],
    *,
    duration_s: float,
    fps: int,
    settle_time_s: float,
    sleep_fn=precise_sleep,
) -> None:
    pose_keys = [key for key in _active_pose_keys(include_gripper=True) if key in current_pose and key in target_pose]
    if not pose_keys:
        raise ValueError("No overlapping PiPER pose keys found between current pose and target pose.")

    steps = max(int(round(duration_s * fps)), 1)
    step_dt_s = duration_s / float(steps) if duration_s > 0 else 0.0
    for step_idx in range(1, steps + 1):
        alpha = _smoothstep(step_idx / steps)
        absolute_pose = {
            key: current_pose[key] + (target_pose[key] - current_pose[key]) * alpha for key in pose_keys
        }
        robot.send_action(_piper_absolute_pose_to_action(robot, absolute_pose))
        sleep_fn(step_dt_s)

    robot.send_action(_piper_absolute_pose_to_action(robot, {key: target_pose[key] for key in pose_keys}))
    sleep_fn(settle_time_s)


@parser.wrap()
def piper_zero_home(cfg: PiperZeroHomeConfig):
    init_logging()
    _validate_piper_zero_home_config(cfg)

    pose_path = _resolve_zero_pose_path(cfg)
    robot = make_robot_from_config(cfg.robot)
    include_gripper = bool(cfg.include_gripper and getattr(robot.config, "sync_gripper", True))

    robot.connect(calibrate=cfg.connect_calibrate)
    try:
        if cfg.mode == "capture":
            pose = _save_zero_pose(robot=robot, pose_path=pose_path, include_gripper=include_gripper)
            logging.info("Captured custom zero pose: %s", pose)
            return pose

        if not pose_path.is_file():
            raise FileNotFoundError(
                f"PiPER zero pose file not found: {pose_path}. "
                "Run this script once with `--mode=capture` first."
            )

        target_pose = _load_zero_pose(pose_path)
        if not include_gripper:
            target_pose = {key: value for key, value in target_pose.items() if key != "gripper.pos"}

        current_pose = _extract_zero_pose_from_observation(robot.get_observation(), include_gripper=include_gripper)
        _move_to_zero_pose(
            robot=robot,
            current_pose=current_pose,
            target_pose=target_pose,
            duration_s=cfg.duration_s,
            fps=cfg.fps,
            settle_time_s=cfg.settle_time_s,
        )
        logging.info("PiPER arm returned to saved zero pose in %.2fs.", cfg.duration_s)
        return target_pose
    finally:
        if robot.is_connected:
            robot.disconnect()


def main():
    register_third_party_plugins()
    piper_zero_home()


if __name__ == "__main__":
    main()
