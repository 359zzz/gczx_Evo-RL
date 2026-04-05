import json
from pathlib import Path
from types import SimpleNamespace

import draccus

from lerobot.scripts.lerobot_piper_zero_home import (
    PiperZeroHomeConfig,
    _load_zero_pose,
    _move_to_zero_pose,
    _piper_absolute_pose_to_action,
    _save_zero_pose,
)


class _StaticPiperRobot:
    robot_type = "piper_follower"
    id = "test_piper"

    def __init__(self, observation: dict[str, float], sync_gripper: bool = True):
        self._observation = observation
        self.config = SimpleNamespace(sync_gripper=sync_gripper)

    def get_observation(self):
        return dict(self._observation)


class _PassthroughPiperRobot:
    robot_type = "piper_follower"
    id = "test_piper"
    is_calibrated = False

    def __init__(self):
        self.config = SimpleNamespace(require_calibration=False, sync_gripper=True)
        self.actions_sent: list[dict[str, float]] = []

    def send_action(self, action):
        action_copy = {key: float(value) for key, value in action.items()}
        self.actions_sent.append(action_copy)
        return action_copy


def test_piper_zero_home_parses_config():
    args = [
        "--mode=capture",
        "--robot.type=piper_follower",
        "--robot.id=my_piper",
        "--robot.port=can0",
        "--duration_s=4.0",
        "--fps=60",
    ]
    cfg = draccus.parse(config_class=PiperZeroHomeConfig, config_path=None, args=args)
    assert cfg.mode == "capture"
    assert cfg.robot.type == "piper_follower"
    assert cfg.robot.port == "can0"
    assert cfg.duration_s == 4.0
    assert cfg.fps == 60


def test_save_and_load_zero_pose(tmp_path: Path):
    robot = _StaticPiperRobot(
        observation={
            "joint_1.pos": 1.0,
            "joint_2.pos": 2.0,
            "joint_3.pos": 3.0,
            "joint_4.pos": 4.0,
            "joint_5.pos": 5.0,
            "joint_6.pos": 6.0,
            "gripper.pos": 7.0,
        }
    )
    pose_path = tmp_path / "zero_pose.json"

    saved_pose = _save_zero_pose(robot=robot, pose_path=pose_path, include_gripper=True)
    loaded_pose = _load_zero_pose(pose_path)

    with open(pose_path) as f:
        payload = json.load(f)

    assert saved_pose == loaded_pose
    assert payload["joint_pos"]["joint_1.pos"] == 1.0
    assert payload["joint_pos"]["gripper.pos"] == 7.0


def test_piper_absolute_pose_to_action_uses_calibration_offsets():
    robot = SimpleNamespace(
        is_calibrated=True,
        config=SimpleNamespace(calibration_scale=1000, require_calibration=True),
        calibration={
            "joint_1.pos": SimpleNamespace(drive_mode=0, homing_offset=10000, range_min=0, range_max=20000),
            "joint_2.pos": SimpleNamespace(drive_mode=1, homing_offset=5000, range_min=0, range_max=10000),
        },
    )

    action = _piper_absolute_pose_to_action(
        robot,
        {
            "joint_1.pos": 13.5,
            "joint_2.pos": 7.0,
        },
    )

    assert action == {
        "joint_1.pos": 3.5,
        "joint_2.pos": -2.0,
    }


def test_move_to_zero_pose_interpolates_and_finishes_at_target():
    robot = _PassthroughPiperRobot()
    current_pose = {
        "joint_1.pos": 0.0,
        "joint_2.pos": 0.0,
        "joint_3.pos": 0.0,
        "joint_4.pos": 0.0,
        "joint_5.pos": 0.0,
        "joint_6.pos": 0.0,
        "gripper.pos": 0.0,
    }
    target_pose = {
        "joint_1.pos": 10.0,
        "joint_2.pos": -10.0,
        "joint_3.pos": 20.0,
        "joint_4.pos": -20.0,
        "joint_5.pos": 30.0,
        "joint_6.pos": -30.0,
        "gripper.pos": 5.0,
    }

    _move_to_zero_pose(
        robot=robot,
        current_pose=current_pose,
        target_pose=target_pose,
        duration_s=0.2,
        fps=10,
        settle_time_s=0.0,
        sleep_fn=lambda _: None,
    )

    assert len(robot.actions_sent) > 1
    assert robot.actions_sent[-1] == target_pose
