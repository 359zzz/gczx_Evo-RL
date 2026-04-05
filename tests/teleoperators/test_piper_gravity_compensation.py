from pathlib import Path

from lerobot.teleoperators.piper_leader.gravity_compensation import _get_urdf_package_dirs


def test_get_urdf_package_dirs_supports_package_uris():
    urdf = Path("/tmp/repo/src/lerobot/assets/piper_description/urdf/piper_no_gripper_description.urdf")

    package_dirs = _get_urdf_package_dirs(urdf)

    assert package_dirs == [
        "/tmp/repo/src/lerobot/assets",
        "/tmp/repo/src/lerobot/assets/piper_description",
    ]
