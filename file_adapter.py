from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from wirecheck import vector

ProgressCallback = Callable[[int, int, str], None]

LSB_MIDPOINT = 2**17
NT_PER_LSB = 100000.0 / 16384.0
STARTUP_CROP_S = 0.1
DEFAULT_BLOCK_SECONDS = 0.5
DEFAULT_MAX_TRAVEL_M = 0.05
DEFAULT_MAX_ALONG_M = 0.2
DEFAULT_Z1_HEIGHT_MM = 10000.0
NOISE_PROBE_HZ = 5.0
DEFAULT_RAW_NAME = "raw.csv"
DEFAULT_POSE_NAME = "pose.csv"
RAW_COLUMNS = ("t_host", "t_mcu", "x", "y", "z")
POSE_COLUMNS = ("t_host", "x", "y", "z", "qw", "qx", "qy", "qz")


@dataclass(frozen=True)
class Position:
    horizontal_m: float
    vertical_m: float
    frame: str = "unspecified"
    sensor_yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        values = (self.horizontal_m, self.vertical_m, self.sensor_yaw_rad)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("position coordinates and yaw must be finite")
        if not self.frame:
            raise ValueError("position frame cannot be empty")


def _read_named_csv(path: Path, columns: tuple[str, ...]) -> NDArray[np.void]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{source}: file not found")
    try:
        table = np.genfromtxt(source, delimiter=",", names=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"{source}: cannot read CSV ({error})") from error
    table = np.atleast_1d(table)
    names = table.dtype.names or ()
    missing = [name for name in columns if name not in names]
    if missing:
        raise ValueError(
            f"{source}: missing column(s) {', '.join(missing)}; "
            f"expected {', '.join(columns)}"
        )
    return table


def quaternion_to_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    q = np.asarray(quaternion, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = (
        np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
    )
    return np.stack(rows, axis=-2)


@dataclass(frozen=True)
class VioTrack:
    time_s: NDArray[np.float64]
    position_m: NDArray[np.float64]
    quaternion: NDArray[np.float64]

    def __post_init__(self) -> None:
        time_s = np.array(self.time_s, dtype=float, copy=True)
        position_m = np.array(self.position_m, dtype=float, copy=True)
        quaternion = np.array(self.quaternion, dtype=float, copy=True)
        if time_s.ndim != 1 or time_s.size < 2:
            raise ValueError("pose track needs at least two timestamps")
        if position_m.shape != (time_s.size, 3):
            raise ValueError("pose positions must have shape (N, 3)")
        if quaternion.shape != (time_s.size, 4):
            raise ValueError("pose quaternions must have shape (N, 4)")
        if np.any(np.diff(time_s) <= 0.0):
            raise ValueError("pose timestamps must increase strictly")
        for value in (time_s, position_m, quaternion):
            value.setflags(write=False)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "position_m", position_m)
        object.__setattr__(self, "quaternion", quaternion)

    @classmethod
    def read(cls, path: Path) -> "VioTrack":
        table = _read_named_csv(path, POSE_COLUMNS)
        time_s = np.asarray(table["t_host"], dtype=float)
        position_m = np.stack(
            [np.asarray(table[name], dtype=float) for name in ("x", "y", "z")], axis=1
        )
        quaternion = np.stack(
            [np.asarray(table[name], dtype=float) for name in ("qw", "qx", "qy", "qz")],
            axis=1,
        )
        finite = np.isfinite(time_s) & np.isfinite(position_m).all(axis=1)
        finite &= np.isfinite(quaternion).all(axis=1)
        time_s, position_m, quaternion = (
            time_s[finite],
            position_m[finite],
            quaternion[finite],
        )
        if time_s.size < 2:
            raise ValueError(f"{path}: fewer than two finite pose rows")
        order = np.argsort(time_s)
        time_s, position_m, quaternion = (
            time_s[order],
            position_m[order],
            quaternion[order],
        )
        dots = np.sum(quaternion[1:] * quaternion[:-1], axis=1)
        signs = np.concatenate([[1.0], np.cumprod(np.where(dots < 0.0, -1.0, 1.0))])
        quaternion = quaternion * signs[:, None]
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        return cls(time_s, position_m, quaternion)

    def at(
        self, time_s: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        query = np.asarray(time_s, dtype=float)
        position = np.stack(
            [np.interp(query, self.time_s, self.position_m[:, axis]) for axis in range(3)],
            axis=1,
        )
        quaternion = np.stack(
            [np.interp(query, self.time_s, self.quaternion[:, axis]) for axis in range(4)],
            axis=1,
        )
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        outside = (query < self.time_s[0]) | (query > self.time_s[-1])
        position[outside] = np.nan
        quaternion[outside] = np.nan
        return position, quaternion


def _analysis_frame(
    rotation: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    forward = -rotation[:, 2]
    horizontal = np.array([forward[0], 0.0, forward[2]])
    if np.linalg.norm(horizontal) < 0.25:
        column = rotation[:, 0]
        horizontal = np.array([-column[0], 0.0, -column[2]])
    across = horizontal / max(float(np.linalg.norm(horizontal)), 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    return across, up, np.cross(across, up)


def load_campaign(
    source: Path,
    *,
    block_seconds: float = DEFAULT_BLOCK_SECONDS,
    lever_arm_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sensor_to_camera: NDArray[np.float64] | None = None,
    max_travel_m: float = DEFAULT_MAX_TRAVEL_M,
    max_along_m: float = DEFAULT_MAX_ALONG_M,
    raw_name: str = DEFAULT_RAW_NAME,
    pose_name: str = DEFAULT_POSE_NAME,
    progress: ProgressCallback | None = None,
) -> vector.RawCollection:
    root = Path(source)
    if not root.is_dir():
        raise ValueError(f"--data must be a directory containing {raw_name} and {pose_name}: {root}")
    if not math.isfinite(block_seconds) or block_seconds <= 0.0:
        raise ValueError("block length must be positive and finite")
    raw = _read_named_csv(root / raw_name, RAW_COLUMNS)
    track = VioTrack.read(root / pose_name)

    t_host = np.asarray(raw["t_host"], dtype=float)
    t_mcu = np.asarray(raw["t_mcu"], dtype=float)
    counts = np.stack([np.asarray(raw[name], dtype=float) for name in ("x", "y", "z")], axis=1)
    keep = t_mcu > STARTUP_CROP_S
    t_host, t_mcu, counts = t_host[keep], t_mcu[keep], counts[keep]
    if t_mcu.size < 2:
        raise ValueError(f"{root}: fewer than two samples after the startup crop")
    if np.any(np.diff(t_mcu) <= 0.0):
        raise ValueError(f"{root}: t_mcu must increase strictly")
    field_nt = (counts - LSB_MIDPOINT) * NT_PER_LSB

    fs_hz = 1.0 / float(np.median(np.diff(t_mcu)))
    block_size = max(256, int(round(block_seconds * fs_hz)))
    positions, quaternions = track.at(t_host)
    rotations = quaternion_to_matrix(quaternions)
    valid = np.asarray(
        np.isfinite(positions).all(axis=1) & np.isfinite(quaternions).all(axis=1),
        dtype=bool,
    )
    first = int(np.argmax(valid))
    if not bool(valid[first]):
        raise ValueError(f"{root}: pose track does not cover the recording")
    across, up, along = _analysis_frame(np.asarray(rotations[first], dtype=float))
    lever = np.asarray(lever_arm_m, dtype=float)
    if sensor_to_camera is None:
        sensor_to_camera = np.eye(3)
    sensor_to_camera = np.asarray(sensor_to_camera, dtype=float)

    origin: NDArray[np.float64] | None = None
    captures: list[vector.RawCapture] = []
    total = t_mcu.size // block_size
    for index in range(total):
        start = index * block_size
        stop = start + block_size
        if not bool(valid[start:stop].all()):
            continue
        middle = start + block_size // 2
        rotation = rotations[middle] @ sensor_to_camera
        sensor_m = positions[middle] + rotations[middle] @ lever
        if origin is None:
            origin = sensor_m
        offset = sensor_m - origin
        travel = float(np.linalg.norm(positions[stop - 1] - positions[start]))
        along_m = float(offset @ along)
        if travel > max_travel_m or abs(along_m) > max_along_m:
            continue
        captures.append(
            vector.RawCapture(
                f"{raw_name}#{index:04d}",
                Position(float(offset @ across), float(offset @ up), "ipad-vio"),
                t_mcu[start:stop].copy(),
                field_nt[start:stop] @ rotation.T,
            )
        )
        if progress is not None:
            progress(index + 1, total, f"block {index:04d}")
    if not captures:
        raise ValueError(
            f"{root}: no usable pose-tagged blocks; check the pose track and "
            "the travel/along-cable limits"
        )
    return vector.RawCollection(root.name, tuple(captures))
