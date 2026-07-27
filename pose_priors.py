"""Pose priors for occluded object tracking.

Two priors are implemented:

* ``HandObjectPosePrior`` propagates the object rotation through a stable
  hand--object transform.  HaWoR can reset its world coordinate system in the
  middle of a sequence, so the palm trajectory is stitched with local hand
  increments and discontinuous increments are ignored.
* ``HistoryMemoryPosePrior`` is a lightweight counterpart of EgoAERO's
  keyframe memory pool.  It keeps low-occlusion RGB-D observations, matches
  them to the current frame, lifts matches to 3D, and estimates the current
  object pose while historical poses remain fixed.
* ``HandMotionPosePrior`` propagates an object from a reliable frame with the
  rigid component of the active contact joints rather than assuming a fixed
  palm--object transform.
* ``GraspMemoryPosePrior`` retrieves hand-articulation-matched, low-occlusion
  historical poses and keeps their propagated hypotheses multimodal.
* ``GraspTypePosePrior`` implements category-specific geometric references for
  pinch, clip, multi-finger grab and power hold.
* ``ContactConsistencyPrior`` implements a lightweight COP-style constraint:
  distances from contact joints to canonical object vertices should remain
  stable over a short grasp window.

The module deliberately contains no SAM3D-specific model code, which makes
the geometry independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


POSE_KEYS = ("rotation", "translation", "scale")
_MCP_INDICES = np.array([5, 9, 13, 17], dtype=np.int64)
_FINGER_CHAINS = {
    "thumb": np.array([1, 2, 3, 4], dtype=np.int64),
    "index": np.array([5, 6, 7, 8], dtype=np.int64),
    "middle": np.array([9, 10, 11, 12], dtype=np.int64),
    "ring": np.array([13, 14, 15, 16], dtype=np.int64),
    "pinky": np.array([17, 18, 19, 20], dtype=np.int64),
}
_FINGER_NAMES = tuple(_FINGER_CHAINS)
_TIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def copy_pose(pose: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(pose[key], dtype=torch.float32).detach().cpu().clone()
        for key in POSE_KEYS
    }


def result_to_pose(result: dict[str, Any], prefer_post_opt: bool = True) -> dict[str, torch.Tensor]:
    prefix = "post_opt_" if prefer_post_opt and all(
        f"post_opt_{key}" in result for key in POSE_KEYS
    ) else ""
    pose = {
        key: torch.as_tensor(result[f"{prefix}{key}"], dtype=torch.float32)
        .detach()
        .cpu()
        .clone()
        for key in POSE_KEYS
    }
    if not torch.isfinite(pose["rotation"]).all() or not torch.isfinite(
        pose["translation"]
    ).all():
        return {
            key: torch.as_tensor(result[key], dtype=torch.float32)
            .detach()
            .cpu()
            .clone()
            for key in POSE_KEYS
        }
    return pose


def load_pose_archive(path: str | Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    archive = _torch_load(path)
    raw_frames = archive.get("frames", archive)
    frames: dict[int, dict[str, Any]] = {}
    for key, value in raw_frames.items():
        try:
            frame_idx = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and all(k in value for k in POSE_KEYS):
            frames[frame_idx] = value
    return frames, archive


def pose_to_matrix(pose: dict[str, Any]) -> np.ndarray:
    quat_wxyz = _as_numpy(pose["rotation"]).reshape(-1, 4)[0].astype(np.float64)
    translation = _as_numpy(pose["translation"]).reshape(-1, 3)[0].astype(np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def matrix_to_pose(matrix: np.ndarray, scale: Any) -> dict[str, torch.Tensor]:
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32).detach().cpu().clone()
    if scale_tensor.ndim == 1:
        scale_tensor = scale_tensor.unsqueeze(0)
    return {
        "rotation": torch.from_numpy(quat.astype(np.float32)).unsqueeze(0),
        "translation": torch.from_numpy(matrix[:3, 3].astype(np.float32)).unsqueeze(0),
        "scale": scale_tensor,
    }


def _apply_camera_motion_to_render_pose(
    motion: np.ndarray,
    pose: dict[str, Any],
) -> np.ndarray:
    """Apply camera-space motion to a pose rendered with row-vector rotation.

    PyTorch3D ``Transform3d`` maps object points as ``v @ R + t``.  A camera
    motion fitted as ``D @ p + d`` must consequently update the stored pose as
    ``R_new = R @ D.T`` and ``t_new = D @ t + d``.  Left-multiplying two
    conventional column-vector pose matrices rotates the wrong object axis.
    """
    pose_matrix = pose_to_matrix(pose)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = pose_matrix[:3, :3] @ motion[:3, :3].T
    result[:3, 3] = (
        motion[:3, :3] @ pose_matrix[:3, 3] + motion[:3, 3]
    )
    return result


def pose_errors(pose: dict[str, Any], reference: dict[str, Any]) -> tuple[float, float]:
    pose_matrix = pose_to_matrix(pose)
    reference_matrix = pose_to_matrix(reference)
    translation_error = float(
        np.linalg.norm(pose_matrix[:3, 3] - reference_matrix[:3, 3])
    )
    rotation_error = float(
        Rotation.from_matrix(
            reference_matrix[:3, :3].T @ pose_matrix[:3, :3]
        ).magnitude()
    )
    return translation_error, rotation_error


def blend_poses(
    first: dict[str, Any], second: dict[str, Any], second_weight: float
) -> dict[str, torch.Tensor]:
    weight = float(np.clip(second_weight, 0.0, 1.0))
    first_matrix = pose_to_matrix(first)
    second_matrix = pose_to_matrix(second)
    rotations = Rotation.from_matrix(
        np.stack([first_matrix[:3, :3], second_matrix[:3, :3]], axis=0)
    )
    rotation = Slerp([0.0, 1.0], rotations)([weight]).as_matrix()[0]
    translation = (
        (1.0 - weight) * first_matrix[:3, 3]
        + weight * second_matrix[:3, 3]
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix_to_pose(matrix, first["scale"])


def _rotation_angle(matrix: np.ndarray) -> float:
    return float(Rotation.from_matrix(matrix).magnitude())


def _palm_transform(joints: np.ndarray) -> np.ndarray | None:
    """Build a right-handed palm frame from MANO joints in PyTorch3D camera axes."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (21, 3) or not np.isfinite(joints).all():
        return None

    origin = joints[_MCP_INDICES].mean(axis=0)
    forward = joints[9] - joints[0]
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return None
    forward /= forward_norm

    across = joints[5] - joints[17]
    across -= forward * np.dot(across, forward)
    across_norm = np.linalg.norm(across)
    if across_norm < 1e-6:
        return None
    across /= across_norm

    normal = np.cross(across, forward)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-6:
        return None
    normal /= normal_norm
    across = np.cross(forward, normal)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.stack([across, forward, normal], axis=1)
    transform[:3, 3] = origin
    return transform


class HandObjectPosePrior:
    """Propagate an anchor object pose using HaWoR palm motion."""

    def __init__(
        self,
        hand_meshes_path: str | Path,
        anchor_frame: int,
        anchor_pose: dict[str, Any],
        hand: str = "right",
        jump_translation_threshold: float = 0.15,
        jump_rotation_threshold_deg: float = 60.0,
        flip_xy_to_pytorch3d: bool = True,
    ) -> None:
        if hand not in {"left", "right"}:
            raise ValueError(f"hand must be left or right, got {hand!r}")
        data = np.load(hand_meshes_path, allow_pickle=False)
        joints = np.asarray(data[f"{hand}_joints"], dtype=np.float64).copy()
        valid = np.asarray(
            data.get(f"{hand}_valid", np.ones(joints.shape[0], dtype=bool)),
            dtype=bool,
        )
        if flip_xy_to_pytorch3d:
            joints[..., :2] *= -1.0

        raw: list[np.ndarray | None] = []
        for frame_idx, frame_joints in enumerate(joints):
            raw.append(_palm_transform(frame_joints) if valid[frame_idx] else None)

        self.transforms: list[np.ndarray | None] = [None] * len(raw)
        self.reset_frames: list[int] = []
        previous_raw: np.ndarray | None = None
        previous_stitched: np.ndarray | None = None
        trans_threshold = float(jump_translation_threshold)
        rot_threshold = np.deg2rad(float(jump_rotation_threshold_deg))

        for frame_idx, current_raw in enumerate(raw):
            if current_raw is None:
                continue
            if previous_raw is None or previous_stitched is None:
                current_stitched = current_raw.copy()
            else:
                camera_increment = current_raw @ np.linalg.inv(previous_raw)
                translation_jump = np.linalg.norm(camera_increment[:3, 3])
                rotation_jump = _rotation_angle(camera_increment[:3, :3])
                if translation_jump > trans_threshold or rotation_jump > rot_threshold:
                    # HaWoR occasionally starts a new SLAM/world segment.  At the
                    # reset boundary assume one-frame continuity, then compose
                    # subsequent increments in the new segment in hand-local axes.
                    self.reset_frames.append(frame_idx)
                    current_stitched = previous_stitched.copy()
                else:
                    local_increment = np.linalg.inv(previous_raw) @ current_raw
                    current_stitched = previous_stitched @ local_increment
            self.transforms[frame_idx] = current_stitched
            previous_raw = current_raw
            previous_stitched = current_stitched

        self.anchor_frame = int(anchor_frame)
        if not 0 <= self.anchor_frame < len(self.transforms):
            raise IndexError(f"hand anchor frame {self.anchor_frame} is out of range")
        anchor_hand = self.transforms[self.anchor_frame]
        if anchor_hand is None:
            raise ValueError(f"hand pose is invalid at anchor frame {self.anchor_frame}")
        self.anchor_pose = copy_pose(anchor_pose)
        self.hand_to_object = np.linalg.inv(anchor_hand) @ pose_to_matrix(anchor_pose)

    def get_pose(self, frame_idx: int) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "mode": "hand",
            "anchor_frame": self.anchor_frame,
            "reset_frames": self.reset_frames,
        }
        if not 0 <= frame_idx < len(self.transforms):
            diagnostics["failure"] = "frame_out_of_range"
            return None, diagnostics
        hand_transform = self.transforms[frame_idx]
        if hand_transform is None:
            diagnostics["failure"] = "invalid_hand_pose"
            return None, diagnostics
        object_transform = hand_transform @ self.hand_to_object
        diagnostics["translation"] = object_transform[:3, 3].tolist()
        return matrix_to_pose(object_transform, self.anchor_pose["scale"]), diagnostics


def _read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    if source.shape[0] < 3:
        return None
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    try:
        u, _, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _ransac_rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    count = source.shape[0]
    if count < 3:
        return None, np.zeros(count, dtype=bool)
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(count, dtype=bool)
    best_median = np.inf
    for _ in range(max(int(iterations), 1)):
        indices = rng.choice(count, size=3, replace=False)
        transform = _rigid_fit(source[indices], target[indices])
        if transform is None:
            continue
        predicted = source @ transform[:3, :3].T + transform[:3, 3]
        residual = np.linalg.norm(predicted - target, axis=1)
        inliers = residual < threshold
        inlier_count = int(inliers.sum())
        if inlier_count < 3:
            continue
        median = float(np.median(residual[inliers]))
        if inlier_count > int(best_inliers.sum()) or (
            inlier_count == int(best_inliers.sum()) and median < best_median
        ):
            best_inliers = inliers
            best_median = median
    if int(best_inliers.sum()) < 3:
        return None, best_inliers
    refined = _rigid_fit(source[best_inliers], target[best_inliers])
    return refined, best_inliers


@dataclass
class _MemoryFrame:
    frame_idx: int
    pose: dict[str, torch.Tensor]
    quality: float
    visible_area: int
    depth_ratio: float
    hand_occlusion: float


class HistoryMemoryPosePrior:
    """Low-occlusion keyframe memory with fixed-history RGB-D pose fitting."""

    def __init__(
        self,
        pose_archive_path: str | Path,
        frames_dir: str | Path,
        masks_root: str | Path,
        pointmap_dir: str | Path,
        object_name: str,
        occluder_masks_root: str | Path | None = None,
        occluder_name: str | None = None,
        history_start_frame: int = 0,
        history_end_frame: int | None = None,
        pool_size: int = 24,
        min_frame_gap: int = 3,
        occluder_dilation_px: int = 5,
        max_keyframes: int = 4,
        lookback: int = 120,
        min_matches: int = 8,
        match_ratio: float = 0.78,
        feature_mask_dilation_px: int = 6,
        ransac_threshold: float = 0.035,
        ransac_iterations: int = 384,
        flip_xy_to_pytorch3d: bool = True,
    ) -> None:
        self.pose_frames, _ = load_pose_archive(pose_archive_path)
        self.frames_dir = Path(frames_dir)
        self.masks_root = Path(masks_root)
        self.pointmap_dir = Path(pointmap_dir)
        self.object_name = object_name
        self.occluder_masks_root = (
            Path(occluder_masks_root) if occluder_masks_root else None
        )
        self.occluder_name = occluder_name
        self.occluder_dilation_px = int(occluder_dilation_px)
        self.max_keyframes = int(max_keyframes)
        self.lookback = int(lookback)
        self.min_matches = int(min_matches)
        self.match_ratio = float(match_ratio)
        self.feature_mask_dilation_px = int(feature_mask_dilation_px)
        self.ransac_threshold = float(ransac_threshold)
        self.ransac_iterations = int(ransac_iterations)
        self.flip_xy_to_pytorch3d = bool(flip_xy_to_pytorch3d)
        self.sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._feature_cache: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
        self._pointmap_cache: dict[int, np.ndarray] = {}

        end = history_end_frame if history_end_frame is not None else 10**9
        candidates: list[_MemoryFrame] = []
        for frame_idx in sorted(self.pose_frames):
            if frame_idx < history_start_frame or frame_idx >= end:
                continue
            try:
                object_mask = self._load_object_mask(frame_idx)
                pointmap = np.load(
                    self._pointmap_path(frame_idx), mmap_mode="r"
                )
            except FileNotFoundError:
                continue
            valid_depth = np.isfinite(pointmap).all(axis=2) & (pointmap[..., 2] > 1e-5)
            visible_area = int(object_mask.sum())
            if visible_area == 0:
                continue
            depth_ratio = float((valid_depth & object_mask).sum()) / visible_area
            hand_occlusion = self._hand_occlusion(frame_idx, object_mask)
            candidates.append(
                _MemoryFrame(
                    frame_idx=frame_idx,
                    pose=result_to_pose(self.pose_frames[frame_idx]),
                    quality=0.0,
                    visible_area=visible_area,
                    depth_ratio=depth_ratio,
                    hand_occlusion=hand_occlusion,
                )
            )

        if not candidates:
            raise ValueError("no usable frames found for history memory pool")
        areas = np.array([frame.visible_area for frame in candidates], dtype=np.float64)
        low, high = np.percentile(areas, [10, 90])
        denominator = max(high - low, 1.0)
        for frame in candidates:
            area_score = float(np.clip((frame.visible_area - low) / denominator, 0.0, 1.0))
            frame.quality = (
                0.45 * area_score
                + 0.20 * frame.depth_ratio
                + 0.35 * (1.0 - frame.hand_occlusion)
            )

        # Select the best observation inside temporal bins.  Pure quality
        # ranking over-fills the pool with early, large silhouettes and loses
        # the recent viewpoints needed by the current frame.
        chronological = sorted(candidates, key=lambda item: item.frame_idx)
        selected: list[_MemoryFrame] = []
        for indices in np.array_split(
            np.arange(len(chronological)), min(int(pool_size), len(chronological))
        ):
            if len(indices) == 0:
                continue
            bin_candidates = [chronological[int(index)] for index in indices]
            candidate = max(bin_candidates, key=lambda item: item.quality)
            if all(
                abs(candidate.frame_idx - existing.frame_idx) >= min_frame_gap
                for existing in selected
            ):
                selected.append(candidate)
        self.pool = sorted(selected, key=lambda item: item.frame_idx)

    def _image_path(self, frame_idx: int) -> Path:
        return self.frames_dir / f"{frame_idx:06d}.png"

    def _mask_path(self, frame_idx: int) -> Path:
        return (
            self.masks_root
            / f"frame_{frame_idx:06d}_masks"
            / f"{self.object_name}.png"
        )

    def _pointmap_path(self, frame_idx: int) -> Path:
        return self.pointmap_dir / f"{frame_idx:06d}_pointmap.npy"

    def _load_object_mask(self, frame_idx: int) -> np.ndarray:
        return _read_mask(self._mask_path(frame_idx))

    def _load_pointmap(self, frame_idx: int) -> np.ndarray:
        if frame_idx not in self._pointmap_cache:
            pointmap = np.load(self._pointmap_path(frame_idx)).astype(np.float64)
            if self.flip_xy_to_pytorch3d:
                pointmap[..., :2] *= -1.0
            self._pointmap_cache[frame_idx] = pointmap
        return self._pointmap_cache[frame_idx]

    def _hand_occlusion(self, frame_idx: int, object_mask: np.ndarray) -> float:
        if self.occluder_masks_root is None or not self.occluder_name:
            return 0.0
        path = (
            self.occluder_masks_root
            / f"frame_{frame_idx:06d}_masks"
            / f"{self.occluder_name}.png"
        )
        if not path.is_file():
            return 0.0
        hand = _read_mask(path, object_mask.shape)
        ignored = _dilate(hand, self.occluder_dilation_px)
        return float((object_mask & ignored).sum()) / max(int(object_mask.sum()), 1)

    def _feature_mask(self, frame_idx: int, object_mask: np.ndarray) -> np.ndarray:
        mask = _dilate(object_mask, self.feature_mask_dilation_px)
        if self.occluder_masks_root is not None and self.occluder_name:
            path = (
                self.occluder_masks_root
                / f"frame_{frame_idx:06d}_masks"
                / f"{self.occluder_name}.png"
            )
            if path.is_file():
                hand = _read_mask(path, object_mask.shape)
                mask &= ~_dilate(hand, self.occluder_dilation_px)
        return (mask.astype(np.uint8) * 255)

    def _features(
        self,
        frame_idx: int,
        image: np.ndarray | None = None,
        object_mask: np.ndarray | None = None,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        if image is None and frame_idx in self._feature_cache:
            return self._feature_cache[frame_idx]
        if image is None:
            image = cv2.imread(str(self._image_path(frame_idx)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(self._image_path(frame_idx))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if object_mask is None:
            object_mask = self._load_object_mask(frame_idx)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        keypoints, descriptors = self.sift.detectAndCompute(
            gray, self._feature_mask(frame_idx, object_mask)
        )
        result = (keypoints, descriptors)
        if image is not None:
            self._feature_cache[frame_idx] = result
        return result

    def _matches(
        self,
        current_frame: int,
        current_image: np.ndarray,
        current_mask: np.ndarray,
        memory: _MemoryFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        current_keypoints, current_descriptors = self._features(
            current_frame, current_image, current_mask
        )
        memory_keypoints, memory_descriptors = self._features(memory.frame_idx)
        if current_descriptors is None or memory_descriptors is None:
            return np.empty((0, 3)), np.empty((0, 3)), 0
        if len(current_descriptors) < 2 or len(memory_descriptors) < 2:
            return np.empty((0, 3)), np.empty((0, 3)), 0

        pairs = self.matcher.knnMatch(memory_descriptors, current_descriptors, k=2)
        good = [first for first, second in pairs if first.distance < self.match_ratio * second.distance]
        if not good:
            return np.empty((0, 3)), np.empty((0, 3)), 0

        memory_pointmap = self._load_pointmap(memory.frame_idx)
        current_pointmap = self._load_pointmap(current_frame)
        object_to_camera = pose_to_matrix(memory.pose)
        camera_to_object = np.linalg.inv(object_to_camera)
        canonical_points: list[np.ndarray] = []
        current_points: list[np.ndarray] = []
        used_current: set[int] = set()
        for match in sorted(good, key=lambda item: item.distance):
            if match.trainIdx in used_current:
                continue
            used_current.add(match.trainIdx)
            memory_xy = np.rint(memory_keypoints[match.queryIdx].pt).astype(int)
            current_xy = np.rint(current_keypoints[match.trainIdx].pt).astype(int)
            mx, my = int(memory_xy[0]), int(memory_xy[1])
            cx, cy = int(current_xy[0]), int(current_xy[1])
            if not (
                0 <= my < memory_pointmap.shape[0]
                and 0 <= mx < memory_pointmap.shape[1]
                and 0 <= cy < current_pointmap.shape[0]
                and 0 <= cx < current_pointmap.shape[1]
            ):
                continue
            memory_point = memory_pointmap[my, mx]
            current_point = current_pointmap[cy, cx]
            if not np.isfinite(memory_point).all() or not np.isfinite(current_point).all():
                continue
            if memory_point[2] <= 1e-5 or current_point[2] <= 1e-5:
                continue
            canonical = camera_to_object[:3, :3] @ memory_point + camera_to_object[:3, 3]
            canonical_points.append(canonical)
            current_points.append(current_point)
        if not canonical_points:
            return np.empty((0, 3)), np.empty((0, 3)), len(good)
        return np.stack(canonical_points), np.stack(current_points), len(good)

    def estimate(
        self,
        frame_idx: int,
        image: np.ndarray,
        object_mask: np.ndarray,
        coarse_pose: dict[str, Any] | None,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any] | None, dict[str, Any]]:
        eligible = [
            frame
            for frame in self.pool
            if frame.frame_idx < frame_idx
            and frame.frame_idx >= frame_idx - self.lookback
        ]
        diagnostics: dict[str, Any] = {
            "mode": "history",
            "pool_frames": [frame.frame_idx for frame in self.pool],
            "eligible_frames": [frame.frame_idx for frame in eligible],
        }
        if not eligible:
            diagnostics["failure"] = "no_eligible_keyframes"
            return None, None, diagnostics

        match_sets = []
        for memory in eligible:
            canonical, current, raw_matches = self._matches(
                frame_idx, image, object_mask, memory
            )
            match_sets.append((memory, canonical, current, raw_matches))
        def relevance(item):
            memory, canonical, _, _ = item
            match_score = min(canonical.shape[0] / max(self.min_matches, 1), 1.0)
            recency_score = np.exp(
                -(frame_idx - memory.frame_idx) / max(self.lookback / 3.0, 1.0)
            )
            rotation_score = 0.5
            if coarse_pose is not None:
                _, rotation_error = pose_errors(memory.pose, coarse_pose)
                rotation_score = np.exp(-rotation_error / np.deg2rad(45.0))
            return float(
                0.40 * match_score
                + 0.25 * memory.quality
                + 0.20 * recency_score
                + 0.15 * rotation_score
            )

        match_sets.sort(key=relevance, reverse=True)
        selected = match_sets[: self.max_keyframes]
        diagnostics["selected_keyframes"] = [item[0].frame_idx for item in selected]
        diagnostics["selection_scores"] = {
            item[0].frame_idx: relevance(item) for item in selected
        }
        diagnostics["matches_per_keyframe"] = {
            item[0].frame_idx: int(item[1].shape[0]) for item in selected
        }

        canonical_sets = [item[1] for item in selected if item[1].shape[0] > 0]
        current_sets = [item[2] for item in selected if item[2].shape[0] > 0]
        if canonical_sets:
            canonical = np.concatenate(canonical_sets, axis=0)
            current = np.concatenate(current_sets, axis=0)
        else:
            canonical = np.empty((0, 3), dtype=np.float64)
            current = np.empty((0, 3), dtype=np.float64)

        if canonical.shape[0] >= self.min_matches:
            transform, inliers = _ransac_rigid_fit(
                canonical,
                current,
                threshold=self.ransac_threshold,
                iterations=self.ransac_iterations,
                seed=frame_idx,
            )
            inlier_count = int(inliers.sum())
            diagnostics["total_matches"] = int(canonical.shape[0])
            diagnostics["inlier_count"] = inlier_count
            if transform is not None and inlier_count >= self.min_matches:
                scale = (
                    coarse_pose["scale"]
                    if coarse_pose is not None
                    else selected[0][0].pose["scale"]
                )
                pose = matrix_to_pose(transform, scale)
                if coarse_pose is not None:
                    translation_error, rotation_error = pose_errors(pose, coarse_pose)
                    diagnostics["coarse_translation_delta"] = translation_error
                    diagnostics["coarse_rotation_delta_deg"] = float(
                        np.rad2deg(rotation_error)
                    )
                    if translation_error > 0.30 or rotation_error > np.deg2rad(100.0):
                        diagnostics["failure"] = "rgbd_fit_rejected_as_implausible"
                    else:
                        inlier_ratio = inlier_count / max(int(canonical.shape[0]), 1)
                        support = min(
                            inlier_count / max(2.0 * self.min_matches, 1.0), 1.0
                        )
                        blend_weight = float(
                            np.clip(inlier_ratio * support, 0.15, 0.60)
                        )
                        # EgoAERO includes E_pose to keep sparse RGB-D matches
                        # from producing a large update.  Apply the equivalent
                        # regularization by blending the raw fit with the
                        # chained coarse pose according to RANSAC support.
                        pose = blend_poses(coarse_pose, pose, blend_weight)
                        final_translation_error, final_rotation_error = pose_errors(
                            pose, coarse_pose
                        )
                        diagnostics["rgbd_fit_blend_weight"] = blend_weight
                        diagnostics["regularized_translation_delta"] = (
                            final_translation_error
                        )
                        diagnostics["regularized_rotation_delta_deg"] = float(
                            np.rad2deg(final_rotation_error)
                        )
                        context = {
                            "canonical_points": canonical[inliers],
                            "current_points": current[inliers],
                            "source": "rgbd_matches",
                        }
                        diagnostics["source"] = "rgbd_matches"
                        return pose, context, diagnostics
                else:
                    context = {
                        "canonical_points": canonical[inliers],
                        "current_points": current[inliers],
                        "source": "rgbd_matches",
                    }
                    diagnostics["source"] = "rgbd_matches"
                    return pose, context, diagnostics

        # Transparent objects often provide too few trustworthy RGB features.
        # Fall back to the best low-occlusion keyframe as a conservative
        # rotation prior; current-frame translation remains the coarse pose.
        best_memory = max(selected, key=relevance)[0]
        fallback = copy_pose(best_memory.pose)
        if coarse_pose is not None:
            fallback["translation"] = copy_pose(coarse_pose)["translation"]
            fallback["scale"] = copy_pose(coarse_pose)["scale"]
        diagnostics["source"] = "low_occlusion_rotation_fallback"
        diagnostics["fallback_keyframe"] = best_memory.frame_idx
        return fallback, None, diagnostics

    def pool_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "frame_idx": frame.frame_idx,
                "quality": frame.quality,
                "visible_area": frame.visible_area,
                "depth_ratio": frame.depth_ratio,
                "hand_occlusion": frame.hand_occlusion,
            }
            for frame in self.pool
        ]


def history_alignment_score(
    pose: dict[str, Any], context: dict[str, Any], sigma: float = 0.035
) -> tuple[float, float]:
    canonical = np.asarray(context["canonical_points"], dtype=np.float64)
    current = np.asarray(context["current_points"], dtype=np.float64)
    if canonical.shape[0] == 0:
        return 0.0, float("inf")
    transform = pose_to_matrix(pose)
    predicted = canonical @ transform[:3, :3].T + transform[:3, 3]
    residual = np.linalg.norm(predicted - current, axis=1)
    median = float(np.median(residual))
    score = float(np.exp(-median / max(float(sigma), 1e-8)))
    return score, median


# ---------------------------------------------------------------------------
# MANO hand sequence shared by the contact-aware priors
# ---------------------------------------------------------------------------


def _safe_unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8 or not np.isfinite(norm):
        return None
    return vector / norm


def _make_frame(
    origin: np.ndarray,
    first_axis: np.ndarray,
    second_axis: np.ndarray,
) -> np.ndarray | None:
    """Create a right-handed frame while keeping ``first_axis`` unchanged."""
    first = _safe_unit(first_axis)
    if first is None:
        return None
    second = np.asarray(second_axis, dtype=np.float64)
    second = second - first * float(np.dot(first, second))
    second = _safe_unit(second)
    if second is None:
        return None
    third = _safe_unit(np.cross(first, second))
    if third is None:
        return None
    second = np.cross(third, first)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.stack([first, second, third], axis=1)
    transform[:3, 3] = np.asarray(origin, dtype=np.float64)
    return transform


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest proper rotation taking one unit vector to another."""
    source_unit = _safe_unit(source)
    target_unit = _safe_unit(target)
    if source_unit is None or target_unit is None:
        return np.eye(3, dtype=np.float64)
    cross = np.cross(source_unit, target_unit)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, source_unit))) > 0.8:
            helper = np.array([0.0, 1.0, 0.0])
        axis = _safe_unit(np.cross(source_unit, helper))
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / sine
    angle = np.arctan2(sine, cosine)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _finger_curls(joints: np.ndarray) -> dict[str, float]:
    curls: dict[str, float] = {}
    for name, chain in _FINGER_CHAINS.items():
        segments = np.diff(joints[chain], axis=0)
        total = 0.0
        for first, second in zip(segments[:-1], segments[1:]):
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator < 1e-10:
                continue
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
            total += float(np.arccos(cosine))
        curls[name] = total
    return curls


class ManoHandSequence:
    """Load camera-space MANO and align its scale with SAM3D/MoGe point maps.

    HaWoR's monocular hand mesh and MoGe point maps have different metric
    scales.  Their image projections are nevertheless shared.  For every
    requested frame we project the MANO vertices into the hand mask and use
    the median ``pointmap_z / mano_z`` ratio.  A short temporal median removes
    transparent-object depth outliers.
    """

    def __init__(
        self,
        hand_meshes_path: str | Path,
        hand: str = "left",
        pointmap_dir: str | Path | None = None,
        hand_masks_root: str | Path | None = None,
        hand_mask_name: str | None = None,
        hand_focal: float = 609.5352935791016,
        coordinate_scale: float = 1.7,
        scale_smoothing_radius: int = 2,
        flip_xy_to_pytorch3d: bool = True,
    ) -> None:
        if hand not in {"left", "right"}:
            raise ValueError(f"hand must be left or right, got {hand!r}")
        data = np.load(hand_meshes_path, allow_pickle=False)
        joints_key = f"{hand}_joints"
        if joints_key not in data:
            available = [key for key in ("left_joints", "right_joints") if key in data]
            raise KeyError(f"{joints_key} is missing; available joint arrays: {available}")
        self.hand = hand
        self.raw_joints = np.asarray(data[joints_key], dtype=np.float64)
        vertices_key = f"{hand}_vertices"
        self.raw_vertices = (
            np.asarray(data[vertices_key], dtype=np.float64)
            if vertices_key in data
            else None
        )
        if "frame_indices" in data:
            frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        else:
            frame_indices = np.arange(self.raw_joints.shape[0], dtype=np.int64)
        if len(frame_indices) != self.raw_joints.shape[0]:
            raise ValueError("frame_indices and MANO arrays have different lengths")
        valid_key = f"{hand}_valid"
        valid = np.asarray(
            data[valid_key] if valid_key in data else np.ones(len(frame_indices)),
            dtype=bool,
        )
        self.frame_to_row = {
            int(frame): int(row)
            for row, frame in enumerate(frame_indices)
            if bool(valid[row])
        }
        self.pointmap_dir = Path(pointmap_dir) if pointmap_dir else None
        self.hand_masks_root = Path(hand_masks_root) if hand_masks_root else None
        self.hand_mask_name = hand_mask_name
        self.hand_focal = float(hand_focal)
        self.coordinate_scale = float(coordinate_scale)
        self.scale_smoothing_radius = max(int(scale_smoothing_radius), 0)
        self.flip_xy_to_pytorch3d = bool(flip_xy_to_pytorch3d)
        self._raw_scale_cache: dict[int, float | None] = {}
        self._smooth_scale_cache: dict[int, float] = {}

    def has_frame(self, frame_idx: int) -> bool:
        return int(frame_idx) in self.frame_to_row

    def _row(self, frame_idx: int) -> int:
        try:
            return self.frame_to_row[int(frame_idx)]
        except KeyError as exc:
            raise IndexError(f"MANO is invalid or unavailable at frame {frame_idx}") from exc

    def _hand_mask_path(self, frame_idx: int) -> Path | None:
        if self.hand_masks_root is None or not self.hand_mask_name:
            return None
        return (
            self.hand_masks_root
            / f"frame_{frame_idx:06d}_masks"
            / f"{self.hand_mask_name}.png"
        )

    def _estimate_raw_scale(self, frame_idx: int) -> float | None:
        frame_idx = int(frame_idx)
        if frame_idx in self._raw_scale_cache:
            return self._raw_scale_cache[frame_idx]
        result: float | None = None
        mask_path = self._hand_mask_path(frame_idx)
        if (
            self.raw_vertices is not None
            and self.pointmap_dir is not None
            and mask_path is not None
            and mask_path.is_file()
            and self.has_frame(frame_idx)
        ):
            pointmap_path = self.pointmap_dir / f"{frame_idx:06d}_pointmap.npy"
            if pointmap_path.is_file():
                mask = _read_mask(mask_path)
                vertices = self.raw_vertices[self._row(frame_idx)]
                height, width = mask.shape
                center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
                safe_z = np.maximum(vertices[:, 2:3], 1e-8)
                pixels = np.rint(
                    vertices[:, :2] / safe_z * self.hand_focal + center
                ).astype(np.int64)
                inside = (
                    (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < height)
                )
                clipped_x = np.clip(pixels[:, 0], 0, width - 1)
                clipped_y = np.clip(pixels[:, 1], 0, height - 1)
                inside &= mask[clipped_y, clipped_x]
                if int(inside.sum()) >= 24:
                    pointmap = np.load(pointmap_path, mmap_mode="r")
                    depth = np.asarray(
                        pointmap[pixels[inside, 1], pixels[inside, 0], 2],
                        dtype=np.float64,
                    )
                    ratios = depth / np.maximum(vertices[inside, 2], 1e-8)
                    ratios = ratios[
                        np.isfinite(ratios) & (ratios > 0.25) & (ratios < 5.0)
                    ]
                    if ratios.size >= 24:
                        lower, upper = np.percentile(ratios, [15.0, 85.0])
                        trimmed = ratios[(ratios >= lower) & (ratios <= upper)]
                        if trimmed.size:
                            result = float(np.median(trimmed))
        self._raw_scale_cache[frame_idx] = result
        return result

    def scale_at(self, frame_idx: int) -> float:
        frame_idx = int(frame_idx)
        if frame_idx in self._smooth_scale_cache:
            return self._smooth_scale_cache[frame_idx]
        estimates = []
        for neighbour in range(
            frame_idx - self.scale_smoothing_radius,
            frame_idx + self.scale_smoothing_radius + 1,
        ):
            if not self.has_frame(neighbour):
                continue
            estimate = self._estimate_raw_scale(neighbour)
            if estimate is not None:
                estimates.append(estimate)
        scale = float(np.median(estimates)) if estimates else self.coordinate_scale
        self._smooth_scale_cache[frame_idx] = scale
        return scale

    def _convert(self, points: np.ndarray, scale: float) -> np.ndarray:
        converted = np.asarray(points, dtype=np.float64).copy() * float(scale)
        if self.flip_xy_to_pytorch3d:
            converted[..., :2] *= -1.0
        return converted

    def joints(self, frame_idx: int, common_scale: float | None = None) -> np.ndarray:
        scale = self.scale_at(frame_idx) if common_scale is None else float(common_scale)
        return self._convert(self.raw_joints[self._row(frame_idx)], scale)

    def vertices(self, frame_idx: int, common_scale: float | None = None) -> np.ndarray:
        if self.raw_vertices is None:
            raise ValueError("MANO vertices are required for this operation")
        scale = self.scale_at(frame_idx) if common_scale is None else float(common_scale)
        return self._convert(self.raw_vertices[self._row(frame_idx)], scale)

    def articulation_descriptor(self, frame_idx: int) -> np.ndarray:
        joints = self.raw_joints[self._row(frame_idx)]
        palm = _palm_transform(joints)
        if palm is None:
            raise ValueError(f"cannot construct palm frame at {frame_idx}")
        scale = max(float(np.linalg.norm(joints[12] - joints[0])), 1e-6)
        local = (joints - palm[:3, 3]) @ palm[:3, :3] / scale
        curls = _finger_curls(joints)
        thumb_distances = np.linalg.norm(joints[_TIP_INDICES[1:]] - joints[4], axis=1) / scale
        descriptor = np.concatenate(
            [
                local.reshape(-1),
                0.35 * np.array([curls[name] for name in _FINGER_NAMES]),
                0.5 * thumb_distances,
            ]
        )
        return descriptor.astype(np.float64)

    def analyze_grasp(
        self, frame_idx: int, override: str = "auto"
    ) -> dict[str, Any]:
        joints = self.raw_joints[self._row(frame_idx)]
        curls = _finger_curls(joints)
        palm_length = max(float(np.linalg.norm(joints[12] - joints[0])), 1e-6)
        thumb_distances = np.linalg.norm(joints[_TIP_INDICES[1:]] - joints[4], axis=1)
        thumb_ratios = thumb_distances / palm_length
        nonthumb_curls = np.array(
            [curls[name] for name in ("index", "middle", "ring", "pinky")]
        )
        curled_count = int((nonthumb_curls > 1.05).sum())
        close_thumb_index = int(np.argmin(thumb_ratios))

        nonthumb_tips = joints[_TIP_INDICES[1:]]
        pairwise = np.linalg.norm(
            nonthumb_tips[:, None, :] - nonthumb_tips[None, :, :], axis=2
        )
        pairwise += np.eye(4) * 1e6
        closest_pair_flat = int(np.argmin(pairwise))
        pair_first, pair_second = np.unravel_index(closest_pair_flat, pairwise.shape)

        if override != "auto":
            grasp_type = override
        elif float(nonthumb_curls.mean()) >= 1.25 and curled_count >= 3:
            grasp_type = "power_hold"
        elif float(thumb_ratios.min()) < 0.48 and curled_count <= 2:
            grasp_type = "pinch"
        elif float(pairwise[pair_first, pair_second] / palm_length) < 0.35 and curled_count <= 2:
            grasp_type = "clip"
        elif curled_count >= 2:
            grasp_type = "grab"
        else:
            grasp_type = "clip"

        if grasp_type == "pinch":
            active_fingers = ["thumb", _FINGER_NAMES[close_thumb_index + 1]]
        elif grasp_type == "clip":
            active_fingers = [
                _FINGER_NAMES[pair_first + 1],
                _FINGER_NAMES[pair_second + 1],
            ]
        elif grasp_type == "grab":
            active_fingers = ["thumb"] + [
                name
                for name in ("index", "middle", "ring", "pinky")
                if curls[name] > 0.75
            ]
        else:
            active_fingers = list(_FINGER_NAMES)

        return {
            "type": grasp_type,
            "curls": {name: float(value) for name, value in curls.items()},
            "thumb_tip_ratios": thumb_ratios.tolist(),
            "curled_nonthumb_count": curled_count,
            "active_fingers": active_fingers,
            "opposing_finger": _FINGER_NAMES[close_thumb_index + 1],
            "clip_fingers": [
                _FINGER_NAMES[pair_first + 1],
                _FINGER_NAMES[pair_second + 1],
            ],
        }

    def contact_joint_indices(self, analysis: dict[str, Any]) -> np.ndarray:
        grasp_type = analysis["type"]
        active = analysis["active_fingers"]
        indices: list[int] = []
        for name in active:
            chain = _FINGER_CHAINS[name]
            if grasp_type == "power_hold":
                indices.extend(chain[1:].tolist())
            elif grasp_type in {"pinch", "clip"}:
                indices.extend(chain[-2:].tolist())
            else:
                indices.extend(chain[1:].tolist())
        if len(set(indices)) < 3:
            indices.extend([5, 9, 13])
        return np.array(sorted(set(indices)), dtype=np.int64)

    def rigid_motion(
        self,
        source_frame: int,
        target_frame: int,
        joint_indices: Iterable[int],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        common_scale = float(
            np.median([self.scale_at(source_frame), self.scale_at(target_frame)])
        )
        source = self.joints(source_frame, common_scale=common_scale)
        target = self.joints(target_frame, common_scale=common_scale)
        indices = np.asarray(list(joint_indices), dtype=np.int64)
        transform = _rigid_fit(source[indices], target[indices])
        diagnostics: dict[str, Any] = {
            "source_frame": int(source_frame),
            "target_frame": int(target_frame),
            "joint_indices": indices.tolist(),
            "mano_to_pointmap_scale": common_scale,
        }
        if transform is None:
            diagnostics["failure"] = "rigid_fit_failed"
            return None, diagnostics

        prediction = source[indices] @ transform[:3, :3].T + transform[:3, 3]
        residuals = np.linalg.norm(prediction - target[indices], axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        inliers = residuals <= median + max(2.5 * mad, 0.0025)
        if int(inliers.sum()) >= 3 and int(inliers.sum()) < len(indices):
            refined = _rigid_fit(source[indices[inliers]], target[indices[inliers]])
            if refined is not None:
                transform = refined
                prediction = source[indices] @ transform[:3, :3].T + transform[:3, 3]
                residuals = np.linalg.norm(prediction - target[indices], axis=1)
        diagnostics["fit_median_mm"] = float(np.median(residuals) * 1000.0)
        diagnostics["fit_max_mm"] = float(np.max(residuals) * 1000.0)
        diagnostics["rotation_delta_deg"] = float(
            np.rad2deg(_rotation_angle(transform[:3, :3]))
        )
        return transform, diagnostics


# ---------------------------------------------------------------------------
# Contact-joint motion propagation
# ---------------------------------------------------------------------------


class HandMotionPosePrior:
    """Propagate a reliable object pose with active contact-joint motion."""

    def __init__(
        self,
        hand_sequence: ManoHandSequence,
        anchor_frame: int,
        anchor_pose: dict[str, Any],
        grasp_type: str = "auto",
        coarse_blend_weight: float = 0.0,
    ) -> None:
        if not hand_sequence.has_frame(anchor_frame):
            raise ValueError(f"hand is unavailable at anchor frame {anchor_frame}")
        self.hand_sequence = hand_sequence
        self.anchor_frame = int(anchor_frame)
        self.anchor_pose = copy_pose(anchor_pose)
        self.grasp_type = grasp_type
        self.coarse_blend_weight = float(np.clip(coarse_blend_weight, 0.0, 1.0))

    def estimate(
        self,
        frame_idx: int,
        coarse_pose: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor] | None, list[dict[str, torch.Tensor]], dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "mode": "hand_motion",
            "anchor_frame": self.anchor_frame,
        }
        if not self.hand_sequence.has_frame(frame_idx):
            diagnostics["failure"] = "invalid_hand_pose"
            return None, [], diagnostics
        analysis = self.hand_sequence.analyze_grasp(frame_idx, self.grasp_type)
        indices = self.hand_sequence.contact_joint_indices(analysis)
        motion, motion_diagnostics = self.hand_sequence.rigid_motion(
            self.anchor_frame, frame_idx, indices
        )
        diagnostics.update(motion_diagnostics)
        diagnostics["grasp"] = analysis
        if motion is None:
            return None, [], diagnostics
        predicted_matrix = _apply_camera_motion_to_render_pose(
            motion, self.anchor_pose
        )
        scale = coarse_pose["scale"] if coarse_pose is not None else self.anchor_pose["scale"]
        predicted = matrix_to_pose(predicted_matrix, scale)
        if coarse_pose is not None and self.coarse_blend_weight > 0.0:
            predicted = blend_poses(
                predicted, coarse_pose, second_weight=self.coarse_blend_weight
            )
            diagnostics["coarse_blend_weight"] = self.coarse_blend_weight
        diagnostics["predicted_translation"] = (
            predicted["translation"].reshape(-1, 3)[0].tolist()
        )
        return predicted, [predicted], diagnostics


# ---------------------------------------------------------------------------
# Sequential PICO palm-guided propagation
# ---------------------------------------------------------------------------


class PicoHandChainPosePrior:
    """Propagate the previous SAM3D pose with a PICO palm increment.

    PICO palm and camera transforms share one world frame.  SAM3D stores its
    rendered rotation for row-vector points, so the stored rotation is
    transposed when entering/leaving conventional column-vector SE(3).
    """

    def __init__(self, pose_npz_path: str | Path, hand: str = "right") -> None:
        if hand not in {"left", "right"}:
            raise ValueError(f"hand must be left or right, got {hand!r}")
        data = np.load(pose_npz_path, allow_pickle=False)
        palm_key = f"{hand}_palm_to_world"
        if "camera_to_world" not in data:
            raise KeyError("camera_to_world is missing from PICO pose archive")
        if palm_key not in data:
            raise KeyError(f"{palm_key} is missing from PICO pose archive")

        self.camera_to_world = np.asarray(
            data["camera_to_world"], dtype=np.float64
        )
        self.palm_to_world = np.asarray(data[palm_key], dtype=np.float64)
        if self.camera_to_world.shape[1:] != (4, 4):
            raise ValueError("camera_to_world must have shape [N, 4, 4]")
        if self.palm_to_world.shape != self.camera_to_world.shape:
            raise ValueError(
                f"{palm_key} and camera_to_world must have the same shape"
            )
        if "frame_indices" in data:
            frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        else:
            frame_indices = np.arange(len(self.camera_to_world), dtype=np.int64)
        if len(frame_indices) != len(self.camera_to_world):
            raise ValueError("frame_indices and PICO transforms have different lengths")
        self.frame_to_row = {
            int(frame_idx): int(row)
            for row, frame_idx in enumerate(frame_indices)
        }
        self.hand = hand
        self.pose_npz_path = str(pose_npz_path)

    def has_frame(self, frame_idx: int) -> bool:
        return int(frame_idx) in self.frame_to_row

    @staticmethod
    def _render_pose_to_column_transform(
        pose: dict[str, Any],
    ) -> np.ndarray:
        stored = pose_to_matrix(pose)
        transform = stored.copy()
        transform[:3, :3] = stored[:3, :3].T
        return transform

    @staticmethod
    def _column_transform_to_render_pose(
        transform: np.ndarray,
        scale: Any,
    ) -> dict[str, torch.Tensor]:
        stored = transform.copy()
        stored[:3, :3] = transform[:3, :3].T
        return matrix_to_pose(stored, scale)

    def estimate(
        self,
        source_frame: int,
        target_frame: int,
        source_pose: dict[str, Any],
        apply_hand_motion: bool = True,
    ) -> tuple[dict[str, torch.Tensor] | None, list[dict[str, torch.Tensor]], dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "mode": "pico_hand_chain",
            "source_frame": int(source_frame),
            "target_frame": int(target_frame),
            "apply_hand_motion": bool(apply_hand_motion),
        }
        if not self.has_frame(source_frame) or not self.has_frame(target_frame):
            diagnostics["failure"] = "frame_out_of_range"
            return None, [], diagnostics

        source_row = self.frame_to_row[int(source_frame)]
        target_row = self.frame_to_row[int(target_frame)]
        source_camera = self.camera_to_world[source_row]
        target_camera = self.camera_to_world[target_row]
        source_object_camera = self._render_pose_to_column_transform(source_pose)
        source_object_world = source_camera @ source_object_camera

        if apply_hand_motion:
            source_palm = self.palm_to_world[source_row]
            target_palm = self.palm_to_world[target_row]
            palm_to_object = np.linalg.inv(source_palm) @ source_object_world
            target_object_world = target_palm @ palm_to_object
            palm_delta = target_palm @ np.linalg.inv(source_palm)
            diagnostics["palm_rotation_delta_deg"] = float(
                np.rad2deg(_rotation_angle(palm_delta[:3, :3]))
            )
            diagnostics["palm_translation_delta_m"] = float(
                np.linalg.norm(palm_delta[:3, 3])
            )
        else:
            # The object is physically static in the PICO world frame.  Only
            # its camera-frame pose changes as the headset camera moves.
            target_object_world = source_object_world

        target_object_camera = np.linalg.inv(target_camera) @ target_object_world
        predicted = self._column_transform_to_render_pose(
            target_object_camera, source_pose["scale"]
        )
        diagnostics["predicted_translation"] = (
            predicted["translation"].reshape(-1, 3)[0].tolist()
        )
        return predicted, [predicted], diagnostics


# ---------------------------------------------------------------------------
# Low-occlusion grasp-pose memory
# ---------------------------------------------------------------------------


@dataclass
class _GraspMemoryFrame:
    frame_idx: int
    pose: dict[str, torch.Tensor]
    descriptor: np.ndarray
    grasp_type: str
    visible_area: int
    proximity: float
    quality: float = 0.0


class GraspMemoryPosePrior:
    """Retrieve similar historical grasps and propagate all top hypotheses."""

    def __init__(
        self,
        hand_sequence: ManoHandSequence,
        pose_archive_path: str | Path,
        masks_root: str | Path,
        object_name: str,
        hand_masks_root: str | Path | None,
        hand_mask_name: str | None,
        history_start_frame: int = 0,
        history_end_frame: int | None = None,
        pool_size: int = 32,
        top_k: int = 4,
        lookback: int = 160,
        grasp_type: str = "auto",
        mask_dilation_px: int = 6,
    ) -> None:
        pose_frames, _ = load_pose_archive(pose_archive_path)
        self.hand_sequence = hand_sequence
        self.masks_root = Path(masks_root)
        self.object_name = object_name
        self.hand_masks_root = Path(hand_masks_root) if hand_masks_root else None
        self.hand_mask_name = hand_mask_name
        self.top_k = max(int(top_k), 1)
        self.lookback = max(int(lookback), 1)
        self.grasp_type = grasp_type
        self.mask_dilation_px = max(int(mask_dilation_px), 0)
        end = history_end_frame if history_end_frame is not None else 10**9

        candidates: list[_GraspMemoryFrame] = []
        for frame_idx in sorted(pose_frames):
            if not (history_start_frame <= frame_idx < end):
                continue
            if not hand_sequence.has_frame(frame_idx):
                continue
            object_path = (
                self.masks_root
                / f"frame_{frame_idx:06d}_masks"
                / f"{self.object_name}.png"
            )
            if not object_path.is_file():
                continue
            object_mask = _read_mask(object_path)
            visible_area = int(object_mask.sum())
            if visible_area == 0:
                continue
            proximity = 0.0
            if self.hand_masks_root is not None and self.hand_mask_name:
                hand_path = (
                    self.hand_masks_root
                    / f"frame_{frame_idx:06d}_masks"
                    / f"{self.hand_mask_name}.png"
                )
                if hand_path.is_file():
                    hand_mask = _read_mask(hand_path, object_mask.shape)
                    proximity = float(
                        (object_mask & _dilate(hand_mask, self.mask_dilation_px)).sum()
                    ) / max(visible_area, 1)
            try:
                descriptor = hand_sequence.articulation_descriptor(frame_idx)
                analysis = hand_sequence.analyze_grasp(frame_idx, grasp_type)
            except (IndexError, ValueError):
                continue
            candidates.append(
                _GraspMemoryFrame(
                    frame_idx=frame_idx,
                    pose=result_to_pose(pose_frames[frame_idx]),
                    descriptor=descriptor,
                    grasp_type=analysis["type"],
                    visible_area=visible_area,
                    proximity=proximity,
                )
            )
        if not candidates:
            raise ValueError("no usable hand-pose memory frames")

        areas = np.array([item.visible_area for item in candidates], dtype=np.float64)
        low, high = np.percentile(areas, [10.0, 90.0])
        denominator = max(float(high - low), 1.0)
        for item in candidates:
            area_score = float(np.clip((item.visible_area - low) / denominator, 0.0, 1.0))
            item.quality = 0.75 * area_score + 0.25 * (1.0 - item.proximity)

        chronological = sorted(candidates, key=lambda item: item.frame_idx)
        selected: list[_GraspMemoryFrame] = []
        for indices in np.array_split(
            np.arange(len(chronological)), min(max(int(pool_size), 1), len(chronological))
        ):
            if len(indices) == 0:
                continue
            selected.append(
                max((chronological[int(index)] for index in indices), key=lambda x: x.quality)
            )
        self.pool = sorted(selected, key=lambda item: item.frame_idx)

    @staticmethod
    def _medoid(priors: list[dict[str, torch.Tensor]]) -> int:
        if len(priors) <= 1:
            return 0
        matrices = [pose_to_matrix(pose) for pose in priors]
        costs = np.zeros(len(priors), dtype=np.float64)
        for first in range(len(priors)):
            for second in range(first + 1, len(priors)):
                translation = np.linalg.norm(
                    matrices[first][:3, 3] - matrices[second][:3, 3]
                ) / 0.05
                rotation = _rotation_angle(
                    matrices[first][:3, :3].T @ matrices[second][:3, :3]
                ) / np.deg2rad(30.0)
                distance = float(translation + rotation)
                costs[first] += distance
                costs[second] += distance
        return int(np.argmin(costs))

    def estimate(
        self,
        frame_idx: int,
        coarse_pose: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor] | None, list[dict[str, torch.Tensor]], dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "mode": "hand_memory",
            "pool_frames": [item.frame_idx for item in self.pool],
        }
        if not self.hand_sequence.has_frame(frame_idx):
            diagnostics["failure"] = "invalid_hand_pose"
            return None, [], diagnostics
        descriptor = self.hand_sequence.articulation_descriptor(frame_idx)
        analysis = self.hand_sequence.analyze_grasp(frame_idx, self.grasp_type)
        eligible = [
            item
            for item in self.pool
            if item.frame_idx < frame_idx and item.frame_idx >= frame_idx - self.lookback
        ]
        if not eligible:
            diagnostics["failure"] = "no_eligible_memory"
            return None, [], diagnostics

        def retrieval_cost(item: _GraspMemoryFrame) -> float:
            descriptor_distance = float(
                np.sqrt(np.mean(np.square(item.descriptor - descriptor)))
            )
            class_penalty = 0.0 if item.grasp_type == analysis["type"] else 0.20
            age_penalty = 0.08 * (frame_idx - item.frame_idx) / self.lookback
            quality_penalty = 0.12 * (1.0 - item.quality)
            return descriptor_distance + class_penalty + age_penalty + quality_penalty

        retrieved = sorted(eligible, key=retrieval_cost)[: self.top_k]
        indices = self.hand_sequence.contact_joint_indices(analysis)
        priors: list[dict[str, torch.Tensor]] = []
        used_frames: list[int] = []
        fit_diagnostics: dict[int, dict[str, Any]] = {}
        for memory in retrieved:
            motion, fit = self.hand_sequence.rigid_motion(
                memory.frame_idx, frame_idx, indices
            )
            fit_diagnostics[memory.frame_idx] = fit
            if motion is None:
                continue
            matrix = _apply_camera_motion_to_render_pose(motion, memory.pose)
            scale = coarse_pose["scale"] if coarse_pose is not None else memory.pose["scale"]
            priors.append(matrix_to_pose(matrix, scale))
            used_frames.append(memory.frame_idx)
        diagnostics.update(
            {
                "grasp": analysis,
                "retrieved_frames": [item.frame_idx for item in retrieved],
                "retrieval_costs": {
                    item.frame_idx: retrieval_cost(item) for item in retrieved
                },
                "used_frames": used_frames,
                "motion_fits": fit_diagnostics,
            }
        )
        if not priors:
            diagnostics["failure"] = "all_memory_motion_fits_failed"
            return None, [], diagnostics
        medoid_index = self._medoid(priors)
        diagnostics["medoid_frame"] = used_frames[medoid_index]
        return priors[medoid_index], priors, diagnostics

    def pool_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "frame_idx": item.frame_idx,
                "quality": item.quality,
                "visible_area": item.visible_area,
                "hand_proximity": item.proximity,
                "grasp_type": item.grasp_type,
            }
            for item in self.pool
        ]


# ---------------------------------------------------------------------------
# User-proposed grasp-category geometric references
# ---------------------------------------------------------------------------


class GraspTypePosePrior:
    """Category-specific object reference for grab/clip/pinch/power hold."""

    def __init__(
        self,
        hand_sequence: ManoHandSequence,
        anchor_frame: int,
        anchor_pose: dict[str, Any],
        canonical_axis: np.ndarray,
        grasp_type: str = "auto",
    ) -> None:
        self.hand_sequence = hand_sequence
        self.anchor_frame = int(anchor_frame)
        self.anchor_pose = copy_pose(anchor_pose)
        axis = _safe_unit(canonical_axis)
        if axis is None:
            raise ValueError("canonical object axis is degenerate")
        self.canonical_axis = axis
        self.grasp_type = grasp_type

    def _grasp_frame(
        self,
        frame_idx: int,
        analysis: dict[str, Any],
        common_scale: float,
    ) -> np.ndarray | None:
        joints = self.hand_sequence.joints(frame_idx, common_scale=common_scale)
        across = joints[5] - joints[17]
        forward = joints[9] - joints[0]
        palm_normal = np.cross(across, forward)
        grasp_type = analysis["type"]
        if grasp_type == "pinch":
            opposing = analysis["opposing_finger"]
            opposing_tip = int(_FINGER_CHAINS[opposing][-1])
            origin = 0.5 * (joints[4] + joints[opposing_tip])
            return _make_frame(origin, joints[opposing_tip] - joints[4], forward)
        if grasp_type == "clip":
            first, second = analysis["clip_fingers"]
            first_tip = int(_FINGER_CHAINS[first][-1])
            second_tip = int(_FINGER_CHAINS[second][-1])
            origin = 0.5 * (joints[first_tip] + joints[second_tip])
            return _make_frame(origin, joints[second_tip] - joints[first_tip], forward)
        if grasp_type == "grab":
            active_tips = [int(_FINGER_CHAINS[name][-1]) for name in analysis["active_fingers"]]
            origin = joints[active_tips].mean(axis=0)
            return _make_frame(origin, palm_normal, across)
        # For a power hold the cup/bottle axis normally follows the hand's
        # wrist-to-finger tube.  Only constrain this axis; axial twist remains
        # deliberately ambiguous for a rotationally symmetric beaker.
        distal = np.array([3, 4, 7, 8, 11, 12, 15, 16, 19, 20], dtype=np.int64)
        origin = joints[distal].mean(axis=0)
        return _make_frame(origin, forward, across)

    def estimate(
        self,
        frame_idx: int,
        coarse_pose: dict[str, Any] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor] | None,
        list[dict[str, torch.Tensor]],
        dict[str, Any] | None,
        dict[str, Any],
    ]:
        diagnostics: dict[str, Any] = {
            "mode": "grasp_type",
            "anchor_frame": self.anchor_frame,
        }
        if not self.hand_sequence.has_frame(frame_idx):
            diagnostics["failure"] = "invalid_hand_pose"
            return None, [], None, diagnostics
        target_analysis = self.hand_sequence.analyze_grasp(frame_idx, self.grasp_type)
        anchor_analysis = self.hand_sequence.analyze_grasp(
            self.anchor_frame,
            target_analysis["type"] if self.grasp_type == "auto" else self.grasp_type,
        )
        common_scale = float(
            np.median(
                [
                    self.hand_sequence.scale_at(self.anchor_frame),
                    self.hand_sequence.scale_at(frame_idx),
                ]
            )
        )
        anchor_frame = self._grasp_frame(
            self.anchor_frame, anchor_analysis, common_scale
        )
        target_frame = self._grasp_frame(frame_idx, target_analysis, common_scale)
        diagnostics["grasp"] = target_analysis
        diagnostics["mano_to_pointmap_scale"] = common_scale
        if anchor_frame is None or target_frame is None:
            diagnostics["failure"] = "grasp_frame_failed"
            return None, [], None, diagnostics

        if target_analysis["type"] == "power_hold":
            rotation_delta = _align_vectors(
                anchor_frame[:3, 0], target_frame[:3, 0]
            )
            translation_delta = (
                target_frame[:3, 3] - rotation_delta @ anchor_frame[:3, 3]
            )
            motion = np.eye(4, dtype=np.float64)
            motion[:3, :3] = rotation_delta
            motion[:3, 3] = translation_delta
            diagnostics["constraint"] = "power_hold_tube_axis"
        else:
            motion = target_frame @ np.linalg.inv(anchor_frame)
            diagnostics["constraint"] = f"{target_analysis['type']}_interaction_frame"

        predicted_matrix = _apply_camera_motion_to_render_pose(
            motion, self.anchor_pose
        )
        scale = coarse_pose["scale"] if coarse_pose is not None else self.anchor_pose["scale"]
        predicted = matrix_to_pose(predicted_matrix, scale)

        # Preserve the cylinder's axial symmetry as four explicit hypotheses.
        priors = []
        for angle in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
            symmetry = Rotation.from_rotvec(self.canonical_axis * angle).as_matrix()
            hypothesis = predicted_matrix.copy()
            hypothesis[:3, :3] = symmetry.T @ predicted_matrix[:3, :3]
            priors.append(matrix_to_pose(hypothesis, scale))
        target_axis = predicted_matrix[:3, :3].T @ self.canonical_axis
        axis_context = {
            "canonical_axis": self.canonical_axis.copy(),
            "target_axis": target_axis,
            "grasp_type": target_analysis["type"],
        }
        diagnostics["target_axis"] = target_axis.tolist()
        return predicted, priors, axis_context, diagnostics


def grasp_axis_score(
    pose: dict[str, Any],
    context: dict[str, Any],
    sigma_deg: float = 20.0,
) -> tuple[float, float]:
    transform = pose_to_matrix(pose)
    canonical = np.asarray(context["canonical_axis"], dtype=np.float64)
    target = _safe_unit(np.asarray(context["target_axis"], dtype=np.float64))
    predicted = _safe_unit(transform[:3, :3].T @ canonical)
    if target is None or predicted is None:
        return 0.0, 180.0
    # A transparent beaker is nearly symmetric under axis reversal in
    # silhouette; do not falsely penalize the sign here.
    cosine = float(np.clip(abs(np.dot(target, predicted)), 0.0, 1.0))
    angle = float(np.arccos(cosine))
    sigma = max(float(np.deg2rad(sigma_deg)), 1e-8)
    return float(np.exp(-angle / sigma)), float(np.rad2deg(angle))


# ---------------------------------------------------------------------------
# COP-style contact-distance consistency
# ---------------------------------------------------------------------------


class ContactConsistencyPrior:
    """Build short-window fingertip-to-object distance signatures."""

    def __init__(
        self,
        hand_sequence: ManoHandSequence,
        mesh_vertices: np.ndarray,
        pose_archive_path: str | Path,
        reference_frames: Iterable[int],
        sample_vertices: int = 768,
        grasp_type: str = "auto",
    ) -> None:
        pose_frames, _ = load_pose_archive(pose_archive_path)
        vertices = np.asarray(mesh_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh_vertices must have shape (N, 3)")
        count = min(max(int(sample_vertices), 64), len(vertices))
        indices = np.linspace(0, len(vertices) - 1, count, dtype=np.int64)
        self.canonical_vertices = vertices[indices]
        self.hand_sequence = hand_sequence
        self.grasp_type = grasp_type
        self.reference_poses = {
            int(frame): result_to_pose(pose_frames[int(frame)])
            for frame in reference_frames
            if int(frame) in pose_frames and hand_sequence.has_frame(int(frame))
        }
        if not self.reference_poses:
            raise ValueError("no valid contact reference frames")

    def context(self, frame_idx: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "contact_reference_frames": sorted(self.reference_poses),
        }
        if not self.hand_sequence.has_frame(frame_idx):
            diagnostics["contact_failure"] = "invalid_hand_pose"
            return None, diagnostics
        analysis = self.hand_sequence.analyze_grasp(frame_idx, self.grasp_type)
        joint_indices = self.hand_sequence.contact_joint_indices(analysis)
        current_points = self.hand_sequence.joints(frame_idx)[joint_indices]
        references = []
        for reference_frame, pose in self.reference_poses.items():
            reference_points = self.hand_sequence.joints(reference_frame)[joint_indices]
            matrix = pose_to_matrix(pose)
            scale = float(_as_numpy(pose["scale"]).reshape(-1)[0])
            object_points = (
                self.canonical_vertices * scale
            ) @ matrix[:3, :3] + matrix[:3, 3]
            distances = np.linalg.norm(
                reference_points[:, None, :] - object_points[None, :, :], axis=2
            )
            references.append(
                {
                    "frame_idx": reference_frame,
                    "distances": distances.astype(np.float32),
                    "surface_distances": distances.min(axis=1).astype(np.float32),
                }
            )
        context = {
            "canonical_vertices": self.canonical_vertices,
            "current_hand_points": current_points,
            "joint_indices": joint_indices,
            "references": references,
        }
        diagnostics["contact_joint_indices"] = joint_indices.tolist()
        diagnostics["contact_grasp_type"] = analysis["type"]
        return context, diagnostics


def contact_consistency_score(
    pose: dict[str, Any],
    context: dict[str, Any],
    sigma: float = 0.025,
) -> tuple[float, float, int]:
    canonical = np.asarray(context["canonical_vertices"], dtype=np.float64)
    hand_points = np.asarray(context["current_hand_points"], dtype=np.float64)
    matrix = pose_to_matrix(pose)
    scale = float(_as_numpy(pose["scale"]).reshape(-1)[0])
    object_points = canonical * scale @ matrix[:3, :3] + matrix[:3, 3]
    distances = np.linalg.norm(
        hand_points[:, None, :] - object_points[None, :, :], axis=2
    )
    best_residual = float("inf")
    best_frame = -1
    for reference in context["references"]:
        reference_distances = np.asarray(reference["distances"], dtype=np.float64)
        vector_residual = float(np.median(np.abs(distances - reference_distances)))
        surface_residual = float(
            np.median(
                np.abs(
                    distances.min(axis=1)
                    - np.asarray(reference["surface_distances"], dtype=np.float64)
                )
            )
        )
        residual = 0.75 * vector_residual + 0.25 * surface_residual
        if residual < best_residual:
            best_residual = residual
            best_frame = int(reference["frame_idx"])
    score = float(np.exp(-best_residual / max(float(sigma), 1e-8)))
    return score, best_residual, best_frame
