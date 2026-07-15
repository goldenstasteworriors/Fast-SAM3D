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

The module deliberately contains no SAM3D-specific model code, which makes
the geometry independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


POSE_KEYS = ("rotation", "translation", "scale")
_MCP_INDICES = np.array([5, 9, 13, 17], dtype=np.int64)


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
