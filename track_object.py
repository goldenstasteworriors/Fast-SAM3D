"""

Fast-SAM3D **batched** variant:
  - Batched sampling: all K pose samples processed in parallel (chunked)
  - Taylor expansion cache (ShortCut_faster): ~2.7x per-sample speedup
  - Optional torch.compile on DiT backbone
  - All Fast-SAM3D acceleration flags (SS cache, SLaT carving, mesh aggregation)

Usage:
    python track_object.py \
    --config checkpoints/hf/pipeline.yaml \
    --vid_dir /path/to/video_dir \
    --masks_root /path/to/video_dir/video_segmentation/masks \
    --object_name cup \
    --init_frame 0 \
    --output_dir /path/to/video_dir/obj_tracking_out \
    --guidance_strength 1 \
    --save_layout \
    --fix_scale_to_init_frame \
    --pose_guidance_strength 0.5 \
    --num_pose_samples 25 \
    --scoring_metric render_iou \
    --pose_selection cluster \
    --cluster_dist_thresh 0.3 \
    --cluster_min_size 3 \
    --cluster_w_rot 1.5 \
    --chain_poses \
    --post_optimize \
    --no-enable_shape_icp \
    --chain_on_diffusion \
    --enable_ss_cache \
    --torch_compile \
    --euler_steps 25

    optionally:
    --rotvel_json /path/to/video_dir/426_motion_stats.json

    Occlusion-aware render IoU scoring (comma-separated names support both hands):
    --occluder_masks_root /path/to/hand_segmentation_review/masks \
    --occluder_mask_names right_hand_sam3 \
    --occluder_dilation_px 5

    With pose_log_likelihood selection (selects pose with highest log p(pose | shape, image)):
    python track_object.py \
    --config checkpoints/hf/pipeline.yaml \
    --vid_dir /path/to/video_dir \
    --object_name pourer \
    --init_frame 37 \
    --frames 37:41 \
    --guidance_strength 1 \
    --save_layout \
    --fix_scale_to_init_frame \
    --scoring_metric render_iou \
    --pose_selection pose_log_likelihood \
    --ll_steps 25 \
    --chain_poses \
    --post_optimize \
    --no-enable_shape_icp \
    --chain_on_diffusion \
    --num_pose_samples_per_pgs 3 \
    --euler_steps 25 \
    --batch_chunk_size 5 \
    --output_dir /path/to/video_dir/pose_ll_test
"""

import sys
import os

os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "notebook"))

import argparse
import cv2
import glob
import json
import time
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from loguru import logger
from omegaconf import OmegaConf

import h5py
import shutil
from scipy.spatial.transform import Rotation as ScipyRotation
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, PointLights, TexturesVertex,
)

from notebook.inference import Inference, load_image, load_mask
from sam3d_objects.pipeline.inference_utils import (
    voxelize_mesh,
    prune_sparse_structure,
    ROTATION_6D_MEAN,
    ROTATION_6D_STD,
)
from sam3d_objects.pipeline.inference_utils import layout_post_optimization
from sam3d_objects.data.dataset.tdfy.pose_target import PoseTargetConverter
from fft.fft2d import calculate_hfer_robust
from pose_log_likelihood import compute_pose_only_log_likelihoods
from pose_priors import (
    ContactConsistencyPrior,
    GraspMemoryPosePrior,
    GraspTypePosePrior,
    HandObjectPosePrior,
    HandMotionPosePrior,
    HistoryMemoryPosePrior,
    ManoHandSequence,
    contact_consistency_score,
    copy_pose,
    grasp_axis_score,
    history_alignment_score,
    load_pose_archive,
    pose_errors,
    result_to_pose,
)


_NEW_HAND_PRIOR_MODES = {
    "hand_motion",
    "hand_memory",
    "grasp_type",
    "hand_contact",
    "hand_fusion",
}


# ------------------------------------------------------------------
# Mesh voxelization helpers (inlined from mesh_to_shape.py)
# ------------------------------------------------------------------

def load_mesh_as_open3d(mesh_path: str) -> o3d.geometry.TriangleMesh:
    tm = trimesh.load(mesh_path, force="mesh")
    logger.info(f"Loaded mesh: {mesh_path}  vertices={len(tm.vertices)}  faces={len(tm.faces)}")
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(tm.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(tm.faces))
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def voxelize(mesh_path: str):
    o3d_mesh = load_mesh_as_open3d(mesh_path)
    mesh_ss, scale, center = voxelize_mesh(o3d_mesh, resolution=64)
    mesh_coords = torch.argwhere(mesh_ss > 0)
    logger.info(f"Voxelized mesh: {mesh_coords.shape[0]} occupied voxels")
    return mesh_ss, mesh_coords.int(), scale, center


def compute_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a_bool = a.bool()
    b_bool = b.bool()
    intersection = (a_bool & b_bool).sum().item()
    union = (a_bool | b_bool).sum().item()
    return intersection / max(union, 1)


def downsample_sparse_structure(coord_batch, max_coords=42000, downsample_factor=2):
    if coord_batch.shape[0] <= max_coords:
        return coord_batch, 1

    coords = coord_batch[:, 1:].float()
    batch_indices = coord_batch[:, 0:1]

    coords_min = coords.min(dim=0)[0]
    coords_max = coords.max(dim=0)[0]
    original_size = coords_max - coords_min + 1

    target_size = original_size / downsample_factor
    offset = (original_size - target_size) / 2
    target_min = coords_min + offset
    target_max = coords_min + offset + target_size - 1

    coords_normalized = (coords - coords_min) / (coords_max - coords_min)
    coords_rescaled = coords_normalized * (target_size - 1) + target_min
    coords_rescaled = torch.round(coords_rescaled).int()
    coords_rescaled = torch.clamp(coords_rescaled, target_min.int(), target_max.int())

    combined = torch.cat([batch_indices, coords_rescaled], dim=1)
    unique_combined = torch.unique(combined, dim=0)

    if unique_combined.shape[0] > max_coords:
        indices = torch.randperm(unique_combined.shape[0], device=coord_batch.device)[:max_coords]
        unique_combined = unique_combined[indices]

    return unique_combined.int(), downsample_factor


# ------------------------------------------------------------------
# Rendering IoU helpers
# ------------------------------------------------------------------

_R_YUP_TO_ZUP_RENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32).T


# ------------------------------------------------------------------
# Camera extrinsics helpers
# ------------------------------------------------------------------

_F_ARKIT_P3D = np.diag([-1.0, 1.0, -1.0, 1.0])


def load_extrinsics_from_hdf5(hdf5_path, frame_indices):
    with h5py.File(hdf5_path, 'r') as f:
        camera_transforms = f['transforms/camera'][:]
    extrinsics = {}
    for idx in frame_indices:
        T_c2w_arkit = camera_transforms[idx].astype(np.float64)
        T_c2w_p3d = T_c2w_arkit @ _F_ARKIT_P3D
        extrinsics[idx] = T_c2w_p3d
    return extrinsics


def _make_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def _T_to_pos_quat_wxyz(T):
    pos = T[:3, 3].copy()
    quat_wxyz = ScipyRotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
    return pos, quat_wxyz


def _quat_to_matrix_np(quat_wxyz):
    return ScipyRotation.from_quat(quat_wxyz, scalar_first=True).as_matrix()


def transform_pose_between_cameras(pose_dict, T_c2w_src, T_c2w_dst):
    quat_np = pose_dict["rotation"].detach().cpu().squeeze().numpy().astype(np.float64)
    trans_np = pose_dict["translation"].detach().cpu().squeeze().numpy().astype(np.float64)

    T_obj_in_src = _make_T(trans_np, _quat_to_matrix_np(quat_np))
    T_obj_in_dst = np.linalg.inv(T_c2w_dst) @ T_c2w_src @ T_obj_in_src

    new_trans, new_quat = _T_to_pos_quat_wxyz(T_obj_in_dst)

    return {
        "rotation": torch.tensor(new_quat, dtype=torch.float32).unsqueeze(0),
        "translation": torch.tensor(new_trans, dtype=torch.float32).unsqueeze(0),
        "scale": pose_dict["scale"].clone(),
    }


def _maybe_transform_pose(pose_dict, prev_frame_idx, curr_frame_idx, extrinsics):
    if extrinsics is None or pose_dict is None or prev_frame_idx == curr_frame_idx:
        return pose_dict
    return transform_pose_between_cameras(
        pose_dict, extrinsics[prev_frame_idx], extrinsics[curr_frame_idx],
    )


def _build_renderable_mesh(init_trimesh_obj, device):
    verts = np.array(init_trimesh_obj.vertices, dtype=np.float32) @ _R_YUP_TO_ZUP_RENDER
    verts_t = torch.from_numpy(verts).to(device)
    faces_t = torch.from_numpy(np.array(init_trimesh_obj.faces, dtype=np.int64)).to(device)
    textures = TexturesVertex(verts_features=torch.ones_like(verts_t)[None])
    return Meshes(verts=[verts_t], faces=[faces_t], textures=textures)


def _build_silhouette_renderer(intrinsics_3x3, width, height, device):
    K = intrinsics_3x3.detach().float().to(device)
    fx = (K[0, 0] * width).item()
    fy = (K[1, 1] * height).item()
    cx = (K[0, 2] * width).item()
    cy = (K[1, 2] * height).item()

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=((height, width),),
        in_ndc=False, device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(height, width), blur_radius=0.0,
        faces_per_pixel=1, max_faces_per_bin=50000, bin_size=0,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )


def _render_silhouette(mesh, pose_dict, intrinsics_3x3, width, height, device, renderer=None):
    quat = pose_dict["rotation"].detach().float().squeeze().to(device)
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    trans = pose_dict["translation"].detach().float().squeeze().to(device)
    if trans.dim() == 1:
        trans = trans.unsqueeze(0)
    sc = pose_dict["scale"].detach().float().squeeze().to(device)
    if sc.dim() == 1:
        sc = sc.unsqueeze(0)

    R_mat = quaternion_to_matrix(quat)
    tfm = compose_transform(scale=sc, rotation=R_mat, translation=trans)
    verts = mesh.verts_packed()
    verts_cam = tfm.transform_points(verts.unsqueeze(0)).squeeze(0)

    mesh_transformed = mesh.clone()
    mesh_transformed = mesh_transformed.update_padded(verts_cam.unsqueeze(0))

    if renderer is None:
        renderer = _build_silhouette_renderer(intrinsics_3x3, width, height, device)

    with torch.no_grad():
        images = renderer(mesh_transformed)

    alpha = images[0, ..., 3].cpu().numpy()
    return alpha


def _compute_binary_iou(render_binary, gt_binary, valid_mask=None):
    render_binary = np.asarray(render_binary, dtype=bool)
    gt_binary = np.asarray(gt_binary, dtype=bool)
    if valid_mask is None:
        valid_mask = np.ones_like(render_binary, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    intersection = (render_binary & gt_binary & valid_mask).sum()
    union = ((render_binary | gt_binary) & valid_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _visible_component_scores(
    render_binary,
    gt_binary,
    occluder_binary=None,
    min_area_px=20,
    bridge_dilation_px=2,
):
    """Score every visible object component and its connection behind an occluder.

    Standard IoU weights components by area and can therefore ignore a small
    visible fragment.  Here every sufficiently large connected component has
    equal importance.  The bridge score additionally requires one connected
    part of the *full* rendered silhouette to touch all visible components
    through the object-or-occluder support region.
    """
    render_binary = np.asarray(render_binary, dtype=bool)
    gt_binary = np.asarray(gt_binary, dtype=bool)
    if occluder_binary is None:
        occluder_binary = np.zeros_like(gt_binary, dtype=bool)
    else:
        occluder_binary = np.asarray(occluder_binary, dtype=bool)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        gt_binary.astype(np.uint8), connectivity=8,
    )
    components = [
        {
            "label": int(index),
            "area": int(stats[index, cv2.CC_STAT_AREA]),
            "centroid": centroids[index].astype(np.float64),
            "mask": labels == index,
        }
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area_px)
    ]
    components.sort(key=lambda item: item["area"], reverse=True)
    if not components:
        return {
            "component_count": 0,
            "component_areas": [],
            "component_centroids": [],
            "component_recalls": [],
            "component_balance": 0.0,
            "component_bridge": 0.0,
        }

    # Use the raw hand mask here.  The dilated mask remains useful for a
    # tolerant global IoU, but must not erase a small, genuinely visible part.
    visible_render = render_binary & ~occluder_binary
    recalls = [
        float((visible_render & item["mask"]).sum()) / max(item["area"], 1)
        for item in components
    ]
    component_balance = float(min(recalls))

    if len(components) == 1:
        component_bridge = recalls[0]
    else:
        dilation = max(int(bridge_dilation_px), 0)
        if dilation > 0:
            kernel_size = 2 * dilation + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            bridge_occluder = cv2.dilate(
                occluder_binary.astype(np.uint8), kernel, iterations=1,
            ).astype(bool)
        else:
            bridge_occluder = occluder_binary
        support_region = gt_binary | bridge_occluder
        supported_render = render_binary & support_region
        bridge_count, bridge_labels = cv2.connectedComponents(
            supported_render.astype(np.uint8), connectivity=8,
        )
        component_bridge = 0.0
        for bridge_label in range(1, bridge_count):
            candidate = bridge_labels == bridge_label
            touch_recalls = [
                float((candidate & item["mask"]).sum()) / max(item["area"], 1)
                for item in components
            ]
            component_bridge = max(component_bridge, min(touch_recalls))

    return {
        "component_count": len(components),
        "component_areas": [item["area"] for item in components],
        "component_centroids": [item["centroid"].tolist() for item in components],
        "component_recalls": recalls,
        "component_balance": component_balance,
        "component_bridge": float(component_bridge),
    }


def compute_render_iou(
    mesh,
    pose_dict,
    intrinsics_3x3,
    gt_mask,
    width,
    height,
    device,
    renderer=None,
    occluder_mask=None,
    occluder_dilation_px=0,
    component_min_area_px=20,
    component_bridge_dilation_px=2,
    return_details=False,
):
    alpha = _render_silhouette(mesh, pose_dict, intrinsics_3x3, width, height, device, renderer=renderer)
    render_binary = alpha > 0.5
    gt_binary = gt_mask > 0
    raw_iou = _compute_binary_iou(render_binary, gt_binary)

    score_iou = raw_iou
    occlusion_applied = False
    ignored_gt_fraction = 0.0
    visible_render_fraction = 1.0
    occluder_pixels = 0
    raw_occluder_binary = None

    if occluder_mask is not None:
        occluder_binary = np.asarray(occluder_mask) > 0
        if occluder_binary.shape != gt_binary.shape:
            occluder_binary = cv2.resize(
                occluder_binary.astype(np.uint8),
                (gt_binary.shape[1], gt_binary.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        raw_occluder_binary = occluder_binary.copy()
        if occluder_dilation_px > 0:
            kernel_size = 2 * int(occluder_dilation_px) + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            occluder_binary = cv2.dilate(
                occluder_binary.astype(np.uint8), kernel, iterations=1,
            ).astype(bool)

        occluder_pixels = int(occluder_binary.sum())
        valid_mask = ~occluder_binary
        valid_union = ((render_binary | gt_binary) & valid_mask).sum()
        if valid_union > 0:
            score_iou = _compute_binary_iou(render_binary, gt_binary, valid_mask)
            occlusion_applied = True

        gt_pixels = int(gt_binary.sum())
        if gt_pixels > 0:
            ignored_gt_fraction = float((gt_binary & occluder_binary).sum()) / float(gt_pixels)
        render_pixels = int(render_binary.sum())
        if render_pixels > 0:
            visible_render_fraction = float((render_binary & valid_mask).sum()) / float(render_pixels)

    component_details = _visible_component_scores(
        render_binary,
        gt_binary,
        occluder_binary=raw_occluder_binary,
        min_area_px=component_min_area_px,
        bridge_dilation_px=component_bridge_dilation_px,
    )

    details = {
        "score_iou": score_iou,
        "raw_iou": raw_iou,
        "occlusion_applied": occlusion_applied,
        "ignored_gt_fraction": ignored_gt_fraction,
        "visible_render_fraction": visible_render_fraction,
        "occluder_pixels": occluder_pixels,
        **component_details,
    }
    return details if return_details else score_iou


def _align_vector_rotation(source, target):
    """Return the minimum 3D rotation taking ``source`` onto ``target``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-8:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, basis)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return ScipyRotation.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / sine
    angle = float(np.arctan2(sine, cosine))
    return ScipyRotation.from_rotvec(angle * axis).as_matrix()


def _component_axis_pose_hypotheses(
    base_pose,
    gt_mask,
    intrinsics_3x3,
    image_hw,
    rendered_canonical_axis,
    depth_angles_deg,
    min_area_px=20,
):
    """Generate poses whose rendered object axis joins visible mask pieces.

    The image-space direction is fixed by the two largest visible components;
    depth remains ambiguous and is represented by explicit hypotheses.
    Translation is recentered on the equal-weight component midpoint so the
    small fragment cannot be overwhelmed by the large one.
    """
    diagnostics = {"active": False, "reason": "insufficient_components"}
    if base_pose is None or gt_mask is None or rendered_canonical_axis is None:
        return [], diagnostics

    gt_binary = np.asarray(gt_mask) > 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        gt_binary.astype(np.uint8), connectivity=8,
    )
    components = [
        (int(stats[index, cv2.CC_STAT_AREA]), centroids[index].astype(np.float64))
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area_px)
    ]
    components.sort(key=lambda item: item[0], reverse=True)
    if len(components) < 2:
        return [], diagnostics

    first, second = components[:2]
    image_direction = second[1] - first[1]
    if float(np.linalg.norm(image_direction)) < 2.0:
        diagnostics["reason"] = "degenerate_component_direction"
        return [], diagnostics

    height, width = image_hw
    intrinsics = np.asarray(
        intrinsics_3x3.detach().cpu() if torch.is_tensor(intrinsics_3x3) else intrinsics_3x3,
        dtype=np.float64,
    ).squeeze()
    fx = float(intrinsics[0, 0] * width)
    fy = float(intrinsics[1, 1] * height)
    cx = float(intrinsics[0, 2] * width)
    cy = float(intrinsics[1, 2] * height)
    screen_axis = np.array(
        [-image_direction[0] / max(fx, 1e-8),
         -image_direction[1] / max(fy, 1e-8),
         0.0],
        dtype=np.float64,
    )
    screen_axis /= max(float(np.linalg.norm(screen_axis)), 1e-12)

    base_rotation_tensor = base_pose["rotation"].detach().cpu()
    base_translation_tensor = base_pose["translation"].detach().cpu()
    base_quaternion = base_rotation_tensor.reshape(-1, 4)[0].numpy().astype(np.float64)
    base_rotation = ScipyRotation.from_quat(
        base_quaternion, scalar_first=True,
    ).as_matrix()
    canonical_axis = np.asarray(rendered_canonical_axis, dtype=np.float64)
    canonical_axis /= max(float(np.linalg.norm(canonical_axis)), 1e-12)
    # PyTorch3D Transform3d uses row vectors: v_camera = v_object @ R + t.
    # In column notation the rendered axis is therefore R.T @ axis.
    source_axis = base_rotation.T @ canonical_axis

    translation = base_translation_tensor.reshape(-1, 3)[0].numpy().astype(np.float64)
    target_center = 0.5 * (first[1] + second[1])
    z = max(float(translation[2]), 1e-4)
    projected_u = cx - fx * float(translation[0]) / z
    projected_v = cy - fy * float(translation[1]) / z
    recentered_translation = translation.copy()
    recentered_translation[0] -= (target_center[0] - projected_u) * z / max(fx, 1e-8)
    recentered_translation[1] -= (target_center[1] - projected_v) * z / max(fy, 1e-8)

    hypotheses = []
    target_axes = []
    for depth_angle_deg in depth_angles_deg:
        angle = float(np.deg2rad(depth_angle_deg))
        target_axis = (
            np.cos(angle) * screen_axis
            + np.sin(angle) * np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        target_axis /= max(float(np.linalg.norm(target_axis)), 1e-12)
        delta = _align_vector_rotation(source_axis, target_axis)
        rotation = base_rotation @ delta.T
        quaternion = ScipyRotation.from_matrix(rotation).as_quat(scalar_first=True)
        hypothesis = copy_pose(base_pose)
        hypothesis["rotation"] = torch.as_tensor(
            quaternion,
            dtype=base_rotation_tensor.dtype,
        ).reshape(base_rotation_tensor.shape)
        hypothesis["translation"] = torch.as_tensor(
            recentered_translation,
            dtype=base_translation_tensor.dtype,
        ).reshape(base_translation_tensor.shape)
        hypotheses.append(hypothesis)
        target_axes.append(target_axis.tolist())

    diagnostics = {
        "active": bool(hypotheses),
        "component_areas": [first[0], second[0]],
        "component_centroids": [first[1].tolist(), second[1].tolist()],
        "image_axis_angle_deg": float(
            np.rad2deg(np.arctan2(image_direction[1], image_direction[0]))
        ),
        "target_center": target_center.tolist(),
        "depth_angles_deg": [float(value) for value in depth_angles_deg],
        "target_axes": target_axes,
    }
    return hypotheses, diagnostics


# ------------------------------------------------------------------
# Clustering-based pose selection
# ------------------------------------------------------------------

def _pose_distance(pose_a, pose_b, w_trans=1.0, w_rot=1.0):
    t_a = pose_a["translation"].detach().cpu().squeeze().numpy().astype(np.float64)
    t_b = pose_b["translation"].detach().cpu().squeeze().numpy().astype(np.float64)
    trans_dist = np.linalg.norm(t_b - t_a)

    q_a = pose_a["rotation"].detach().cpu().squeeze().numpy().astype(np.float64)
    q_b = pose_b["rotation"].detach().cpu().squeeze().numpy().astype(np.float64)
    r_a = ScipyRotation.from_quat([q_a[1], q_a[2], q_a[3], q_a[0]])
    r_b = ScipyRotation.from_quat([q_b[1], q_b[2], q_b[3], q_b[0]])
    rot_dist = (r_a.inv() * r_b).magnitude()

    return w_trans * trans_dist + w_rot * rot_dist


def _compute_pose_distance_matrix(candidates, w_trans=1.0, w_rot=1.0):
    K = len(candidates)
    full_matrix = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            d = _pose_distance(candidates[i], candidates[j], w_trans, w_rot)
            full_matrix[i, j] = d
            full_matrix[j, i] = d

    from scipy.spatial.distance import squareform
    condensed = squareform(full_matrix)
    return condensed, full_matrix


def cluster_pose_candidates(
    candidates,
    scoring_metric="shape_iou",
    dist_thresh=0.5,
    min_cluster_size=1,
    w_trans=1.0,
    w_rot=1.0,
):
    from scipy.cluster.hierarchy import linkage, fcluster

    K = len(candidates)

    if K <= 1:
        return 0, {"num_clusters": 1, "cluster_sizes": [K], "selected_cluster": 0}
    if K == 2:
        scores = []
        for c in candidates:
            if scoring_metric in c:
                scores.append(c[scoring_metric])
            elif "shape_iou" in c:
                scores.append(c["shape_iou"])
            else:
                scores.append(0.0)
        best = int(np.argmax(scores))
        return best, {"num_clusters": 1, "cluster_sizes": [2], "selected_cluster": 0}

    condensed, full_matrix = _compute_pose_distance_matrix(candidates, w_trans, w_rot)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=dist_thresh, criterion='distance')

    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    cluster_members = {}
    for idx, lab in enumerate(labels):
        cluster_members.setdefault(lab, []).append(idx)

    scores = []
    for c in candidates:
        if scoring_metric in c:
            scores.append(c[scoring_metric])
        elif "shape_iou" in c:
            scores.append(c["shape_iou"])
        else:
            scores.append(0.0)
    scores = np.array(scores)

    cluster_ranking = []
    for lab, members in cluster_members.items():
        size = len(members)
        avg_score = scores[members].mean()
        cluster_ranking.append((lab, size, avg_score, members))

    cluster_ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)

    valid_clusters = [c for c in cluster_ranking if c[1] >= min_cluster_size]
    if not valid_clusters:
        valid_clusters = cluster_ranking

    selected_lab, selected_size, selected_avg_score, selected_members = valid_clusters[0]

    member_scores = [(m, scores[m]) for m in selected_members]
    member_scores.sort(key=lambda x: x[1], reverse=True)
    best_idx = member_scores[0][0]

    cluster_sizes = sorted([len(m) for m in cluster_members.values()], reverse=True)

    cluster_info = {
        "num_clusters": num_clusters,
        "cluster_sizes": cluster_sizes,
        "selected_cluster": int(selected_lab),
        "selected_cluster_size": selected_size,
        "selected_cluster_avg_score": float(selected_avg_score),
        "best_candidate_idx": best_idx,
        "best_candidate_score": float(scores[best_idx]),
        "dist_thresh": dist_thresh,
        "labels": labels.tolist(),
    }

    return best_idx, cluster_info


# ------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------

def get_frame_mask_dir(masks_root, frame_idx):
    return os.path.join(masks_root, f"frame_{frame_idx:06d}_masks")


def discover_frames(vid_dir, object_name, frame_range=None, masks_root=None, frames_dir=None):
    if frames_dir is None:
        frames_dir = os.path.join(vid_dir, "all_frames")
    if masks_root is None:
        masks_root = os.path.join(vid_dir, "video_segmentation", "masks")

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    results = []
    for fpath in frame_files:
        fname = os.path.basename(fpath)
        frame_idx = int(os.path.splitext(fname)[0])

        if frame_range is not None:
            start, end = frame_range
            if frame_idx < start or frame_idx >= end:
                continue

        mask_dir = get_frame_mask_dir(masks_root, frame_idx)
        mask_path = os.path.join(mask_dir, f"{object_name}.png")
        if os.path.isfile(mask_path):
            results.append((frame_idx, fpath, mask_path))

    return results


def load_frame_result_from_layout(masks_root, object_name, frame_idx=0):
    frame_dir = get_frame_mask_dir(masks_root, frame_idx)
    layout_path = os.path.join(frame_dir, "layout.json")
    if not os.path.isfile(layout_path):
        logger.warning(f"Frame {frame_idx} layout.json not found at {layout_path}")
        return None

    with open(layout_path) as f:
        layout = json.load(f)

    target_mesh = f"{object_name}.obj"
    obj_entry = None
    for obj in layout.get("objects", []):
        if obj.get("mesh_obj") == target_mesh:
            obj_entry = obj
            break

    if obj_entry is None:
        logger.warning(f"Object '{object_name}' not found in frame {frame_idx} layout.json")
        return None

    l2s = obj_entry["local_to_scene"]
    result = {
        "translation": torch.tensor(l2s["translation"], dtype=torch.float32).unsqueeze(0),
        "rotation": torch.tensor(l2s["quat_wxyz"], dtype=torch.float32).unsqueeze(0),
        "scale": torch.tensor(l2s["scale"], dtype=torch.float32).unsqueeze(0),
        "shape_iou": 1.0,
    }

    intr = layout.get("intrinsics_normalized")
    if intr:
        K = torch.zeros(3, 3)
        K[0, 0] = intr["fx_norm"]
        K[1, 1] = intr["fy_norm"]
        K[0, 2] = intr["cx_norm"]
        K[1, 2] = intr["cy_norm"]
        K[2, 2] = 1.0
        result["intrinsics"] = K.unsqueeze(0)

    return result


# ------------------------------------------------------------------
# Decoder inversion
# ------------------------------------------------------------------

@torch.no_grad()
def check_encoder_available(pipeline):
    enc = pipeline.models["ss_encoder"] if "ss_encoder" in pipeline.models else None
    if enc is None:
        return False
    try:
        for p in enc.parameters():
            if p.numel() > 0 and p.abs().sum() > 0:
                return True
        return False
    except Exception:
        return False


def encode_target_occupancy(pipeline, target_occ, device,
                            opt_steps=1000, lr=0.05, verbose=True):
    ss_decoder = pipeline.models["ss_decoder"]
    target = target_occ.float().unsqueeze(1).to(device)

    if check_encoder_available(pipeline):
        ss_encoder = pipeline.models["ss_encoder"]
        with torch.no_grad():
            result = ss_encoder(target)
            z_target = result["mean"] if isinstance(result, dict) else result
        logger.info(f"[Encoder] z_target shape: {z_target.shape}")
        return z_target.detach(), "encoder"

    logger.info(f"SS encoder not available -- optimizing z_target through decoder ({opt_steps} steps)")

    z_target = torch.randn(1, 8, 16, 16, 16, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z_target], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_steps)

    best_loss = float("inf")
    best_z = None

    for step in range(opt_steps):
        optimizer.zero_grad()
        decoded = ss_decoder(z_target)

        loss = F.binary_cross_entropy_with_logits(decoded, target)
        reg_loss = 0.001 * (z_target ** 2).mean()
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_z = z_target.detach().clone()

        if verbose and (step % 100 == 0 or step == opt_steps - 1):
            with torch.no_grad():
                recon_occ = (ss_decoder(z_target) > 0).squeeze()
                target_bool = target.squeeze().bool()
                intersection = (recon_occ & target_bool).sum().item()
                union = (recon_occ | target_bool).sum().item()
                iou = intersection / max(union, 1)
            logger.info(
                f"  Step {step:4d}: loss={total_loss.item():.4f}, "
                f"recon_IoU={iou:.4f}, lr={scheduler.get_last_lr()[0]:.6f}"
            )

    logger.info(f"[Decoder Inversion] best_loss={best_loss:.4f}, z_target shape={best_z.shape}")
    return best_z, "decoder_inversion"


# ------------------------------------------------------------------
# Pose encoding (decoded pose -> latent dict)
# ------------------------------------------------------------------

_POSE_LATENT_KEYS = frozenset({
    "6drotation_normalized", "translation", "scale", "translation_scale",
})


def _pose_guidance_alpha(
    latent_key,
    rotation_strength,
    translation_strength,
    scale_strength,
):
    if latent_key == "6drotation_normalized":
        return rotation_strength
    if latent_key in ("translation", "translation_scale"):
        return translation_strength
    if latent_key == "scale":
        return scale_strength
    return 0.0


def encode_pose_to_latent(
    rotation, translation, scale,
    scene_scale, scene_shift, device=None,
):
    rotation = torch.as_tensor(rotation, dtype=torch.float32)
    translation = torch.as_tensor(translation, dtype=torch.float32)
    scale = torch.as_tensor(scale, dtype=torch.float32)

    if device is not None:
        rotation = rotation.to(device)
        translation = translation.to(device)
        scale = scale.to(device)
    dev = rotation.device

    scene_scale_t = torch.as_tensor(scene_scale, dtype=torch.float32, device=dev)
    scene_shift_t = torch.as_tensor(scene_shift, dtype=torch.float32, device=dev)

    if rotation.ndim == 1:
        rotation = rotation.unsqueeze(0)
    if translation.ndim == 1:
        translation = translation.unsqueeze(0)
    if scale.ndim == 1:
        scale = scale.unsqueeze(0)
    if scale.shape[-1] == 1:
        scale = scale.expand(-1, 3)

    if rotation.ndim == 2:
        rotation = rotation.unsqueeze(1)
    if translation.ndim == 2:
        translation = translation.unsqueeze(1)
    if scale.ndim == 2:
        scale = scale.unsqueeze(1)

    pose_target_dict = PoseTargetConverter.dicts_instance_pose_to_pose_target(
        pose_target_convention="ScaleShiftInvariant",
        instance_scale_l2c=scale,
        instance_position_l2c=translation,
        instance_quaternion_l2c=rotation,
        scene_scale=scene_scale_t,
        scene_shift=scene_shift_t,
    )

    quat = pose_target_dict["x_instance_rotation"]
    R = quaternion_to_matrix(quat)
    rot_6d = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
    rot_6d_norm = (rot_6d - ROTATION_6D_MEAN.to(dev)) / ROTATION_6D_STD.to(dev)

    ssi_scale = pose_target_dict["x_instance_scale"]
    scale_latent = torch.log(ssi_scale.clamp(min=1e-8))

    trans_latent = pose_target_dict["x_instance_translation"]

    trans_scale = pose_target_dict["x_translation_scale"]
    trans_scale_latent = torch.log(trans_scale.clamp(min=1e-8))

    return {
        "6drotation_normalized": rot_6d_norm,
        "translation": trans_latent,
        "scale": scale_latent,
        "translation_scale": trans_scale_latent,
    }


def _prepare_pose_target_latent(pose_target, ss_input_dict, device):
    if pose_target is None:
        return None

    if _POSE_LATENT_KEYS.issubset(pose_target.keys()):
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                for k, v in pose_target.items() if k in _POSE_LATENT_KEYS}

    assert "rotation" in pose_target and "translation" in pose_target and "scale" in pose_target, (
        f"Decoded pose_target must have 'rotation', 'translation', 'scale'. Got: {list(pose_target.keys())}"
    )
    scene_scale = pose_target.get("scene_scale", ss_input_dict.get("pointmap_scale"))
    scene_shift = pose_target.get("scene_shift", ss_input_dict.get("pointmap_shift"))
    return encode_pose_to_latent(
        rotation=pose_target["rotation"],
        translation=pose_target["translation"],
        scale=pose_target["scale"],
        scene_scale=scene_scale,
        scene_shift=scene_shift,
        device=device,
    )


# ------------------------------------------------------------------
# Batched guided sampling (with Taylor cache support)
# ------------------------------------------------------------------

def batched_guided_sample_sparse_structure(
    pipeline, ss_input_dict, z_target_flat, mesh_ss,
    num_samples=1,
    guidance_strength=0.7, pose_guidance_strength=0.0, pose_target=None,
    pose_translation_guidance_strength=None,
    pose_scale_guidance_strength=None,
    base_seed=42, inference_steps=None, verbose=True,
    pose_sde_strength=0.0,
    chunk_size=None,
):
    """Batched guided sampling with Taylor cache support.

    Runs K=num_samples in parallel (or chunked) forward passes.
    When the generator is ShortCut_faster, the Taylor expansion cache
    is re-initialized per chunk before the Euler loop.

    Returns list of K tuples (return_dict, shape_iou), one per sample.
    """
    ss_generator = pipeline.models["ss_generator"]
    ss_decoder = pipeline.models["ss_decoder"]

    ss_generator.no_shortcut = True
    ss_generator.reverse_fn.strength = pipeline.ss_cfg_strength
    ss_generator.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm

    prev_steps = ss_generator.inference_steps
    if inference_steps:
        ss_generator.inference_steps = inference_steps

    device = ss_input_dict["image"].device
    alpha = guidance_strength
    alpha_pose = pose_guidance_strength
    alpha_pose_translation = (
        alpha_pose
        if pose_translation_guidance_strength is None
        else float(pose_translation_guidance_strength)
    )
    alpha_pose_scale = (
        alpha_pose
        if pose_scale_guidance_strength is None
        else float(pose_scale_guidance_strength)
    )
    K = num_samples

    if chunk_size is None:
        chunk_size = K

    # Resolve one or more pose targets.  Multiple hand/grasp hypotheses are
    # assigned round-robin so guided diffusion actually generates candidates
    # around every mode instead of only using extra modes during ranking.
    pose_target_latent = None
    pose_target_count = 0
    if max(alpha_pose, alpha_pose_translation, alpha_pose_scale) > 0 and pose_target is not None:
        pose_targets = (
            list(pose_target)
            if isinstance(pose_target, (list, tuple))
            else [pose_target]
        )
        pose_target_count = len(pose_targets)
        latent_targets = [
            _prepare_pose_target_latent(target, ss_input_dict, device)
            for target in pose_targets
        ]
        assignments = [index % pose_target_count for index in range(K)]
        pose_target_latent = {
            key: torch.cat(
                [latent_targets[target_index][key] for target_index in assignments],
                dim=0,
            )
            for key in _POSE_LATENT_KEYS
        }

    # Embed conditions ONCE at batch=1, then expand per chunk
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            condition_args_1, condition_kwargs = pipeline.get_condition_input(
                pipeline.condition_embedders["ss_condition_embedder"],
                ss_input_dict,
                pipeline.ss_condition_input_mapping,
            )

    # Time schedule (shared)
    t_seq, d = ss_generator._prepare_t_and_d()

    # Detect ShortCut_faster (has cache_dic attribute)
    _is_faster_gen = hasattr(ss_generator, 'cache_dic')

    # Generate per-seed noise (exact reproducibility with sequential version)
    assert pipeline.is_mm_dit(), "Expected multi-modal DiT"
    latent_shape_1 = {
        k: (1,) + (v.pos_emb.shape[0], v.input_layer.in_features)
        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
    }

    all_noise = {}
    for ki in range(K):
        torch.manual_seed(base_seed + ki)
        noise_ki = ss_generator._generate_noise(latent_shape_1, device)
        for k in noise_ki:
            all_noise.setdefault(k, []).append(noise_ki[k])
    all_noise = {k: torch.cat(v, dim=0) for k, v in all_noise.items()}

    if verbose:
        parts = [f"K={K}", f"{len(t_seq)-1} steps", f"chunk_size={chunk_size}",
                 f"alpha_shape={alpha:.2f}"]
        if max(alpha_pose, alpha_pose_translation, alpha_pose_scale) > 0:
            parts.append(
                "alpha_pose="
                f"R{alpha_pose:.2f}/T{alpha_pose_translation:.2f}/S{alpha_pose_scale:.2f}"
            )
            if pose_target_count > 1:
                parts.append(f"pose_modes={pose_target_count}")
        if pose_sde_strength > 0:
            parts.append(f"pose_sde={pose_sde_strength:.3f}")
        if _is_faster_gen:
            parts.append("taylor_cache=ON")
        logger.info(f"Batched guided sampling: {', '.join(parts)}")

    # Process in chunks
    all_x_t_chunks = []

    for chunk_start in range(0, K, chunk_size):
        chunk_end = min(chunk_start + chunk_size, K)
        chunk_K = chunk_end - chunk_start

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
                x_t = {k: v[chunk_start:chunk_end].clone() for k, v in all_noise.items()}

                z_noise = x_t["shape"].clone()
                z_target = z_target_flat.expand(chunk_K, -1, -1).to(device, dtype=z_noise.dtype)

                pose_noise = {}
                if pose_target_latent is not None:
                    for k in _POSE_LATENT_KEYS:
                        pose_noise[k] = x_t[k].clone()

                condition_args = tuple(
                    a.expand(chunk_K, *a.shape[1:]) if isinstance(a, torch.Tensor) else a
                    for a in condition_args_1
                )

                # Initialize Taylor cache for this chunk (ShortCut_faster)
                if _is_faster_gen:
                    from cache_utils_ss import cache_init as _ss_cache_init
                    num_euler_steps = len(t_seq) - 1
                    if ss_generator.ss_params is None:
                        ss_generator.ss_params = {
                            'ss_cache_stride': 3,
                            'ss_warmup': 2,
                            'ss_order': 1,
                            'ss_momentum_beta': 0.5,
                        }
                    ss_generator.cache_dic, ss_generator.current = _ss_cache_init(
                        num_euler_steps,
                        cache_interval=ss_generator.ss_params['ss_cache_stride'],
                        max_order=ss_generator.ss_params['ss_order'],
                        first_enhance=ss_generator.ss_params['ss_warmup'],
                        end_enhance=num_euler_steps,
                    )
                    ss_generator.prev_v = None

                for t0, t1 in zip(t_seq[:-1], t_seq[1:]):
                    dt = t1 - t0
                    velocity = ss_generator._generate_dynamics(
                        x_t, t0, d, *condition_args, **condition_kwargs
                    )

                    x_next = {}
                    for k in x_t:
                        x_next[k] = x_t[k] + velocity[k] * dt

                    t_next = t1

                    # Shape guidance
                    if alpha > 0:
                        z_ref = (1.0 - t_next) * z_noise + t_next * z_target
                        x_next["shape"] = (1.0 - alpha) * x_next["shape"] + alpha * z_ref

                    # Pose guidance
                    if pose_target_latent is not None:
                        for k in _POSE_LATENT_KEYS:
                            alpha_key = _pose_guidance_alpha(
                                k,
                                alpha_pose,
                                alpha_pose_translation,
                                alpha_pose_scale,
                            )
                            if alpha_key <= 0:
                                continue
                            pt = pose_target_latent[k][chunk_start:chunk_end].to(
                                x_next[k].dtype
                            )
                            z_ref_pose = (1.0 - t_next) * pose_noise[k] + t_next * pt
                            x_next[k] = (1.0 - alpha_key) * x_next[k] + alpha_key * z_ref_pose

                    # SDE noise injection on pose channels only
                    if pose_sde_strength > 0:
                        sigma = 1.0 - t_next
                        if sigma > 1e-3:
                            var_t = sigma / max(1.0 - sigma, dt)
                            std_dev_t = (var_t ** 0.5) * pose_sde_strength
                            noise_scale = std_dev_t * (dt ** 0.5)
                            for k in _POSE_LATENT_KEYS:
                                x_next[k] = x_next[k] + noise_scale * torch.randn_like(x_next[k])

                    x_t = x_next

                all_x_t_chunks.append(x_t)

    # Concatenate all chunks
    x_t_all = {k: torch.cat([c[k] for c in all_x_t_chunks], dim=0) for k in all_x_t_chunks[0]}

    # Batched shape decode
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            shape_latent = x_t_all["shape"]
            shape_cube = shape_latent.permute(0, 2, 1).contiguous().view(K, 8, 16, 16, 16)
            ss = ss_decoder(shape_cube)

    # Per-sample coord extraction + shape IoU
    results = []
    for b in range(K):
        ret_b = {k: v[b:b+1] for k, v in x_t_all.items()}

        coords_b = torch.argwhere(ss[b] > 0)[:, [1, 2, 3]]
        batch_col = torch.full((coords_b.shape[0], 1), 0, dtype=torch.int32, device=device)
        coords_b = torch.cat([batch_col, coords_b], dim=1).int()

        ret_b["coords_original"] = coords_b
        if pipeline.downsample_ss_dist > 0:
            coords_b = prune_sparse_structure(
                coords_b, max_neighbor_axes_dist=pipeline.downsample_ss_dist,
            )
        coords_b, downsample_factor = downsample_sparse_structure(coords_b)
        ret_b["coords"] = coords_b
        ret_b["downsample_factor"] = downsample_factor

        # Shape IoU
        g_coords = ret_b["coords_original"]
        guided_ss = torch.zeros(1, 64, 64, 64, dtype=torch.long, device=device)
        if g_coords.shape[0] > 0:
            guided_ss[0, g_coords[:, 1].long(), g_coords[:, 2].long(), g_coords[:, 3].long()] = 1
        shape_iou = compute_iou(mesh_ss.to(device), guided_ss)

        results.append((ret_b, shape_iou))

    ss_generator.inference_steps = prev_steps

    if verbose:
        logger.info(f"Batched sampling complete: {K} samples")

    return results


# ------------------------------------------------------------------
# Sequential guided sampling (reference / fallback)
# ------------------------------------------------------------------

def guided_sample_sparse_structure(
    pipeline, ss_input_dict, z_target_flat, mesh_ss,
    guidance_strength=0.7, pose_guidance_strength=0.0, pose_target=None,
    pose_translation_guidance_strength=None,
    pose_scale_guidance_strength=None,
    seed=42, inference_steps=None, verbose=True,
    pose_sde_strength=0.0,
    condition_args=None, condition_kwargs=None,
):
    """Single-sample sequential guided sampling with Taylor cache support."""
    ss_generator = pipeline.models["ss_generator"]
    ss_decoder = pipeline.models["ss_decoder"]

    ss_generator.no_shortcut = True
    ss_generator.reverse_fn.strength = pipeline.ss_cfg_strength
    ss_generator.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm

    prev_steps = ss_generator.inference_steps
    if inference_steps:
        ss_generator.inference_steps = inference_steps

    img = ss_input_dict["image"]
    bs = img.shape[0]
    device = img.device
    alpha = guidance_strength
    alpha_pose = pose_guidance_strength
    alpha_pose_translation = (
        alpha_pose
        if pose_translation_guidance_strength is None
        else float(pose_translation_guidance_strength)
    )
    alpha_pose_scale = (
        alpha_pose
        if pose_scale_guidance_strength is None
        else float(pose_scale_guidance_strength)
    )

    pose_target_latent = None
    if max(alpha_pose, alpha_pose_translation, alpha_pose_scale) > 0 and pose_target is not None:
        pose_target_latent = _prepare_pose_target_latent(
            pose_target, ss_input_dict, device,
        )

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            assert pipeline.is_mm_dit(), "Expected multi-modal DiT"
            latent_shape_dict = {
                k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
            }

            if condition_args is None or condition_kwargs is None:
                condition_args, condition_kwargs = pipeline.get_condition_input(
                    pipeline.condition_embedders["ss_condition_embedder"],
                    ss_input_dict,
                    pipeline.ss_condition_input_mapping,
                )

            torch.manual_seed(seed)
            x_t = ss_generator._generate_noise(latent_shape_dict, device)

            z_noise = x_t["shape"].clone()
            z_target = z_target_flat.to(device, dtype=z_noise.dtype)

            pose_noise = {}
            if pose_target_latent is not None:
                for k in _POSE_LATENT_KEYS:
                    pose_noise[k] = x_t[k].clone()

            t_seq, d = ss_generator._prepare_t_and_d()

            # Initialize Taylor expansion cache for ShortCut_faster
            _is_faster_gen = hasattr(ss_generator, 'cache_dic')
            if _is_faster_gen:
                from cache_utils_ss import cache_init as _ss_cache_init
                num_euler_steps = len(t_seq) - 1
                if ss_generator.ss_params is None:
                    ss_generator.ss_params = {
                        'ss_cache_stride': 3,
                        'ss_warmup': 2,
                        'ss_order': 1,
                        'ss_momentum_beta': 0.5,
                    }
                ss_generator.cache_dic, ss_generator.current = _ss_cache_init(
                    num_euler_steps,
                    cache_interval=ss_generator.ss_params['ss_cache_stride'],
                    max_order=ss_generator.ss_params['ss_order'],
                    first_enhance=ss_generator.ss_params['ss_warmup'],
                    end_enhance=num_euler_steps,
                )
                ss_generator.prev_v = None

            if verbose:
                parts = [f"{len(t_seq)-1} steps", f"alpha_shape={alpha:.2f}"]
                if max(alpha_pose, alpha_pose_translation, alpha_pose_scale) > 0:
                    parts.append(
                        "alpha_pose="
                        f"R{alpha_pose:.2f}/T{alpha_pose_translation:.2f}/S{alpha_pose_scale:.2f}"
                    )
                if pose_sde_strength > 0:
                    parts.append(f"pose_sde={pose_sde_strength:.3f}")
                logger.info(f"Guided sampling: {', '.join(parts)}")

            for t0, t1 in zip(t_seq[:-1], t_seq[1:]):
                dt = t1 - t0
                velocity = ss_generator._generate_dynamics(
                    x_t, t0, d, *condition_args, **condition_kwargs
                )

                x_next = {}
                for k in x_t:
                    x_next[k] = x_t[k] + velocity[k] * dt

                t_next = t1

                if alpha > 0:
                    z_ref = (1.0 - t_next) * z_noise + t_next * z_target
                    x_next["shape"] = (1.0 - alpha) * x_next["shape"] + alpha * z_ref

                if pose_target_latent is not None:
                    for k in _POSE_LATENT_KEYS:
                        alpha_key = _pose_guidance_alpha(
                            k,
                            alpha_pose,
                            alpha_pose_translation,
                            alpha_pose_scale,
                        )
                        if alpha_key <= 0:
                            continue
                        z_ref_pose = (1.0 - t_next) * pose_noise[k] + t_next * pose_target_latent[k].to(x_next[k].dtype)
                        x_next[k] = (1.0 - alpha_key) * x_next[k] + alpha_key * z_ref_pose

                if pose_sde_strength > 0:
                    sigma = 1.0 - t_next
                    if sigma > 1e-3:
                        var_t = sigma / max(1.0 - sigma, dt)
                        std_dev_t = (var_t ** 0.5) * pose_sde_strength
                        noise_scale = std_dev_t * (dt ** 0.5)
                        for k in _POSE_LATENT_KEYS:
                            x_next[k] = x_next[k] + noise_scale * torch.randn_like(x_next[k])

                x_t = x_next

            # Decode shape to coords
            return_dict = x_t
            shape_latent = return_dict["shape"]
            ss = ss_decoder(
                shape_latent.permute(0, 2, 1)
                .contiguous()
                .view(shape_latent.shape[0], 8, 16, 16, 16)
            )
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

            return_dict["coords_original"] = coords
            original_count = coords.shape[0]
            if pipeline.downsample_ss_dist > 0:
                coords = prune_sparse_structure(
                    coords, max_neighbor_axes_dist=pipeline.downsample_ss_dist,
                )
            coords, downsample_factor = downsample_sparse_structure(coords)
            return_dict["coords"] = coords
            return_dict["downsample_factor"] = downsample_factor

    ss_generator.inference_steps = prev_steps

    # Compute shape IoU
    g_coords = return_dict["coords_original"]
    guided_ss = torch.zeros(1, 64, 64, 64, dtype=torch.long, device=device)
    guided_ss[0, g_coords[:, 1].long(), g_coords[:, 2].long(), g_coords[:, 3].long()] = 1
    shape_iou = compute_iou(mesh_ss.to(device), guided_ss)

    if verbose:
        logger.info(
            f"  Final: {coords.shape[0]} coords "
            f"(from {original_count}, factor={downsample_factor}), "
            f"shape_IoU={shape_iou:.4f}"
        )

    return return_dict, shape_iou


# ------------------------------------------------------------------
# Single-frame guided pose (batched)
# ------------------------------------------------------------------

def guided_predict_pose(
    pipeline, mesh_ss, rgba, z_target_flat, device,
    guidance_strength=0.7, pose_guidance_strength=0.0, pose_target=None,
    pose_translation_guidance_strength=None,
    pose_scale_guidance_strength=None,
    seed=42, post_optimize=False,
    init_trimesh=None, fixed_scale=None,
    pose_sde_strength=0.0,
    num_pose_samples=1,
    scoring_metric="shape_iou",
    render_mesh=None,
    gt_mask=None,
    occluder_mask=None,
    occluder_dilation_px=0,
    render_raw_iou_score_weight=0.0,
    component_balance_score_weight=0.0,
    component_bridge_score_weight=0.0,
    component_min_area_px=20,
    component_bridge_dilation_px=2,
    component_axis_guidance=False,
    component_axis_base_pose=None,
    component_axis_rendered_canonical_axis=None,
    component_axis_depth_angles_deg=None,
    image_hw=None,
    enable_shape_icp=True,
    pose_selection="greedy",
    cluster_dist_thresh=0.5,
    cluster_min_size=1,
    cluster_w_trans=1.0,
    cluster_w_rot=1.0,
    batch_chunk_size=8,
    ss_inference_steps=None,
    ll_steps=25,
    num_pose_samples_per_pgs=None,
    candidate_pose_prior=None,
    candidate_pose_priors=None,
    prior_translation_score_weight=0.0,
    prior_rotation_score_weight=0.0,
    history_context=None,
    history_alignment_score_weight=0.0,
    history_alignment_sigma=0.035,
    contact_context=None,
    contact_consistency_score_weight=0.0,
    contact_consistency_sigma=0.025,
    grasp_axis_context=None,
    grasp_axis_score_weight=0.0,
    grasp_axis_sigma_deg=20.0,
):
    # Preprocess (once -- independent of seed)
    pointmap_dict = pipeline.compute_pointmap(rgba)
    pointmap = pointmap_dict["pointmap"]
    ss_input_dict = pipeline.preprocess_image(
        rgba, pipeline.ss_preprocessor, pointmap=pointmap,
    )
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)

    # Intrinsics
    if "intrinsics" in pointmap_dict:
        intrinsics = pointmap_dict["intrinsics"].detach().cpu()
    else:
        loaded_img = pipeline.image_to_float(rgba)
        loaded_img_t = torch.from_numpy(loaded_img).permute(2, 0, 1).contiguous()[:3]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=pipeline.dtype):
                dm_output = pipeline.depth_model(loaded_img_t)
        intrinsics = dm_output["intrinsics"].detach().cpu()

    component_axis_hypotheses = []
    component_axis_diagnostics = {"active": False, "reason": "disabled"}
    if component_axis_guidance:
        axis_base_pose = component_axis_base_pose
        if axis_base_pose is None and pose_target is not None:
            axis_base_pose = pose_target[0] if isinstance(pose_target, list) else pose_target
        angles = (
            component_axis_depth_angles_deg
            if component_axis_depth_angles_deg is not None
            else [-70.0, -50.0, -30.0, -10.0, 0.0, 10.0, 30.0, 50.0, 70.0]
        )
        component_axis_hypotheses, component_axis_diagnostics = (
            _component_axis_pose_hypotheses(
                axis_base_pose,
                gt_mask,
                intrinsics.squeeze(),
                image_hw,
                component_axis_rendered_canonical_axis,
                angles,
                min_area_px=component_min_area_px,
            )
        )
        if component_axis_hypotheses:
            # Replace image-ambiguous hand hypotheses.  Each depth angle now
            # receives diffusion samples instead of being considered only in
            # post-hoc scoring.
            pose_target = component_axis_hypotheses
            logger.info(
                "  Component-axis guidance: "
                f"{len(component_axis_hypotheses)} modes, "
                f"image_angle={component_axis_diagnostics['image_axis_angle_deg']:.1f}deg"
            )

    # Build renderer once for render_iou scoring
    frame_renderer = None
    if scoring_metric == "render_iou" and render_mesh is not None and gt_mask is not None:
        H, W = image_hw
        frame_renderer = _build_silhouette_renderer(intrinsics.squeeze(), W, H, device)

    # Batched sampling: all K seeds in one (or chunked) forward pass
    if num_pose_samples_per_pgs is not None:
        # PGS sweep: sample num_pose_samples_per_pgs candidates at each pgs level
        pgs_levels = [i * 0.1 for i in range(10)]
        batched_results = []
        seed_offset = 0
        for pgs_val in pgs_levels:
            chunk_results = batched_guided_sample_sparse_structure(
                pipeline, ss_input_dict, z_target_flat, mesh_ss,
                num_samples=num_pose_samples_per_pgs,
                guidance_strength=guidance_strength,
                pose_guidance_strength=pgs_val,
                pose_target=pose_target,
                pose_translation_guidance_strength=pose_translation_guidance_strength,
                pose_scale_guidance_strength=pose_scale_guidance_strength,
                base_seed=seed + seed_offset,
                inference_steps=ss_inference_steps,
                verbose=False,
                pose_sde_strength=pose_sde_strength,
                chunk_size=batch_chunk_size,
            )
            batched_results.extend(chunk_results)
            seed_offset += num_pose_samples_per_pgs
        num_pose_samples = len(batched_results)
        logger.info(f"  PGS sweep: {num_pose_samples} samples across {len(pgs_levels)} levels {pgs_levels}")
    else:
        batched_results = batched_guided_sample_sparse_structure(
            pipeline, ss_input_dict, z_target_flat, mesh_ss,
            num_samples=num_pose_samples,
            guidance_strength=guidance_strength,
            pose_guidance_strength=pose_guidance_strength,
            pose_target=pose_target,
            pose_translation_guidance_strength=pose_translation_guidance_strength,
            pose_scale_guidance_strength=pose_scale_guidance_strength,
            base_seed=seed,
            inference_steps=ss_inference_steps,
            verbose=False,
            pose_sde_strength=pose_sde_strength,
            chunk_size=batch_chunk_size,
        )

    best_result = None
    best_score = -float("inf")
    all_samples = []

    for k, (guided_return_dict, shape_iou) in enumerate(batched_results):
        sample_seed = seed + k

        # Snapshot x1 latent BEFORE pose_decoder overwrites translation/scale
        x1_latent = {
            k_: guided_return_dict[k_].detach().clone().cpu()
            for k_ in guided_return_dict
            if k_ not in ("coords_original", "coords", "downsample_factor")
        }

        # Decode pose
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            guided_return_dict.update(pipeline.pose_decoder(
                guided_return_dict,
                scene_scale=pointmap_scale,
                scene_shift=pointmap_shift,
            ))
        if "scale" in guided_return_dict and "downsample_factor" in guided_return_dict:
            guided_return_dict["scale"] = guided_return_dict["scale"] * guided_return_dict["downsample_factor"]

        # Override scale before post-optimization
        if fixed_scale is not None:
            guided_return_dict["scale"] = fixed_scale.to(guided_return_dict["scale"].device)

        pose = {
            k_: guided_return_dict[k_].detach().cpu()
            for k_ in ("translation", "rotation", "scale")
            if k_ in guided_return_dict
        }

        result = {
            **pose,
            "shape_iou": shape_iou,
            "intrinsics": intrinsics,
            "sample_seed": sample_seed,
            "x1_latent": x1_latent,
        }

        # Compute render IoU if requested
        if scoring_metric == "render_iou" and render_mesh is not None and gt_mask is not None:
            H, W = image_hw
            if "post_opt_rotation" in result:
                pose_for_render = {
                    "rotation": result["post_opt_rotation"],
                    "translation": result["post_opt_translation"],
                    "scale": result["post_opt_scale"],
                }
            else:
                pose_for_render = {
                    "rotation": result["rotation"],
                    "translation": result["translation"],
                    "scale": result["scale"],
                }
            riou_details = compute_render_iou(
                render_mesh, pose_for_render, intrinsics.squeeze(),
                gt_mask, W, H, device, renderer=frame_renderer,
                occluder_mask=occluder_mask,
                occluder_dilation_px=occluder_dilation_px,
                component_min_area_px=component_min_area_px,
                component_bridge_dilation_px=component_bridge_dilation_px,
                return_details=True,
            )
            result["render_iou"] = riou_details["score_iou"]
            result["render_iou_raw"] = riou_details["raw_iou"]
            result["occlusion_applied"] = riou_details["occlusion_applied"]
            result["occlusion_ignored_gt_fraction"] = riou_details["ignored_gt_fraction"]
            result["occlusion_visible_render_fraction"] = riou_details["visible_render_fraction"]
            result["occluder_pixels"] = riou_details["occluder_pixels"]
            result["visible_component_count"] = riou_details["component_count"]
            result["visible_component_areas"] = riou_details["component_areas"]
            result["visible_component_recalls"] = riou_details["component_recalls"]
            result["visible_component_balance"] = riou_details["component_balance"]
            result["occlusion_bridge_score"] = riou_details["component_bridge"]
            if component_axis_hypotheses:
                result["component_axis_hypothesis"] = (
                    k % len(component_axis_hypotheses)
                )
                result["component_axis_depth_angle_deg"] = float(
                    component_axis_diagnostics["depth_angles_deg"]
                    [result["component_axis_hypothesis"]]
                )

        # Score selection.  Render IoU remains the image evidence; optional
        # hand/history terms only act as soft priors on candidate ranking.
        if scoring_metric == "render_iou" and "render_iou" in result:
            base_score = result["render_iou"]
        else:
            base_score = result["shape_iou"]
        score = float(base_score)
        result["base_selection_score"] = float(base_score)
        if scoring_metric == "render_iou" and "render_iou" in result:
            score += float(render_raw_iou_score_weight) * float(
                result.get("render_iou_raw", 0.0)
            )
            score += float(component_balance_score_weight) * float(
                result.get("visible_component_balance", 0.0)
            )
            score += float(component_bridge_score_weight) * float(
                result.get("occlusion_bridge_score", 0.0)
            )

        active_priors = candidate_pose_priors
        if active_priors is None and candidate_pose_prior is not None:
            active_priors = [candidate_pose_prior]
        if active_priors:
            prior_errors = [pose_errors(pose, prior) for prior in active_priors]
            weighted_errors = [
                float(prior_translation_score_weight) * translation_error
                + float(prior_rotation_score_weight) * rotation_error
                for translation_error, rotation_error in prior_errors
            ]
            if max(
                float(prior_translation_score_weight),
                float(prior_rotation_score_weight),
            ) > 0.0:
                best_prior_index = int(np.argmin(weighted_errors))
            else:
                # Diagnostics-only fallback with commensurate 5 cm / 30 degree
                # scales.  It does not alter selection when both weights are 0.
                best_prior_index = int(
                    np.argmin(
                        [
                            translation_error / 0.05
                            + rotation_error / np.deg2rad(30.0)
                            for translation_error, rotation_error in prior_errors
                        ]
                    )
                )
            translation_error, rotation_error = prior_errors[best_prior_index]
            result["pose_prior_hypothesis_count"] = len(active_priors)
            result["pose_prior_best_hypothesis"] = best_prior_index
            result["pose_prior_translation_error"] = translation_error
            result["pose_prior_rotation_error_deg"] = float(np.rad2deg(rotation_error))
            score -= weighted_errors[best_prior_index]

        if history_context is not None:
            alignment_score, median_residual = history_alignment_score(
                pose, history_context,
                sigma=history_alignment_sigma,
            )
            result["history_alignment_score"] = alignment_score
            result["history_alignment_median_m"] = median_residual
            score += float(history_alignment_score_weight) * alignment_score

        if contact_context is not None:
            contact_score, contact_residual, contact_reference = (
                contact_consistency_score(
                    pose,
                    contact_context,
                    sigma=contact_consistency_sigma,
                )
            )
            result["contact_consistency_score"] = contact_score
            result["contact_consistency_residual_m"] = contact_residual
            result["contact_consistency_reference_frame"] = contact_reference
            score += float(contact_consistency_score_weight) * contact_score

        if grasp_axis_context is not None:
            axis_score, axis_error_deg = grasp_axis_score(
                pose,
                grasp_axis_context,
                sigma_deg=grasp_axis_sigma_deg,
            )
            result["grasp_axis_score"] = axis_score
            result["grasp_axis_error_deg"] = axis_error_deg
            score += float(grasp_axis_score_weight) * axis_score

        result["selection_score"] = score
        all_samples.append(result)

        if num_pose_samples > 1:
            logger.info(
                f"    sample {k+1}/{num_pose_samples} (seed={sample_seed}): "
                f"base={base_score:.4f}, score={score:.4f}"
            )

        if score > best_score:
            best_score = score
            best_result = result

    # Candidate selection
    if num_pose_samples > 1:
        if pose_selection == "pose_log_likelihood":
            logger.info(f"  Computing pose-only log-likelihoods for {len(all_samples)} candidates (steps={ll_steps})...")
            log_liks = compute_pose_only_log_likelihoods(
                pipeline, ss_input_dict, all_samples, steps=ll_steps,
            )
            for sample, ll in zip(all_samples, log_liks):
                sample["pose_log_likelihood"] = ll
            best_idx = int(np.argmax(log_liks))
            best_result = all_samples[best_idx]
            best_score = log_liks[best_idx]
            logger.info(
                f"  Pose log-likelihood selection: best sample {best_idx} "
                f"(seed={best_result['sample_seed']}, "
                f"pose_log_p={best_score:.4f})"
            )
        elif pose_selection == "cluster":
            best_idx, cluster_info = cluster_pose_candidates(
                all_samples,
                scoring_metric="selection_score",
                dist_thresh=cluster_dist_thresh,
                min_cluster_size=cluster_min_size,
                w_trans=cluster_w_trans,
                w_rot=cluster_w_rot,
            )
            best_result = all_samples[best_idx]
            best_result["cluster_info"] = cluster_info
            logger.info(
                f"  Cluster selection: {cluster_info['num_clusters']} clusters, "
                f"sizes={cluster_info['cluster_sizes']}, "
                f"selected cluster size={cluster_info['selected_cluster_size']}, "
                f"best candidate idx={best_idx} "
                f"(seed={best_result['sample_seed']}, "
                f"score={cluster_info['best_candidate_score']:.4f})"
            )
        else:
            logger.info(
                f"  Greedy best of {num_pose_samples}: "
                f"score={best_score:.4f} (seed={best_result['sample_seed']})"
            )
        best_result["all_samples"] = all_samples

    best_result["component_axis_diagnostics"] = component_axis_diagnostics

    # Run post-optimization on the best sample only
    if post_optimize and init_trimesh is not None:
        try:
            mesh_copy = deepcopy(init_trimesh)
            postopt_pose = {
                k_: best_result[k_].to(device)
                for k_ in ("translation", "rotation", "scale")
                if k_ in best_result
            }
            postopt = pipeline.run_post_optimization(
                mesh_copy,
                intrinsics.to(device),
                deepcopy(postopt_pose),
                ss_input_dict,
                force_alignment=True,
                fixed_scale=fixed_scale,
                Enable_visible_ICP=False,
                Enable_shape_ICP=enable_shape_icp,
            )
            for k_ in ("translation", "rotation", "scale"):
                if k_ in postopt:
                    v = postopt[k_]
                    best_result[f"post_opt_{k_}"] = v.detach().cpu() if hasattr(v, "cpu") else v
            if "iou" in postopt:
                best_result["post_opt_iou"] = postopt["iou"]
        except Exception as e:
            logger.warning(f"Post-optimization failed: {e}")

        # Hard override: ensure fixed_scale is the final scale
        if fixed_scale is not None:
            fixed_cpu = fixed_scale.detach().cpu()
            if "post_opt_scale" in best_result:
                best_result["post_opt_scale"] = fixed_cpu.clone()

    return best_result


# ------------------------------------------------------------------
# Layout / combined visualization helpers
# ------------------------------------------------------------------

P3D_TO_ISAAC = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32).T


def _to_tensor(v):
    if isinstance(v, torch.Tensor):
        return v.float()
    return torch.tensor(v, dtype=torch.float32)


def _get_best_pose(result):
    if "post_opt_rotation" in result:
        rot = result["post_opt_rotation"]
        trans = result["post_opt_translation"]
        has_nan = (torch.isnan(rot).any() if hasattr(rot, 'isnan') else False) or \
                  (torch.isnan(trans).any() if hasattr(trans, 'isnan') else False)
        if not has_nan:
            return {
                "rotation": result["post_opt_rotation"],
                "translation": result["post_opt_translation"],
                "scale": result["post_opt_scale"],
            }
        logger.warning("Post-opt pose contains NaN, falling back to diffusion pose")
        return {
            "rotation": result["rotation"],
            "translation": result["translation"],
            "scale": result["scale"],
            "_copied_from_diffusion": True,
        }
    return {
        "rotation": result["rotation"],
        "translation": result["translation"],
        "scale": result["scale"],
    }


def _get_diffusion_pose(result):
    return {
        "rotation": result["rotation"],
        "translation": result["translation"],
        "scale": result["scale"],
    }


def compute_new_quat(mesh_vertices_yup, rotation, translation, scale):
    rot = _to_tensor(rotation)
    if rot.dim() == 1:
        rot = rot.unsqueeze(0)
    trans = _to_tensor(translation)
    if trans.dim() == 1:
        trans = trans.unsqueeze(0)
    sc = _to_tensor(scale)
    if sc.dim() == 1:
        sc = sc.unsqueeze(0)

    old_verts = mesh_vertices_yup.copy()
    verts_zup = mesh_vertices_yup.astype(np.float32) @ _R_YUP_TO_ZUP
    verts_t = torch.from_numpy(verts_zup).float()

    R_l2c = quaternion_to_matrix(rot)
    l2c = compose_transform(scale=sc, rotation=R_l2c, translation=trans)
    new_verts = (
        l2c.transform_points(verts_t.unsqueeze(0)).squeeze(0).numpy() @ P3D_TO_ISAAC
    )

    matrix_proc, _, _ = trimesh.registration.procrustes(
        old_verts, new_verts, reflection=False, return_cost=True,
    )
    return ScipyRotation.from_matrix(matrix_proc[:3, :3]).as_quat(scalar_first=True)


def _build_intrinsics_dict(intrinsics):
    if intrinsics is None:
        return None
    K = intrinsics.detach().cpu().numpy() if hasattr(intrinsics, "cpu") else np.array(intrinsics)
    return {
        "fx_norm": float(K[0, 0]),
        "fy_norm": float(K[1, 1]),
        "cx_norm": float(K[0, 2]),
        "cy_norm": float(K[1, 2]),
    }


def _build_pose_dict(pose, init_trimesh, include_matrix=False):
    rot = _to_tensor(pose["rotation"])
    if rot.dim() == 1:
        rot = rot.unsqueeze(0)
    trans = _to_tensor(pose["translation"])
    if trans.dim() == 1:
        trans = trans.unsqueeze(0)
    sc = _to_tensor(pose["scale"])
    if sc.dim() == 1:
        sc = sc.unsqueeze(0)

    new_quat = compute_new_quat(init_trimesh.vertices.copy(), rot, trans, sc)
    q_wxyz = rot.detach().cpu().flatten().tolist()
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

    d = {
        "translation": trans.detach().cpu().flatten().tolist(),
        "scale": sc.detach().cpu().flatten().tolist(),
        "quat_wxyz": q_wxyz,
        "new_quat": new_quat.tolist(),
        "quat_xyzw": q_xyzw,
    }
    if include_matrix:
        R_mat = quaternion_to_matrix(rot.detach())
        tfm = compose_transform(scale=sc.detach(), rotation=R_mat, translation=trans.detach())
        M = tfm.get_matrix().detach().cpu().squeeze(0)
        d["matrix_4x4_row_major"] = [float(v) for v in M.reshape(-1).tolist()]
    return d


def save_combined_visualization(
    output_dir, mesh_path, object_name, all_results,
    frame_indices, init_trimesh, plane_dir=None, plane_pose=None,
):
    combined_dir = os.path.join(output_dir, "combined_visualization")
    os.makedirs(combined_dir, exist_ok=True)

    objects_list = []

    for idx, frame_idx in enumerate(sorted(frame_indices)):
        result = all_results.get(frame_idx)
        if result is None:
            continue

        obj_name = f"{object_name}_frame{frame_idx}"

        best_dict = _build_pose_dict(_get_best_pose(result), init_trimesh)
        diff_dict = _build_pose_dict(_get_diffusion_pose(result), init_trimesh)

        entry = {
            "index": idx,
            "frame_idx": frame_idx,
            "camera_frame": f"cam{frame_idx}",
            "mesh_obj": f"{obj_name}.obj",
            "local_to_scene": best_dict,
            "diffusion_pose": diff_dict,
        }
        intrinsics_dict = _build_intrinsics_dict(result.get("intrinsics"))
        if intrinsics_dict is not None:
            entry["intrinsics_normalized"] = intrinsics_dict
        objects_list.append(entry)
        logger.info(f"  Frame {frame_idx}: added to combined layout")

    if plane_dir and os.path.isdir(plane_dir):
        plane_src = os.path.join(plane_dir, "plane.obj")
        if os.path.isfile(plane_src):
            plane_dst = os.path.join(combined_dir, "plane")
            os.makedirs(plane_dst, exist_ok=True)
            shutil.copy(plane_src, os.path.join(plane_dst, "plane.obj"))
            plane_entry = {
                "index": len(objects_list),
                "mesh_obj": "plane.obj",
                "local_to_scene": plane_pose or {
                    "translation": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "new_quat": [1, 0, 0, 0],
                },
            }
            objects_list.append(plane_entry)

    layout = {
        "frame": "per_camera",
        "note": (
            f"Guided pose tracking visualization. "
            f"Frames: {sorted(frame_indices)}. "
            f"Each object's pose is in its own camera frame."
        ),
        "objects": objects_list,
    }

    layout_path = os.path.join(combined_dir, "layout.json")
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    logger.info(f"Saved combined layout ({len(objects_list)} objects) to: {layout_path}")
    return layout_path


# ------------------------------------------------------------------
# Video processing (bidirectional from init_frame)
# ------------------------------------------------------------------

def process_video(args):
    # If num_pose_samples_per_pgs is set, compute effective num_pose_samples
    if args.num_pose_samples_per_pgs is not None:
        args.num_pose_samples = args.num_pose_samples_per_pgs * 10
        logger.info(f"PGS sweep mode: {args.num_pose_samples_per_pgs} samples/level × 10 levels = {args.num_pose_samples} total")

    device = torch.device(args.device)
    init_frame = args.init_frame
    try:
        component_axis_depth_angles_deg = [
            float(value.strip())
            for value in args.component_axis_depth_angles_deg.split(",")
            if value.strip()
        ]
    except ValueError:
        logger.error("--component_axis_depth_angles_deg must be comma-separated numbers")
        return
    if args.component_axis_guidance and not component_axis_depth_angles_deg:
        logger.error("--component_axis_guidance requires at least one depth angle")
        return

    masks_root = args.masks_root if args.masks_root else os.path.join(args.vid_dir, "video_segmentation", "masks")
    occluder_mask_names = [
        name.strip() for name in (args.occluder_mask_names or "").split(",")
        if name.strip()
    ]
    if bool(args.occluder_masks_root) != bool(occluder_mask_names):
        logger.error("--occluder_masks_root and --occluder_mask_names must be provided together")
        return
    if args.occluder_masks_root and not os.path.isdir(args.occluder_masks_root):
        logger.error(f"Occluder masks root not found: {args.occluder_masks_root}")
        return
    if args.occluder_dilation_px < 0:
        logger.error("--occluder_dilation_px must be non-negative")
        return
    if occluder_mask_names:
        logger.info(
            f"Occlusion-aware render IoU: root={args.occluder_masks_root}, "
            f"names={occluder_mask_names}, dilation={args.occluder_dilation_px}px"
        )
    if args.mesh:
        mesh_path = args.mesh
    else:
        init_frame_dir = get_frame_mask_dir(masks_root, init_frame)
        mesh_path = os.path.join(
            init_frame_dir, args.object_name, f"{args.object_name}.obj"
        )
    logger.info(f"Init mesh: {mesh_path}")

    frame_range = None
    if args.frames:
        parts = args.frames.split(":")
        frame_range = (int(parts[0]), int(parts[1]))
        logger.info(f"Frame range: {frame_range[0]} to {frame_range[1]}")

    frames = discover_frames(args.vid_dir, args.object_name, frame_range, masks_root=masks_root)
    logger.info(f"Found {len(frames)} frames with masks for '{args.object_name}'")
    if not frames:
        logger.error("No frames found. Check vid_dir and object_name.")
        return

    frame_lookup = {fidx: (fidx, img, msk) for fidx, img, msk in frames}
    all_frame_indices = sorted(frame_lookup.keys())

    extrinsics = None
    if args.hdf5_path is not None:
        extrinsics = load_extrinsics_from_hdf5(args.hdf5_path, all_frame_indices)
        logger.info(f"Loaded camera extrinsics for {len(extrinsics)} frames from {args.hdf5_path}")
    elif args.chain_poses:
        logger.info("No --hdf5_path provided; pose chaining will not account for camera motion.")

    if init_frame not in frame_lookup:
        logger.error(f"Init frame {init_frame} not found among discovered frames: "
                     f"{all_frame_indices[:5]}...{all_frame_indices[-5:]}")
        return

    pose_archive_frames = {}
    if args.pose_archive is not None:
        if not os.path.isfile(args.pose_archive):
            logger.error(f"Pose archive not found: {args.pose_archive}")
            return
        pose_archive_frames, _ = load_pose_archive(args.pose_archive)
        logger.info(
            f"Loaded pose archive with {len(pose_archive_frames)} frames: "
            f"{args.pose_archive}"
        )

    if (
        args.pose_prior_mode == "hand"
        or args.pose_prior_mode in _NEW_HAND_PRIOR_MODES
    ) and not args.hand_meshes:
        logger.error(f"--pose_prior_mode {args.pose_prior_mode} requires --hand_meshes")
        return
    if args.pose_prior_mode == "history":
        if not args.pose_archive:
            logger.error("--pose_prior_mode history requires --pose_archive")
            return
        if not args.history_pointmap_dir:
            logger.error("--pose_prior_mode history requires --history_pointmap_dir")
            return
    if args.pose_prior_mode in {"hand_memory", "hand_contact", "hand_fusion"}:
        if not args.pose_archive:
            logger.error(f"--pose_prior_mode {args.pose_prior_mode} requires --pose_archive")
            return

    # ---- Load pipeline (Fast-SAM3D style) ----
    logger.info("Loading pipeline...")
    config_path = os.path.join(project_root, args.config)
    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)

    # Fast-SAM3D acceleration config
    if args.enable_acceleration:
        args.enable_ss_cache = True
        args.enable_slat_carving = True
        args.enable_mesh_aggregation = True
    if args.enable_ss_cache:
        config['ss_generator_config_path'] = "ss_generator_faster.yaml"
    if args.enable_slat_carving:
        config['slat_generator_config_path'] = "slat_generator_faster.yaml"

    logger.info(f"[Fast-SAM3D] SS:{args.enable_ss_cache}, "
                f"SLaT:{args.enable_slat_carving}, "
                f"Mesh:{args.enable_mesh_aggregation}")

    inference = Inference(config, compile=False, args=args)
    pipeline = inference._pipeline

    # Wire up post-optimization method
    if pipeline.layout_post_optimization_method is None:
        pipeline.layout_post_optimization_method = layout_post_optimization
        logger.info("Wired layout_post_optimization_method onto pipeline")

    # Set Euler steps override
    ss_gen = pipeline.models["ss_generator"]
    if args.euler_steps is not None:
        ss_gen.inference_steps = args.euler_steps
        logger.info(f"Set euler steps to {args.euler_steps}")
    else:
        logger.info(f"Using default euler steps: {ss_gen.inference_steps}")

    # Set ss_params on the generator for ShortCut_faster cache
    if hasattr(ss_gen, 'ss_params') and ss_gen.ss_params is None:
        ss_gen.ss_params = {
            'ss_cache_stride': args.ss_cache_stride,
            'ss_warmup': args.ss_warmup,
            'ss_order': args.ss_order,
            'ss_momentum_beta': args.ss_momentum_beta,
        }
        logger.info(f"Set ss_params on generator: {ss_gen.ss_params}")

    logger.info("Pipeline loaded.")

    # ---- torch.compile backbone (one-time) ----
    if args.torch_compile:
        import time as _time
        logger.info("Compiling DiT backbone with torch.compile(mode='default')...")
        _t0 = _time.perf_counter()
        ss_gen.reverse_fn.backbone = torch.compile(
            ss_gen.reverse_fn.backbone, mode="default"
        )
        logger.info(f"torch.compile call took {_time.perf_counter() - _t0:.1f}s")
        # Trigger compilation with a dummy forward pass at batch=num_pose_samples
        logger.info("Running compile warmup (first forward triggers Triton codegen)...")
        _t0 = _time.perf_counter()
        _warmup_K = min(args.num_pose_samples, args.batch_chunk_size)
        _backbone = ss_gen.reverse_fn.backbone
        _lat_shape = {
            k: (1,) + (v.pos_emb.shape[0], v.input_layer.in_features)
            for k, v in _backbone.latent_mapping.items()
        }
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
                _warmup_noise = [ss_gen._generate_noise(_lat_shape, device) for _ in range(_warmup_K)]
                _warmup_x = {k: torch.cat([n[k] for n in _warmup_noise], dim=0) for k in _warmup_noise[0]}
                _dummy_rgba = np.zeros((480, 640, 4), dtype=np.uint8)
                _dummy_rgba[:, :, 3] = 255
                _pm = pipeline.compute_pointmap(_dummy_rgba)
                _ss_inp = pipeline.preprocess_image(_dummy_rgba, pipeline.ss_preprocessor, pointmap=_pm["pointmap"])
                _cond_args, _cond_kw = pipeline.get_condition_input(
                    pipeline.condition_embedders["ss_condition_embedder"],
                    _ss_inp, pipeline.ss_condition_input_mapping,
                )
                _cond_args_K = tuple(
                    a.expand(_warmup_K, *a.shape[1:]) if isinstance(a, torch.Tensor) else a
                    for a in _cond_args
                )
                ss_gen.reverse_fn.strength = pipeline.ss_cfg_strength
                ss_gen.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm
                _t_seq, _d = ss_gen._prepare_t_and_d()
                # For ShortCut_faster, initialize cache for warmup step
                if hasattr(ss_gen, 'cache_dic'):
                    from cache_utils_ss import cache_init as _ss_cache_init
                    ss_gen.cache_dic, ss_gen.current = _ss_cache_init(
                        len(_t_seq) - 1,
                        cache_interval=ss_gen.ss_params['ss_cache_stride'],
                        max_order=ss_gen.ss_params['ss_order'],
                        first_enhance=ss_gen.ss_params['ss_warmup'],
                        end_enhance=len(_t_seq) - 1,
                    )
                    ss_gen.prev_v = None
                _ = ss_gen._generate_dynamics(_warmup_x, _t_seq[0], _d, *_cond_args_K, **_cond_kw)
        del _warmup_noise, _warmup_x, _cond_args, _cond_kw, _cond_args_K, _ss_inp, _pm, _dummy_rgba
        torch.cuda.empty_cache()
        logger.info(f"Compile warmup done in {_time.perf_counter() - _t0:.1f}s")

    # ---- Voxelize mesh ----
    mesh_ss, mesh_coords, scale, center = voxelize(mesh_path)
    logger.info(f"Voxelized: {mesh_ss.sum().item()} occupied voxels")

    # ---- Decoder inversion ----
    z_target_5d, encode_method = encode_target_occupancy(
        pipeline, mesh_ss, device,
        opt_steps=args.latent_opt_steps,
        lr=args.latent_opt_lr,
    )
    z_target_flat = z_target_5d.view(1, 8, -1).permute(0, 2, 1).contiguous()
    logger.info(f"z_target ready ({encode_method}), shape: {z_target_flat.shape}")

    init_trimesh = (
        trimesh.load(mesh_path, force="mesh")
        if (
            args.post_optimize
            or args.save_layout
            or args.scoring_metric == "render_iou"
            or args.component_axis_guidance
            or args.pose_prior_mode in {"grasp_type", "hand_contact", "hand_fusion"}
        )
        else None
    )
    rendered_mesh_vertices = None
    rendered_canonical_axis = None
    if init_trimesh is not None:
        rendered_mesh_vertices = (
            np.asarray(init_trimesh.vertices, dtype=np.float64)
            @ _R_YUP_TO_ZUP_RENDER.astype(np.float64)
        )
        centered_rendered_vertices = (
            rendered_mesh_vertices - rendered_mesh_vertices.mean(axis=0)
        )
        _, _, rendered_principal_axes = np.linalg.svd(
            centered_rendered_vertices, full_matrices=False,
        )
        rendered_canonical_axis = rendered_principal_axes[0]

    if init_frame in pose_archive_frames:
        init_frame_result = deepcopy(pose_archive_frames[init_frame])
        init_pose_source = f"pose archive {args.pose_archive}"
    else:
        init_frame_result = load_frame_result_from_layout(
            masks_root, args.object_name, frame_idx=init_frame,
        )
        init_pose_source = "layout.json"
    if init_frame_result is None:
        logger.error(f"Could not load frame {init_frame} pose from layout.json. "
                     f"Run single-frame prediction for frame {init_frame} first.")
        return
    init_frame_scale = result_to_pose(init_frame_result)["scale"]
    logger.info(f"Loaded frame {init_frame} pose from {init_pose_source} "
                f"(scale={init_frame_scale.squeeze().tolist()})")

    if init_frame != 0 and not args.chain_poses:
        logger.warning(f"--init_frame {init_frame} without --chain_poses: "
                       f"pose guidance will use the same init frame pose for every frame. "
                       f"Consider adding --chain_poses for sequential propagation.")

    plane_pose = None
    if args.save_layout and args.plane_dir:
        init_frame_dir = get_frame_mask_dir(masks_root, init_frame)
        init_frame_layout_path = os.path.join(init_frame_dir, "layout.json")
        if os.path.isfile(init_frame_layout_path):
            with open(init_frame_layout_path) as f:
                init_frame_layout = json.load(f)
            for obj in init_frame_layout.get("objects", []):
                if obj.get("mesh_obj") == "plane.obj":
                    plane_pose = obj["local_to_scene"]
                    break

    # Load rotation velocity for adaptive pose guidance
    rotvel_dict = None
    if args.rotvel_json is not None:
        with open(args.rotvel_json) as f:
            raw = json.load(f)
        rotvel_dict = {int(entry["frame"]): abs(entry["rotation_deg"]) for entry in raw if "rotation_deg" in entry}
        logger.info(f"Loaded rotation velocity for {len(rotvel_dict)} frames from {args.rotvel_json}")
    # adaptive strength: --pg_ceil at 0 deg/frame, linear with --pg_slope, floor at --pg_floor.
    # Defaults reproduce the original formula: max(0.1, 0.7 - 0.09 * v).
    adaptive_pose_strength = lambda v: max(args.pg_floor, args.pg_ceil - args.pg_slope * v)
    logger.info(f"adaptive_pose_strength formula: max({args.pg_floor}, {args.pg_ceil} - {args.pg_slope} * v)")

    pose_target_from_file = None
    if args.pose_guidance_strength > 0 and args.pose_target_file:
        pose_target_data = torch.load(args.pose_target_file, map_location="cpu")
        pose_target_from_file = {
            k: v for k, v in pose_target_data.items()
            if k in _POSE_LATENT_KEYS or k in ("rotation", "translation", "scale",
                                                 "scene_scale", "scene_shift")
        }
        logger.info(f"Loaded pose target from {args.pose_target_file} (keys: {list(pose_target_from_file.keys())})")

    if args.chain_poses:
        if args.pose_guidance_strength <= 0:
            logger.warning("--chain_poses requires --pose_guidance_strength > 0, "
                           "setting pose_guidance_strength to 0.5")
            args.pose_guidance_strength = 0.5

    pose_prior_provider = None
    hand_sequence = None
    hand_motion_provider = None
    hand_memory_provider = None
    grasp_type_provider = None
    contact_provider = None
    history_pool_summary = None
    hand_memory_pool_summary = None

    hand_anchor_frame = (
        init_frame if args.hand_anchor_frame is None else args.hand_anchor_frame
    )
    hand_anchor_pose = None
    if args.pose_prior_mode == "hand" or args.pose_prior_mode in _NEW_HAND_PRIOR_MODES:
        if hand_anchor_frame in pose_archive_frames:
            hand_anchor_pose = result_to_pose(pose_archive_frames[hand_anchor_frame])
        elif hand_anchor_frame == init_frame:
            hand_anchor_pose = result_to_pose(init_frame_result)
        else:
            logger.error(
                f"Hand anchor frame {hand_anchor_frame} is unavailable; provide it "
                "through --pose_archive or use --init_frame"
            )
            return

    if args.pose_prior_mode == "hand":
        try:
            pose_prior_provider = HandObjectPosePrior(
                args.hand_meshes,
                anchor_frame=hand_anchor_frame,
                anchor_pose=hand_anchor_pose,
                hand=args.hand_side,
                jump_translation_threshold=args.hand_jump_translation_threshold,
                jump_rotation_threshold_deg=args.hand_jump_rotation_threshold_deg,
                flip_xy_to_pytorch3d=args.hand_flip_xy_to_pytorch3d,
            )
        except (OSError, KeyError, ValueError, IndexError) as exc:
            logger.error(f"Failed to initialize hand pose prior: {exc}")
            return
        logger.info(
            f"Hand-object pose prior: hand={args.hand_side}, "
            f"anchor={hand_anchor_frame}, detected resets="
            f"{pose_prior_provider.reset_frames}"
        )
    elif args.pose_prior_mode == "history":
        history_occluder_name = args.history_occluder_name
        if history_occluder_name is None and occluder_mask_names:
            history_occluder_name = occluder_mask_names[0]
        try:
            pose_prior_provider = HistoryMemoryPosePrior(
                pose_archive_path=args.pose_archive,
                frames_dir=os.path.join(args.vid_dir, "all_frames"),
                masks_root=masks_root,
                pointmap_dir=args.history_pointmap_dir,
                object_name=args.object_name,
                occluder_masks_root=args.occluder_masks_root,
                occluder_name=history_occluder_name,
                history_start_frame=args.history_start_frame,
                history_end_frame=args.history_end_frame,
                pool_size=args.history_pool_size,
                min_frame_gap=args.history_min_frame_gap,
                occluder_dilation_px=args.occluder_dilation_px,
                max_keyframes=args.history_max_keyframes,
                lookback=args.history_lookback,
                min_matches=args.history_min_matches,
                match_ratio=args.history_match_ratio,
                feature_mask_dilation_px=args.history_feature_mask_dilation_px,
                ransac_threshold=args.history_ransac_threshold,
                ransac_iterations=args.history_ransac_iterations,
                flip_xy_to_pytorch3d=args.history_flip_xy_to_pytorch3d,
            )
        except (OSError, KeyError, ValueError) as exc:
            logger.error(f"Failed to initialize history pose prior: {exc}")
            return
        history_pool_summary = pose_prior_provider.pool_summary()
        logger.info(
            "History memory pool: "
            + ", ".join(
                f"f{item['frame_idx']}(q={item['quality']:.3f}, "
                f"occ={item['hand_occlusion']:.2f})"
                for item in history_pool_summary
            )
        )
    elif args.pose_prior_mode in _NEW_HAND_PRIOR_MODES:
        hand_mask_name = args.hand_mask_name
        if hand_mask_name is None and occluder_mask_names:
            hand_mask_name = occluder_mask_names[0]
        hand_pointmap_dir = args.hand_pointmap_dir
        if hand_pointmap_dir is None:
            hand_pointmap_dir = args.history_pointmap_dir
        if hand_pointmap_dir is None:
            hand_pointmap_dir = os.path.join(args.vid_dir, "all_frames")
        try:
            hand_sequence = ManoHandSequence(
                args.hand_meshes,
                hand=args.hand_side,
                pointmap_dir=hand_pointmap_dir,
                hand_masks_root=args.occluder_masks_root,
                hand_mask_name=hand_mask_name,
                hand_focal=args.hand_camera_focal,
                coordinate_scale=args.hand_coordinate_scale,
                scale_smoothing_radius=args.hand_scale_smoothing_radius,
                flip_xy_to_pytorch3d=args.hand_flip_xy_to_pytorch3d,
            )
            if args.pose_prior_mode in {"hand_motion", "hand_contact", "hand_fusion"}:
                hand_motion_provider = HandMotionPosePrior(
                    hand_sequence,
                    anchor_frame=hand_anchor_frame,
                    anchor_pose=hand_anchor_pose,
                    grasp_type=args.hand_grasp_type,
                    coarse_blend_weight=args.hand_coarse_blend_weight,
                )
            if args.pose_prior_mode in {"hand_memory", "hand_fusion"}:
                hand_memory_provider = GraspMemoryPosePrior(
                    hand_sequence,
                    pose_archive_path=args.pose_archive,
                    masks_root=masks_root,
                    object_name=args.object_name,
                    hand_masks_root=args.occluder_masks_root,
                    hand_mask_name=hand_mask_name,
                    history_start_frame=args.hand_memory_start_frame,
                    history_end_frame=args.hand_memory_end_frame,
                    pool_size=args.hand_memory_pool_size,
                    top_k=args.hand_memory_top_k,
                    lookback=args.hand_memory_lookback,
                    grasp_type=args.hand_grasp_type,
                    mask_dilation_px=args.occluder_dilation_px,
                )
                hand_memory_pool_summary = hand_memory_provider.pool_summary()
            if args.pose_prior_mode in {"grasp_type", "hand_fusion"}:
                grasp_type_provider = GraspTypePosePrior(
                    hand_sequence,
                    anchor_frame=hand_anchor_frame,
                    anchor_pose=hand_anchor_pose,
                    canonical_axis=rendered_canonical_axis,
                    grasp_type=args.hand_grasp_type,
                )
            if args.pose_prior_mode in {"hand_contact", "hand_fusion"}:
                if args.contact_reference_frames:
                    reference_frames = [
                        int(value.strip())
                        for value in args.contact_reference_frames.split(",")
                        if value.strip()
                    ]
                else:
                    reference_frames = sorted(
                        frame
                        for frame in pose_archive_frames
                        if hand_anchor_frame - args.contact_reference_window < frame
                        <= hand_anchor_frame
                    )
                contact_provider = ContactConsistencyPrior(
                    hand_sequence,
                    mesh_vertices=rendered_mesh_vertices,
                    pose_archive_path=args.pose_archive,
                    reference_frames=reference_frames,
                    sample_vertices=args.contact_sample_vertices,
                    grasp_type=args.hand_grasp_type,
                )
        except (OSError, KeyError, ValueError, IndexError) as exc:
            logger.error(f"Failed to initialize {args.pose_prior_mode}: {exc}")
            return
        logger.info(
            f"Hand-aware pose prior: mode={args.pose_prior_mode}, "
            f"hand={args.hand_side}, anchor={hand_anchor_frame}, "
            f"scale@anchor={hand_sequence.scale_at(hand_anchor_frame):.4f}"
        )
        if hand_memory_pool_summary is not None:
            logger.info(
                "Grasp memory pool: "
                + ", ".join(
                    f"f{item['frame_idx']}({item['grasp_type']},q={item['quality']:.2f})"
                    for item in hand_memory_pool_summary
                )
            )

    render_mesh = None
    image_hw = None
    if args.scoring_metric == "render_iou":
        render_mesh = _build_renderable_mesh(init_trimesh, device)
        first_image = load_image(frames[0][1])
        image_hw = (first_image.shape[0], first_image.shape[1])
        logger.info(f"Render IoU scoring: mesh built, image size={image_hw}")

    # Split frames into backward and forward passes
    backward_indices = sorted(
        [fidx for fidx in all_frame_indices if fidx < init_frame],
        reverse=True,
    )
    forward_indices = sorted(
        [fidx for fidx in all_frame_indices if fidx > init_frame],
    )

    total_to_process = len(backward_indices) + len(forward_indices)
    logger.info(f"Init frame: {init_frame} | "
                f"Backward: {len(backward_indices)} frames ({backward_indices[:3]}{'...' if len(backward_indices) > 3 else ''}) | "
                f"Forward: {len(forward_indices)} frames ({forward_indices[:3]}{'...' if len(forward_indices) > 3 else ''})")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    processed_count = [0]

    def _process_single_frame(frame_idx, pose_target, pass_name):
        processed_count[0] += 1
        fidx, image_path, mask_path = frame_lookup[frame_idx]

        logger.info(
            f"[{processed_count[0]}/{total_to_process}] ({pass_name}) Frame {frame_idx}: "
            f"{os.path.basename(image_path)} + {os.path.basename(mask_path)}"
        )

        image = load_image(image_path)
        mask = load_mask(mask_path)

        if mask.sum() == 0:
            logger.warning(f"  Frame {frame_idx}: empty mask -- skipping, reusing previous pose target")
            return None

        occluder_mask = None
        if occluder_mask_names:
            combined_occluder = np.zeros_like(mask, dtype=bool)
            found_occluders = []
            for occluder_name in occluder_mask_names:
                occluder_path = os.path.join(
                    args.occluder_masks_root,
                    f"frame_{frame_idx:06d}_masks",
                    f"{occluder_name}.png",
                )
                if not os.path.isfile(occluder_path):
                    continue
                frame_occluder = load_mask(occluder_path) > 0
                if frame_occluder.shape != mask.shape:
                    frame_occluder = cv2.resize(
                        frame_occluder.astype(np.uint8),
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                combined_occluder |= frame_occluder
                found_occluders.append(occluder_name)
            if found_occluders:
                occluder_mask = combined_occluder
                logger.info(
                    f"  Occluder masks: {found_occluders}, "
                    f"pixels={int(occluder_mask.sum())} before dilation"
                )
            else:
                logger.warning(
                    f"  Frame {frame_idx}: no occluder masks found; using raw render IoU"
                )

        frame_pose_target = pose_target
        candidate_pose_prior = None
        candidate_pose_priors = None
        history_context = None
        contact_context = None
        grasp_axis_context = None
        pose_prior_diagnostics = {"mode": "none"}
        decoded_coarse_pose = (
            pose_target
            if pose_target is not None
            and all(key in pose_target for key in ("rotation", "translation", "scale"))
            else None
        )
        hand_prior_in_range = (
            (args.hand_prior_start_frame is None or frame_idx >= args.hand_prior_start_frame)
            and (args.hand_prior_end_frame is None or frame_idx < args.hand_prior_end_frame)
        )
        if args.pose_prior_mode == "hand":
            prior_pose, pose_prior_diagnostics = pose_prior_provider.get_pose(frame_idx)
            if prior_pose is not None:
                if args.hand_use_chain_translation and decoded_coarse_pose is not None:
                    chain_pose = copy_pose(decoded_coarse_pose)
                    prior_pose["translation"] = chain_pose["translation"]
                    prior_pose["scale"] = chain_pose["scale"]
                    pose_prior_diagnostics["translation_source"] = "chained_pose"
                else:
                    pose_prior_diagnostics["translation_source"] = "hand_propagation"
                frame_pose_target = prior_pose
                candidate_pose_prior = prior_pose
        elif args.pose_prior_mode == "history":
            prior_pose, history_context, pose_prior_diagnostics = (
                pose_prior_provider.estimate(
                    frame_idx,
                    image,
                    mask > 0,
                    decoded_coarse_pose,
                )
            )
            if prior_pose is not None:
                frame_pose_target = prior_pose
                candidate_pose_prior = prior_pose
                candidate_pose_priors = [prior_pose]
        elif args.pose_prior_mode in _NEW_HAND_PRIOR_MODES and not hand_prior_in_range:
            pose_prior_diagnostics = {
                "mode": args.pose_prior_mode,
                "active": False,
                "reason": "outside_hand_interaction_range",
                "active_range": [
                    args.hand_prior_start_frame,
                    args.hand_prior_end_frame,
                ],
            }
        elif args.pose_prior_mode == "hand_motion":
            prior_pose, priors, pose_prior_diagnostics = (
                hand_motion_provider.estimate(frame_idx, decoded_coarse_pose)
            )
            if prior_pose is not None:
                frame_pose_target = prior_pose
                candidate_pose_prior = prior_pose
                candidate_pose_priors = priors
        elif args.pose_prior_mode == "hand_memory":
            prior_pose, priors, pose_prior_diagnostics = (
                hand_memory_provider.estimate(frame_idx, decoded_coarse_pose)
            )
            if prior_pose is not None:
                frame_pose_target = prior_pose
                candidate_pose_prior = prior_pose
                candidate_pose_priors = priors
        elif args.pose_prior_mode == "grasp_type":
            prior_pose, priors, grasp_axis_context, pose_prior_diagnostics = (
                grasp_type_provider.estimate(frame_idx, decoded_coarse_pose)
            )
            if prior_pose is not None:
                frame_pose_target = prior_pose
                candidate_pose_prior = prior_pose
                candidate_pose_priors = priors
        elif args.pose_prior_mode == "hand_contact":
            prior_pose, priors, pose_prior_diagnostics = (
                hand_motion_provider.estimate(frame_idx, decoded_coarse_pose)
            )
            contact_context, contact_diagnostics = contact_provider.context(frame_idx)
            pose_prior_diagnostics.update(contact_diagnostics)
            if prior_pose is not None:
                frame_pose_target = prior_pose
                candidate_pose_prior = prior_pose
                candidate_pose_priors = priors
        elif args.pose_prior_mode == "hand_fusion":
            prior_pose, priors, pose_prior_diagnostics = (
                hand_memory_provider.estimate(frame_idx, decoded_coarse_pose)
            )
            if prior_pose is None:
                prior_pose, priors, motion_diagnostics = (
                    hand_motion_provider.estimate(frame_idx, decoded_coarse_pose)
                )
                pose_prior_diagnostics["motion_fallback"] = motion_diagnostics
            type_pose, type_priors, grasp_axis_context, type_diagnostics = (
                grasp_type_provider.estimate(frame_idx, decoded_coarse_pose)
            )
            pose_prior_diagnostics["grasp_type_prior"] = type_diagnostics
            contact_context, contact_diagnostics = contact_provider.context(frame_idx)
            pose_prior_diagnostics.update(contact_diagnostics)
            if type_pose is not None:
                priors = list(priors) + list(type_priors)
            if prior_pose is None and type_pose is not None:
                prior_pose = type_pose
            if prior_pose is not None:
                frame_pose_target = prior_pose
                candidate_pose_prior = prior_pose
                candidate_pose_priors = priors or [prior_pose]

        if (
            args.hand_multimodal_guidance
            and candidate_pose_priors is not None
            and len(candidate_pose_priors) > 1
            and args.pose_prior_mode in _NEW_HAND_PRIOR_MODES
        ):
            frame_pose_target = candidate_pose_priors
            pose_prior_diagnostics["diffusion_pose_hypotheses"] = len(
                candidate_pose_priors
            )

        if args.pose_prior_mode != "none":
            logger.info(
                "  Pose prior: "
                + json.dumps(pose_prior_diagnostics, ensure_ascii=False)
            )

        rgba = np.concatenate(
            [image[..., :3], (mask.astype(np.uint8) * 255)[..., None]], axis=-1
        )

        # Inject HFER and acceleration params
        hfer = calculate_hfer_robust(mask_path)
        if hasattr(inference, 'get_hfer'):
            inference.get_hfer(hfer)
        if hasattr(inference, 'get_params'):
            inference.get_params(args)

        # Run guided prediction (batched)
        torch.cuda.synchronize()
        _t0 = time.perf_counter()

        # Compute per-frame pose guidance strength
        # rotvel_dict[n] = rotation from frame n to n+1
        # Forward (finding n+1): use rotvel[n] = rotvel[frame_idx - 1]
        # Backward (finding n-1): use rotvel[n-1] = rotvel[frame_idx]
        rotvel_key = (frame_idx - 1) if pass_name == "forward" else frame_idx
        if rotvel_dict is not None and rotvel_key in rotvel_dict:
            frame_pose_strength = adaptive_pose_strength(rotvel_dict[rotvel_key])
            logger.info(f"  rotvel[{rotvel_key}]={rotvel_dict[rotvel_key]:.2f} deg/frame → pose_guidance_strength={frame_pose_strength:.3f}")
        else:
            frame_pose_strength = args.pose_guidance_strength

        if args.pose_translation_guidance_strength is None:
            frame_translation_strength = (
                0.0 if args.pose_prior_mode == "hand" else frame_pose_strength
            )
        else:
            frame_translation_strength = args.pose_translation_guidance_strength
        if args.pose_scale_guidance_strength is None:
            frame_scale_strength = (
                0.0 if args.fix_scale_to_init_frame else frame_translation_strength
            )
        else:
            frame_scale_strength = args.pose_scale_guidance_strength

        result = guided_predict_pose(
            pipeline, mesh_ss, rgba, z_target_flat, device,
            guidance_strength=args.guidance_strength,
            pose_guidance_strength=frame_pose_strength,
            pose_target=frame_pose_target,
            pose_translation_guidance_strength=frame_translation_strength,
            pose_scale_guidance_strength=frame_scale_strength,
            seed=args.seed,
            post_optimize=args.post_optimize,
            init_trimesh=init_trimesh,
            fixed_scale=init_frame_scale if args.fix_scale_to_init_frame else None,
            pose_sde_strength=args.pose_sde_strength,
            num_pose_samples=args.num_pose_samples,
            scoring_metric=args.scoring_metric,
            render_mesh=render_mesh,
            gt_mask=mask,
            occluder_mask=occluder_mask,
            occluder_dilation_px=args.occluder_dilation_px,
            render_raw_iou_score_weight=args.render_raw_iou_score_weight,
            component_balance_score_weight=args.component_balance_score_weight,
            component_bridge_score_weight=args.component_bridge_score_weight,
            component_min_area_px=args.component_min_area_px,
            component_bridge_dilation_px=args.component_bridge_dilation_px,
            component_axis_guidance=(
                args.component_axis_guidance and hand_prior_in_range
            ),
            component_axis_base_pose=decoded_coarse_pose,
            component_axis_rendered_canonical_axis=rendered_canonical_axis,
            component_axis_depth_angles_deg=component_axis_depth_angles_deg,
            image_hw=image_hw,
            enable_shape_icp=args.enable_shape_icp,
            pose_selection=args.pose_selection,
            cluster_dist_thresh=args.cluster_dist_thresh,
            cluster_min_size=args.cluster_min_size,
            cluster_w_trans=args.cluster_w_trans,
            cluster_w_rot=args.cluster_w_rot,
            batch_chunk_size=args.batch_chunk_size,
            ss_inference_steps=args.ss_inference_steps,
            ll_steps=args.ll_steps,
            num_pose_samples_per_pgs=args.num_pose_samples_per_pgs,
            candidate_pose_prior=candidate_pose_prior,
            candidate_pose_priors=candidate_pose_priors,
            prior_translation_score_weight=args.prior_translation_score_weight,
            prior_rotation_score_weight=args.prior_rotation_score_weight,
            history_context=history_context,
            history_alignment_score_weight=args.history_alignment_score_weight,
            history_alignment_sigma=args.history_alignment_sigma,
            contact_context=contact_context,
            contact_consistency_score_weight=args.contact_consistency_score_weight,
            contact_consistency_sigma=args.contact_consistency_sigma,
            grasp_axis_context=grasp_axis_context,
            grasp_axis_score_weight=args.grasp_axis_score_weight,
            grasp_axis_sigma_deg=args.grasp_axis_sigma_deg,
        )

        torch.cuda.synchronize()
        result["frame_time_s"] = time.perf_counter() - _t0
        result["pose_prior_diagnostics"] = pose_prior_diagnostics
        result.pop("x1_latent", None)  # strip large latent before saving

        # Save all K samples to a separate file
        if args.num_pose_samples > 1 and "all_samples" in result:
            all_samples_data = result.pop("all_samples")
            # Strip large latent tensors to keep saved files lean
            for s in all_samples_data:
                s.pop("x1_latent", None)
            samples_save = {
                "frame_idx": frame_idx,
                "num_samples": len(all_samples_data),
                "scoring_metric": args.scoring_metric,
                "pose_selection": args.pose_selection,
                "best_seed": result["sample_seed"],
                "samples": all_samples_data,
            }
            if "cluster_info" in result:
                samples_save["cluster_info"] = result.pop("cluster_info")
            samples_path = os.path.join(args.output_dir, f"frame_{frame_idx:06d}_samples.pt")
            torch.save(samples_save, samples_path)

        # Log pose
        t = result["translation"].squeeze().numpy()
        s = result["scale"].squeeze().numpy()
        iou = result["shape_iou"]
        msg = f"  pose: t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}], s={s[0]:.4f}, IoU={iou:.4f}"
        if "post_opt_iou" in result:
            msg += f", post_opt_IoU={result['post_opt_iou']:.4f}"
        if "render_iou" in result:
            if result.get("occlusion_applied", False):
                msg += (
                    f", render_IoU(raw->visible)={result['render_iou_raw']:.4f}"
                    f"->{result['render_iou']:.4f}, "
                    f"ignored_gt={100.0 * result['occlusion_ignored_gt_fraction']:.1f}%"
                )
            else:
                msg += f", render_IoU={result['render_iou']:.4f}"
        if "selection_score" in result:
            msg += f", selection={result['selection_score']:.4f}"
        if "pose_prior_rotation_error_deg" in result:
            msg += f", prior_R_err={result['pose_prior_rotation_error_deg']:.1f}deg"
        if "history_alignment_median_m" in result:
            msg += f", history_med={result['history_alignment_median_m']:.4f}m"
        if "contact_consistency_residual_m" in result:
            msg += (
                f", contact={result['contact_consistency_residual_m']:.4f}m"
                f"@f{result['contact_consistency_reference_frame']}"
            )
        if "grasp_axis_error_deg" in result:
            msg += f", axis_err={result['grasp_axis_error_deg']:.1f}deg"
        if result.get("visible_component_count", 0) > 1:
            recalls = "/".join(
                f"{value:.2f}" for value in result["visible_component_recalls"]
            )
            msg += (
                f", component_recall={recalls}"
                f", bridge={result['occlusion_bridge_score']:.2f}"
            )
        if "component_axis_depth_angle_deg" in result:
            msg += f", mask_axis_depth={result['component_axis_depth_angle_deg']:.0f}deg"
        logger.info(msg)

        all_results[frame_idx] = result

        return result

    # ---- Backward pass: K-1 -> 0 ----
    if backward_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKWARD PASS: frame {init_frame} -> frame {backward_indices[-1]}")
        logger.info(f"{'='*60}")
        bwd_pose_target = _get_best_pose(init_frame_result) if args.chain_poses else pose_target_from_file
        bwd_pose_frame = init_frame

        for frame_idx in backward_indices:
            if args.chain_poses:
                bwd_pose_target_in_curr = _maybe_transform_pose(
                    bwd_pose_target, bwd_pose_frame, frame_idx, extrinsics,
                )
            elif extrinsics is not None and pose_target_from_file is not None:
                bwd_pose_target_in_curr = _maybe_transform_pose(
                    pose_target_from_file, init_frame, frame_idx, extrinsics,
                )
            else:
                bwd_pose_target_in_curr = bwd_pose_target

            result = _process_single_frame(frame_idx, bwd_pose_target_in_curr, "backward")
            if result is None:
                continue
            if args.chain_poses:
                if args.chain_on_diffusion:
                    bwd_pose_target = _get_diffusion_pose(result)
                else:
                    bwd_pose_target = _get_best_pose(result)
                bwd_pose_frame = frame_idx

    # ---- Forward pass: K+1 -> N ----
    if forward_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"FORWARD PASS: frame {init_frame} -> frame {forward_indices[-1]}")
        logger.info(f"{'='*60}")
        fwd_pose_target = _get_best_pose(init_frame_result) if args.chain_poses else pose_target_from_file
        fwd_pose_frame = init_frame

        for frame_idx in forward_indices:
            if args.chain_poses:
                fwd_pose_target_in_curr = _maybe_transform_pose(
                    fwd_pose_target, fwd_pose_frame, frame_idx, extrinsics,
                )
            elif extrinsics is not None and pose_target_from_file is not None:
                fwd_pose_target_in_curr = _maybe_transform_pose(
                    pose_target_from_file, init_frame, frame_idx, extrinsics,
                )
            else:
                fwd_pose_target_in_curr = fwd_pose_target

            result = _process_single_frame(frame_idx, fwd_pose_target_in_curr, "forward")
            if result is None:
                continue
            if args.chain_poses:
                if args.chain_on_diffusion:
                    fwd_pose_target = _get_diffusion_pose(result)
                else:
                    fwd_pose_target = _get_best_pose(result)
                fwd_pose_frame = frame_idx

    # Always include init frame result
    all_results[init_frame] = init_frame_result

    # Save
    output = {
        "mesh_path": mesh_path,
        "object_name": args.object_name,
        "init_frame": init_frame,
        "euler_steps": args.euler_steps if args.euler_steps is not None else ss_gen.inference_steps,
        "guidance_strength": args.guidance_strength,
        "pose_guidance_strength": args.pose_guidance_strength,
        "pose_translation_guidance_strength": args.pose_translation_guidance_strength,
        "pose_scale_guidance_strength": args.pose_scale_guidance_strength,
        "pose_sde_strength": args.pose_sde_strength,
        "num_pose_samples": args.num_pose_samples,
        "scoring_metric": args.scoring_metric,
        "occluder_masks_root": args.occluder_masks_root,
        "occluder_mask_names": occluder_mask_names,
        "occluder_dilation_px": args.occluder_dilation_px,
        "render_raw_iou_score_weight": args.render_raw_iou_score_weight,
        "component_balance_score_weight": args.component_balance_score_weight,
        "component_bridge_score_weight": args.component_bridge_score_weight,
        "component_min_area_px": args.component_min_area_px,
        "component_bridge_dilation_px": args.component_bridge_dilation_px,
        "component_axis_guidance": args.component_axis_guidance,
        "component_axis_depth_angles_deg": component_axis_depth_angles_deg,
        "pose_selection": args.pose_selection,
        "cluster_dist_thresh": args.cluster_dist_thresh if args.pose_selection == "cluster" else None,
        "cluster_min_size": args.cluster_min_size if args.pose_selection == "cluster" else None,
        "latent_opt_steps": args.latent_opt_steps,
        "encode_method": encode_method,
        "hdf5_path": args.hdf5_path,
        "enable_ss_cache": args.enable_ss_cache,
        "torch_compile": args.torch_compile,
        "batch_chunk_size": args.batch_chunk_size,
        "pose_prior_mode": args.pose_prior_mode,
        "pose_archive": args.pose_archive,
        "hand_meshes": args.hand_meshes,
        "hand_side": (
            args.hand_side
            if args.pose_prior_mode == "hand" or args.pose_prior_mode in _NEW_HAND_PRIOR_MODES
            else None
        ),
        "hand_anchor_frame": (
            args.hand_anchor_frame
            if args.pose_prior_mode == "hand" or args.pose_prior_mode in _NEW_HAND_PRIOR_MODES
            else None
        ),
        "hand_grasp_type": (
            args.hand_grasp_type if args.pose_prior_mode in _NEW_HAND_PRIOR_MODES else None
        ),
        "hand_multimodal_guidance": args.hand_multimodal_guidance,
        "hand_prior_range": [
            args.hand_prior_start_frame,
            args.hand_prior_end_frame,
        ],
        "hand_memory_pool": hand_memory_pool_summary,
        "prior_translation_score_weight": args.prior_translation_score_weight,
        "prior_rotation_score_weight": args.prior_rotation_score_weight,
        "history_alignment_score_weight": args.history_alignment_score_weight,
        "contact_consistency_score_weight": args.contact_consistency_score_weight,
        "grasp_axis_score_weight": args.grasp_axis_score_weight,
        "history_memory_pool": history_pool_summary,
        "frames": all_results,
    }
    save_path = os.path.join(args.output_dir, "guided_poses.pt")
    torch.save(output, save_path)
    logger.info(f"Saved {len(all_results)} frame results to {save_path}")

    if args.save_layout and init_trimesh is not None:
        frame_indices = sorted(all_results.keys())
        save_combined_visualization(
            args.output_dir, mesh_path, args.object_name,
            all_results, frame_indices, init_trimesh,
            plane_dir=args.plane_dir,
            plane_pose=plane_pose,
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Guided pose prediction for video (Fast-SAM3D batched, bidirectional from arbitrary init frame)."
    )
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml",
                        help="Path to pipeline config YAML (relative to project root)")
    parser.add_argument("--vid_dir", required=True, help="Path to video directory")
    parser.add_argument("--object_name", required=True, help="Object name (e.g. bottle_0)")
    parser.add_argument("--masks_root", default=None,
                        help="Override masks root directory. Default: <vid_dir>/video_segmentation/masks")
    parser.add_argument("--occluder_masks_root", default=None,
                        help="Per-frame occluder mask root, using frame_XXXXXX_masks/<name>.png")
    parser.add_argument("--occluder_mask_names", default=None,
                        help="Comma-separated occluder mask names used as ignored regions for render IoU")
    parser.add_argument("--occluder_dilation_px", type=int, default=0,
                        help="Dilate combined occluder mask by this many pixels before render IoU scoring")
    parser.add_argument("--render_raw_iou_score_weight", type=float, default=0.0,
                        help="Additional candidate reward for full, non-occlusion-masked silhouette IoU")
    parser.add_argument("--component_balance_score_weight", type=float, default=0.0,
                        help="Reward the minimum recall across all visible object-mask components")
    parser.add_argument("--component_bridge_score_weight", type=float, default=0.0,
                        help="Reward one rendered silhouette connecting visible components through the occluder")
    parser.add_argument("--component_min_area_px", type=int, default=20,
                        help="Minimum object-mask connected-component area used by balanced scoring")
    parser.add_argument("--component_bridge_dilation_px", type=int, default=2,
                        help="Small boundary tolerance used only for the occlusion bridge test")
    parser.add_argument("--component_axis_guidance", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Generate diffusion pose modes from the line joining the two largest visible mask components")
    parser.add_argument("--component_axis_depth_angles_deg",
                        default="-70,-50,-30,-10,0,10,30,50,70",
                        help="Comma-separated object-axis depth angles for component-axis pose modes")
    parser.add_argument("--mesh", default=None, help="Override init mesh path")
    parser.add_argument("--output_dir", default="guided_pose_output", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--init_frame", type=int, default=0,
                        help="Anchor frame index (default 0)")
    parser.add_argument("--guidance_strength", type=float, default=0.7, help="Shape guidance alpha")
    parser.add_argument("--pose_guidance_strength", type=float, default=0.0,
                        help="Pose guidance alpha")
    parser.add_argument("--pose_translation_guidance_strength", type=float, default=None,
                        help="Translation/translation-scale guidance alpha. Default: use pose guidance alpha; hand prior mode defaults to 0.")
    parser.add_argument("--pose_scale_guidance_strength", type=float, default=None,
                        help="Scale guidance alpha. Default: use translation alpha, or 0 with fixed scale.")
    parser.add_argument("--pose_target_file", default=None,
                        help="Path to .pt file with target pose")
    parser.add_argument("--pose_archive", default=None,
                        help="Existing guided_poses.pt used for a non-layout init frame and prior anchors/keyframes")
    parser.add_argument("--pose_prior_mode", default="none",
                        choices=[
                            "none", "hand", "history", "hand_motion",
                            "hand_memory", "grasp_type", "hand_contact",
                            "hand_fusion",
                        ],
                        help=(
                            "Additional pose prior. New hand modes provide active-joint "
                            "motion, grasp memory, category geometry, COP-style contact, "
                            "or their fusion."
                        ))
    parser.add_argument("--prior_translation_score_weight", type=float, default=0.0,
                        help="Candidate penalty per metre from the active pose prior")
    parser.add_argument("--prior_rotation_score_weight", type=float, default=0.0,
                        help="Candidate penalty per radian from the active pose prior")
    parser.add_argument("--contact_consistency_score_weight", type=float, default=0.0,
                        help="Positive candidate reward for COP-style contact consistency")
    parser.add_argument("--contact_consistency_sigma", type=float, default=0.025,
                        help="Metre scale converting contact residual to a score")
    parser.add_argument("--grasp_axis_score_weight", type=float, default=0.0,
                        help="Positive candidate reward for grasp-category axis agreement")
    parser.add_argument("--grasp_axis_sigma_deg", type=float, default=20.0,
                        help="Angular scale for grasp-axis candidate scoring")

    # --- Hand-object relative pose prior ---
    parser.add_argument("--hand_meshes", default=None,
                        help="HaWoR all_hand_meshes.npz")
    parser.add_argument("--hand_side", default="right", choices=["left", "right"])
    parser.add_argument("--hand_anchor_frame", type=int, default=None,
                        help="Reliable grasp frame defining T_hand_object; defaults to init frame")
    parser.add_argument("--hand_use_chain_translation", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use chained SAM3D translation while taking rotation from the hand prior")
    parser.add_argument("--hand_flip_xy_to_pytorch3d", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Convert HaWoR camera x/y axes to PyTorch3D axes")
    parser.add_argument("--hand_jump_translation_threshold", type=float, default=0.15,
                        help="Metres above which a HaWoR coordinate reset is stitched")
    parser.add_argument("--hand_jump_rotation_threshold_deg", type=float, default=60.0,
                        help="Degrees above which a HaWoR coordinate reset is stitched")

    # --- Camera-space MANO contact and grasp priors ---
    parser.add_argument("--hand_pointmap_dir", default=None,
                        help="MoGe point maps used to align monocular MANO scale")
    parser.add_argument("--hand_mask_name", default=None,
                        help="Hand mask filename stem; defaults to the first occluder")
    parser.add_argument("--hand_camera_focal", type=float, default=609.5352935791016,
                        help="Focal length used for HaWoR camera-space projection")
    parser.add_argument("--hand_coordinate_scale", type=float, default=1.7,
                        help="Fallback MANO-to-pointmap scale when depth calibration fails")
    parser.add_argument("--hand_scale_smoothing_radius", type=int, default=2,
                        help="Temporal median radius for MANO depth-scale calibration")
    parser.add_argument("--hand_grasp_type", default="auto",
                        choices=["auto", "grab", "clip", "pinch", "power_hold"],
                        help="Automatic or forced grasp-category geometric reference")
    parser.add_argument("--hand_prior_start_frame", type=int, default=None,
                        help="First frame of the hand-interaction segment (inclusive)")
    parser.add_argument("--hand_prior_end_frame", type=int, default=None,
                        help="End of the hand-interaction segment (exclusive)")
    parser.add_argument("--hand_coarse_blend_weight", type=float, default=0.0,
                        help="Blend hand-motion target toward the chained SAM3D pose")
    parser.add_argument("--hand_multimodal_guidance", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Distribute diffusion candidates across all hand/grasp pose hypotheses")
    parser.add_argument("--hand_memory_start_frame", type=int, default=0)
    parser.add_argument("--hand_memory_end_frame", type=int, default=None)
    parser.add_argument("--hand_memory_pool_size", type=int, default=32)
    parser.add_argument("--hand_memory_top_k", type=int, default=4)
    parser.add_argument("--hand_memory_lookback", type=int, default=160)
    parser.add_argument("--contact_reference_frames", default=None,
                        help="Comma-separated reliable frames for contact signatures")
    parser.add_argument("--contact_reference_window", type=int, default=4,
                        help="Number of archive frames ending at the hand anchor when no list is given")
    parser.add_argument("--contact_sample_vertices", type=int, default=768,
                        help="Canonical mesh vertices used by contact-distance scoring")

    # --- EgoAERO-inspired low-occlusion history memory ---
    parser.add_argument("--history_pointmap_dir", default=None,
                        help="Directory containing NNNNNN_pointmap.npy RGB-D point maps")
    parser.add_argument("--history_occluder_name", default=None,
                        help="Hand mask name for memory quality; defaults to first occluder mask name")
    parser.add_argument("--history_start_frame", type=int, default=0)
    parser.add_argument("--history_end_frame", type=int, default=None)
    parser.add_argument("--history_pool_size", type=int, default=24)
    parser.add_argument("--history_min_frame_gap", type=int, default=3)
    parser.add_argument("--history_max_keyframes", type=int, default=4)
    parser.add_argument("--history_lookback", type=int, default=120)
    parser.add_argument("--history_min_matches", type=int, default=8)
    parser.add_argument("--history_match_ratio", type=float, default=0.78)
    parser.add_argument("--history_feature_mask_dilation_px", type=int, default=6)
    parser.add_argument("--history_ransac_threshold", type=float, default=0.035)
    parser.add_argument("--history_ransac_iterations", type=int, default=384)
    parser.add_argument("--history_flip_xy_to_pytorch3d", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--history_alignment_score_weight", type=float, default=0.0,
                        help="Positive candidate reward for RGB-D history alignment")
    parser.add_argument("--history_alignment_sigma", type=float, default=0.035,
                        help="Metre scale for converting median history residual to a score")
    parser.add_argument("--latent_opt_steps", type=int, default=1000, help="Decoder inversion steps")
    parser.add_argument("--latent_opt_lr", type=float, default=0.05, help="Decoder inversion LR")
    parser.add_argument("--post_optimize", action="store_true", help="Run post-optimization")
    parser.add_argument("--frames", default=None, help="Frame range as 'start:end'")
    parser.add_argument("--save_layout", action="store_true",
                        help="Save per-frame layout.json and combined visualization")
    parser.add_argument("--plane_dir", default=None, help="Path to plane mesh directory")
    parser.add_argument("--fix_scale_to_init_frame", action="store_true",
                        help="Fix scale to init frame's scale")
    parser.add_argument("--fix_scale_to_frame0", action="store_true",
                        dest="fix_scale_to_init_frame",
                        help=argparse.SUPPRESS)
    parser.add_argument("--chain_poses", action="store_true",
                        help="Chain previous frame's pose as target for next frame")
    parser.add_argument("--chain_on_diffusion", action="store_true", default=False,
                        help="When chaining, use raw diffusion pose instead of post-optimized")
    parser.add_argument("--pose_sde_strength", type=float, default=0.0,
                        help="SDE noise strength for pose channels")
    parser.add_argument("--num_pose_samples", type=int, default=1,
                        help="Number of pose samples (K) per frame")
    parser.add_argument("--num_pose_samples_per_pgs", type=int, default=None,
                        help="Samples per PGS level (0.0, 0.1, ..., 0.9). "
                             "Overrides --num_pose_samples. Total = this * 10.")
    parser.add_argument("--scoring_metric", default="shape_iou",
                        choices=["render_iou", "shape_iou"],
                        help="Metric for selecting best of K samples")
    parser.add_argument("--pose_selection", default="greedy",
                        choices=["greedy", "cluster", "pose_log_likelihood"],
                        help="How to select best of K pose samples")
    parser.add_argument("--ll_steps", type=int, default=25,
                        help="Number of ODE integration steps for pose log-likelihood scoring")
    parser.add_argument("--cluster_dist_thresh", type=float, default=0.5,
                        help="Distance threshold for clustering")
    parser.add_argument("--cluster_min_size", type=int, default=1,
                        help="Minimum cluster size")
    parser.add_argument("--cluster_w_trans", type=float, default=1.0,
                        help="Translation distance weight in clustering")
    parser.add_argument("--cluster_w_rot", type=float, default=0.5,
                        help="Rotation distance weight in clustering")
    parser.add_argument("--enable_shape_icp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable ICP in post-optimization")
    parser.add_argument("--hdf5_path", default=None,
                        help="Path to HDF5 file with camera extrinsics")
    parser.add_argument("--rotvel_json", default=None,
                        help="Path to rotation velocity JSON (frame_idx → degrees/frame). "
                            "When provided, pose_guidance_strength is set per-frame via "
                            "max(--pg_floor, --pg_ceil - --pg_slope * v).")
    parser.add_argument("--pg_floor", type=float, default=0.1,
                        help="Floor for adaptive pose_guidance_strength formula (used with --rotvel_json).")
    parser.add_argument("--pg_ceil", type=float, default=0.7,
                        help="Ceiling (intercept) for adaptive pose_guidance_strength formula.")
    parser.add_argument("--pg_slope", type=float, default=0.09,
                        help="Slope (per deg/frame) for adaptive pose_guidance_strength formula.")

    parser.add_argument("--ss_inference_steps", type=int, default=None,
                        help="Override SS generator inference steps for guided sampling")

    # --- Batching ---
    parser.add_argument("--batch_chunk_size", type=int, default=8,
                        help="Batch chunk size for batched sampling (adjust based on GPU VRAM)")
    parser.add_argument("--euler_steps", type=int, default=None,
                        help="Number of Euler steps (default: pipeline default)")

    # --- torch.compile ---
    parser.add_argument("--torch_compile", action="store_true", default=False,
                        help="Apply torch.compile(mode='default') to DiT backbone. "
                             "One-time ~60s compile warmup at startup.")

    # --- Fast-SAM3D acceleration: SS cache ---
    parser.add_argument("--ss_cache_stride", type=int, default=3)
    parser.add_argument("--ss_warmup", type=int, default=2)
    parser.add_argument("--ss_order", type=int, default=1)
    parser.add_argument("--ss_momentum_beta", type=float, default=0.5)

    # --- Fast-SAM3D acceleration: SLaT ---
    parser.add_argument("--slat_thresh", type=float, default=0.5)
    parser.add_argument("--slat_warmup", type=int, default=2)
    parser.add_argument("--slat_carving_ratio", type=float, default=0.15)

    # --- Fast-SAM3D acceleration: Mesh ---
    parser.add_argument("--mesh_spectral_threshold_low", type=float, default=0.5)
    parser.add_argument("--mesh_spectral_threshold_high", type=float, default=0.7)

    # --- Acceleration toggles ---
    parser.add_argument("--enable_ss_cache", action="store_true")
    parser.add_argument("--enable_slat_carving", action="store_true")
    parser.add_argument("--enable_mesh_aggregation", action="store_true")
    parser.add_argument("--enable_acceleration", action="store_true",
                        help="Enable all acceleration stages (SS + SLaT + Mesh)")

    args = parser.parse_args()

    process_video(args)


if __name__ == "__main__":
    main()
