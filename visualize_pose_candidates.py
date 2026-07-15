#!/usr/bin/env python3
"""Render saved pose candidates as per-frame contact sheets.

The tracker stores diffusion pose candidates in ``frame_*_samples.pt``.
This utility projects the fixed object mesh with each saved pose, overlays the
rendered silhouette on the RGB frame, and reports both raw silhouette IoU and
occlusion-aware visible-region IoU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from track_object import (
    _build_renderable_mesh,
    _build_silhouette_renderer,
    _compute_binary_iou,
    _render_silhouette,
)


CLUSTER_COLORS = [
    (255, 128, 0),
    (0, 180, 255),
    (180, 0, 255),
    (0, 210, 120),
    (255, 80, 160),
    (160, 160, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize every saved pose candidate for selected frames."
    )
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--masks_root", required=True)
    parser.add_argument("--occluder_masks_root", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_name", default="beaker")
    parser.add_argument("--occluder_name", default="right_hand_sam3")
    parser.add_argument("--occluder_dilation_px", type=int, default=5)
    parser.add_argument("--cluster_min_size", type=int, default=3)
    parser.add_argument("--grid_cols", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_binary_mask(path: Path, height: int, width: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    size = 2 * radius + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def select_by_cluster(
    scores: list[float], labels: list[int], min_cluster_size: int
) -> tuple[int, int]:
    members: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        members.setdefault(int(label), []).append(idx)

    ranking = []
    for label, indices in members.items():
        average = float(np.mean([scores[idx] for idx in indices]))
        ranking.append((label, len(indices), average, indices))
    ranking.sort(key=lambda item: (item[1], item[2]), reverse=True)

    valid = [item for item in ranking if item[1] >= min_cluster_size]
    if not valid:
        valid = ranking
    selected_label, _, _, selected_members = valid[0]
    selected_idx = max(selected_members, key=lambda idx: scores[idx])
    return selected_idx, selected_label


def add_mask_overlay(
    image: np.ndarray,
    render_mask: np.ndarray,
    gt_mask: np.ndarray,
    ignored_mask: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    render_color = np.zeros_like(output)
    render_color[:] = (0, 220, 0)
    output[render_mask] = cv2.addWeighted(
        output[render_mask], 0.62, render_color[render_mask], 0.38, 0
    )

    ignored_color = np.zeros_like(output)
    ignored_color[:] = (0, 0, 255)
    output[ignored_mask] = cv2.addWeighted(
        output[ignored_mask], 0.72, ignored_color[ignored_mask], 0.28, 0
    )

    render_contours, _ = cv2.findContours(
        render_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    gt_contours, _ = cv2.findContours(
        gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    ignored_contours, _ = cv2.findContours(
        ignored_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, render_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(output, gt_contours, -1, (255, 255, 0), 2)
    cv2.drawContours(output, ignored_contours, -1, (0, 0, 255), 2)
    return output


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.62,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_cell(
    image: np.ndarray,
    render_mask: np.ndarray,
    gt_mask: np.ndarray,
    ignored_mask: np.ndarray,
    candidate_idx: int,
    sample_seed: int,
    cluster_label: int,
    raw_iou: float,
    visible_iou: float,
    raw_selected_idx: int,
    visible_selected_idx: int,
) -> np.ndarray:
    cell = add_mask_overlay(image, render_mask, gt_mask, ignored_mask)
    cv2.rectangle(cell, (0, 0), (cell.shape[1], 67), (0, 0, 0), -1)

    markers = []
    if candidate_idx == raw_selected_idx:
        markers.append("RAW")
    if candidate_idx == visible_selected_idx:
        markers.append("VIS")
    marker_text = "+".join(markers)

    first_line = f"#{candidate_idx:02d} seed={sample_seed} C{cluster_label}"
    if marker_text:
        first_line += f" {marker_text}"
    put_text(cell, first_line, (10, 25))
    put_text(cell, f"IoU raw={raw_iou:.3f} vis={visible_iou:.3f}", (10, 54))

    cluster_color = CLUSTER_COLORS[(cluster_label - 1) % len(CLUSTER_COLORS)]
    cv2.rectangle(cell, (2, 2), (cell.shape[1] - 3, cell.shape[0] - 3), cluster_color, 4)
    if candidate_idx == raw_selected_idx:
        cv2.rectangle(cell, (7, 7), (cell.shape[1] - 8, cell.shape[0] - 8), (0, 255, 255), 5)
    if candidate_idx == visible_selected_idx:
        cv2.rectangle(cell, (13, 13), (cell.shape[1] - 14, cell.shape[0] - 14), (255, 0, 255), 5)
    return cell


def make_header(
    width: int,
    frame_idx: int,
    num_samples: int,
    overlap_fraction: float,
    raw_selected_idx: int,
    raw_selected_seed: int,
    visible_selected_idx: int,
    visible_selected_seed: int,
) -> np.ndarray:
    header = np.full((126, width, 3), 24, dtype=np.uint8)
    put_text(
        header,
        f"Frame {frame_idx:03d} | {num_samples} diffusion pose candidates | hand/object overlap={100.0 * overlap_fraction:.1f}%",
        (18, 32),
        scale=0.82,
    )
    put_text(
        header,
        f"RAW-selected: #{raw_selected_idx:02d} seed={raw_selected_seed} (yellow) | visible-IoU reselected: #{visible_selected_idx:02d} seed={visible_selected_seed} (magenta)",
        (18, 72),
        scale=0.72,
    )
    put_text(
        header,
        "green=rendered mesh silhouette | cyan=object mask | red=ignored hand mask | outer color=SE(3) cluster",
        (18, 108),
        scale=0.66,
    )
    return header


def make_overview(previews: list[tuple[int, np.ndarray]], output_path: Path) -> None:
    target_width = 1200
    resized = []
    for frame_idx, image in previews:
        height = int(round(image.shape[0] * target_width / image.shape[1]))
        small = cv2.resize(image, (target_width, height), interpolation=cv2.INTER_AREA)
        put_text(small, f"frame {frame_idx:03d}", (18, 32), scale=0.8)
        resized.append(small)

    blank = np.full_like(resized[0], 24)
    while len(resized) % 2:
        resized.append(blank.copy())
    rows = [cv2.hconcat(resized[i : i + 2]) for i in range(0, len(resized), 2)]
    overview = cv2.vconcat(rows)
    if not cv2.imwrite(str(output_path), overview, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise RuntimeError(f"Failed to write {output_path}")


def main() -> None:
    args = parse_args()
    if args.grid_cols <= 0:
        raise ValueError("--grid_cols must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    mesh_obj = trimesh.load(args.mesh, force="mesh", process=False)
    render_mesh = _build_renderable_mesh(mesh_obj, device)

    all_rows: list[dict] = []
    frame_summaries = []
    previews: list[tuple[int, np.ndarray]] = []

    for frame_idx in args.frames:
        samples_path = Path(args.samples_dir) / f"frame_{frame_idx:06d}_samples.pt"
        data = torch.load(samples_path, map_location="cpu", weights_only=False)
        samples = data["samples"]
        if not samples:
            raise ValueError(f"No samples in {samples_path}")

        image_path = Path(args.frames_dir) / f"{frame_idx:06d}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]

        gt_path = (
            Path(args.masks_root)
            / f"frame_{frame_idx:06d}_masks"
            / f"{args.object_name}.png"
        )
        occluder_path = (
            Path(args.occluder_masks_root)
            / f"frame_{frame_idx:06d}_masks"
            / f"{args.occluder_name}.png"
        )
        gt_mask = load_binary_mask(gt_path, height, width)
        occluder_mask = load_binary_mask(occluder_path, height, width)
        ignored_mask = dilate_mask(occluder_mask, args.occluder_dilation_px)
        valid_mask = ~ignored_mask
        overlap_fraction = float((gt_mask & ignored_mask).sum()) / max(int(gt_mask.sum()), 1)

        intrinsics = samples[0]["intrinsics"].detach().float().squeeze()
        renderer = _build_silhouette_renderer(intrinsics, width, height, device)

        rendered_masks = []
        raw_scores = []
        visible_scores = []
        for sample in samples:
            pose = {
                "translation": sample["translation"],
                "rotation": sample["rotation"],
                "scale": sample["scale"],
            }
            alpha = _render_silhouette(
                render_mesh,
                pose,
                intrinsics,
                width,
                height,
                device,
                renderer=renderer,
            )
            render_mask = alpha > 0.5
            rendered_masks.append(render_mask)
            raw_scores.append(_compute_binary_iou(render_mask, gt_mask))
            visible_scores.append(_compute_binary_iou(render_mask, gt_mask, valid_mask))

        cluster_info = data.get("cluster_info", {})
        labels = cluster_info.get("labels", [1] * len(samples))
        if len(labels) != len(samples):
            raise ValueError(f"Cluster labels do not match candidates in {samples_path}")
        raw_selected_idx = int(cluster_info.get("best_candidate_idx", np.argmax(raw_scores)))
        visible_selected_idx, visible_selected_cluster = select_by_cluster(
            visible_scores, labels, args.cluster_min_size
        )

        cells = []
        for candidate_idx, (sample, render_mask, raw_iou, visible_iou, label) in enumerate(
            zip(samples, rendered_masks, raw_scores, visible_scores, labels)
        ):
            cells.append(
                draw_cell(
                    image,
                    render_mask,
                    gt_mask,
                    ignored_mask,
                    candidate_idx,
                    int(sample["sample_seed"]),
                    int(label),
                    raw_iou,
                    visible_iou,
                    raw_selected_idx,
                    visible_selected_idx,
                )
            )
            all_rows.append(
                {
                    "frame_idx": frame_idx,
                    "candidate_idx": candidate_idx,
                    "sample_seed": int(sample["sample_seed"]),
                    "cluster_label": int(label),
                    "raw_iou": raw_iou,
                    "visible_iou": visible_iou,
                    "raw_selected": candidate_idx == raw_selected_idx,
                    "visible_selected": candidate_idx == visible_selected_idx,
                    "translation": sample["translation"].detach().cpu().flatten().tolist(),
                    "rotation_wxyz": sample["rotation"].detach().cpu().flatten().tolist(),
                    "scale": sample["scale"].detach().cpu().flatten().tolist(),
                }
            )

        rows = int(np.ceil(len(cells) / args.grid_cols))
        blank = np.zeros_like(cells[0])
        while len(cells) < rows * args.grid_cols:
            cells.append(blank.copy())
        grid_rows = [
            cv2.hconcat(cells[row * args.grid_cols : (row + 1) * args.grid_cols])
            for row in range(rows)
        ]
        grid = cv2.vconcat(grid_rows)
        header = make_header(
            grid.shape[1],
            frame_idx,
            len(samples),
            overlap_fraction,
            raw_selected_idx,
            int(samples[raw_selected_idx]["sample_seed"]),
            visible_selected_idx,
            int(samples[visible_selected_idx]["sample_seed"]),
        )
        contact_sheet = cv2.vconcat([header, grid])

        png_path = output_dir / f"frame_{frame_idx:06d}_candidates.png"
        if not cv2.imwrite(str(png_path), contact_sheet):
            raise RuntimeError(f"Failed to write {png_path}")
        preview_width = 1600
        preview_height = int(round(contact_sheet.shape[0] * preview_width / contact_sheet.shape[1]))
        preview = cv2.resize(
            contact_sheet, (preview_width, preview_height), interpolation=cv2.INTER_AREA
        )
        preview_path = output_dir / f"frame_{frame_idx:06d}_candidates_preview.jpg"
        if not cv2.imwrite(
            str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"Failed to write {preview_path}")
        previews.append((frame_idx, contact_sheet))

        frame_summaries.append(
            {
                "frame_idx": frame_idx,
                "num_samples": len(samples),
                "cluster_sizes": cluster_info.get("cluster_sizes"),
                "hand_object_overlap_fraction": overlap_fraction,
                "raw_selected_idx": raw_selected_idx,
                "raw_selected_seed": int(samples[raw_selected_idx]["sample_seed"]),
                "raw_selected_iou": raw_scores[raw_selected_idx],
                "visible_selected_idx": visible_selected_idx,
                "visible_selected_seed": int(samples[visible_selected_idx]["sample_seed"]),
                "visible_selected_cluster": visible_selected_cluster,
                "visible_selected_iou": visible_scores[visible_selected_idx],
                "selection_changed": raw_selected_idx != visible_selected_idx,
                "contact_sheet": png_path.name,
                "preview": preview_path.name,
            }
        )
        torch.cuda.empty_cache()

    make_overview(previews, output_dir / "overview.jpg")

    csv_path = output_dir / "candidate_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame_idx",
            "candidate_idx",
            "sample_seed",
            "cluster_label",
            "raw_iou",
            "visible_iou",
            "raw_selected",
            "visible_selected",
            "translation",
            "rotation_wxyz",
            "scale",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "samples_dir": str(Path(args.samples_dir).resolve()),
        "frames_dir": str(Path(args.frames_dir).resolve()),
        "masks_root": str(Path(args.masks_root).resolve()),
        "occluder_masks_root": str(Path(args.occluder_masks_root).resolve()),
        "mesh": str(Path(args.mesh).resolve()),
        "frames": args.frames,
        "occluder_name": args.occluder_name,
        "occluder_dilation_px": args.occluder_dilation_px,
        "cluster_min_size": args.cluster_min_size,
        "note": (
            "Pose candidates come from the saved tracking run. Raw and visible IoU "
            "are recomputed here; visible IoU ignores the dilated hand-mask region."
        ),
        "frame_summaries": frame_summaries,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(json.dumps(frame_summaries, indent=2))


if __name__ == "__main__":
    main()
