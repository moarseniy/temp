#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_REQUIRED_LABELS = {
    "passport_details",
    "gender",
    "birth_date",
    "snils",
    "oms",
    "work_place",
    "job_title",
    "patient_name",
    "patient_reg_address",
    "patient_res_address",
    "patient_phone_number",
}

DEFAULT_IMPORTANT_LABEL = "important"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}

EPS = 1e-6


# -----------------------------
# Label Studio loading
# -----------------------------


def basename_from_ls_image_path(path: str) -> str:
    """Return file name from Label Studio data.image path."""
    return Path(str(path).split("?d=")[-1]).name


def rect_percent_to_quad(value: dict[str, Any], original_width: int, original_height: int):
    """
    Convert Label Studio rectangle result to an image-space quad.

    Label Studio stores x/y/width/height in percentages and rotation in degrees.
    Rotation is applied around the rectangle top-left anchor.
    """
    x0 = value["x"] / 100.0 * original_width
    y0 = value["y"] / 100.0 * original_height
    w = value["width"] / 100.0 * original_width
    h = value["height"] / 100.0 * original_height
    rotation = float(value.get("rotation", 0.0) or 0.0)

    theta = np.deg2rad(rotation)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    quad = []
    for dx, dy in [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]:
        x = x0 + dx * cos_t - dy * sin_t
        y = y0 + dx * sin_t + dy * cos_t
        quad.append([x, y])

    return quad


def safe_filename_part(value: str) -> str:
    value = Path(str(value)).name.strip()
    value = re.sub(r"[^A-Za-zА-Яа-яЁё0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "unknown"


def make_vis_output_filename(filename: str, file_upload_by_file: dict[str, str]) -> str:
    upload_name = file_upload_by_file.get(filename)
    if not upload_name:
        return filename

    src = Path(filename)
    upload_part = safe_filename_part(upload_name)
    return f"{src.stem}_{upload_part}{src.suffix}"


def load_gt_label_studio(
    path: Path,
    required_labels: set[str],
    important_label: str,
    labels_filter: set[str] | None = None,
):
    """
    Load Label Studio rectanglelabels from annotations with was_cancelled == false.

    Categories:
    - required GT: labels from required_labels. They create TP/FN.
    - important regions: label == important_label. They restrict where FP is counted.
    - ignored GT: all other rectanglelabels. They do not create TP/FN and suppress FP.

    Prediction outside important regions is neutral and is not counted as FP.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_gt_by_file: dict[str, list[dict[str, Any]]] = {}
    ignored_gt_by_file: dict[str, list[dict[str, Any]]] = {}
    important_gt_by_file: dict[str, list[dict[str, Any]]] = {}
    all_gt_by_file: dict[str, list[dict[str, Any]]] = {}
    file_upload_by_file: dict[str, str] = {}
    valid_annotated_files: set[str] = set()

    stats = {
        "tasks_total": 0,
        "annotations_total": 0,
        "annotations_cancelled": 0,
        "annotations_used_was_cancelled_false": 0,
        "rectangle_results_used": 0,
        "rectangle_results_filtered_by_label": 0,
        "required_results_used": 0,
        "ignored_results_used": 0,
        "important_results_used": 0,
        "valid_annotated_files_count": 0,
    }

    for task in data:
        stats["tasks_total"] += 1
        image_path = task.get("data", {}).get("image")
        if not image_path:
            continue

        filename = basename_from_ls_image_path(image_path)
        required_gt_by_file.setdefault(filename, [])
        ignored_gt_by_file.setdefault(filename, [])
        important_gt_by_file.setdefault(filename, [])
        all_gt_by_file.setdefault(filename, [])

        if task.get("file_upload"):
            file_upload_by_file[filename] = task["file_upload"]

        for ann in task.get("annotations", []):
            stats["annotations_total"] += 1
            if ann.get("was_cancelled", False):
                stats["annotations_cancelled"] += 1
                continue

            stats["annotations_used_was_cancelled_false"] += 1
            valid_annotated_files.add(filename)

            for res in ann.get("result", []):
                if res.get("type") != "rectanglelabels":
                    continue

                value = res.get("value", {})
                rect_labels = list(value.get("rectanglelabels", []))

                if labels_filter is not None and not any(label in labels_filter for label in rect_labels):
                    stats["rectangle_results_filtered_by_label"] += 1
                    continue

                if "original_width" not in res or "original_height" not in res:
                    raise ValueError(
                        f"Label Studio result for file {filename!r} has no original_width/original_height"
                    )

                quad = rect_percent_to_quad(
                    value=value,
                    original_width=res["original_width"],
                    original_height=res["original_height"],
                )

                is_important = important_label in rect_labels
                is_required = any(label in required_labels for label in rect_labels)

                gt_item = {
                    "quad": quad,
                    "labels": rect_labels,
                    "is_required": is_required,
                    "is_important": is_important,
                    "rotation": float(value.get("rotation", 0.0) or 0.0),
                    "ls_result_id": res.get("id"),
                }

                all_gt_by_file[filename].append(gt_item)
                stats["rectangle_results_used"] += 1

                # A rectangle may technically have several labels.
                # If it contains both a required label and the important label,
                # keep it in both groups instead of letting important override required.
                if is_important:
                    important_gt_by_file[filename].append(gt_item)
                    stats["important_results_used"] += 1
                if is_required:
                    required_gt_by_file[filename].append(gt_item)
                    stats["required_results_used"] += 1
                if not is_required and not is_important:
                    ignored_gt_by_file[filename].append(gt_item)
                    stats["ignored_results_used"] += 1

    stats["valid_annotated_files_count"] = len(valid_annotated_files)

    return {
        "required_gt_by_file": required_gt_by_file,
        "ignored_gt_by_file": ignored_gt_by_file,
        "important_gt_by_file": important_gt_by_file,
        "all_gt_by_file": all_gt_by_file,
        "file_upload_by_file": file_upload_by_file,
        "valid_annotated_files": valid_annotated_files,
        "label_studio_stats": stats,
    }


# -----------------------------
# Prediction loading
# -----------------------------


def normalize_pred_item(item):
    """
    Supported prediction formats:
    - [[x,y], [x,y], [x,y], [x,y]]
    - [x1,y1,x2,y2]
    - {"quad": ...}
    - {"bbox": ...}
    - {"box": ...}
    """
    raw_item = item
    pred_label = None
    pred_score = None

    if isinstance(item, dict):
        pred_label = item.get("label") or item.get("class") or item.get("category")
        pred_score = item.get("score") or item.get("confidence") or item.get("prob")

        if "quad" in item:
            item = item["quad"]
        elif "bbox" in item:
            item = item["bbox"]
        elif "box" in item:
            item = item["box"]
        else:
            raise ValueError(f"Unknown prediction item dict format: {item}")

    if len(item) == 4 and all(isinstance(p, (int, float)) for p in item):
        x1, y1, x2, y2 = item
        quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif len(item) == 4 and all(hasattr(p, "__len__") and len(p) == 2 for p in item):
        quad = item
    else:
        raise ValueError(f"Unknown prediction item format: {item}")

    return {
        "quad": quad,
        "label": pred_label,
        "score": pred_score,
        "raw": raw_item,
    }


def load_predictions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Predictions JSON must be a dict: filename/path -> list of boxes")

    pred_by_file: dict[str, list[dict[str, Any]]] = {}

    for key, items in data.items():
        filename = Path(str(key)).name
        pred_by_file.setdefault(filename, [])

        if items is None:
            continue
        if not isinstance(items, list):
            raise ValueError(f"Predictions for {key!r} must be a list, got {type(items).__name__}")

        for item in items:
            pred_by_file[filename].append(normalize_pred_item(item))

    return pred_by_file


# -----------------------------
# Geometry
# -----------------------------


def polygon_area(quad) -> float:
    pts = np.asarray(quad, dtype=np.float32)
    return float(abs(cv2.contourArea(pts)))


def polygon_intersection_area(quad_a, quad_b) -> float:
    a = np.asarray(quad_a, dtype=np.float32)
    b = np.asarray(quad_b, dtype=np.float32)

    if polygon_area(a) <= 0 or polygon_area(b) <= 0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(a, b)
    return float(max(inter_area, 0.0))


def polygon_iou(quad_a, quad_b) -> float:
    area_a = polygon_area(quad_a)
    area_b = polygon_area(quad_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0

    inter_area = polygon_intersection_area(quad_a, quad_b)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


def polygon_intersects(quad_a, quad_b) -> bool:
    return polygon_intersection_area(quad_a, quad_b) > EPS


def union_iou_raster(gt_quads, pred_quads) -> float:
    """
    Approximate IoU between union(GT quads) and union(pred quads).

    This is used for the single IoU logic, but supports task-specific cases:
    - many small predictions inside one large GT;
    - one prediction covering several GT boxes.
    """
    if not gt_quads or not pred_quads:
        return 0.0

    all_pts = []
    for quad in list(gt_quads) + list(pred_quads):
        pts = np.asarray(quad, dtype=np.float32)
        if pts.size:
            all_pts.append(pts)

    if not all_pts:
        return 0.0

    all_pts = np.vstack(all_pts)
    x_min = int(np.floor(np.min(all_pts[:, 0]))) - 2
    y_min = int(np.floor(np.min(all_pts[:, 1]))) - 2
    x_max = int(np.ceil(np.max(all_pts[:, 0]))) + 2
    y_max = int(np.ceil(np.max(all_pts[:, 1]))) + 2

    width = max(1, x_max - x_min + 1)
    height = max(1, y_max - y_min + 1)

    # Prevent very large masks. Downscaling is acceptable for visualization/evaluation quads.
    max_pixels = 8_000_000
    scale = 1.0
    if width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        width = max(1, int(np.ceil(width * scale)))
        height = max(1, int(np.ceil(height * scale)))

    def to_mask_points(quad):
        pts = np.asarray(quad, dtype=np.float32).copy()
        pts[:, 0] = (pts[:, 0] - x_min) * scale
        pts[:, 1] = (pts[:, 1] - y_min) * scale
        return np.round(pts).astype(np.int32)

    gt_mask = np.zeros((height, width), dtype=np.uint8)
    pred_mask = np.zeros((height, width), dtype=np.uint8)

    for quad in gt_quads:
        cv2.fillPoly(gt_mask, [to_mask_points(quad)], 1)
    for quad in pred_quads:
        cv2.fillPoly(pred_mask, [to_mask_points(quad)], 1)

    inter = int(np.count_nonzero((gt_mask > 0) & (pred_mask > 0)))
    union = int(np.count_nonzero((gt_mask > 0) | (pred_mask > 0)))
    if union <= 0:
        return 0.0

    return float(inter / union)


def pred_inside_any_region(pred, regions) -> bool:
    return any(polygon_intersects(pred["quad"], region["quad"]) for region in regions)


# -----------------------------
# Matching
# -----------------------------


def connected_required_pred_components(required_gt_items, pred_items):
    """
    Build connected components in a bipartite graph:
    required GT nodes are connected to prediction nodes if their polygons intersect.
    """
    gt_to_preds: dict[int, set[int]] = defaultdict(set)
    pred_to_gts: dict[int, set[int]] = defaultdict(set)

    for gt_idx, gt in enumerate(required_gt_items):
        for pred_idx, pred in enumerate(pred_items):
            if polygon_intersects(gt["quad"], pred["quad"]):
                gt_to_preds[gt_idx].add(pred_idx)
                pred_to_gts[pred_idx].add(gt_idx)

    visited_gt: set[int] = set()
    visited_pred: set[int] = set()
    components = []

    for start_gt in range(len(required_gt_items)):
        if start_gt in visited_gt:
            continue

        queue = deque([("gt", start_gt)])
        comp_gts: set[int] = set()
        comp_preds: set[int] = set()

        while queue:
            kind, idx = queue.popleft()
            if kind == "gt":
                if idx in visited_gt:
                    continue
                visited_gt.add(idx)
                comp_gts.add(idx)
                for pred_idx in gt_to_preds.get(idx, set()):
                    if pred_idx not in visited_pred:
                        queue.append(("pred", pred_idx))
            else:
                if idx in visited_pred:
                    continue
                visited_pred.add(idx)
                comp_preds.add(idx)
                for gt_idx in pred_to_gts.get(idx, set()):
                    if gt_idx not in visited_gt:
                        queue.append(("gt", gt_idx))

        components.append((sorted(comp_gts), sorted(comp_preds)))

    # Prediction components with no required GT are not needed for TP/FN matching;
    # they are handled later as FP/neutral candidates.
    return components


def match_image_business(
    required_gt_items,
    ignored_gt_items,
    important_gt_items,
    pred_items,
    iou_threshold: float,
):
    """
    Single business IoU logic.

    TP/FN:
    - Required GT is matched if the IoU between union(required GT component)
      and union(predictions in that component) is >= iou_threshold.
    - This supports both many-pred-to-one-GT and one-pred-to-many-GT cases.

    FP:
    - Prediction already matched to required GT => not FP.
    - Prediction outside all important regions => neutral, not FP.
    - Prediction inside ignored GT => neutral, not FP.
    - Remaining prediction inside important regions => FP.
    """
    required_matched_by: dict[int, dict[str, Any]] = {}
    pred_matched_required: dict[int, list[dict[str, Any]]] = defaultdict(list)
    component_matches = []

    components = connected_required_pred_components(required_gt_items, pred_items)

    for comp_idx, (gt_indices, pred_indices) in enumerate(components):
        if not gt_indices or not pred_indices:
            continue

        gt_quads = [required_gt_items[i]["quad"] for i in gt_indices]
        pred_quads = [pred_items[i]["quad"] for i in pred_indices]
        score = union_iou_raster(gt_quads=gt_quads, pred_quads=pred_quads)

        component_info = {
            "component_idx": comp_idx,
            "gt_indices": gt_indices,
            "pred_indices": pred_indices,
            "iou": score,
            "matched": score >= iou_threshold,
        }
        component_matches.append(component_info)

        if score >= iou_threshold:
            for gt_idx in gt_indices:
                required_matched_by[gt_idx] = component_info
            for pred_idx in pred_indices:
                for gt_idx in gt_indices:
                    pred_matched_required[pred_idx].append(
                        {
                            "gt_idx": gt_idx,
                            "gt_labels": required_gt_items[gt_idx].get("labels", []),
                            "iou": score,
                            "component_idx": comp_idx,
                        }
                    )

    matched_gt = set(required_matched_by.keys())
    matched_pred = set(pred_matched_required.keys())

    fn_indices = [
        idx for idx in range(len(required_gt_items))
        if idx not in matched_gt
    ]

    pred_neutral_ignored: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pred_neutral_outside_important: dict[int, dict[str, Any]] = {}
    fp_indices = []

    for pred_idx, pred in enumerate(pred_items):
        if pred_idx in matched_pred:
            continue

        # important is an FP mask. Outside it, prediction is ignored for FP.
        if important_gt_items and not pred_inside_any_region(pred, important_gt_items):
            pred_neutral_outside_important[pred_idx] = {
                "reason": "outside_important_regions",
            }
            continue

        # If important markup is missing for the file, keep old behavior:
        # prediction can still become FP unless it overlaps ignored GT.
        for ignored_idx, ignored_gt in enumerate(ignored_gt_items):
            if polygon_intersects(pred["quad"], ignored_gt["quad"]):
                pred_neutral_ignored[pred_idx].append(
                    {
                        "ignored_gt_idx": ignored_idx,
                        "gt_labels": ignored_gt.get("labels", []),
                        "reason": "inside_ignored_gt",
                    }
                )

        if pred_idx in pred_neutral_ignored:
            continue

        fp_indices.append(pred_idx)

    tp = len(matched_gt)
    fn = len(fn_indices)
    fp = len(fp_indices)

    matches = []
    for gt_idx, comp in required_matched_by.items():
        matches.append(
            {
                "gt_idx": gt_idx,
                "gt_labels": required_gt_items[gt_idx].get("labels", []),
                "pred_indices": comp["pred_indices"],
                "iou": comp["iou"],
                "component_idx": comp["component_idx"],
            }
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
        "component_matches": component_matches,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "matched_pred_indices": sorted(matched_pred),
        "neutral_ignored_pred_indices": sorted(pred_neutral_ignored.keys()),
        "neutral_outside_important_pred_indices": sorted(pred_neutral_outside_important.keys()),
        "pred_matched_required": {str(k): v for k, v in pred_matched_required.items()},
    }


def safe_div(a, b):
    return a / b if b else 0.0


def update_required_label_stats(per_label, required_gt_items, matches, fn_indices):
    for gt in required_gt_items:
        labels = [label for label in gt.get("labels", []) if label in per_label]
        for label in labels:
            per_label[label]["gt"] += 1

    for match in matches:
        labels = [label for label in match.get("gt_labels", []) if label in per_label]
        for label in labels:
            per_label[label]["tp"] += 1

    for idx in fn_indices:
        if idx < 0 or idx >= len(required_gt_items):
            continue
        labels = [label for label in required_gt_items[idx].get("labels", []) if label in per_label]
        for label in labels:
            per_label[label]["fn"] += 1


def finalize_required_label_stats(per_label):
    result = {}
    for label, stats in sorted(per_label.items()):
        gt = stats["gt"]
        tp = stats["tp"]
        fn = stats["fn"]
        result[label] = {
            "gt": gt,
            "tp": tp,
            "fn": fn,
            "recall": safe_div(tp, tp + fn),
        }
    return result


def choose_files(valid_annotated_files, pred_by_file, eval_scope: str):
    gt_files = set(valid_annotated_files)
    pred_files = set(pred_by_file.keys())

    if eval_scope == "pred":
        # Focus on predictions, but do not evaluate files without valid LS annotations.
        return pred_files & gt_files
    if eval_scope == "gt":
        return gt_files
    if eval_scope == "intersection":
        return gt_files & pred_files
    if eval_scope == "all":
        return gt_files | pred_files
    raise ValueError(f"Unknown eval_scope: {eval_scope}")


def evaluate(
    required_gt_by_file,
    ignored_gt_by_file,
    important_gt_by_file,
    pred_by_file,
    valid_annotated_files: set[str],
    iou_threshold: float,
    required_labels: set[str],
    important_label: str,
    eval_scope: str,
):
    selected_files = choose_files(
        valid_annotated_files=valid_annotated_files,
        pred_by_file=pred_by_file,
        eval_scope=eval_scope,
    )
    all_files = sorted(selected_files)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_required_gt = 0
    total_ignored_gt = 0
    total_important_gt = 0
    total_pred = 0
    total_neutral_ignored_pred = 0
    total_neutral_outside_important_pred = 0

    per_label = {
        label: {"gt": 0, "tp": 0, "fn": 0}
        for label in sorted(required_labels)
    }

    per_file = []

    for filename in all_files:
        required_gt_items = required_gt_by_file.get(filename, [])
        ignored_gt_items = ignored_gt_by_file.get(filename, [])
        important_gt_items = important_gt_by_file.get(filename, [])
        pred_items = pred_by_file.get(filename, [])

        match = match_image_business(
            required_gt_items=required_gt_items,
            ignored_gt_items=ignored_gt_items,
            important_gt_items=important_gt_items,
            pred_items=pred_items,
            iou_threshold=iou_threshold,
        )

        tp = match["tp"]
        fp = match["fp"]
        fn = match["fn"]

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_required_gt += len(required_gt_items)
        total_ignored_gt += len(ignored_gt_items)
        total_important_gt += len(important_gt_items)
        total_pred += len(pred_items)
        total_neutral_ignored_pred += len(match["neutral_ignored_pred_indices"])
        total_neutral_outside_important_pred += len(match["neutral_outside_important_pred_indices"])

        update_required_label_stats(
            per_label=per_label,
            required_gt_items=required_gt_items,
            matches=match["matches"],
            fn_indices=match["fn_indices"],
        )

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_file.append(
            {
                "file": filename,
                "required_gt": len(required_gt_items),
                "ignored_gt": len(ignored_gt_items),
                "important_regions": len(important_gt_items),
                "pred": len(pred_items),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "neutral_ignored_pred": len(match["neutral_ignored_pred_indices"]),
                "neutral_outside_important_pred": len(match["neutral_outside_important_pred_indices"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matches": match["matches"],
                "component_matches": match["component_matches"],
                "fp_indices": match["fp_indices"],
                "fn_indices": match["fn_indices"],
                "matched_pred_indices": match["matched_pred_indices"],
                "neutral_ignored_pred_indices": match["neutral_ignored_pred_indices"],
                "neutral_outside_important_pred_indices": match["neutral_outside_important_pred_indices"],
                "has_important_regions": bool(important_gt_items),
            }
        )

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    valid_files = set(valid_annotated_files)
    pred_files = set(pred_by_file.keys())
    pred_without_valid_ls = sorted(pred_files - valid_files)
    ls_without_pred = sorted(valid_files - pred_files)

    important_files = {f for f, items in important_gt_by_file.items() if items}
    selected_without_important = sorted(set(all_files) - important_files)

    summary = {
        "policy": "required_labels_with_ignored_gt_and_important_fp_mask",
        "required_labels": sorted(required_labels),
        "important_label": important_label,
        "eval_scope": eval_scope,
        "match_metric": "iou",
        "match_logic": "component_union_iou",
        "iou_threshold": iou_threshold,
        "evaluated_files_count": len(all_files),
        "evaluated_files": all_files,
        "valid_label_studio_files_count": len(valid_files),
        "prediction_files_count": len(pred_files),
        "prediction_files_without_valid_label_studio_count": len(pred_without_valid_ls),
        "prediction_files_without_valid_label_studio": pred_without_valid_ls,
        "label_studio_files_without_predictions_count": len(ls_without_pred),
        "label_studio_files_without_predictions": ls_without_pred,
        "evaluated_files_without_important_regions_count": len(selected_without_important),
        "evaluated_files_without_important_regions": selected_without_important,
        "required_gt": total_required_gt,
        "ignored_gt": total_ignored_gt,
        "important_regions": total_important_gt,
        "pred": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "neutral_ignored_pred": total_neutral_ignored_pred,
        "neutral_outside_important_pred": total_neutral_outside_important_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_required_label": finalize_required_label_stats(per_label),
    }

    return summary, per_file


# -----------------------------
# Visualization
# -----------------------------


def draw_quads(draw: ImageDraw.ImageDraw, items, color: str, width: int):
    for item in items:
        quad = item["quad"]
        pts = [(int(round(x)), int(round(y))) for x, y in quad]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color, width=width)


def draw_indexed_quads(draw: ImageDraw.ImageDraw, items, indices, color: str, width: int):
    for idx in indices:
        if idx < 0 or idx >= len(items):
            continue
        quad = items[idx]["quad"]
        pts = [(int(round(x)), int(round(y))) for x, y in quad]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color, width=width)


def should_visualize_file(file_metrics: dict, vis_filter: str) -> bool:
    if vis_filter == "all":
        return True
    if vis_filter == "errors":
        return file_metrics["fp"] > 0 or file_metrics["fn"] > 0
    if vis_filter == "fp":
        return file_metrics["fp"] > 0
    if vis_filter == "fn":
        return file_metrics["fn"] > 0
    raise ValueError(f"Unknown vis_filter: {vis_filter}")


def find_image_path(images_dir: Path, filename: str) -> Path | None:
    direct = images_dir / filename
    if direct.exists():
        return direct

    filename_lower = filename.lower()
    for path in images_dir.rglob("*"):
        if path.is_file() and path.name.lower() == filename_lower:
            return path

    stem = Path(filename).stem.lower()
    candidates = []
    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem.lower() == stem:
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    return None


def draw_visualizations(
    required_gt_by_file,
    ignored_gt_by_file,
    important_gt_by_file,
    pred_by_file,
    per_file,
    images_dir: Path,
    output_dir: Path,
    vis_mode: str,
    vis_filter: str,
    file_upload_by_file: dict[str, str] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_upload_by_file = file_upload_by_file or {}

    saved = 0
    skipped = 0

    for metrics in per_file:
        filename = metrics["file"]
        if not should_visualize_file(metrics, vis_filter):
            continue

        image_path = find_image_path(images_dir, filename)
        if image_path is None:
            print(f"[VIS SKIP] image not found: {images_dir / filename}")
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        required_gt_items = required_gt_by_file.get(filename, [])
        ignored_gt_items = ignored_gt_by_file.get(filename, [])
        important_gt_items = important_gt_by_file.get(filename, [])
        pred_items = pred_by_file.get(filename, [])

        if vis_mode == "gt":
            # Important regions: cyan. Ignored GT: yellow. Required GT: green.
            draw_quads(draw, important_gt_items, color="cyan", width=2)
            draw_quads(draw, ignored_gt_items, color="yellow", width=2)
            draw_quads(draw, required_gt_items, color="lime", width=3)

        elif vis_mode == "gt_pred":
            # Important regions: cyan. Ignored GT: yellow. Required GT: green. Predictions: red.
            draw_quads(draw, important_gt_items, color="cyan", width=2)
            draw_quads(draw, ignored_gt_items, color="yellow", width=2)
            draw_quads(draw, required_gt_items, color="lime", width=3)
            draw_quads(draw, pred_items, color="red", width=3)

        elif vis_mode == "errors":
            # Only errors that affect statistics:
            # FP pred inside important regions: red.
            # FN required GT: blue.
            draw_indexed_quads(draw, pred_items, metrics["fp_indices"], color="red", width=4)
            draw_indexed_quads(draw, required_gt_items, metrics["fn_indices"], color="blue", width=4)

        else:
            raise ValueError(f"Unknown vis_mode: {vis_mode}")

        out_filename = make_vis_output_filename(filename, file_upload_by_file)
        image.save(output_dir / out_filename)
        saved += 1

    print()
    print(f"[VIS] saved={saved} skipped={skipped}")
    print(f"[VIS] mode={vis_mode} filter={vis_filter}")
    print(f"[VIS] output={output_dir}")


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Business detection metrics for Label Studio rectanglelabels and quad predictions. "
            "Only required labels create TP/FN. The 'important' label is an FP mask."
        )
    )

    parser.add_argument("--gt", required=True, help="Label Studio JSON")
    parser.add_argument("--pred", required=True, help="Predictions JSON: filename/path -> list of quads/boxes")
    parser.add_argument("--output", default="detection_metrics.json", help="Output metrics JSON")

    parser.add_argument(
        "--required-labels",
        nargs="*",
        default=sorted(DEFAULT_REQUIRED_LABELS),
        help="Labels that are mandatory for the business metric.",
    )
    parser.add_argument(
        "--important-label",
        default=DEFAULT_IMPORTANT_LABEL,
        help="Label Studio rectanglelabel used as FP mask. Predictions outside these regions are not FP.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional legacy filter: load only these Label Studio labels before required/ignored/important split.",
    )

    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for component union matching")

    parser.add_argument(
        "--eval-scope",
        choices=["pred", "gt", "intersection", "all"],
        default="pred",
        help=(
            "Which files to include in metrics: "
            "pred = prediction files with valid Label Studio annotations; "
            "gt = all valid Label Studio files; "
            "intersection = files present in both valid Label Studio and predictions; "
            "all = union of valid Label Studio files and prediction files."
        ),
    )

    parser.add_argument("--images-dir", default=None, help="Folder with source images")
    parser.add_argument("--vis-output-dir", default=None, help="Folder for visualizations")
    parser.add_argument(
        "--vis-mode",
        choices=["gt", "gt_pred", "errors"],
        default="gt_pred",
        help=(
            "gt = important cyan + ignored yellow + required green; "
            "gt_pred = GT + predictions red; "
            "errors = only FP red + FN blue"
        ),
    )
    parser.add_argument(
        "--vis-filter",
        choices=["all", "errors", "fp", "fn"],
        default="all",
        help="Which visualizations to save",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    required_labels = set(args.required_labels or [])
    labels_filter = set(args.labels) if args.labels else None

    gt_data = load_gt_label_studio(
        path=Path(args.gt),
        required_labels=required_labels,
        important_label=args.important_label,
        labels_filter=labels_filter,
    )
    pred_by_file = load_predictions(Path(args.pred))

    summary, per_file = evaluate(
        required_gt_by_file=gt_data["required_gt_by_file"],
        ignored_gt_by_file=gt_data["ignored_gt_by_file"],
        important_gt_by_file=gt_data["important_gt_by_file"],
        pred_by_file=pred_by_file,
        valid_annotated_files=gt_data["valid_annotated_files"],
        iou_threshold=args.iou,
        required_labels=required_labels,
        important_label=args.important_label,
        eval_scope=args.eval_scope,
    )
    summary["label_studio"] = gt_data["label_studio_stats"]

    result = {
        "summary": summary,
        "per_file": per_file,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if args.images_dir and args.vis_output_dir:
        draw_visualizations(
            required_gt_by_file=gt_data["required_gt_by_file"],
            ignored_gt_by_file=gt_data["ignored_gt_by_file"],
            important_gt_by_file=gt_data["important_gt_by_file"],
            pred_by_file=pred_by_file,
            per_file=per_file,
            images_dir=Path(args.images_dir),
            output_dir=Path(args.vis_output_dir),
            vis_mode=args.vis_mode,
            vis_filter=args.vis_filter,
            file_upload_by_file=gt_data["file_upload_by_file"],
        )

    print()
    print("Detection metrics")
    print("=================")
    print(f"Policy:        {summary['policy']}")
    print(f"Eval scope:    {summary['eval_scope']}")
    print(f"Match metric:  {summary['match_metric']}")
    print(f"Match logic:   {summary['match_logic']}")
    print(f"IoU threshold: {summary['iou_threshold']}")
    print(f"Files:         {summary['evaluated_files_count']}")
    print(f"Required GT:   {summary['required_gt']}")
    print(f"Ignored GT:    {summary['ignored_gt']}")
    print(f"Important:     {summary['important_regions']}")
    print(f"Pred:          {summary['pred']}")
    print(f"TP:            {summary['tp']}")
    print(f"FP:            {summary['fp']}")
    print(f"FN:            {summary['fn']}")
    print(f"Neutral ignored pred:           {summary['neutral_ignored_pred']}")
    print(f"Neutral outside important pred: {summary['neutral_outside_important_pred']}")
    print(f"Precision:     {summary['precision']:.4f}")
    print(f"Recall:        {summary['recall']:.4f}")
    print(f"F1:            {summary['f1']:.4f}")
    print()
    print(f"Valid LS annotations used:        {summary['label_studio']['annotations_used_was_cancelled_false']}")
    print(f"Cancelled LS annotations skipped: {summary['label_studio']['annotations_cancelled']}")
    print(f"Evaluated files without important regions: {summary['evaluated_files_without_important_regions_count']}")
    print(f"Prediction files without valid LS:         {summary['prediction_files_without_valid_label_studio_count']}")
    print()
    print(f"Saved details: {args.output}")


if __name__ == "__main__":
    main()
