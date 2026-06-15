#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union as shapely_unary_union
except Exception:  # pragma: no cover
    ShapelyPolygon = None
    shapely_unary_union = None


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
    Convert Label Studio rectangle result to image-space quad.

    Label Studio stores x/y/width/height in percentages and rotation in degrees.
    The rotation is applied around the rectangle top-left anchor.
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


def item_is_required(labels: list[str], required_labels: set[str]) -> bool:
    return any(label in required_labels for label in labels)


def load_gt_label_studio(
    path: Path,
    required_labels: set[str],
    labels_filter: set[str] | None = None,
):
    """
    Load only Label Studio annotations with was_cancelled == false.

    Required labels create TP/FN. All other rectanglelabels are ignored regions:
    they do not create TP/FN, but predictions overlapping them are not counted as FP.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_gt_by_file: dict[str, list[dict[str, Any]]] = {}
    ignored_gt_by_file: dict[str, list[dict[str, Any]]] = {}
    all_gt_by_file: dict[str, list[dict[str, Any]]] = {}
    file_upload_by_file: dict[str, str] = {}
    valid_annotated_files: set[str] = set()

    tasks_total = 0
    annotations_total = 0
    annotations_cancelled = 0
    annotations_used = 0
    rectangle_results_used = 0
    rectangle_results_filtered_by_label = 0

    for task in data:
        tasks_total += 1
        image_path = task.get("data", {}).get("image")
        if not image_path:
            continue

        filename = basename_from_ls_image_path(image_path)
        required_gt_by_file.setdefault(filename, [])
        ignored_gt_by_file.setdefault(filename, [])
        all_gt_by_file.setdefault(filename, [])

        if task.get("file_upload"):
            file_upload_by_file[filename] = task["file_upload"]

        for ann in task.get("annotations", []):
            annotations_total += 1
            if ann.get("was_cancelled", False):
                annotations_cancelled += 1
                continue

            annotations_used += 1
            valid_annotated_files.add(filename)

            for res in ann.get("result", []):
                if res.get("type") != "rectanglelabels":
                    continue

                value = res.get("value", {})
                rect_labels = list(value.get("rectanglelabels", []))

                if labels_filter is not None and not any(label in labels_filter for label in rect_labels):
                    rectangle_results_filtered_by_label += 1
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

                gt_item = {
                    "quad": quad,
                    "labels": rect_labels,
                    "is_required": item_is_required(rect_labels, required_labels),
                    "rotation": float(value.get("rotation", 0.0) or 0.0),
                    "ls_result_id": res.get("id"),
                }

                all_gt_by_file[filename].append(gt_item)
                if gt_item["is_required"]:
                    required_gt_by_file[filename].append(gt_item)
                else:
                    ignored_gt_by_file[filename].append(gt_item)

                rectangle_results_used += 1

    stats = {
        "tasks_total": tasks_total,
        "annotations_total": annotations_total,
        "annotations_cancelled": annotations_cancelled,
        "annotations_used_was_cancelled_false": annotations_used,
        "rectangle_results_used": rectangle_results_used,
        "rectangle_results_filtered_by_label": rectangle_results_filtered_by_label,
        "valid_annotated_files_count": len(valid_annotated_files),
    }

    return {
        "required_gt_by_file": required_gt_by_file,
        "ignored_gt_by_file": ignored_gt_by_file,
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

    if polygon_area(a) <= EPS or polygon_area(b) <= EPS:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(a, b)
    return float(max(inter_area, 0.0))


def polygons_intersect(quad_a, quad_b) -> bool:
    return polygon_intersection_area(quad_a, quad_b) > EPS


def _shapely_polygon(quad):
    if ShapelyPolygon is None:
        return None
    try:
        poly = ShapelyPolygon([(float(x), float(y)) for x, y in quad])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= EPS:
            return None
        return poly
    except Exception:
        return None


def _union_iou_shapely(quads_a, quads_b) -> float | None:
    if ShapelyPolygon is None or shapely_unary_union is None:
        return None

    polys_a = [_shapely_polygon(q) for q in quads_a]
    polys_b = [_shapely_polygon(q) for q in quads_b]
    polys_a = [p for p in polys_a if p is not None]
    polys_b = [p for p in polys_b if p is not None]

    if not polys_a or not polys_b:
        return 0.0

    union_a = shapely_unary_union(polys_a)
    union_b = shapely_unary_union(polys_b)

    if union_a.is_empty or union_b.is_empty or union_a.area <= EPS or union_b.area <= EPS:
        return 0.0

    inter = union_a.intersection(union_b).area
    union = union_a.union(union_b).area
    if union <= EPS:
        return 0.0

    return float(inter / union)


def _union_iou_raster(quads_a, quads_b) -> float:
    all_quads = list(quads_a) + list(quads_b)
    if not all_quads:
        return 0.0

    pts = np.asarray([pt for quad in all_quads for pt in quad], dtype=np.float32)
    x_min = int(np.floor(np.min(pts[:, 0]))) - 2
    y_min = int(np.floor(np.min(pts[:, 1]))) - 2
    x_max = int(np.ceil(np.max(pts[:, 0]))) + 2
    y_max = int(np.ceil(np.max(pts[:, 1]))) + 2

    width = max(1, x_max - x_min + 1)
    height = max(1, y_max - y_min + 1)

    max_pixels = 8_000_000
    scale = 1.0
    if width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        width = max(1, int(np.ceil(width * scale)))
        height = max(1, int(np.ceil(height * scale)))

    def to_mask_points(quad):
        arr = np.asarray(quad, dtype=np.float32).copy()
        arr[:, 0] = (arr[:, 0] - x_min) * scale
        arr[:, 1] = (arr[:, 1] - y_min) * scale
        return np.round(arr).astype(np.int32)

    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)

    for quad in quads_a:
        cv2.fillPoly(mask_a, [to_mask_points(quad)], 1)
    for quad in quads_b:
        cv2.fillPoly(mask_b, [to_mask_points(quad)], 1)

    inter = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    union = int(np.count_nonzero((mask_a > 0) | (mask_b > 0)))

    return float(inter / union) if union else 0.0


def union_iou(quads_a, quads_b) -> float:
    """
    IoU between unions of two polygon sets.

    This is the only matching score used by the script. It supports:
    - one prediction covering several GT boxes;
    - several predictions jointly covering one GT box;
    - ordinary one-to-one matches.
    """
    exact = _union_iou_shapely(quads_a, quads_b)
    if exact is not None:
        return exact
    return _union_iou_raster(quads_a, quads_b)


# -----------------------------
# Business IoU matching
# -----------------------------


def build_required_components(required_gt_items, pred_items):
    """
    Build connected components over required GT and predictions.

    An edge exists when a required GT quad and a prediction quad have positive intersection.
    Each component is then evaluated by IoU(union(required_gt), union(pred)).
    """
    n_gt = len(required_gt_items)
    n_pred = len(pred_items)
    total = n_gt + n_pred

    adjacency = [[] for _ in range(total)]

    for gt_idx, gt in enumerate(required_gt_items):
        for pred_idx, pred in enumerate(pred_items):
            if polygons_intersect(gt["quad"], pred["quad"]):
                a = gt_idx
                b = n_gt + pred_idx
                adjacency[a].append(b)
                adjacency[b].append(a)

    seen = [False] * total
    components = []

    for start in range(total):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        gt_indices = []
        pred_indices = []

        while stack:
            node = stack.pop()
            if node < n_gt:
                gt_indices.append(node)
            else:
                pred_indices.append(node - n_gt)

            for nxt in adjacency[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)

        components.append(
            {
                "gt_indices": sorted(gt_indices),
                "pred_indices": sorted(pred_indices),
            }
        )

    return components


def pred_overlaps_any_ignored(pred, ignored_gt_items) -> bool:
    return any(polygons_intersect(pred["quad"], gt["quad"]) for gt in ignored_gt_items)


def match_image_business_iou(required_gt_items, ignored_gt_items, pred_items, iou_threshold: float):
    """
    One matching logic, one score: component-level IoU.

    Business rules:
    - only required GT creates TP/FN;
    - ignored GT creates no TP/FN;
    - predictions that overlap ignored GT are neutral, not FP;
    - required GT and predictions are matched by IoU between their connected-component unions;
    - files/predictions outside valid Label Studio scope are controlled by eval-scope.
    """
    components = build_required_components(required_gt_items, pred_items)

    matched_required_gt = set()
    matched_required_pred = set()
    matches = []
    component_records = []

    for comp_idx, comp in enumerate(components):
        gt_indices = comp["gt_indices"]
        pred_indices = comp["pred_indices"]

        if not gt_indices or not pred_indices:
            score = 0.0
            passed = False
        else:
            gt_quads = [required_gt_items[i]["quad"] for i in gt_indices]
            pred_quads = [pred_items[i]["quad"] for i in pred_indices]
            score = union_iou(gt_quads, pred_quads)
            passed = score >= iou_threshold

        component_records.append(
            {
                "component_idx": comp_idx,
                "gt_indices": gt_indices,
                "pred_indices": pred_indices,
                "iou": score,
                "matched": passed,
            }
        )

        if not passed:
            continue

        for gt_idx in gt_indices:
            matched_required_gt.add(gt_idx)
            gt = required_gt_items[gt_idx]
            matches.append(
                {
                    "gt_idx": gt_idx,
                    "gt_labels": gt.get("labels", []),
                    "pred_indices": pred_indices,
                    "iou": score,
                    "component_idx": comp_idx,
                }
            )

        for pred_idx in pred_indices:
            matched_required_pred.add(pred_idx)

    fn_indices = [idx for idx in range(len(required_gt_items)) if idx not in matched_required_gt]

    neutral_ignored_pred_indices = []
    fp_indices = []

    for pred_idx, pred in enumerate(pred_items):
        if pred_idx in matched_required_pred:
            continue
        if pred_overlaps_any_ignored(pred, ignored_gt_items):
            neutral_ignored_pred_indices.append(pred_idx)
        else:
            fp_indices.append(pred_idx)

    tp = len(matched_required_gt)
    fp = len(fp_indices)
    fn = len(fn_indices)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
        "components": component_records,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "neutral_ignored_pred_indices": neutral_ignored_pred_indices,
        "pred_matched_required_indices": sorted(matched_required_pred),
    }


# -----------------------------
# Evaluation
# -----------------------------


def safe_div(a, b):
    return a / b if b else 0.0


def update_required_label_stats(per_label: dict[str, dict[str, int]], required_gt_items, fn_indices, required_labels: set[str]):
    fn_set = set(fn_indices)
    for gt_idx, gt in enumerate(required_gt_items):
        required_labels_on_item = [label for label in gt.get("labels", []) if label in required_labels]
        for label in required_labels_on_item:
            per_label[label]["gt"] += 1
            if gt_idx in fn_set:
                per_label[label]["fn"] += 1
            else:
                per_label[label]["tp"] += 1


def finalize_required_label_stats(per_label: dict[str, dict[str, int]]):
    result = {}
    for label in sorted(per_label.keys()):
        item = dict(per_label[label])
        item["recall"] = safe_div(item["tp"], item["tp"] + item["fn"])
        result[label] = item
    return result


def select_eval_files(valid_annotated_files: set[str], prediction_files: set[str], eval_scope: str):
    if eval_scope == "pred":
        # Prediction-focused, but still only inside valid Label Studio scope.
        return sorted(valid_annotated_files & prediction_files)
    if eval_scope == "gt":
        return sorted(valid_annotated_files)
    raise ValueError(f"Unknown eval_scope: {eval_scope}")


def evaluate(
    required_gt_by_file,
    ignored_gt_by_file,
    pred_by_file,
    valid_annotated_files: set[str],
    iou_threshold: float,
    required_labels: set[str],
    eval_scope: str,
):
    prediction_files = set(pred_by_file.keys())
    all_files = select_eval_files(
        valid_annotated_files=valid_annotated_files,
        prediction_files=prediction_files,
        eval_scope=eval_scope,
    )

    pred_only_files = sorted(prediction_files - valid_annotated_files)
    gt_only_files = sorted(valid_annotated_files - prediction_files)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_required_gt = 0
    total_ignored_gt = 0
    total_pred = 0
    total_neutral_ignored_pred = 0

    per_label = defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0})
    per_file = []

    for filename in all_files:
        required_gt_items = required_gt_by_file.get(filename, [])
        ignored_gt_items = ignored_gt_by_file.get(filename, [])
        pred_items = pred_by_file.get(filename, [])

        match_result = match_image_business_iou(
            required_gt_items=required_gt_items,
            ignored_gt_items=ignored_gt_items,
            pred_items=pred_items,
            iou_threshold=iou_threshold,
        )

        tp = match_result["tp"]
        fp = match_result["fp"]
        fn = match_result["fn"]

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_required_gt += len(required_gt_items)
        total_ignored_gt += len(ignored_gt_items)
        total_pred += len(pred_items)
        total_neutral_ignored_pred += len(match_result["neutral_ignored_pred_indices"])

        update_required_label_stats(
            per_label=per_label,
            required_gt_items=required_gt_items,
            fn_indices=match_result["fn_indices"],
            required_labels=required_labels,
        )

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_file.append(
            {
                "file": filename,
                "required_gt": len(required_gt_items),
                "ignored_gt": len(ignored_gt_items),
                "pred": len(pred_items),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "neutral_ignored_pred": len(match_result["neutral_ignored_pred_indices"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matches": match_result["matches"],
                "components": match_result["components"],
                "fp_indices": match_result["fp_indices"],
                "fn_indices": match_result["fn_indices"],
                "neutral_ignored_pred_indices": match_result["neutral_ignored_pred_indices"],
                "pred_matched_required_indices": match_result["pred_matched_required_indices"],
                "has_errors": fp > 0 or fn > 0,
            }
        )

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    summary = {
        "policy": "business_required_labels_with_ignored_gt",
        "eval_scope": eval_scope,
        "match_metric": "iou",
        "match_logic": "component_union_iou",
        "iou_threshold": iou_threshold,
        "required_labels": sorted(required_labels),
        "evaluated_files_count": len(all_files),
        "valid_label_studio_files_count": len(valid_annotated_files),
        "prediction_files_count": len(pred_by_file),
        "prediction_only_files_count": len(pred_only_files),
        "prediction_only_files_ignored_count": len(pred_only_files),
        "prediction_only_files_ignored": pred_only_files,
        "label_studio_files_without_predictions_count": len(gt_only_files),
        "label_studio_files_without_predictions_evaluated_count": len(gt_only_files if eval_scope == "gt" else []),
        "label_studio_files_without_predictions_ignored_count": len([] if eval_scope == "gt" else gt_only_files),
        "label_studio_files_without_predictions": gt_only_files,
        "required_gt": total_required_gt,
        "ignored_gt": total_ignored_gt,
        "pred": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "neutral_ignored_pred": total_neutral_ignored_pred,
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

    metrics_by_file = {item["file"]: item for item in per_file}

    saved = 0
    skipped = 0

    for filename, metrics in metrics_by_file.items():
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
        pred_items = pred_by_file.get(filename, [])

        if vis_mode == "gt":
            # Required GT: green. Ignored GT: yellow.
            draw_quads(draw, ignored_gt_items, color="yellow", width=2)
            draw_quads(draw, required_gt_items, color="lime", width=3)

        elif vis_mode == "gt_pred":
            # Required GT: green. Ignored GT: yellow. Predictions: red.
            draw_quads(draw, ignored_gt_items, color="yellow", width=2)
            draw_quads(draw, required_gt_items, color="lime", width=3)
            draw_quads(draw, pred_items, color="red", width=3)

        elif vis_mode == "errors":
            # FP pred: red. FN required GT: blue.
            draw_indexed_quads(draw, pred_items, metrics["fp_indices"], color="red", width=4)
            draw_indexed_quads(draw, required_gt_items, metrics["fn_indices"], color="blue", width=4)

        else:
            raise ValueError(f"Unknown vis_mode: {vis_mode}")

        out_filename = make_vis_output_filename(filename, file_upload_by_file)
        out_path = output_dir / out_filename
        image.save(out_path)
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
            "Business IoU detector evaluator for Label Studio rectanglelabels and quad predictions. "
            "Only required labels create TP/FN; other valid labels are ignore-regions."
        )
    )

    parser.add_argument("--gt", required=True, help="Label Studio JSON")
    parser.add_argument("--pred", required=True, help="Predictions JSON: filename/path -> list of quads/boxes")
    parser.add_argument("--output", default="detection_metrics.json", help="Output metrics JSON")

    parser.add_argument(
        "--required-labels",
        nargs="*",
        default=sorted(DEFAULT_REQUIRED_LABELS),
        help="Labels mandatory for the business metric.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional legacy filter: load only these Label Studio labels before required/ignored split.",
    )
    parser.add_argument(
        "--eval-scope",
        choices=["pred", "gt"],
        default="pred",
        help=(
            "pred = evaluate only files that are both in predictions and valid non-cancelled Label Studio annotations; "
            "gt = evaluate all valid non-cancelled Label Studio files, missing predictions become FN."
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help=(
            "IoU threshold for component-level union IoU. "
            "Several small predictions inside one GT and one prediction over several GT are evaluated as unions."
        ),
    )

    parser.add_argument("--images-dir", default=None, help="Folder with source images")
    parser.add_argument("--vis-output-dir", default=None, help="Folder for visualizations")
    parser.add_argument(
        "--vis-mode",
        choices=["gt", "gt_pred", "errors"],
        default="gt_pred",
        help=(
            "gt = required GT green + ignored GT yellow; "
            "gt_pred = GT + predictions red; "
            "errors = FP red + FN required GT blue"
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
        labels_filter=labels_filter,
    )
    pred_by_file = load_predictions(Path(args.pred))

    summary, per_file = evaluate(
        required_gt_by_file=gt_data["required_gt_by_file"],
        ignored_gt_by_file=gt_data["ignored_gt_by_file"],
        pred_by_file=pred_by_file,
        valid_annotated_files=gt_data["valid_annotated_files"],
        iou_threshold=args.iou,
        required_labels=required_labels,
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
    print(f"Pred:          {summary['pred']}")
    print(f"TP:            {summary['tp']}")
    print(f"FP:            {summary['fp']}")
    print(f"FN:            {summary['fn']}")
    print(f"Neutral pred:  {summary['neutral_ignored_pred']}")
    print(f"Precision:     {summary['precision']:.4f}")
    print(f"Recall:        {summary['recall']:.4f}")
    print(f"F1:            {summary['f1']:.4f}")
    print()
    print(f"Valid LS annotations used:        {summary['label_studio']['annotations_used_was_cancelled_false']}")
    print(f"Cancelled LS annotations skipped: {summary['label_studio']['annotations_cancelled']}")
    print(f"Prediction-only files ignored:    {summary['prediction_only_files_ignored_count']}")
    print(f"LS files without predictions:     {summary['label_studio_files_without_predictions_count']}")
    print(f"LS-only files ignored by scope:   {summary['label_studio_files_without_predictions_ignored_count']}")
    print()
    print(f"Saved details: {args.output}")


if __name__ == "__main__":
    main()
