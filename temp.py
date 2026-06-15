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
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def basename_from_ls_image_path(path: str) -> str:
    """
    Label Studio may store image path as '/data/upload/...' or with '?d=...'.
    This function returns the final file name used to match predictions/images.
    """
    return Path(str(path).split("?d=")[-1]).name


def rect_percent_to_quad(value: dict[str, Any], original_width: int, original_height: int):
    """
    Convert Label Studio rectangle result to image-space quad.

    Label Studio stores x/y/width/height in percentages and rotation in degrees.
    For rectanglelabels, rotation is applied around the rectangle top-left anchor.
    """
    x0 = value["x"] / 100.0 * original_width
    y0 = value["y"] / 100.0 * original_height
    w = value["width"] / 100.0 * original_width
    h = value["height"] / 100.0 * original_height
    rotation = float(value.get("rotation", 0.0) or 0.0)

    theta = np.deg2rad(rotation)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    rel_points = [
        (0.0, 0.0),
        (w, 0.0),
        (w, h),
        (0.0, h),
    ]

    quad = []
    for dx, dy in rel_points:
        # Image coordinate system: X grows right, Y grows down.
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
    Loads only Label Studio annotations with was_cancelled == false.

    Output:
    - required_gt_by_file: GT items whose rectanglelabels contain one of required_labels.
    - ignored_gt_by_file: all other valid rectanglelabels; they are ignore-regions.
    - all_gt_by_file: required + ignored.
    - valid_annotated_files: files that have at least one non-cancelled annotation object.
      These are the files evaluated by default. Prediction-only files are ignored.
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


def normalize_pred_item(item):
    """
    Supported prediction formats:
    - [[x,y], [x,y], [x,y], [x,y]]
    - [x1,y1,x2,y2]
    - {"quad": ...}
    - {"bbox": ...}
    - {"box": ...}

    Prediction labels are not required. The evaluator treats predictions as PII regions.
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
        quad = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]
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


def polygon_area(quad):
    pts = np.asarray(quad, dtype=np.float32)
    return abs(cv2.contourArea(pts))


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


def polygon_intersection_over_gt(pred_quad, gt_quad) -> float:
    """
    Coverage-style metric for cases where one prediction may contain several GT boxes.
    score = intersection_area / gt_area.
    """
    gt_area = polygon_area(gt_quad)
    pred_area = polygon_area(pred_quad)

    if gt_area <= 0 or pred_area <= 0:
        return 0.0

    inter_area = polygon_intersection_area(pred_quad, gt_quad)
    if inter_area <= 0:
        return 0.0

    return float(inter_area / gt_area)


def polygon_match_score(pred_quad, gt_quad, match_metric: str) -> float:
    if match_metric == "iou":
        return polygon_iou(pred_quad, gt_quad)
    if match_metric == "intersection":
        return polygon_intersection_over_gt(pred_quad, gt_quad)
    raise ValueError(f"Unknown match_metric: {match_metric}")


def match_image_business(
    required_gt_items,
    ignored_gt_items,
    pred_items,
    threshold: float,
    match_metric: str,
):
    """
    Business matching rules:
    - required GT matched by at least one pred => TP.
    - required GT with no pred => FN.
    - ignored GT does not create TP/FN.
    - pred matched to ignored GT only => neutral, not FP.
    - pred not matched to any required or ignored GT => FP.
    - one pred may match multiple GT boxes.
    """
    required_matched_by: dict[int, list[dict[str, Any]]] = {}
    pred_matched_required: dict[int, list[dict[str, Any]]] = {}
    pred_matched_ignored: dict[int, list[dict[str, Any]]] = {}

    for gt_idx, gt in enumerate(required_gt_items):
        for pred_idx, pred in enumerate(pred_items):
            score = polygon_match_score(
                pred_quad=pred["quad"],
                gt_quad=gt["quad"],
                match_metric=match_metric,
            )
            if score >= threshold:
                required_matched_by.setdefault(gt_idx, []).append(
                    {
                        "pred_idx": pred_idx,
                        "score": score,
                        match_metric: score,
                    }
                )
                pred_matched_required.setdefault(pred_idx, []).append(
                    {
                        "gt_idx": gt_idx,
                        "gt_labels": gt.get("labels", []),
                        "score": score,
                        match_metric: score,
                    }
                )

    for gt_idx, gt in enumerate(ignored_gt_items):
        for pred_idx, pred in enumerate(pred_items):
            score = polygon_match_score(
                pred_quad=pred["quad"],
                gt_quad=gt["quad"],
                match_metric=match_metric,
            )
            if score >= threshold:
                pred_matched_ignored.setdefault(pred_idx, []).append(
                    {
                        "ignored_gt_idx": gt_idx,
                        "gt_labels": gt.get("labels", []),
                        "score": score,
                        match_metric: score,
                    }
                )

    matched_required_gt = set(required_matched_by.keys())
    matched_required_pred = set(pred_matched_required.keys())
    matched_ignored_pred = set(pred_matched_ignored.keys())
    matched_any_pred = matched_required_pred | matched_ignored_pred

    fn_indices = [
        idx for idx in range(len(required_gt_items))
        if idx not in matched_required_gt
    ]
    fp_indices = [
        idx for idx in range(len(pred_items))
        if idx not in matched_any_pred
    ]
    neutral_ignored_pred_indices = sorted(matched_ignored_pred - matched_required_pred)

    matches = []
    for gt_idx, pred_matches in required_matched_by.items():
        best = max(pred_matches, key=lambda x: x["score"])
        gt = required_gt_items[gt_idx]
        matches.append(
            {
                "gt_idx": gt_idx,
                "gt_labels": gt.get("labels", []),
                "pred_idx": best["pred_idx"],
                "score": best["score"],
                match_metric: best["score"],
                "all_pred_matches": pred_matches,
            }
        )

    tp = len(matched_required_gt)
    fp = len(fp_indices)
    fn = len(fn_indices)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "neutral_ignored_pred_indices": neutral_ignored_pred_indices,
        "pred_matched_required_indices": sorted(matched_required_pred),
        "pred_matched_ignored_indices": sorted(matched_ignored_pred),
    }


def safe_div(a, b):
    return a / b if b else 0.0


def update_required_label_stats(
    per_label: dict[str, dict[str, int]],
    required_gt_items,
    fn_indices,
    required_labels: set[str],
):
    fn_set = set(fn_indices)
    for gt_idx, gt in enumerate(required_gt_items):
        labels = gt.get("labels", [])
        required_labels_on_item = [label for label in labels if label in required_labels]

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


def select_eval_files(
    valid_annotated_files: set[str],
    pred_files: set[str],
    eval_scope: str,
):
    """
    Select the file universe for metrics.

    gt:           valid Label Studio files only. Missing predictions create FN.
    pred:         prediction files only. Label Studio files without predictions are ignored.
    intersection: files present both in valid Label Studio annotations and predictions.
    all:          union of valid Label Studio files and prediction files.
    """
    if eval_scope == "gt":
        return sorted(valid_annotated_files)
    if eval_scope == "pred":
        return sorted(pred_files)
    if eval_scope == "intersection":
        return sorted(valid_annotated_files & pred_files)
    if eval_scope == "all":
        return sorted(valid_annotated_files | pred_files)
    raise ValueError(f"Unknown eval_scope: {eval_scope}")


def evaluate(
    required_gt_by_file,
    ignored_gt_by_file,
    pred_by_file,
    valid_annotated_files: set[str],
    threshold: float,
    match_metric: str,
    required_labels: set[str],
    eval_scope: str = "gt",
):
    pred_files = set(pred_by_file.keys())
    all_files = select_eval_files(
        valid_annotated_files=valid_annotated_files,
        pred_files=pred_files,
        eval_scope=eval_scope,
    )
    selected_files = set(all_files)

    pred_only_files = sorted(pred_files - valid_annotated_files)
    gt_only_files = sorted(valid_annotated_files - pred_files)
    pred_only_files_evaluated = sorted(selected_files & set(pred_only_files))
    gt_only_files_evaluated = sorted(selected_files & set(gt_only_files))

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

        match_result = match_image_business(
            required_gt_items=required_gt_items,
            ignored_gt_items=ignored_gt_items,
            pred_items=pred_items,
            threshold=threshold,
            match_metric=match_metric,
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
            per_label,
            required_gt_items,
            match_result["fn_indices"],
            required_labels,
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
                "fp_indices": match_result["fp_indices"],
                "fn_indices": match_result["fn_indices"],
                "neutral_ignored_pred_indices": match_result["neutral_ignored_pred_indices"],
                "pred_matched_required_indices": match_result["pred_matched_required_indices"],
                "pred_matched_ignored_indices": match_result["pred_matched_ignored_indices"],
                "has_errors": fp > 0 or fn > 0,
            }
        )

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    summary = {
        "policy": "required_labels_with_ignored_other_gt",
        "eval_scope": eval_scope,
        "match_metric": match_metric,
        "threshold": threshold,
        "iou_threshold": threshold if match_metric == "iou" else None,
        "intersection_threshold": threshold if match_metric == "intersection" else None,
        "required_labels": sorted(required_labels),
        "evaluated_files_count": len(all_files),
        "valid_label_studio_files_count": len(valid_annotated_files),
        "prediction_files_count": len(pred_by_file),
        "prediction_only_files_total_count": len(pred_only_files),
        "prediction_only_files_evaluated_count": len(pred_only_files_evaluated),
        "prediction_only_files_evaluated": pred_only_files_evaluated,
        "prediction_only_files_ignored_count": len(set(pred_only_files) - selected_files),
        "prediction_only_files_ignored": sorted(set(pred_only_files) - selected_files),
        "label_studio_files_without_predictions_count": len(gt_only_files),
        "label_studio_files_without_predictions": gt_only_files,
        "label_studio_files_without_predictions_evaluated_count": len(gt_only_files_evaluated),
        "label_studio_files_without_predictions_evaluated": gt_only_files_evaluated,
        "label_studio_files_without_predictions_ignored_count": len(set(gt_only_files) - selected_files),
        "label_studio_files_without_predictions_ignored": sorted(set(gt_only_files) - selected_files),
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


def draw_quads(draw: ImageDraw.ImageDraw, items, color: str, width: int):
    for item in items:
        quad = item["quad"]
        pts = [(int(round(x)), int(round(y))) for x, y in quad]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color, width=width)


def draw_indexed_quads(
    draw: ImageDraw.ImageDraw,
    items,
    indices,
    color: str,
    width: int,
):
    for idx in indices:
        if idx < 0 or idx >= len(items):
            continue
        item = items[idx]
        quad = item["quad"]
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
            # Only business errors:
            # FP pred outside any valid GT: red.
            # FN required GT: blue.
            draw_indexed_quads(
                draw,
                pred_items,
                metrics["fp_indices"],
                color="red",
                width=4,
            )
            draw_indexed_quads(
                draw,
                required_gt_items,
                metrics["fn_indices"],
                color="blue",
                width=4,
            )

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detection metrics for Label Studio rectanglelabels and quad predictions. "
            "Business policy: only required labels create TP/FN; all other valid GT labels are ignore-regions."
        )
    )

    parser.add_argument("--gt", required=True, help="Label Studio JSON")
    parser.add_argument("--pred", required=True, help="Predictions JSON: filename/path -> list of quads/boxes")
    parser.add_argument("--output", default="detection_metrics.json", help="Output metrics JSON")

    parser.add_argument(
        "--required-labels",
        nargs="*",
        default=sorted(DEFAULT_REQUIRED_LABELS),
        help="Labels that are mandatory for the business metric. Defaults to patient-critical fields.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional legacy filter: load only these Label Studio labels before required/ignored split.",
    )

    parser.add_argument(
        "--match-metric",
        choices=["iou", "intersection"],
        default="iou",
        help="iou = polygon IoU; intersection = intersection_area / gt_area",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="Threshold for --match-metric iou")
    parser.add_argument(
        "--intersection-threshold",
        type=float,
        default=0.8,
        help="Threshold for --match-metric intersection",
    )

    parser.add_argument(
        "--eval-scope",
        choices=["gt", "pred", "intersection", "all"],
        default="gt",
        help=(
            "Which files to include in metrics: "
            "gt = valid Label Studio files only; "
            "pred = prediction files only; "
            "intersection = only files present in both valid Label Studio and predictions; "
            "all = union of valid Label Studio files and prediction files."
        ),
    )
    parser.add_argument(
        "--include-pred-only-files",
        action="store_true",
        help=(
            "Deprecated compatibility flag. If set together with default --eval-scope gt, "
            "it behaves like --eval-scope all."
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

    threshold = args.iou if args.match_metric == "iou" else args.intersection_threshold

    gt_data = load_gt_label_studio(
        path=Path(args.gt),
        required_labels=required_labels,
        labels_filter=labels_filter,
    )
    pred_by_file = load_predictions(Path(args.pred))

    eval_scope = args.eval_scope
    if args.include_pred_only_files and eval_scope == "gt":
        eval_scope = "all"

    summary, per_file = evaluate(
        required_gt_by_file=gt_data["required_gt_by_file"],
        ignored_gt_by_file=gt_data["ignored_gt_by_file"],
        pred_by_file=pred_by_file,
        valid_annotated_files=gt_data["valid_annotated_files"],
        threshold=threshold,
        match_metric=args.match_metric,
        required_labels=required_labels,
        eval_scope=eval_scope,
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
    print(f"Threshold:     {summary['threshold']}")
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
    print(f"Valid LS annotations used:      {summary['label_studio']['annotations_used_was_cancelled_false']}")
    print(f"Cancelled LS annotations skipped: {summary['label_studio']['annotations_cancelled']}")
    print(f"Prediction-only files evaluated: {summary['prediction_only_files_evaluated_count']}")
    print(f"Prediction-only files ignored:   {summary['prediction_only_files_ignored_count']}")
    print(f"LS-only files evaluated:         {summary['label_studio_files_without_predictions_evaluated_count']}")
    print(f"LS-only files ignored:           {summary['label_studio_files_without_predictions_ignored_count']}")
    print()
    print(f"Saved details: {args.output}")


if __name__ == "__main__":
    main()
