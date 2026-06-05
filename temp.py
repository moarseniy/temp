#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

def basename_from_ls_image_path(path: str) -> str:
    return Path(path.split("?d=")[-1]).name

def rect_percent_to_quad(value: dict, original_width: int, original_height: int):
    """
    Convert Label Studio rectangle result to image-space quad.

    Label Studio stores x/y/width/height in percentages and rotation in degrees.
    Rotation is applied around the top-left corner of the rectangle, which is also
    the anchor used in annotation.result[]['value'].
    """
    x0 = value["x"] / 100.0 * original_width
    y0 = value["y"] / 100.0 * original_height
    w = value["width"] / 100.0 * original_width
    h = value["height"] / 100.0 * original_height
    rotation = float(value.get("rotation", 0.0) or 0.0)

    theta = np.deg2rad(rotation)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    # Positive rotation in image coordinates is clockwise because Y grows down.
    rel_points = [
        (0.0, 0.0),
        (w, 0.0),
        (w, h),
        (0.0, h),
    ]

    quad = []
    for dx, dy in rel_points:
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

def load_gt_label_studio(path: Path, labels: set[str] = None):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    gt_by_file = {}
    file_upload_by_file = {}

    for task in data:
        image_path = task["data"]["image"]
        filename = basename_from_ls_image_path(image_path)

        gt_by_file.setdefault(filename, [])
        if task.get("file_upload"):
            file_upload_by_file[filename] = task["file_upload"]

        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue

            for res in ann.get("result", []):
                if res.get("type") != "rectanglelabels":
                    continue

                value = res["value"]
                rect_labels = value.get("rectanglelabels", [])

                if labels is not None and not any(label in labels for label in rect_labels):
                    continue

                quad = rect_percent_to_quad(
                    value=value,
                    original_width=res["original_width"],
                    original_height=res["original_height"],
                )

                gt_by_file[filename].append(
                    {
                        "quad": quad,
                        "labels": rect_labels,
                    }
                )

    return gt_by_file, file_upload_by_file

def normalize_pred_item(item):
    """
    Поддерживает:
    - [[x,y], [x,y], [x,y], [x,y]]
    - [x1,y1,x2,y2]
    - {"quad": ...}
    - {"bbox": ...}
    - {"box": ...}
    """
    if isinstance(item, dict):
        if "quad" in item:
            return item["quad"]
        if "bbox" in item:
            item = item["bbox"]
        elif "box" in item:
            item = item["box"]
        else:
            raise ValueError(f"Unknown prediction item dict format: {item}")

    if len(item) == 4 and all(isinstance(p, (int, float)) for p in item):
        x1, y1, x2, y2 = item
        return [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]

    if len(item) == 4 and all(len(p) == 2 for p in item):
        return item

    raise ValueError(f"Unknown prediction item format: {item}")

def load_predictions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pred_by_file = {}

    for key, items in data.items():
        filename = Path(key).name
        pred_by_file.setdefault(filename, [])

        for item in items:
            quad = normalize_pred_item(item)
            pred_by_file[filename].append({"quad": quad})

    return pred_by_file

def polygon_area(quad):
    pts = np.asarray(quad, dtype=np.float32)
    return abs(cv2.contourArea(pts))

def polygon_iou(quad_a, quad_b):
    a = np.asarray(quad_a, dtype=np.float32)
    b = np.asarray(quad_b, dtype=np.float32)

    area_a = polygon_area(a)
    area_b = polygon_area(b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)

def match_image(gt_items, pred_items, iou_threshold: float):
    """
    Many-to-one matching for depersonalization/detection coverage.

    Правила:
    - GT считается найденным, если есть хотя бы один pred с IoU >= threshold.
    - Один pred может покрыть несколько GT.
    - FP — pred, который не покрыл ни одного GT.
    """

    gt_matched_by = {}
    pred_matched_to = {}

    for gt_idx, gt in enumerate(gt_items):
        for pred_idx, pred in enumerate(pred_items):
            iou = polygon_iou(pred["quad"], gt["quad"])

            if iou >= iou_threshold:
                gt_matched_by.setdefault(gt_idx, []).append(
                    {
                        "pred_idx": pred_idx,
                        "iou": iou,
                    }
                )
                pred_matched_to.setdefault(pred_idx, []).append(
                    {
                        "gt_idx": gt_idx,
                        "iou": iou,
                    }
                )

    matched_gt = set(gt_matched_by.keys())
    matched_pred = set(pred_matched_to.keys())

    tp = len(matched_gt)
    fn_indices = [
        idx for idx in range(len(gt_items))
        if idx not in matched_gt
    ]

    fp_indices = [
        idx for idx in range(len(pred_items))
        if idx not in matched_pred
    ]

    fp = len(fp_indices)
    fn = len(fn_indices)

    matches = []

    for gt_idx, pred_matches in gt_matched_by.items():
        best = max(pred_matches, key=lambda x: x["iou"])
        matches.append(
            {
                "gt_idx": gt_idx,
                "pred_idx": best["pred_idx"],
                "iou": best["iou"],
                "all_pred_matches": pred_matches,
            }
        )

    return tp, fp, fn, matches, fp_indices, fn_indices

def safe_div(a, b):
    return a / b if b else 0.0

def evaluate(gt_by_file, pred_by_file, iou_threshold: float):
    # Всегда считаем только файлы, которые есть в predictions.
    all_files = sorted(pred_by_file.keys())

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_file = []

    for filename in all_files:
        gt_items = gt_by_file.get(filename, [])
        pred_items = pred_by_file.get(filename, [])

        tp, fp, fn, matches, fp_indices, fn_indices = match_image(
            gt_items=gt_items,
            pred_items=pred_items,
            iou_threshold=iou_threshold,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_file.append(
            {
                "file": filename,
                "gt": len(gt_items),
                "pred": len(pred_items),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matches": matches,
                # "matches": [
                #     {
                #         "pred_idx": p,
                #         "gt_idx": g,
                #         "iou": iou,
                #     }
                #     for p, g, iou in matches
                # ],
                "fp_indices": fp_indices,
                "fn_indices": fn_indices,
            }
        )

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    summary = {
        "iou_threshold": iou_threshold,
        "files_count": len(all_files),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return summary, per_file

def draw_quads(draw: ImageDraw.ImageDraw, items, color: str, width: int):
    for item in items:
        quad = item["quad"]
        pts = [(int(round(x)), int(round(y))) for x, y in quad]
        draw.line(pts + [pts[0]], fill=color, width=width)

def draw_indexed_quads(
    draw: ImageDraw.ImageDraw,
    items,
    indices,
    color: str,
    width: int,
):
    for idx in indices:
        item = items[idx]
        quad = item["quad"]
        pts = [(int(round(x)), int(round(y))) for x, y in quad]
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

def draw_visualizations(
    gt_by_file,
    pred_by_file,
    per_file,
    images_dir: Path,
    output_dir: Path,
    vis_mode: str,
    vis_filter: str,
    file_upload_by_file: dict[str, str] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_upload_by_file = file_upload_by_file or {}

    metrics_by_file = {item["file"]: item for item in per_file}

    saved = 0
    skipped = 0

    for filename, metrics in metrics_by_file.items():
        if not should_visualize_file(metrics, vis_filter):
            continue

        image_path = images_dir / filename

        if not image_path.exists():
            print(f"[VIS SKIP] image not found: {image_path}")
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        gt_items = gt_by_file.get(filename, [])
        pred_items = pred_by_file.get(filename, [])

        if vis_mode == "gt":
            # Все GT зелёным.
            draw_quads(draw, gt_items, color="lime", width=3)

        elif vis_mode == "gt_pred":
            # Все GT зелёным, все pred красным.
            draw_quads(draw, gt_items, color="lime", width=3)
            draw_quads(draw, pred_items, color="red", width=3)

        elif vis_mode == "errors":
            # Только ошибочные:
            # FP pred — красным.
            # FN gt — синим.
            draw_indexed_quads(
                draw,
                pred_items,
                metrics["fp_indices"],
                color="red",
                width=4,
            )
            draw_indexed_quads(
                draw,
                gt_items,
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", required=True, help="Label Studio JSON")
    parser.add_argument("--pred", required=True, help="Predictions JSON")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output", default="detection_metrics.json")

    parser.add_argument(
        "--images-dir",
        default=None,
        help="Папка с исходными картинками",
    )

    parser.add_argument(
        "--vis-output-dir",
        default=None,
        help="Папка для сохранения визуализаций",
    )

    parser.add_argument(
        "--vis-mode",
        choices=["gt", "gt_pred", "errors"],
        default="gt_pred",
        help=(
            "Что рисовать: "
            "gt = только GT; "
            "gt_pred = GT зелёным + pred красным; "
            "errors = только FP/FN: FP красным, FN синим"
        ),
    )

    parser.add_argument(
        "--vis-filter",
        choices=["all", "errors", "fp", "fn"],
        default="all",
        help=(
            "Какие картинки сохранять: "
            "all = все; "
            "errors = только где есть FP или FN; "
            "fp = только где есть FP; "
            "fn = только где есть FN"
        ),
    )

    args = parser.parse_args()

    labels = set(args.labels) if args.labels else None

    gt_by_file, file_upload_by_file = load_gt_label_studio(Path(args.gt), labels=labels)
    pred_by_file = load_predictions(Path(args.pred))

    summary, per_file = evaluate(
        gt_by_file=gt_by_file,
        pred_by_file=pred_by_file,
        iou_threshold=args.iou,
    )

    result = {
        "summary": summary,
        "per_file": per_file,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if args.images_dir and args.vis_output_dir:
        draw_visualizations(
            gt_by_file=gt_by_file,
            pred_by_file=pred_by_file,
            per_file=per_file,
            images_dir=Path(args.images_dir),
            output_dir=Path(args.vis_output_dir),
            vis_mode=args.vis_mode,
            vis_filter=args.vis_filter,
            file_upload_by_file=file_upload_by_file,
        )

    print()
    print("Detection metrics")
    print("=================")
    print(f"IoU threshold: {summary['iou_threshold']}")
    print(f"Files:         {summary['files_count']}")
    print(f"TP:            {summary['tp']}")
    print(f"FP:            {summary['fp']}")
    print(f"FN:            {summary['fn']}")
    print(f"Precision:     {summary['precision']:.4f}")
    print(f"Recall:        {summary['recall']:.4f}")
    print(f"F1:            {summary['f1']:.4f}")
    print()
    print(f"Saved details: {args.output}")

if __name__ == "__main__":
    main()