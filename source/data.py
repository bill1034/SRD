"""Audited 2-D OBB input, exact isotropic preprocessing and rotated evaluation.

Coordinates are image coordinates: x right, y down, angles clockwise in radians.
Every tunable option is read from cfg so it can be exposed in notebook cell two.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def _data_numpy(value, dtype=np.float32):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _data_array(value, columns, name):
    array = _data_numpy(value)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite N x {columns} array")
    return array.copy()


def _data_labels(value, count, classes, name):
    if not isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) or len(value) != count:
        raise ValueError(f"{name}: labels length does not match annotations")
    output = []
    for label in value:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if isinstance(label, str):
            if label not in classes:
                raise ValueError(f"{name}: unknown class {label!r}")
            output.append(classes.index(label))
        elif isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_)) and 0 <= label < len(classes):
            output.append(int(label))
        else:
            raise ValueError(f"{name}: invalid zero-based label {label!r}")
    return np.asarray(output, dtype=np.int64)


def obb_polygons(boxes):
    """Return N x 8 clockwise corners without changing the input box geometry."""
    boxes = _data_array(boxes, 5, "boxes")
    if not len(boxes):
        return np.empty((0, 8), dtype=np.float32)
    local = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]], np.float32)
    points = local[None] * boxes[:, None, 2:4]
    cosine, sine = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    x = cosine[:, None] * points[:, :, 0] - sine[:, None] * points[:, :, 1]
    y = sine[:, None] * points[:, :, 0] + cosine[:, None] * points[:, :, 1]
    return (np.stack([x, y], axis=-1) + boxes[:, None, :2]).reshape(-1, 8)


def _data_canonical(boxes):
    boxes = boxes.copy()
    swap = boxes[:, 2] < boxes[:, 3]
    boxes[swap, 2:4] = boxes[swap, 2:4][:, ::-1]
    boxes[swap, 4] += np.pi / 2
    boxes[:, 4] = (boxes[:, 4] + np.pi / 2) % np.pi - np.pi / 2
    return boxes


def _data_bounds(points, width, height, tolerance, name):
    points = np.asarray(points).reshape(-1, 2)
    if len(points) and (np.any(points < -tolerance) or np.any(points[:, 0] > width + tolerance) or np.any(points[:, 1] > height + tolerance)):
        raise ValueError(f"{name}: annotation extent is outside image bounds {width} x {height}")


def _data_annotations(record, classes, width, height, cfg):
    name = str(record["id"])
    if ("boxes" in record) == ("polygons" in record):
        raise ValueError(f"{name}: provide exactly one annotation field: boxes or polygons")
    tolerance = float(cfg.get("annotation_bounds_tolerance", 1.0))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("annotation_bounds_tolerance must be finite and non-negative")
    if "boxes" in record:
        boxes = _data_array(record["boxes"], 5, name)
        if len(boxes) and (np.any(boxes[:, 2:4] <= 0) or np.any(boxes[:, 4] < -np.pi / 2 - 1e-6) or np.any(boxes[:, 4] >= np.pi / 2 + 1e-6)):
            raise ValueError(f"{name}: boxes need positive extents and le90 angles in radians")
        _data_bounds(obb_polygons(boxes), width, height, tolerance, name)
        boxes = _data_canonical(boxes)
    else:
        polygons = _data_array(record["polygons"], 8, name)
        _data_bounds(polygons, width, height, tolerance, name)
        converted = []
        for polygon in polygons.reshape(-1, 4, 2):
            edges = np.roll(polygon, -1, axis=0) - polygon
            cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
            if not (np.all(cross > 1e-6) or np.all(cross < -1e-6)):
                raise ValueError(f"{name}: polygon must be strictly convex with ordered, distinct corners")
            (cx, cy), (bw, bh), angle = cv2.minAreaRect(polygon.astype(np.float32))
            if min(bw, bh) <= 0:
                raise ValueError(f"{name}: polygon has zero area")
            converted.append([cx, cy, bw, bh, math.radians(angle)])
        boxes = _data_canonical(np.asarray(converted, dtype=np.float32).reshape(-1, 5))
    if "labels" not in record:
        raise ValueError(f"{name}: labels are missing")
    labels = _data_labels(record["labels"], len(boxes), classes, name)
    difficult = np.asarray(record.get("difficult", [0] * len(boxes)))
    if difficult.shape != (len(boxes),) or not np.isin(difficult, [0, 1, 2]).all():
        raise ValueError(f"{name}: difficult must have one 0/1/2 value per annotation")
    ignored = difficult.astype(bool)
    return boxes[~ignored], labels[~ignored], boxes[ignored], labels[ignored]


def _data_rgb(path):
    try:
        with Image.open(path) as source:
            # Keep raw image coordinates; do not silently rotate labeled EXIF images.
            return np.asarray(source.convert("RGB")).copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to decode image {path}: {error}") from error


def _data_classes(value):
    if not isinstance(value, (list, tuple)) or not value or any(not isinstance(c, str) or not c.strip() for c in value) or len(set(value)) != len(value):
        raise ValueError("classes must be a non-empty list of distinct class names")
    return list(value)


def _data_dota(root, cfg):
    classes = _data_classes(cfg.get("classes"))
    group_path = Path(cfg.get("groups_csv") or root / "groups.csv")
    groups = {}
    if group_path.is_file():
        with group_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"image_id", "group_id"}.issubset(reader.fieldnames or []):
                raise ValueError("groups.csv requires image_id and group_id columns")
            for row in reader:
                key, value = row["image_id"].strip(), row["group_id"].strip()
                if not key or not value or key in groups:
                    raise ValueError("groups.csv has empty or duplicate image_id/group_id entries")
                groups[key] = value
    records = []
    for split in ("train", "val", "test"):
        image_root = root / split / "images"
        if not image_root.exists():
            continue
        images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for image in images:
            relative = image.relative_to(image_root)
            key = f"{split}/{relative.with_suffix('').as_posix()}"
            annotation = root / split / "annfiles" / relative.with_suffix(".txt")
            if not annotation.is_file():
                raise ValueError(f"{key}: missing annotation file {annotation}; missing labels are not negatives")
            polygons, labels, difficult = [], [], []
            for line_number, line in enumerate(annotation.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip() or line.startswith(("imagesource:", "gsd:")):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    raise ValueError(f"{annotation}:{line_number}: expected 8 coordinates, class and difficult")
                try:
                    polygons.append([float(v) for v in parts[:8]])
                    difficult.append(int(parts[-1]))
                except ValueError as error:
                    raise ValueError(f"{annotation}:{line_number}: invalid annotation number") from error
                labels.append(" ".join(parts[8:-1]))
            group = groups.get(key, groups.get(relative.as_posix(), groups.get(relative.with_suffix('').as_posix())))
            records.append(dict(id=key, image=str(image.resolve()), split=split, group_id=group,
                                polygons=polygons, labels=labels, difficult=difficult, reviewed=True))
    return classes, records


def load_data(cfg, required_splits=("train", "val")):
    """Validate a JSON manifest or DOTA split tree; return classes, records, audit.

    A manifest requires classes and records. Each record requires id, image, split,
    group_id, boxes OR polygons, labels and reviewed=true. DOTA empty annotation
    files explicitly represent reviewed negatives; missing files always fail.
    Training requires train/val by default; standalone evaluation can request only
    its selected split, while all supplied records still receive integrity checks.
    """
    if not isinstance(required_splits, (tuple, list)) or any(split not in ("train", "val", "test") for split in required_splits):
        raise ValueError("required_splits must list train, val and/or test")
    manifest_path = cfg.get("manifest_path") or cfg.get("manifest")
    if manifest_path:
        manifest_path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        classes = _data_classes(payload.get("classes"))
        if cfg.get("classes") and list(cfg["classes"]) != classes:
            raise ValueError("cfg classes disagree with manifest classes or their order")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("manifest records must be a list")
        base = manifest_path.parent
    else:
        if not cfg.get("data_root"):
            raise ValueError("Set manifest or data_root to labeled training data")
        base = Path(cfg["data_root"]).expanduser().resolve()
        classes, raw_records = _data_dota(base, cfg)
    records, ids, group_splits, pixel_splits = [], set(), {}, {}
    pixel_groups = {}
    split_counts = dict(train=0, val=0, test=0)
    automatic_groups, polygon_count, duplicate_within_split = 0, 0, 0
    for raw in raw_records:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"].strip():
            raise ValueError("Every record needs a non-empty string id")
        record = dict(raw)
        key, split = record["id"], record.get("split")
        if key in ids:
            raise ValueError(f"Duplicate record id {key!r}")
        ids.add(key)
        if split not in split_counts:
            raise ValueError(f"{key}: split must be train, val or test")
        if record.get("reviewed") is not True:
            raise ValueError(f"{key}: reviewed must explicitly be true")
        group = record.get("group_id")
        if group is None or str(group).strip() == "":
            if not cfg.get("allow_image_groups", False):
                raise ValueError(f"{key}: group_id is required; enable allow_image_groups only for an explicit image-level split")
            group = f"image:{key}"
            automatic_groups += 1
        if isinstance(group, (list, dict, bool)):
            raise ValueError(f"{key}: group_id must be a string or scalar patient identifier")
        group = str(group)
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"{key}: group {group!r} crosses data splits")
        group_splits[group] = split
        if not isinstance(record.get("image"), str) or not record["image"]:
            raise ValueError(f"{key}: image path is missing")
        image = Path(record["image"]).expanduser()
        image = (base / image).resolve() if not image.is_absolute() else image.resolve()
        if image.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError(f"{key}: image must be PNG/JPG/JPEG")
        rgb = _data_rgb(image)
        height, width = rgb.shape[:2]
        digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
        if digest in pixel_splits:
            if pixel_splits[digest] != split:
                raise ValueError(f"{key}: duplicate decoded image pixels cross data splits")
            if pixel_groups[digest] != group:
                raise ValueError(f"{key}: duplicate decoded image pixels belong to different groups; this would bypass same-group reference exclusion")
            duplicate_within_split += 1
        pixel_splits[digest] = split
        pixel_groups[digest] = group
        boxes, labels, ignored_boxes, ignored_labels = _data_annotations(record, classes, width, height, cfg)
        polygon_count += len(record.get("polygons", []))
        record.update(image=str(image), group_id=group, boxes=boxes, labels=labels,
                      ignored_boxes=ignored_boxes, ignored_labels=ignored_labels,
                      width=width, height=height, pixel_sha256=digest)
        records.append(record)
        split_counts[split] += 1
    missing_splits = [split for split in required_splits if not split_counts[split]]
    if missing_splits:
        raise ValueError(f"Required data splits are empty or missing: {', '.join(missing_splits)}")
    audit = dict(split_counts=split_counts, group_count=len(group_splits),
                 grouping="per-image" if automatic_groups == len(records) else ("mixed-explicit-and-image" if automatic_groups else "provided-groups"),
                 automatic_image_groups=automatic_groups, duplicate_pixels_within_split=duplicate_within_split,
                 polygon_to_minimum_area_rectangle_count=polygon_count,
                 negative_images=sum(len(r["boxes"]) == 0 and len(r["ignored_boxes"]) == 0 for r in records),
                 annotation_count=sum(len(r["boxes"]) for r in records),
                 ignored_annotation_count=sum(len(r["ignored_boxes"]) for r in records),
                 pixel_hash="SHA256(decoded RGB shape and pixels)", coordinates="xywha, radians clockwise, le90")
    return classes, records, audit


class OBBDataset(Dataset):
    """Isotropic affine resize and right/bottom pad; no annotation truncation."""
    def __init__(self, records, cfg, training=False):
        self.records = list(records)
        self.training = bool(training)
        self.horizontal_flip_prob = float(cfg.get("train_horizontal_flip_prob", .25))
        self.vertical_flip_prob = float(cfg.get("train_vertical_flip_prob", .25))
        if not 0 <= self.horizontal_flip_prob <= 1 or not 0 <= self.vertical_flip_prob <= 1:
            raise ValueError("Training flip probabilities must be finite numbers between 0 and 1")
        self.size = int(cfg.get("image_size", 1024))
        self.mean = np.asarray(cfg.get("image_mean", [123.675, 116.28, 103.53]), np.float32)
        self.std = np.asarray(cfg.get("image_std", [58.395, 57.12, 57.375]), np.float32)
        if self.size <= 0 or self.size % 32 or self.mean.shape != (3,) or self.std.shape != (3,) or np.any(self.std <= 0) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("image_size must be a positive multiple of 32; RGB mean/std must be finite with positive std")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        rgb = _data_rgb(record["image"])
        height, width = rgb.shape[:2]
        boxes = _data_array(record.get("boxes", []), 5, "boxes")
        ignored = _data_array(record.get("ignored_boxes", []), 5, "ignored_boxes")
        horizontal = self.training and np.random.random() < self.horizontal_flip_prob
        vertical = self.training and np.random.random() < self.vertical_flip_prob
        if horizontal:
            rgb = np.flip(rgb, axis=1)
            for annotation in (boxes, ignored):
                # Continuous box coordinates span [0, width], unlike pixel indices.
                annotation[:, 0] = width - annotation[:, 0]
                annotation[:, 4] = -annotation[:, 4]
        if vertical:
            rgb = np.flip(rgb, axis=0)
            for annotation in (boxes, ignored):
                annotation[:, 1] = height - annotation[:, 1]
                annotation[:, 4] = -annotation[:, 4]
        if horizontal or vertical:
            boxes, ignored = _data_canonical(boxes), _data_canonical(ignored)
        scale = self.size / max(height, width)
        rh, rw = min(self.size, math.ceil(height * scale)), min(self.size, math.ceil(width * scale))
        # One scalar transform avoids anisotropic box errors caused by rounding
        # independent cv2.resize dimensions. Pixel centers use half-pixel mapping.
        matrix = np.array([[scale, 0, (scale - 1) * .5], [0, scale, (scale - 1) * .5]], np.float32)
        resized = cv2.warpAffine(rgb.astype(np.float32), matrix, (rw, rh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in self.mean))
        canvas = np.zeros((self.size, self.size, 3), np.float32)
        canvas[:rh, :rw] = (resized - self.mean) / self.std
        boxes[:, :4] *= scale
        ignored[:, :4] *= scale
        meta = dict(img_shape=(rh, rw, 3), ori_shape=(height, width, 3), pad_shape=(self.size, self.size, 3),
                    scale_factor=np.array([scale] * 4, np.float32), filename=str(record["image"]),
                    ori_filename=Path(record["image"]).name, group_id=record.get("group_id", f"inference:{record['id']}"),
                    sample_id=record["id"], flip=bool(horizontal or vertical),
                    flip_direction="diagonal" if horizontal and vertical else ("horizontal" if horizontal else ("vertical" if vertical else None)),
                    img_norm_cfg=dict(mean=self.mean.copy(), std=self.std.copy(), to_rgb=True))
        return dict(img=torch.from_numpy(canvas.transpose(2, 0, 1).copy()), meta=meta,
                    boxes=torch.from_numpy(boxes), labels=torch.from_numpy(_data_numpy(record.get("labels", []), np.int64)),
                    ignored_boxes=torch.from_numpy(ignored))


def collate_obbs(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    return dict(img=torch.stack([item["img"] for item in items]), metas=[item["meta"] for item in items],
                boxes=[item["boxes"] for item in items], labels=[item["labels"] for item in items],
                ignored_boxes=[item["ignored_boxes"] for item in items])


def inference_records(directory, exclude_dirs=()):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Inference directory does not exist: {directory}")
    excluded = tuple(Path(path).expanduser().resolve() for path in exclude_dirs)
    records = []
    for path in sorted(directory.rglob("*")):
        resolved = path.resolve()
        if any(resolved == parent or parent in resolved.parents for parent in excluded):
            continue
        if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            key = path.relative_to(directory).as_posix()
            rgb = _data_rgb(path)
            records.append(dict(id=key, image=str(path), group_id=f"inference:{key}",
                                width=rgb.shape[1], height=rgb.shape[0],
                                boxes=np.empty((0, 5), np.float32), labels=np.empty(0, np.int64)))
    if not records:
        raise ValueError("Inference directory has no PNG/JPG/JPEG images")
    return records


def _data_predictions(predictions, record_ids, num_classes):
    if isinstance(predictions, dict):
        entries = [dict(value, id=key) for key, value in predictions.items()]
    else:
        entries = list(predictions)
    output = {}
    for item in entries:
        key = item.get("id", item.get("sample_id"))
        if key not in record_ids or key in output:
            raise ValueError(f"Unknown or duplicate prediction id {key!r}")
        boxes = _data_array(item.get("boxes", []), 5, f"prediction {key}")
        scores = _data_numpy(item.get("scores", []))
        if scores.shape != (len(boxes),) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError(f"{key}: prediction scores must be finite probabilities")
        if len(boxes) and np.any(boxes[:, 2:4] <= 0):
            raise ValueError(f"{key}: prediction box extents must be positive")
        labels = _data_labels(item.get("labels", []), len(boxes), list(range(num_classes)), f"prediction {key}")
        output[key] = dict(boxes=boxes, scores=scores, labels=labels)
    return output


def _data_iou(boxes, targets, device):
    if not len(boxes) or not len(targets):
        return np.zeros((len(boxes), len(targets)), np.float32)
    left = torch.as_tensor(boxes, dtype=torch.float32, device=device).contiguous()
    right = torch.as_tensor(targets, dtype=torch.float32, device=device).contiguous()
    return box_iou_rotated(left, right, clockwise=True).cpu().numpy()


def _data_match(ranked, ground_truth, ignored_truth, threshold, device, overlap_cache=None):
    matched = {key: np.zeros(len(value), dtype=bool) for key, value in ground_truth.items()}
    tp, fp, scores = [], [], []
    # Compute each image's pairwise matrix in one operator call, rather than one
    # CUDA/CPU launch per detection. Keep global score order for matching.
    if overlap_cache is None:
        overlap_cache = {}
    if not overlap_cache:
        image_rows = {}
        for rank, (_, key, box) in enumerate(ranked):
            image_rows.setdefault(key, []).append((rank, box))
        overlaps, ignored_overlaps = {}, {}
        for key, entries in image_rows.items():
            boxes = np.stack([entry[1] for entry in entries])
            regular = _data_iou(boxes, ground_truth[key], device)
            difficult = _data_iou(boxes, ignored_truth[key], device)
            for row, (rank, _) in enumerate(entries):
                overlaps[rank] = regular[row]
                ignored_overlaps[rank] = difficult[row]
        overlap_cache.update(regular=overlaps, difficult=ignored_overlaps)
    overlaps, ignored_overlaps = overlap_cache["regular"], overlap_cache["difficult"]
    for rank, (score, key, _) in enumerate(ranked):
        ious = overlaps[rank]
        available = np.where(~matched[key], ious, -1.)
        best = int(np.argmax(available)) if len(available) else -1
        if best >= 0 and available[best] >= threshold:
            matched[key][best] = True
            tp.append(1.)
            fp.append(0.)
        elif len(ignored_overlaps[rank]) and np.max(ignored_overlaps[rank]) >= threshold:
            continue
        else:
            tp.append(0.)
            fp.append(1.)
        scores.append(score)
    return np.asarray(tp), np.asarray(fp), np.asarray(scores)


def _data_ap101(tp, fp, count):
    if count == 0:
        return None
    if len(tp) == 0:
        return 0.
    cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cumulative_tp / count
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(float).eps)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    values = np.zeros(101)
    valid = indices < len(precision)
    values[valid] = precision[indices[valid]]
    return float(values.mean())


def evaluate_obbs(predictions, records, num_classes, cfg):
    """101-point rotated AP at IoUs .50:.05:.95, plus micro precision/recall.

    Predictions and ground truth are in original image coordinates. Classes with
    no non-ignored ground truth are excluded from mAP and reported as None.
    This transparent evaluator is not claimed to reproduce an unavailable
    proprietary baseline evaluator. No COCO maxDet truncation is applied.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    records = list(records)
    keys = [r["id"] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Evaluation records contain duplicate ids")
    normalized = _data_predictions(predictions, set(keys), num_classes)
    device = str(cfg.get("eval_device", "cpu"))
    operating_iou = float(cfg.get("metric_match_iou", cfg.get("eval_iou_threshold", .5)))
    score_cut = float(cfg.get("metric_score_threshold", cfg.get("eval_score_threshold", .5)))
    if not 0 < operating_iou <= 1 or not 0 <= score_cut <= 1:
        raise ValueError("Evaluation IoU/score thresholds are outside valid ranges")
    thresholds = np.round(np.arange(.5, .951, .05), 2)
    ap_matrix = np.full((num_classes, len(thresholds)), np.nan)
    class_counts, operating_counts = {}, []
    for label in range(num_classes):
        truth, ignored = {}, {}
        for record in records:
            boxes = _data_array(record.get("boxes", []), 5, record["id"])
            labels = _data_labels(record.get("labels", []), len(boxes), list(range(num_classes)), record["id"])
            ignored_boxes = _data_array(record.get("ignored_boxes", []), 5, record["id"])
            ignored_labels = _data_labels(record.get("ignored_labels", []), len(ignored_boxes), list(range(num_classes)), record["id"])
            truth[record["id"]] = boxes[labels == label]
            ignored[record["id"]] = ignored_boxes[ignored_labels == label]
        count = sum(len(value) for value in truth.values())
        class_counts[str(label)] = count
        ranked = [(float(score), key, box) for key, prediction in normalized.items()
                  for box, score, cls in zip(prediction["boxes"], prediction["scores"], prediction["labels"]) if cls == label]
        ranked.sort(key=lambda value: -value[0])
        overlap_cache = {}
        for column, threshold in enumerate(thresholds):
            tp, fp, _ = _data_match(ranked, truth, ignored, float(threshold), device, overlap_cache)
            ap = _data_ap101(tp, fp, count)
            if ap is not None:
                ap_matrix[label, column] = ap
        tp, fp, scores = _data_match(ranked, truth, ignored, operating_iou, device, overlap_cache)
        selected = scores >= score_cut
        operating_counts.append((float(tp[selected].sum()), float(fp[selected].sum()), count))
    valid = np.isfinite(ap_matrix[:, 0])
    tp_count, fp_count, gt_count = np.sum(np.asarray(operating_counts), axis=0)
    return {"mAP50:95": float(ap_matrix[valid].mean()) if valid.any() else None,
            "mAP50": float(ap_matrix[valid, 0].mean()) if valid.any() else None,
            "mAP75": float(ap_matrix[valid, 5].mean()) if valid.any() else None,
            "AP50": float(ap_matrix[valid, 0].mean()) if valid.any() else None,
            "AP75": float(ap_matrix[valid, 5].mean()) if valid.any() else None,
            "class_AP": {str(i): float(ap_matrix[i].mean()) if valid[i] else None for i in range(num_classes)},
            "class_AP50": {str(i): float(ap_matrix[i, 0]) if valid[i] else None for i in range(num_classes)},
            "class_gt_counts": class_counts, "precision": float(tp_count / max(tp_count + fp_count, 1)),
            "recall": float(tp_count / max(gt_count, 1)), "true_positives": int(tp_count),
            "false_positives": int(fp_count), "ground_truth_count": int(gt_count),
            "score_threshold": score_cut, "operating_iou": operating_iou,
            "iou_thresholds": thresholds.tolist(), "interpolation": "101-point",
            "max_detections": None, "images": len(records)}


def save_predictions(predictions, records, classes, output_dir, cfg):
    """Write predictions.json and optionally clockwise-box overlays; return JSON path."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {record["id"]: record for record in records}
    normalized = _data_predictions(predictions, set(lookup), len(classes))
    entries = []
    overlay_dir = output_dir / "overlays"
    if cfg.get("save_overlays", True):
        overlay_dir.mkdir(exist_ok=True)
    for key, prediction in normalized.items():
        polygons = obb_polygons(prediction["boxes"])
        entries.append(dict(id=key, boxes=prediction["boxes"].tolist(), scores=prediction["scores"].tolist(),
                            labels=prediction["labels"].tolist(), polygons=polygons.tolist()))
        if cfg.get("save_overlays", True):
            canvas = Image.fromarray(_data_rgb(lookup[key]["image"]))
            draw = ImageDraw.Draw(canvas)
            for polygon, score, label in zip(polygons.reshape(-1, 4, 2), prediction["scores"], prediction["labels"]):
                if score < float(cfg.get("overlay_score_threshold", .5)):
                    continue
                points = [tuple(float(v) for v in point) for point in polygon]
                draw.line(points + points[:1], fill=(0, 255, 0), width=int(cfg.get("overlay_line_width", 2)))
                draw.text(points[0], f"{classes[label]} {score:.3f}", fill=(255, 255, 0))
            filename = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20] + ".png"
            canvas.save(overlay_dir / filename)
            entries[-1]["overlay"] = f"overlays/{filename}"
    target = output_dir / "predictions.json"
    target.write_text(json.dumps(dict(classes=list(classes), coordinates="original-image xywha, radians clockwise, le90",
                                     predictions=entries), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return str(target)
