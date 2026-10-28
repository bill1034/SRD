"""Behavioral checks for annotation integrity, coordinate geometry and AP."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cache" / "dependencies"))


@pytest.fixture
def api():
    path = ROOT / "source" / "data.py"
    assert path.exists(), "Data implementation has not been created yet"
    spec = importlib.util.spec_from_file_location("resroit_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(tmp_path, records=None):
    for index, name in enumerate(("a.png", "b.png", "c.png")):
        Image.new("RGB", (101, 61), (index * 70, 10, 20)).save(tmp_path / name)
    if records is None:
        records = [
            dict(id="a", image="a.png", split="train", group_id="patient-a",
                 boxes=[[50, 30, 20, 10, .4]], labels=["cell"], reviewed=True),
            dict(id="b", image="b.png", split="val", group_id="patient-b",
                 boxes=[], labels=[], reviewed=True),
        ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    return path, records


def test_manifest_retains_reviewed_negative_and_original_group(api, tmp_path):
    path, _ = manifest(tmp_path)
    classes, records, audit = api.load_data(dict(manifest=str(path)))
    assert classes == ["cell"]
    assert records[0]["group_id"] == "patient-a"
    assert records[1]["boxes"].shape == (0, 5)
    assert records[0]["width"] == 101 and records[0]["height"] == 61
    assert audit["split_counts"] == {"train": 1, "val": 1, "test": 0}


def test_pipeline_configuration_names_are_honored(api, tmp_path):
    path, _ = manifest(tmp_path)
    _, records, _ = api.load_data(dict(manifest_path=str(path)))
    pred = dict(id="a", boxes=[[50, 30, 20, 10, .4]], scores=[.7], labels=[0])
    metrics = api.evaluate_obbs([pred], records, 1, dict(metric_score_threshold=.8, metric_match_iou=.75))
    assert metrics["true_positives"] == 0
    assert metrics["score_threshold"] == .8
    assert metrics["operating_iou"] == .75
    assert metrics["mAP50"] == metrics["AP50"]


def test_nonfinite_annotation_tolerance_cannot_disable_bounds_check(api, tmp_path):
    path, records = manifest(tmp_path)
    records[0]["boxes"][0][0] = -100
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    with pytest.raises(ValueError, match="tolerance"):
        api.load_data(dict(manifest_path=str(path), annotation_bounds_tolerance=float("nan")))


@pytest.mark.parametrize("change,match", [
    (lambda r: r[1].pop("boxes"), "annotation|boxes|polygons"),
    (lambda r: r[1].pop("labels"), "labels"),
    (lambda r: r[1].update(reviewed=False), "reviewed"),
    (lambda r: r[1].pop("group_id"), "group"),
    (lambda r: r[1].update(group_id="patient-a"), "group.*split|split.*group"),
    (lambda r: r[0].update(labels=[1]), "label|class"),
    (lambda r: r[0].update(boxes=[[0, 0, 20, 10, .4]]), "bound|extent"),
])
def test_manifest_rejects_missing_or_invalid_supervision(api, tmp_path, change, match):
    path, records = manifest(tmp_path)
    change(records)
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        api.load_data(dict(manifest=str(path)))


def test_duplicate_decoded_pixels_across_splits_rejected(api, tmp_path):
    path, _ = manifest(tmp_path)
    Image.open(tmp_path / "a.png").save(tmp_path / "b.png", compress_level=0)
    with pytest.raises(ValueError, match="pixel|duplicate"):
        api.load_data(dict(manifest=str(path)))


def test_duplicate_training_pixels_cannot_claim_different_patient_group(api, tmp_path):
    path, records = manifest(tmp_path)
    Image.open(tmp_path / "a.png").save(tmp_path / "c.png", compress_level=0)
    records.append(dict(records[0], id="c", image="c.png", group_id="different-patient"))
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.*group|pixel.*group"):
        api.load_data(dict(manifest_path=str(path)))


def test_duplicate_training_pixels_with_same_group_are_audited(api, tmp_path):
    path, records = manifest(tmp_path)
    Image.open(tmp_path / "a.png").save(tmp_path / "c.png", compress_level=0)
    records.append(dict(records[0], id="c", image="c.png"))
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    _, loaded, audit = api.load_data(dict(manifest_path=str(path)))
    assert loaded[0]["group_id"] == loaded[2]["group_id"] == "patient-a"
    assert audit["duplicate_pixels_within_split"] == 1


def test_image_groups_only_when_explicitly_allowed(api, tmp_path):
    path, records = manifest(tmp_path)
    for record in records:
        record.pop("group_id")
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    _, loaded, audit = api.load_data(dict(manifest=str(path), allow_image_groups=True))
    assert loaded[0]["group_id"] != loaded[1]["group_id"]
    assert audit["grouping"] == "per-image"


def test_convex_polygon_conversion_and_self_intersection_rejection(api, tmp_path):
    path, records = manifest(tmp_path)
    records[0].pop("boxes")
    records[0]["polygons"] = [[20, 10, 60, 10, 60, 30, 20, 30]]
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    _, loaded, _ = api.load_data(dict(manifest=str(path)))
    np.testing.assert_allclose(loaded[0]["boxes"][0], [40, 20, 40, 20, 0], atol=1e-5)
    records[0]["polygons"] = [[20, 10, 60, 30, 60, 10, 20, 30]]
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    with pytest.raises(ValueError, match="convex|polygon"):
        api.load_data(dict(manifest=str(path)))


def test_dota_negative_is_empty_file_difficult_is_ignored(api, tmp_path):
    for index, split in enumerate(("train", "val")):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "annfiles").mkdir()
        Image.new("RGB", (80, 60), (index * 80, 0, 0)).save(tmp_path / split / "images" / "a.png")
        annotation = "10 10 30 10 30 20 10 20 cell 0\n40 10 60 10 60 20 40 20 cell 1\n" if index == 0 else ""
        (tmp_path / split / "annfiles" / "a.txt").write_text(annotation)
    (tmp_path / "groups.csv").write_text("image_id,group_id\ntrain/a,p1\nval/a,p2\n")
    _, records, _ = api.load_data(dict(data_root=str(tmp_path), classes=["cell"]))
    assert records[0]["boxes"].shape == (1, 5)
    assert records[0]["ignored_boxes"].shape == (1, 5)
    assert records[1]["boxes"].shape == (0, 5)
    (tmp_path / "val" / "annfiles" / "a.txt").unlink()
    with pytest.raises(ValueError, match="annotation|missing"):
        api.load_data(dict(data_root=str(tmp_path), classes=["cell"]))


def test_dota_preserves_multiword_dental_class_names(api, tmp_path):
    for index, split in enumerate(("train", "val")):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "annfiles").mkdir()
        Image.new("RGB", (80, 60), (index * 80, 0, 0)).save(tmp_path / split / "images" / "a.png")
        (tmp_path / split / "annfiles" / "a.txt").write_text("10 10 30 10 30 20 10 20 Retained Root 0\n")
    _, records, _ = api.load_data(dict(data_root=str(tmp_path), classes=["Retained Root"], allow_image_groups=True))
    assert records[0]["labels"].tolist() == [0]


def test_test_only_manifest_can_be_evaluated_without_training_data(api, tmp_path):
    path, records = manifest(tmp_path)
    records = [dict(records[0], split="test")]
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    with pytest.raises(ValueError, match="train|val"):
        api.load_data(dict(manifest_path=str(path)))
    _, loaded, audit = api.load_data(dict(manifest_path=str(path)), required_splits=("test",))
    assert len(loaded) == 1 and loaded[0]["split"] == "test"
    assert audit["split_counts"] == {"train": 0, "val": 0, "test": 1}


def test_test_only_dota_can_be_evaluated_without_training_data(api, tmp_path):
    (tmp_path / "test" / "images").mkdir(parents=True)
    (tmp_path / "test" / "annfiles").mkdir()
    Image.new("RGB", (80, 60), (60, 0, 0)).save(tmp_path / "test" / "images" / "a.png")
    (tmp_path / "test" / "annfiles" / "a.txt").write_text("10 10 30 10 30 20 10 20 cell 0\n")
    (tmp_path / "groups.csv").write_text("image_id,group_id\ntest/a,test-patient\n")
    cfg = dict(data_root=str(tmp_path), classes=["cell"])
    with pytest.raises(ValueError, match="train|val"):
        api.load_data(cfg)
    _, loaded, audit = api.load_data(cfg, required_splits=("test",))
    assert loaded[0]["group_id"] == "test-patient"
    assert audit["split_counts"]["test"] == 1


def test_dataset_exact_isotropic_round_trip_and_dense_annotations(api, tmp_path):
    path, records = manifest(tmp_path)
    records[0]["boxes"] *= 211
    records[0]["labels"] *= 211
    path.write_text(json.dumps(dict(classes=["cell"], records=records)), encoding="utf-8")
    _, loaded, _ = api.load_data(dict(manifest=str(path)))
    dataset = api.OBBDataset(loaded, dict(image_size=128))
    item = dataset[0]
    assert item["img"].shape == (3, 128, 128)
    assert item["boxes"].shape == (211, 5)
    assert item["meta"]["group_id"] == "patient-a"
    assert item["meta"]["ori_shape"] == (61, 101, 3)
    scale = item["meta"]["scale_factor"][0]
    np.testing.assert_allclose(item["meta"]["scale_factor"], [128 / 101] * 4)
    np.testing.assert_allclose(item["boxes"][0, :4].numpy() / scale, [50, 30, 20, 10], atol=1e-5)
    assert item["boxes"][0, 4].item() == pytest.approx(.4)
    batch = api.collate_obbs([item, dataset[1]])
    assert batch["img"].shape == (2, 3, 128, 128)
    assert batch["boxes"][1].shape == (0, 5)


def test_inference_recursive_ids_do_not_collide_and_need_no_labels(api, tmp_path):
    for name in ("one/a.png", "two/a.jpg"):
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (30, 20)).save(p)
    records = api.inference_records(tmp_path)
    assert len({r["id"] for r in records}) == 2
    assert len(records) == 2
    assert api.OBBDataset(records, dict(image_size=64))[0]["boxes"].shape == (0, 5)


def test_inference_excludes_nested_output_but_keeps_same_named_input(api, tmp_path):
    for name in ("a.png", "other/a.png", "output/overlays/a.png"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (30, 20)).save(path)
    records = api.inference_records(tmp_path, exclude_dirs=(tmp_path / "output",))
    assert {record["id"] for record in records} == {"a.png", "other/a.png"}


@pytest.mark.parametrize("direction,box_key,expected", [
    ("horizontal", "boxes", [44, 24, 12, 6, -.3]),
    ("horizontal", "ignored_boxes", [22, 44, 10, 4, .4]),
    ("vertical", "boxes", [20, 40, 12, 6, -.3]),
    ("vertical", "ignored_boxes", [42, 20, 10, 4, .4]),
])
def test_training_flip_reflects_image_and_ordinary_or_ignore_geometry(api, tmp_path, direction, box_key, expected):
    y, x = np.mgrid[:64, :64]
    pixels = np.stack([x * 3, y * 3, x + y], axis=-1).astype(np.uint8)
    image_path = tmp_path / "flip.png"
    Image.fromarray(pixels).save(image_path)
    record = dict(id="flip", image=str(image_path), group_id="case", labels=[0],
                  boxes=[[20, 24, 12, 6, .3]], ignored_boxes=[[42, 44, 10, 4, -.4]])
    cfg = dict(image_size=64, train_horizontal_flip_prob=float(direction == "horizontal"),
               train_vertical_flip_prob=float(direction == "vertical"))
    plain = api.OBBDataset([record], cfg)[0]
    flipped = api.OBBDataset([record], cfg, training=True)[0]
    np.testing.assert_allclose(flipped[box_key][0], expected, atol=1e-6)
    torch.testing.assert_close(flipped["img"], torch.flip(plain["img"], [2 if direction == "horizontal" else 1]))
    np.testing.assert_allclose(plain["boxes"][0], [20, 24, 12, 6, .3], atol=1e-6)


def eval_record():
    return dict(id="a", boxes=np.array([[50, 30, 20, 10, .4]], np.float32), labels=np.array([0]),
                ignored_boxes=np.zeros((0, 5), np.float32), ignored_labels=np.zeros(0, np.int64))


def test_rotation_ap_matches_hand_computed_ranked_precision(api):
    gt = eval_record()
    pred = dict(id="a", boxes=[[10, 10, 5, 4, 0], [50, 30, 20, 10, .4]], scores=[.99, .9], labels=[0, 0])
    scores = api.evaluate_obbs([pred], [gt], 2, dict(eval_score_threshold=.5, eval_iou_threshold=.5))
    assert scores["mAP50:95"] == pytest.approx(.5)
    assert scores["AP50"] == pytest.approx(.5)
    assert scores["AP75"] == pytest.approx(.5)
    assert scores["precision"] == pytest.approx(.5)
    assert scores["recall"] == pytest.approx(1.)
    assert scores["class_AP"]["1"] is None


def test_ignored_box_does_not_count_as_false_positive(api):
    gt = eval_record()
    gt["ignored_boxes"] = np.array([[10, 10, 5, 4, 0]], np.float32)
    gt["ignored_labels"] = np.array([0])
    pred = dict(id="a", boxes=[[10, 10, 5, 4, 0], [50, 30, 20, 10, .4]], scores=[.99, .9], labels=[0, 0])
    scores = api.evaluate_obbs({"a": pred}, [gt], 1, {})
    assert scores["mAP50:95"] == pytest.approx(1.)
    assert scores["precision"] == pytest.approx(1.)


def test_dense_matching_uses_available_eligible_ground_truth(api):
    gt = eval_record()
    gt["boxes"] = np.array([[50, 30, 20, 10, 0], [55, 30, 20, 10, 0]], np.float32)
    gt["labels"] = np.array([0, 0])
    pred = dict(id="a", boxes=[[50, 30, 20, 10, 0], [51, 30, 20, 10, 0]], scores=[.99, .9], labels=[0, 0])
    scores = api.evaluate_obbs([pred], [gt], 1, {})
    assert scores["AP50"] == pytest.approx(1.)
    assert scores["recall"] == pytest.approx(1.)


def test_evaluation_rejects_unknown_ids_instead_of_silently_dropping(api):
    with pytest.raises(ValueError, match="unknown|record|id"):
        api.evaluate_obbs([dict(id="other", boxes=[], scores=[], labels=[])], [eval_record()], 1, {})


def test_save_predictions_preserves_original_coords_and_clockwise_corners(api, tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (100, 60)).save(image_path)
    pred = dict(id="case/a", boxes=[[50, 30, 20, 10, 0]], scores=[.9], labels=[0])
    target = api.save_predictions([pred], [dict(id="case/a", image=str(image_path))],
                                  ["cell"], tmp_path / "out", dict(save_overlays=True))
    payload = json.loads(Path(target).read_text(encoding="utf-8"))
    np.testing.assert_allclose(payload["predictions"][0]["polygons"][0], [40, 25, 60, 25, 60, 35, 40, 35])
    assert len(list((tmp_path / "out" / "overlays").glob("*.png"))) == 1
