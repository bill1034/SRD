# Spectral Reference Detector (SRD)

Official code release for **SRD: A Reference-Grounded Perception Tool for Agentic Medical Image Inspection**.

[English](README.md) | [简体中文](README_zh-CN.md)

SRD is an oriented medical object detector that augments an established two-stage rotated detector with three evidence modules: conditional spectral adaptation, frozen Bayesian reference read, and fixed-graph evidence refinement.

> This initial release provides the model implementation, notebook workflow, tests, and configuration templates. Medical datasets and trained checkpoints are not redistributed.

## Overview

Oriented findings in panoramic dental radiographs and microscopy can be difficult to separate using local appearance alone. SRD combines three complementary evidence sources:

- **Conditional spectral adaptation** applies an orientation-aware Fourier residual to selected aligned RoIs.
- **Frozen Bayesian reference read** retrieves class-conditioned Student-t evidence from a training-only reference bank with a null option and capacity constraints.
- **Fixed-graph evidence refinement** reconciles local, reference, and neighboring evidence through a strongly convex smooth-TV objective before classification and oriented-box regression.

The detector also exposes confidence, non-null reference support, and refinement magnitude for downstream inspection policies. The released experiments evaluate the single-view detector; policy-level view selection, verification, and stopping remain outside the current release.

## Architecture

![SRD architecture](assets/srd_architecture.png)

## Main Result Profile

![Clean-test metric profiles](assets/main_results.png)

The figure summarizes the reported clean-test comparison across DentalOPG, AU-OPG, and OCD. Exact reproduction requires the original datasets, frozen splits, validation-selected checkpoints, and the environment listed below.

## Environment

The verified environment is Windows x64 with:

- Python 3.11
- PyTorch 2.1.1 and CUDA 11.8
- MMCV-full 1.7.2
- MMDetection 2.28.2
- MMRotate 0.3.4
- e2cnn 0.2.3

The pinned Windows environment can be installed with:

```powershell
.\scripts\setup_environment.ps1 -Python 'C:\path\to\python.exe'
```

The CUDA-enabled `RiRoIAlignRotated` path is required for complete training and inference. CPU execution is suitable for selected mathematical and data tests, not as a full detector fallback.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/bill1034/SRD.git
cd SRD
```

### 2. Prepare a manifest

Copy `configs/manifest.example.json` and replace the example records with your own leakage-safe train/validation/test partitions. Each record requires a unique ID, an image path, split, independent group ID, oriented boxes, labels, and a reviewed flag.

Boxes use `[center_x, center_y, width, height, angle]` in original-image pixels. Angles are clockwise radians in the `le90` convention.

### 3. Configure SRD

`configs/paper_config.py` records the three-module paper configuration. Set at least:

```python
CFG["manifest_path"] = "path/to/manifest.json"
CFG["output_dir"] = "path/to/output"
```

The optional legacy quality head is retained for checkpoint compatibility but disabled in the paper configuration.

### 4. Run the notebook

```powershell
jupyter lab notebooks\SRD.ipynb
```

The notebook supports training, resume, evaluation, and image-only inference. Validation mAP is used for checkpoint selection. Test evaluation must be run only after the checkpoint is frozen.

### 5. Image-only inference

With a trained SRD checkpoint:

```python
CFG["mode"] = "infer"
CFG["checkpoint_path"] = "path/to/stage_B_best.pth"
CFG["inference_dir"] = "path/to/images"
CFG["output_dir"] = "path/to/inference_results"
```

Predictions are exported as JSON with original-image oriented boxes, polygons, labels, scores, and sample IDs. Optional overlay images can also be generated.

## Tests

In the pinned environment:

```powershell
.\scripts\run_tests.ps1 -Python '.\.venv-srd\Scripts\python.exe'
```

The current release contains 69 tests covering data auditing, rotated evaluation, the three evidence modules, checkpoint contracts, staged training, and notebook structure.

## Repository Structure

```text
SRD/
├── assets/                 # Architecture and result figures
├── configs/                # Paper configuration and manifest example
├── notebooks/SRD.ipynb     # End-to-end research workflow
├── scripts/                # Environment setup and test entry points
├── source/                 # Data, model, reasoning, and pipeline code
├── tests/                  # Regression and contract tests
├── CITATION.cff
├── LICENSE
├── NOTICE
└── requirements.txt
```

## Data and Checkpoints

Datasets, patient-level metadata, predictions, and trained checkpoints are not included. Obtain each dataset from its original provider and comply with its license, access conditions, and privacy requirements. The training-only reference bank is stored inside an SRD checkpoint and must never be updated with validation or test cases.

## Citation

Paper citation metadata will be added after publication. Until then, please cite the software metadata in `CITATION.cff`.

```bibtex
@software{yuan2026srd,
  author = {Yuan, Jiachen and Shi, Yanqin and Cai, Anteng and Ye, Zihan and An, Junshu and Xin, Ning and Zhao, Lingxiao},
  title  = {SRD: A Reference-Grounded Perception Tool for Agentic Medical Image Inspection},
  year   = {2026},
  note   = {Open-source software release}
}
```

## Acknowledgements

SRD builds on [MMDetection](https://github.com/open-mmlab/mmdetection), [MMRotate](https://github.com/open-mmlab/mmrotate), [ReDet](https://arxiv.org/abs/2103.07733), and [RoI Transformer](https://arxiv.org/abs/1812.00155). Third-party components remain under their original licenses.

## License

SRD is released under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Medical Disclaimer

This repository is intended for research use only. Its models and outputs are not a substitute for professional diagnosis, clinical judgment, or patient care.
