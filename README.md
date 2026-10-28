# Spectral Reference Detector (SRD)

Official code release for **SRD: A Reference-Grounded Perception Tool for Agentic Medical Image Inspection**.

[English](README.md) | [简体中文](README_zh-CN.md)

SRD is an oriented medical object detector that augments an established two-stage rotated detector with three evidence modules: conditional spectral adaptation, frozen Bayesian reference read, and fixed-graph evidence refinement.

## Overview

- **Conditional spectral adaptation** applies an orientation-aware Fourier residual to selected aligned RoIs.
- **Frozen Bayesian reference read** retrieves class-conditioned Student-t evidence from a training-only reference bank with a null option and capacity constraints.
- **Fixed-graph evidence refinement** reconciles local, reference, and neighboring evidence through a strongly convex smooth-TV objective before classification and oriented-box regression.

## Architecture

![SRD architecture](assets/srd_architecture.png)

## Main Result Profile

![Clean-test metric profiles](assets/main_results.png)

## Environment

Requirements:

- Python 3.11
- PyTorch 2.1.1 and CUDA 11.8
- MMCV-full 1.7.2
- MMDetection 2.28.2
- MMRotate 0.3.4
- e2cnn 0.2.3

```powershell
.\scripts\setup_environment.ps1 -Python 'C:\path\to\python.exe'
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/bill1034/SRD.git
cd SRD
```

### 2. Prepare a manifest

Copy `configs/manifest.example.json` and add image paths, splits, group IDs, oriented boxes, and labels.

Boxes use `[center_x, center_y, width, height, angle]` in original-image pixels. Angles are clockwise radians in the `le90` convention.

### 3. Configure SRD

Set the paths in `configs/paper_config.py`:

```python
CFG["manifest_path"] = "path/to/manifest.json"
CFG["output_dir"] = "path/to/output"
```

### 4. Run the notebook

```powershell
jupyter lab notebooks\SRD.ipynb
```

The notebook supports training, resume, evaluation, and image-only inference. Validation mAP is used for checkpoint selection.

### 5. Image-only inference

With a trained SRD checkpoint:

```python
CFG["mode"] = "infer"
CFG["checkpoint_path"] = "path/to/stage_B_best.pth"
CFG["inference_dir"] = "path/to/images"
CFG["output_dir"] = "path/to/inference_results"
```

## Tests

```powershell
.\scripts\run_tests.ps1 -Python '.\.venv-srd\Scripts\python.exe'
```

## Data and Checkpoints

Datasets and trained checkpoints are not included.

## Citation

```bibtex
@software{yuan2026srd,
  author = {Yuan, Jiachen and Shi, Yanqin and Cai, Anteng and Ye, Zihan and An, Junshu and Xin, Ning and Zhao, Lingxiao},
  title  = {SRD: A Reference-Grounded Perception Tool for Agentic Medical Image Inspection},
  year   = {2026},
  note   = {Open-source software release}
}
```

## Acknowledgements

SRD builds on [MMDetection](https://github.com/open-mmlab/mmdetection), [MMRotate](https://github.com/open-mmlab/mmrotate), [ReDet](https://arxiv.org/abs/2103.07733), and [RoI Transformer](https://arxiv.org/abs/1812.00155).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Medical Disclaimer

For research use only; not for clinical diagnosis.
