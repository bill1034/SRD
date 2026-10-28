# Spectral Reference Detector（SRD）

论文 **SRD: A Reference-Grounded Perception Tool for Agentic Medical Image Inspection** 的官方代码仓库。

[English](README.md) | [简体中文](README_zh-CN.md)

SRD 面向医学图像旋转目标检测，在成熟的两阶段旋转检测器上加入三个证据模块：条件频谱适配、冻结贝叶斯参考读取和固定图证据细化。

## 方法概览

- **条件频谱适配**：仅对被选中的对齐 RoI 施加方向感知的傅里叶残差。
- **冻结贝叶斯参考读取**：从仅由训练集构建的参考库中读取类别条件 Student-t 证据，并通过 null 选项和容量约束拒绝不可靠匹配。
- **固定图证据细化**：通过强凸 smooth-TV 目标协调局部、参考和邻域证据，再进入分类与旋转框回归。

## 架构

![SRD architecture](assets/srd_architecture.png)

## 主结果概览

![Clean-test metric profiles](assets/main_results.png)

## 环境

已验证环境：Windows x64、Python 3.11、PyTorch 2.1.1、CUDA 11.8、MMCV-full 1.7.2、MMDetection 2.28.2、MMRotate 0.3.4 和 e2cnn 0.2.3。

```powershell
.\scripts\setup_environment.ps1 -Python 'C:\path\to\python.exe'
```

## 快速使用

```bash
git clone https://github.com/bill1034/SRD.git
cd SRD
```

1. 复制并修改 `configs/manifest.example.json`，填写自己的训练、验证和测试记录。
2. 在 `configs/paper_config.py` 或 Notebook 的配置单元中设置 `manifest_path` 和 `output_dir`。
3. 启动 `jupyter lab notebooks\SRD.ipynb`。
4. 使用验证集 mAP 选择 checkpoint。

旋转框格式为 `[中心x, 中心y, 宽, 高, 角度]`，坐标使用原图像素，角度为顺时针弧度并采用 `le90` 约定。数据划分必须使用互不重叠的病例或采集组。

## 测试

```powershell
.\scripts\run_tests.ps1 -Python '.\.venv-srd\Scripts\python.exe'
```

## 数据与权重

仓库不包含数据集和训练权重。

## 引用

引用信息见 `CITATION.cff`。

## 许可证与医学声明

SRD 采用 Apache License 2.0，详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。仅限研究用途，不用于临床诊断。
