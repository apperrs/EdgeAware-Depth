
# EdgeAware-Depth

## Overview

We propose **EdgeAware-Depth**, a self-supervised monocular depth estimation model that generates depth maps with sharp geometric boundaries and global structural consistency. Built upon the Monodepth2 framework, EdgeAware-Depth tackles the long‑standing challenges of structural collapse, blurry edges, and depth bleeding in large textureless regions commonly encountered in outdoor scenes.

## Key Features

- **Bottleneck Global Rectification (BGR):** A Vision Transformer (ViT) at the encoder‑decoder bottleneck captures long‑range dependencies, providing globally coherent features that effectively correct macroscopic scene structures.
- **Depth‑Aware Multi‑Scale Attention Module (DAMSAM):** A dedicated decoder module that fuses multi‑scale feature pyramids, edge‑aware attention maps, and an adaptive gating mechanism. It dynamically prioritizes depth discontinuities and suppresses over‑smoothing artifacts, resulting in noticeably sharper object boundaries.
- **Sub‑pixel Convolution‑based Upsampling Mechanism (SPCU):** Replaces standard transposed convolutions with PixelShuffle‑based upsampling to eliminate checkerboard artifacts. By explicitly redistributing feature channels into spatial positions, SPCU preserves high‑frequency details and maintains crisp depth boundaries even in geometrically complex regions.

## Dataset

We evaluate on two challenging outdoor benchmarks:

- **KITTI** Download from: [http://www.cvlibs.net/datasets/kitti/raw_data.php](http://www.cvlibs.net/datasets/kitti/raw_data.php)
- **Cityscapes** Download from: [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)

## Environment Requirements

This project is developed with Python 3.8.10, PyTorch 2.4.1, and CUDA 12.8. Main required packages:

```
torch>=2.0.0
torchvision
opencv-python
matplotlib
numpy
pillow
```

Set up your environment using:

```bash
pip install torch>=2.0.0 torchvision opencv-python matplotlib numpy pillow
```

## Training Process

Train EdgeAware-Depth from scratch on monocular video sequences using the standard self‑supervised pipeline. Example command:

```bash
python3 train.py --model_name mono_model --log_dir log --batch_size 16 --num_epochs 20
```

## Evaluation
To prepare the ground truth depth maps run:
```bash
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```

To evaluate on the test set or generate depth maps for unseen images:

```bash
python3 evaluate_depth.py --load_weights_folder  log/mono_model/models/weights_19 --eval_mono
```

## Prediction for a single image
You can predict scaled disparity for a single image with:

```bash
python3 test_simple.py --image_path assets/t1.png --model_name mono_640x192
```
