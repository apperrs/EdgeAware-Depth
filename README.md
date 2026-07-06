
# EdgeAware-Depth

## Overview

We propose **EdgeAware-Depth**, a self-supervised monocular depth estimation model that generates depth maps with sharp geometric boundaries and global structural consistency. Built upon the Monodepth2 framework, EdgeAware-Depth tackles the long‑standing challenges of structural collapse, blurry edges, and depth bleeding in large textureless regions commonly encountered in outdoor scenes.

## Key Features

- **Bottleneck Global Rectification (BGR):** A Vision Transformer (ViT) at the encoder‑decoder bottleneck captures long‑range dependencies, providing globally coherent features that effectively correct macroscopic scene structures.
- **Depth‑Aware Multi‑Scale Attention Module (DAMSAM):** A dedicated decoder module that fuses multi‑scale feature pyramids, edge‑aware attention maps, and an adaptive gating mechanism. It dynamically prioritizes depth discontinuities and suppresses over‑smoothing artifacts, resulting in noticeably sharper object boundaries.
- **Sub‑pixel Convolution‑based Upsampling Mechanism (SPCU):** Replaces standard transposed convolutions with PixelShuffle‑based upsampling to eliminate checkerboard artifacts. By explicitly redistributing feature channels into spatial positions, SPCU preserves high‑frequency details and maintains crisp depth boundaries even in geometrically complex regions.

## Dataset

We evaluate on two challenging outdoor benchmarks:

### KITTI training data

You can download the entire raw KITTI dataset by running:

```bash
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```

Then unzip with:

```bash
cd kitti_data
unzip "*.zip"
cd ..
```

**Warning:** the dataset is about 175 GB, so make sure you have enough space to unzip as well.

Our default settings expect that you have converted the png images to jpeg with this command, which also deletes the raw KITTI .png files:

```bash
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

### Cityscapes

First, download `leftImg8bit_sequence_trainvaltest.zip` and `camera_trainvaltest.zip` from the [Cityscapes website](https://www.cityscapes-dataset.com/) and unzip them into the folder `/path/to/cityscapes`. Then preprocess the Cityscapes dataset using the following command:

```bash
cd cityscapes
python3 prepare_cityscapes.py \
  --img_height 512 \
  --img_width 1024 \
  --dataset_dir /home/datasets/cityscapes \
  --dump_root /home/datasets/cityscapes_preprocessed \
  --seq_length 3 \
  --num_threads 8
```

Remember to modify `--dataset_dir` and `--dump_root` to point to your own paths.

The ground truth depth files are provided by ManyDepth at [this link](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip). Download and unzip them into `splits/cityscapes`.

## Splits

The train/test/validation splits are defined in the `splits/` and `cityscapes/splits` folders. You can also train a model using Zhou's subset of the standard Eigen split (see [SfMLearner](https://github.com/tinghuiz/SfMLearner)).

## Environment Requirements

This project is developed with Python 3.8.10, PyTorch 2.4.1, and CUDA 12.8. Main required packages:

```
torch
torchvision
opencv-python
matplotlib
numpy
pillow
mmcv
```

Set up your environment using:

```bash
pip3 install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements.txt
pip3 install -U openmim
mim install mmcv-full
```

## Pretrained Weights

Pretrained weights are currently available by contacting the authors. They will be uploaded to Google Drive upon acceptance of the manuscript.

## Training Process

Train EdgeAware-Depth from scratch on monocular video sequences using the standard self‑supervised pipeline.

**On KITTI:**

```bash
python3 train.py --model_name mono_model --log_dir log --batch_size 16 --num_epochs 20
```

**On Cityscapes:**

```bash
cd cityscapes
python3 train.py \
    --data_path datasets/cityscapes \
    --data_path_pre datasets/cityscapes_preprocessed \
    --dataset cityscapes \
    --log_dir log \
    --exp_name edgeaware_cityscapes \
    --width 640 \
    --height 192 \
    --num_scales 4 \
    --num_layers 18 \
    --batch_size 64 \
    --lr_sche_type cosine \
    --learning_rate 1e-4 \
    --eta_min 1e-6 \
    --num_epochs 20 \
    --decay_step 15 \
    --decay_rate 0.1 \
```

## Evaluation
**Evaluate on KITTI:**

```bash
python3 evaluate_depth.py --load_weights_folder log/mono_model/models/weights_19 --eval_mono
```

**Evaluate on Cityscapes:**

```bash
cd cityscapes
python3 evaluate_depth.py \
    --pretrained_path log/weights \
    --batch_size 24 \
    --cityscapes_path /home/datasets/cityscapes
```

## Prediction for a single image

**On KITTI:**

```bash
python3 test_simple.py --image_path assets/test_image.jpg --model_name mono_640x192
```

**On Cityscapes:**

```bash
cd cityscapes
python3 test_simple.py \
    --image_path folder/test_image.jpg \
    --pretrained_path log/ckpt.pth \
    --save_npy
```
## License
This project is licensed under the [MIT License](https://github.com/apperrs/EdgeAware-Depth/blob/main/LICENSE).
