# Asym-CBAM for Sugarcane Leaf Disease Classification

This archive contains the source code associated with the manuscript
"Direction-Aware Asymmetric Spatial Attention for Sugarcane Leaf Disease
Classification Using ConvNeXt-Tiny."

## Main file

`Asym_CBAM_1x5_3seed.py` implements the ConvNeXt-Tiny backbone with the
complete Asym-CBAM module. The spatial attention branch uses learnable
`1 x 5` and `5 x 1` convolutions.

## Dataset

The image dataset is not redistributed with this software. Download the
Sugarcane Leaf Disease Dataset from the original source cited in the
manuscript. Organise the images into the following class folders:

```text
dataset_root/
Healthy/
Mosaic/
Red Rot/
Rust/
Yellow/
```

Folder names with spaces, underscores, or hyphens are matched automatically.

## Experimental protocol

- Image size: 224 x 224 pixels
- Backbone: ImageNet-1K-pretrained ConvNeXt-Tiny
- Asymmetric kernel size: 5
- Seeds: 42, 0, and 1
- Split ratio: 70% training, 20% validation, and 10% testing
- Optimizer: AdamW
- Initial learning rate: 5e-5
- Weight decay: 0.01
- Batch size: 32
- Maximum epochs: 50
- Scheduler: cosine annealing
- Label smoothing: 0.1

For each seed, the code creates a reproducible stratified data partition and
uses the same seed for stochastic training operations. Consequently, the
three seeds correspond to three independently generated stratified splits.
Results are saved and reported separately for every seed before calculating
the multi-seed summary. The highest result reported in the manuscript was
observed for seed 0; the code does not assume or hard-code that result.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the available CPU or CUDA environment
if a platform-specific installation is required.

## Usage

```bash
python Asym_CBAM_1x5_3seed.py \
  --data_root "/path/to/Sugarcane Leaf Disease Dataset" \
  --epochs 50 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --num_workers 0 \
  --save_dir "./outputs_asym1x5_3seed"
```

On Windows Command Prompt, the command can be entered on one line:

```text
python Asym_CBAM_1x5_3seed.py --data_root "C:\path\to\dataset" --epochs 50 --batch_size 32 --lr 5e-5 --weight_decay 0.01 --num_workers 0 --save_dir "outputs_asym1x5_3seed"
```

## Outputs

The script saves the best checkpoint and evaluation outputs for each seed,
including classification metrics, confusion matrices, ROC curves, training
curves, confidence plots, and Grad-CAM visualisations. It also produces a
multi-seed summary.

## Grad-CAM target layer

Grad-CAM uses the depthwise convolutional layer within the final ConvNeXt
block of Stage 4:

```python
model.features[7][2].block[0]
```

## Licence

See `LICENSE`.
