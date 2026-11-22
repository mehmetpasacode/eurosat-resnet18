# EuroSAT Land-Cover Classification with ResNet-18

A Bilkent University CS-484 project that classifies **64 × 64 RGB Sentinel-2 tiles** into ten land-cover classes using a custom CIFAR-style ResNet-18. The course study used Optuna to explore six training and model hyperparameters. The repository provides scripts for training, held-out evaluation, and inference, with a verified CPU Docker build.

[Method](#method) · [Results](#results) · [External comparisons](#external-comparisons) · [Quickstart](#quickstart) · [Docker](#docker-usage) · [Credits](#credits-and-permissions)

## Method

| Component | Implementation |
| --- | --- |
| Data and split | EuroSAT RGB contains 27,000 labeled tiles across ten classes. The code stratifies by class into 21,600 training, 2,700 validation, and 2,700 test images, using seed 42 by default. [EuroSAT dataset](https://github.com/phelber/EuroSAT) |
| Model | A stride-1 3 × 3 convolution, batch normalization, and ReLU feed four residual stages with two blocks each. The stem omits max pooling. Stages 2–4 downsample with stride 2 and projection shortcuts. At width 1.0 the stages have 64, 128, 256, and 512 channels. Global average pooling, dropout, and a linear layer produce ten logits. [Model implementation](model.py) |
| Preprocessing | Training uses horizontal and vertical flips, rotation up to 10°, color jitter, tensor conversion, and Gaussian noise. Validation and test use tensor conversion only. Custom inference converts images to RGB and resizes them to 64 × 64. [Data pipeline](data.py) |
| Training | The current code initializes from scratch and uses AdamW, cosine learning-rate decay, and cross-entropy with 0.05 label smoothing. It selects `best.pt` by validation accuracy and saves the run configuration, split indices, metric history, and final weights. [Training script](train.py) |
| Evaluation | The evaluator uses the saved test indices and reports top-1 accuracy, per-class precision, recall and F1, and a row-normalized confusion matrix. Inference exports predicted classes and softmax confidence scores. [Evaluation](evaluate.py) · [Inference](infer.py) |

## Results

### Historical course experiments

The CS-484 final report (Sections V–VI) records the following scores. The corrected pipeline has passed synthetic checks. Reproducing the report’s dataset accuracy remains pending.

| Experiment | Epochs | Batch | Learning rate | Weight decay | Dropout | Width | Reported accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 10 | 64 | 0.001 | 0.0001 | 0.20 | 1.00 | **96.19%** |
| Fewer epochs | 5 | 64 | 0.001 | 0.0001 | 0.20 | 1.00 | 92.41% |
| Higher learning rate | 10 | 64 | 0.1 | 0.0001 | 0.20 | 1.00 | 91.41% |
| Optuna-tuned | 10 | 32 | 0.0052 | 0.000001024 | 0.14 | 0.75 | **97.90%** |

The report describes a 30-trial Optuna search and separately names **97.90%** as the best tuned result. Figure 4 displays **96.63% overall test accuracy** and **87.26% confidence-filtered accuracy**. The run associations for these scores are unestablished.

The report discusses an **85% performance threshold** and a **0.85 prediction-confidence cutoff**. The precise definition, denominator, and coverage of the confidence-filtered scores cannot be recovered. The table reproduces the report's scores, whose interpretation as standard accuracy remains unverified.

| Configuration | Reported confidence-filtered score |
| --- | ---: |
| Baseline | 85.0% |
| Width 0.15 | 80.0% |
| Zero weight decay | 81.0% |
| Batch size 256 | 79.7% |
| Dropout 0.70 | 78.6% |
| Optuna-tuned | 87.9% |

### Archive and runtime verification

The course team's archived `history.csv` records ten epochs of training and validation loss and accuracy, validation macro precision, recall and F1, and the learning rate. Peak validation accuracy was **92.00%**.

- The archived confusion matrix is labeled **91.41%**. Separately, the archived prediction CSV contains **2,590 correct predictions out of 2,700 (95.93%)**. The artifact-to-checkpoint and artifact-to-run associations remain unverified.
- The archived `best.pt` loaded strictly on CPU as a ten-class, width-1.0 model. A checkpoint matching the width-0.75 tuned configuration is missing.
- Local and Docker CPU checks passed a synthetic forward pass and one AdamW step. The Linux AMD64 image also passed imports for all five modules. Real-data evaluation requires the matching EuroSAT directory and run artifacts.

## External comparisons

These are the original repositories’ reported test scores. Split files, training protocols, and initial weights differ across projects, so the figures provide context for the course results.

| Model | Test accuracy | EuroSAT RGB protocol | Source |
| --- | ---: | --- | --- |
| MobileNetV2 | 95.26% | Stratified 70/15/15, 4,050 test images. ImageNet weights, classifier-head training with a frozen backbone followed by full fine-tuning. | [AhmetServet/eurosat-cnn](https://github.com/AhmetServet/eurosat-cnn#results-summary) |
| VGG16-BN | 96.69% | Same 70/15/15 split and ImageNet transfer-learning protocol as MobileNetV2 above. | [AhmetServet/eurosat-cnn](https://github.com/AhmetServet/eurosat-cnn#results-summary) |
| EfficientNet-B0 | 98.59% | 15% test holdout, 4,050 images. ImageNet initialization, 224 × 224 input, 30 epochs. | [Reza-Asiyabi/Kaggle_EuroSAT_Classification](https://github.com/Reza-Asiyabi/Kaggle_EuroSAT_Classification#results) |
| ViT-Base/16 at 224 px | 98.67% | Label-balanced proportions on an 80/10/10 split. ImageNet-21k pretraining and ImageNet-1k fine-tuning before EuroSAT fine-tuning. | [chathumal93/EuroSat-RGB-Classifiers](https://github.com/chathumal93/EuroSat-RGB-Classifiers#models-performance) |

## Quickstart

Run from the repository root with Python 3.10 and a compatible PyTorch/torchvision pair. Local CPU execution was checked with PyTorch 2.5.1 and torchvision 0.20.1. CLI flags were checked locally and in Docker. The dataset-dependent commands below require local inputs.

```sh
python -m pip install -r requirements.txt
python train.py --help
python evaluate.py --help
python infer.py --help
```

Place the RGB tiles in an `ImageFolder` tree such as `data/EuroSAT/AnnualCrop/` and `data/EuroSAT/Forest/`, with one subdirectory per class.

```sh
python train.py --data_dir data/EuroSAT --epochs 10 --out_dir outputs/baseline
python evaluate.py --data_root data/EuroSAT --out_dir outputs/baseline
python infer.py --data_root data/EuroSAT --out_dir outputs/baseline
python infer.py --images_glob "examples/*.jpg" --out_dir outputs/baseline
```

Evaluation and held-out inference require `best.pt` and `splits_80_10_10.npz` from the same run and the same dataset file ordering. Keep the relative paths fixed because the split uses positional indices. Custom inference only needs `best.pt` and local images. Each training run should use its own writable output directory.

## Docker Usage

Use Docker with a Linux engine. These PowerShell examples run from the repository root and target Linux AMD64. The image uses Python 3.10 on Debian Bookworm slim, PyTorch 2.5.1 and torchvision 0.20.1 from the official CPU wheel index, and pinned versions of the other direct dependencies. [PyTorch CPU installation reference](https://pytorch.org/get-started/previous-versions/#v251)

### Build and smoke test

```powershell
docker build --platform linux/amd64 -t eurosat-resnet18:cpu .
docker run --rm eurosat-resnet18:cpu python -c "import torch, torchvision; from model import resnet18_cifar; torch.set_num_threads(1); assert torch.version.cuda is None; m = resnet18_cifar(10).eval(); x = torch.zeros(2, 3, 64, 64); y = m(x); assert y.shape == (2, 10) and torch.isfinite(y).all(); print('CPU forward pass OK:', y.shape)"
```

The smoke test runs a forward pass on synthetic images. It needs no bind mounts.

### Train and evaluate

Set `$dataset` to the RGB EuroSAT root with one subdirectory per class. Docker mounts it read-only. The writable run directory holds the checkpoint, split indices, logs, and evaluation artifacts.

```powershell
$dataset = (Resolve-Path -LiteralPath './data/EuroSAT').Path
$run = (New-Item -ItemType Directory -Force -Path './outputs/docker-baseline').FullName
docker run --rm --mount "type=bind,source=$dataset,target=/dataset,readonly" --mount "type=bind,source=$run,target=/run-output" eurosat-resnet18:cpu python train.py --data_dir /dataset --epochs 10 --workers 0 --out_dir /run-output
docker run --rm --mount "type=bind,source=$dataset,target=/dataset,readonly" --mount "type=bind,source=$run,target=/run-output" eurosat-resnet18:cpu python evaluate.py --data_root /dataset --workers 0 --out_dir /run-output
```

Evaluation and held-out inference require `best.pt` and `splits_80_10_10.npz` from the same run, with the original dataset file ordering. To use an existing run, set `$run` to its absolute directory path.

### Run inference

Use the same dataset and run mounts for held-out inference. For custom images, mount the image directory read-only. Custom inference needs `best.pt` and writes its CSV into the mounted run directory.

```powershell
docker run --rm --mount "type=bind,source=$dataset,target=/dataset,readonly" --mount "type=bind,source=$run,target=/run-output" eurosat-resnet18:cpu python infer.py --data_root /dataset --out_dir /run-output
$images = (Resolve-Path -LiteralPath './examples').Path
docker run --rm --mount "type=bind,source=$images,target=/images,readonly" --mount "type=bind,source=$run,target=/run-output" eurosat-resnet18:cpu python infer.py --images_glob '/images/*.jpg' --out_dir /run-output
```

## Reproducibility limits

- The archived training, validation, and test subsets shared one dataset object. Setting the validation or test transform replaced the training transform with `ToTensor()`. The current pipeline uses separate dataset instances and supports `--no-augmentation` for a tensor-only control. The exact code revision and effective transforms behind each reported score remain unverified.
- The archive contains neither the Optuna driver nor trial records. The report discusses pretraining, while the available training code initializes from scratch. Historical weight initialization and per-run checkpoint provenance still need confirmation.
- Image-level stratification preserves class proportions. Geographic separation, performance on new regions, and variability across repeated seeds remain unverified. The workflow uses RGB imagery.

## Credits and permissions

This work originated as a Bilkent University CS-484 team project. **Mehmet Paşa** maintains this repository. Historical results come from the team's final report, *Landmark Classification Using ResNet-18 CIFAR Model*.

- Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth developed EuroSAT. [EuroSAT paper](https://arxiv.org/abs/1709.00029)
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun introduced residual learning for image recognition. [ResNet paper](https://arxiv.org/abs/1512.03385)
- Takuya Akiba and collaborators introduced Optuna. [Optuna paper](https://arxiv.org/abs/1907.10902)
- The course report cites J. R. Terven’s EuroSAT comparison. [Referenced benchmark repository](https://github.com/jrterven/eurosat_classification)

This repository has no project license. Confirm publication and redistribution permissions with the collaborators and any applicable course or institutional policy before sharing the code, report, figures, or checkpoints. EuroSAT imagery has separate upstream terms. [EuroSAT license and Sentinel data terms](https://github.com/phelber/EuroSAT#license)
