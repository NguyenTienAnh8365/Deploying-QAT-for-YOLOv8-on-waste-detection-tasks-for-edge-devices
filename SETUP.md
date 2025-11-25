# Setup & Training Guide

This document provides instructions for setting up the environment, preparing datasets, downloading pretrained checkpoints, and training the YOLOv8-based Vietnam Trash Classification model. It covers all steps needed to reproduce results from this repository, including model compression, pruning, fine-tuning, knowledge distillation, and quantization.

---

## Table of Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Pretrained Checkpoints](#pretrained-checkpoints)
4. [Training](#training)
5. [Pruning](#pruning)
6. [Fine-Tuning](#fine-tuning)
7. [Knowledge Distillation](#knowledge-distillation)
8. [Quantization](#quantization)



## Installation <a id="installation"></a>

---

## Datasets preparation <a id="dataset-preparation"></a>

---

## Pretrained Checkpoints <a id="pretrained-checkpoints"></a>

---

## Training <a id="training"></a>

### 1. Standard Training

```bash
python train.py --data trash.yaml --weights yolov8n.pt --epochs 200 --batch 32 --project . --name train-normal
```

### 2. Sparsity Training

```bash
python train.py --data trash.yaml --weights runs/train-normal/weights/best.pt --epochs 50 --batch 32 --sr 0.01 --project . --name train-sparsity
```

### 3. Pruning <a id="fine-tuning"></a>

```bash
python prune.py --data trash.yaml --weights runs/train-sparsity/weights/last.pt --prune-ratio 0.6 --save-dir weights/
```

### 4. Fine-Tuning <a id="fine-tuning"></a>

```bash
python train.py --data trash.yaml --weights weights/pruned.pt --epochs 150 --batch 32 --project . --name train-finetune
```

### 5. Knowledge Distillation <a id="knowledge-distillation"></a>

```bash
python distill.py --teacher runs/train-normal/weights/best.pt --student weights/pruned.pt --data trash.yaml --epochs 120
```

### 6. Quantization <a id="quantization"></a>

```bash
yolo export model=weights/pruned.pt format=onnx device=0 imgsz=640 precision=fp16
```

--- 
## Evaluation / Inference
