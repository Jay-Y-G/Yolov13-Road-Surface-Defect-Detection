# Cloud-ready project copy

This folder contains the YOLOv13 source, road dataset, pretrained nano weights, and a deterministic 80/10/10 train/validation/test split.

## Dataset

The `data.yaml` points to `datasets/VOCdevkit` and uses the YOLO labels paired with the images. `data/` preserves the supplied source images, labels, and VOC XML annotations.

## Training

Run `python train.py` from this folder after installing the dependencies in `requirements.txt` and the PyTorch build appropriate for the cloud GPU. The script saves outputs under `runs/`, which is excluded by `.gitignore`.

The source project includes `trained_weights/best.pt` from the supplied training run for reference. The training script starts from `yolov13n.pt`.

Large local outputs, videos, virtual environments, and Python caches are excluded from Git.
