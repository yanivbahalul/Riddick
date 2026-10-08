# Low-Light Enhancement (PyTorch)

A PyTorch project for enhancing low-light images.

## Project Structure

```
low-light-enhancement-pytorch/
│
├── data/
├── models/
├── outputs/
├── train.py
├── inference.py
├── dataset.py
├── requirements.txt
└── README.md
```

- `data/` — training data. Expected layout: `data/low/` and `data/high/` with matching filenames.
- `models/` — saved model checkpoints.
- `outputs/` — enhanced images produced by `inference.py`.
- `train.py` — trains the enhancement model.
- `inference.py` — runs a trained model on new images.
- `dataset.py` — dataset loading utilities.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data-dir data --checkpoint-dir models --epochs 50
```

## Inference

```bash
python inference.py --input path/to/image_or_dir --checkpoint models/model_epoch50.pth --output-dir outputs
```
