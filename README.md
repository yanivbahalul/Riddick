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

## Dataset

The images in `data/` are from the **LOL (LOw-Light) Dataset**, introduced in the paper
["Deep Retinex Decomposition for Low-Light Enhancement"](https://arxiv.org/abs/1808.04560) (Chen Wei et al., BMVC 2018).

Downloaded from: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

- `data/low/` + `data/high/` — 485 training pairs (`our485` in the original dataset)
- `data/eval15_low/` + `data/eval15_high/` — 15 held-out test pairs (`eval15` in the original dataset), not used for training

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
