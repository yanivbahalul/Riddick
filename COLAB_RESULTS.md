# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-15 12:52

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 10.0 --tv-weight 5.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight5"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight5
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight5/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=10.74  SSIM=0.5042
111.png: PSNR=13.27  SSIM=0.5314
146.png: PSNR=13.59  SSIM=0.6397
179.png: PSNR=13.15  SSIM=0.5354
22.png: PSNR=11.87  SSIM=0.5083
23.png: PSNR=10.89  SSIM=0.3789
493.png: PSNR=13.13  SSIM=0.3610
547.png: PSNR=15.53  SSIM=0.4640
55.png: PSNR=10.79  SSIM=0.3861
665.png: PSNR=15.08  SSIM=0.3555
669.png: PSNR=15.78  SSIM=0.4773
748.png: PSNR=15.83  SSIM=0.5010
778.png: PSNR=13.72  SSIM=0.4418
780.png: PSNR=12.36  SSIM=0.3513
79.png: PSNR=12.38  SSIM=0.5350

Average over 15 images: PSNR=13.21  SSIM=0.4647
