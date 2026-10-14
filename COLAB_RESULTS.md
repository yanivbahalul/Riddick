# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-14 10:10

Command: python train_unsupervised.py --epochs 150 --spatial-weight 5.0 --color-weight 1.0 --tv-weight 1.0 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_spatialweight5"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_spatialweight5
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_spatialweight5/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=9.98  SSIM=0.4621
111.png: PSNR=15.10  SSIM=0.5062
146.png: PSNR=13.48  SSIM=0.6184
179.png: PSNR=9.05  SSIM=0.4428
22.png: PSNR=11.70  SSIM=0.4787
23.png: PSNR=11.40  SSIM=0.3396
493.png: PSNR=8.40  SSIM=0.2314
547.png: PSNR=11.55  SSIM=0.3902
55.png: PSNR=11.70  SSIM=0.3359
665.png: PSNR=12.11  SSIM=0.2985
669.png: PSNR=14.71  SSIM=0.4326
748.png: PSNR=10.93  SSIM=0.4029
778.png: PSNR=9.92  SSIM=0.3358
780.png: PSNR=8.24  SSIM=0.2634
79.png: PSNR=15.39  SSIM=0.5311

Average over 15 images: PSNR=11.58  SSIM=0.4046
