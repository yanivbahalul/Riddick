# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-17 00:38

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 10.0 --tv-weight 160.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight160"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight160
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight160/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=15.75  SSIM=0.7422
111.png: PSNR=11.84  SSIM=0.6516
146.png: PSNR=15.90  SSIM=0.7513
179.png: PSNR=21.13  SSIM=0.7362
22.png: PSNR=12.94  SSIM=0.6493
23.png: PSNR=7.53  SSIM=0.3220
493.png: PSNR=16.02  SSIM=0.5380
547.png: PSNR=18.17  SSIM=0.6536
55.png: PSNR=7.30  SSIM=0.3148
665.png: PSNR=9.85  SSIM=0.3475
669.png: PSNR=13.40  SSIM=0.6053
748.png: PSNR=17.96  SSIM=0.6869
778.png: PSNR=13.09  SSIM=0.4378
780.png: PSNR=21.96  SSIM=0.6234
79.png: PSNR=11.59  SSIM=0.6801

Average over 15 images: PSNR=14.29  SSIM=0.5827
