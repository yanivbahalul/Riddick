# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-16 21:40

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 10.0 --tv-weight 20.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight20"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight20
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight20/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=12.80  SSIM=0.6515
111.png: PSNR=14.47  SSIM=0.6305
146.png: PSNR=16.75  SSIM=0.7291
179.png: PSNR=13.86  SSIM=0.5788
22.png: PSNR=14.10  SSIM=0.6362
23.png: PSNR=10.27  SSIM=0.4241
493.png: PSNR=19.29  SSIM=0.5939
547.png: PSNR=19.69  SSIM=0.5753
55.png: PSNR=9.81  SSIM=0.4170
665.png: PSNR=14.15  SSIM=0.4242
669.png: PSNR=16.92  SSIM=0.5521
748.png: PSNR=20.30  SSIM=0.6097
778.png: PSNR=16.52  SSIM=0.5547
780.png: PSNR=19.34  SSIM=0.4913
79.png: PSNR=13.43  SSIM=0.6100

Average over 15 images: PSNR=15.45  SSIM=0.5652
