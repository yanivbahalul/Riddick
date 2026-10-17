# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-17 08:26

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 20.0 --tv-weight 40.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color20"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color20
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color20/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=17.99  SSIM=0.7279
111.png: PSNR=16.15  SSIM=0.6889
146.png: PSNR=21.05  SSIM=0.7619
179.png: PSNR=14.58  SSIM=0.6003
22.png: PSNR=17.02  SSIM=0.6966
23.png: PSNR=9.51  SSIM=0.3982
493.png: PSNR=18.08  SSIM=0.5865
547.png: PSNR=23.27  SSIM=0.6243
55.png: PSNR=9.10  SSIM=0.3909
665.png: PSNR=12.07  SSIM=0.4032
669.png: PSNR=17.84  SSIM=0.6012
748.png: PSNR=21.40  SSIM=0.6509
778.png: PSNR=15.25  SSIM=0.5246
780.png: PSNR=21.65  SSIM=0.5454
79.png: PSNR=16.26  SSIM=0.7181

Average over 15 images: PSNR=16.75  SSIM=0.5946
