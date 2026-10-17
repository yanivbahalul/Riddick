# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-17 07:27

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 5.0 --tv-weight 40.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color5"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color5
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color5/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=19.41  SSIM=0.7683
111.png: PSNR=15.86  SSIM=0.7056
146.png: PSNR=22.28  SSIM=0.7849
179.png: PSNR=15.23  SSIM=0.6324
22.png: PSNR=17.18  SSIM=0.7125
23.png: PSNR=9.15  SSIM=0.3960
493.png: PSNR=18.87  SSIM=0.6154
547.png: PSNR=24.39  SSIM=0.6456
55.png: PSNR=8.81  SSIM=0.3882
665.png: PSNR=11.74  SSIM=0.4105
669.png: PSNR=17.94  SSIM=0.6336
748.png: PSNR=23.22  SSIM=0.6907
778.png: PSNR=15.38  SSIM=0.5333
780.png: PSNR=24.55  SSIM=0.5739
79.png: PSNR=15.89  SSIM=0.7331

Average over 15 images: PSNR=17.33  SSIM=0.6149
