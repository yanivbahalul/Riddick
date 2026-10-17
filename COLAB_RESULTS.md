# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-17 09:26

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 40.0 --tv-weight 40.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color40"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color40
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color40/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=17.37  SSIM=0.7104
111.png: PSNR=15.70  SSIM=0.6762
146.png: PSNR=20.43  SSIM=0.7458
179.png: PSNR=14.99  SSIM=0.5917
22.png: PSNR=16.52  SSIM=0.6830
23.png: PSNR=9.36  SSIM=0.3911
493.png: PSNR=17.61  SSIM=0.5712
547.png: PSNR=23.01  SSIM=0.6177
55.png: PSNR=8.97  SSIM=0.3839
665.png: PSNR=11.97  SSIM=0.3935
669.png: PSNR=17.32  SSIM=0.5891
748.png: PSNR=21.14  SSIM=0.6418
778.png: PSNR=14.97  SSIM=0.5163
780.png: PSNR=20.94  SSIM=0.5374
79.png: PSNR=15.76  SSIM=0.7048

Average over 15 images: PSNR=16.40  SSIM=0.5836
