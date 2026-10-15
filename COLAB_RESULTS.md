# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-15 10:23

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 10.0 --tv-weight 1.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_exptarget045"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_exptarget045
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_exptarget045/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=10.51  SSIM=0.4740
111.png: PSNR=13.10  SSIM=0.5007
146.png: PSNR=12.96  SSIM=0.6110
179.png: PSNR=12.66  SSIM=0.5150
22.png: PSNR=11.54  SSIM=0.4818
23.png: PSNR=10.52  SSIM=0.3488
493.png: PSNR=12.14  SSIM=0.3198
547.png: PSNR=15.26  SSIM=0.4439
55.png: PSNR=10.46  SSIM=0.3514
665.png: PSNR=14.46  SSIM=0.3337
669.png: PSNR=15.34  SSIM=0.4527
748.png: PSNR=15.74  SSIM=0.4771
778.png: PSNR=12.64  SSIM=0.4032
780.png: PSNR=11.74  SSIM=0.3323
79.png: PSNR=12.24  SSIM=0.5145

Average over 15 images: PSNR=12.75  SSIM=0.4373
