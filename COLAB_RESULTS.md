# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-14 17:07

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 10.0 --tv-weight 1.0 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_colorweight10"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_colorweight10
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_colorweight10/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=10.05  SSIM=0.4733
111.png: PSNR=15.00  SSIM=0.5179
146.png: PSNR=13.53  SSIM=0.6312
179.png: PSNR=9.03  SSIM=0.4470
22.png: PSNR=12.23  SSIM=0.4952
23.png: PSNR=12.53  SSIM=0.3659
493.png: PSNR=9.82  SSIM=0.2859
547.png: PSNR=11.69  SSIM=0.4020
55.png: PSNR=12.75  SSIM=0.3714
665.png: PSNR=13.81  SSIM=0.3254
669.png: PSNR=15.18  SSIM=0.4478
748.png: PSNR=11.12  SSIM=0.4198
778.png: PSNR=11.07  SSIM=0.3756
780.png: PSNR=9.44  SSIM=0.2912
79.png: PSNR=15.40  SSIM=0.5395

Average over 15 images: PSNR=12.18  SSIM=0.4259
