# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-14 21:48

Command: python train_unsupervised.py --epochs 50 --lr 1e-5 --data-dir data/real_night --init-checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_colorweight10/model_epoch150.pth" --spatial-weight 1.0 --color-weight 10.0 --tv-weight 1.0 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step16_finetuned"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step16_finetuned
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step16_finetuned/model_epoch50.pth" --curvenet
Eval result:
1.png: PSNR=10.29  SSIM=0.4828
111.png: PSNR=15.26  SSIM=0.5247
146.png: PSNR=13.91  SSIM=0.6373
179.png: PSNR=9.20  SSIM=0.4544
22.png: PSNR=12.43  SSIM=0.5003
23.png: PSNR=12.64  SSIM=0.3702
493.png: PSNR=10.04  SSIM=0.2972
547.png: PSNR=11.81  SSIM=0.4065
55.png: PSNR=12.85  SSIM=0.3778
665.png: PSNR=14.18  SSIM=0.3319
669.png: PSNR=15.35  SSIM=0.4538
748.png: PSNR=11.30  SSIM=0.4269
778.png: PSNR=11.30  SSIM=0.3828
780.png: PSNR=9.53  SSIM=0.2939
79.png: PSNR=15.54  SSIM=0.5429

Average over 15 images: PSNR=12.38  SSIM=0.4322
