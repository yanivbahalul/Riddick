# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-15 15:49

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 10.0 --tv-weight 10.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight10"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight10
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tvweight10/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=11.02  SSIM=0.5344
111.png: PSNR=13.50  SSIM=0.5497
146.png: PSNR=14.55  SSIM=0.6668
179.png: PSNR=13.08  SSIM=0.5387
22.png: PSNR=12.79  SSIM=0.5480
23.png: PSNR=11.07  SSIM=0.4158
493.png: PSNR=16.37  SSIM=0.4598
547.png: PSNR=16.47  SSIM=0.4888
55.png: PSNR=10.80  SSIM=0.4148
665.png: PSNR=16.06  SSIM=0.3905
669.png: PSNR=16.20  SSIM=0.4958
748.png: PSNR=16.80  SSIM=0.5197
778.png: PSNR=16.42  SSIM=0.5194
780.png: PSNR=14.33  SSIM=0.3824
79.png: PSNR=12.43  SSIM=0.5419

Average over 15 images: PSNR=14.13  SSIM=0.4978
