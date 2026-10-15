# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-15 09:13

Command: python train.py --epochs 150 --hidden-channels 64 --loss ssim --augment --lr-schedule --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step8_lrschedule"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step8_lrschedule
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step8_lrschedule/model_epoch150.pth" --hidden-channels 64
Eval result:
1.png: PSNR=19.23  SSIM=0.8390
111.png: PSNR=22.86  SSIM=0.9014
146.png: PSNR=23.23  SSIM=0.8626
179.png: PSNR=12.23  SSIM=0.6290
22.png: PSNR=23.19  SSIM=0.8643
23.png: PSNR=13.10  SSIM=0.7550
493.png: PSNR=18.79  SSIM=0.5967
547.png: PSNR=18.59  SSIM=0.8146
55.png: PSNR=12.90  SSIM=0.7547
665.png: PSNR=21.20  SSIM=0.7647
669.png: PSNR=25.79  SSIM=0.8698
748.png: PSNR=18.74  SSIM=0.8205
778.png: PSNR=21.60  SSIM=0.7258
780.png: PSNR=16.27  SSIM=0.6688
79.png: PSNR=22.81  SSIM=0.8934

Average over 15 images: PSNR=19.37  SSIM=0.7840
