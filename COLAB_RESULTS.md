# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-16 20:10

Command: python train.py --epochs 9 --hidden-channels 64 --loss ssim --augment --unet --init-checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_unet_hc64/model_epoch91.pth" --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_unet_hc64_resumed"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_unet_hc64_resumed
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_unet_hc64_resumed/model_epoch9.pth" --unet --hidden-channels 64
Eval result:
1.png: PSNR=18.47  SSIM=0.8287
111.png: PSNR=23.57  SSIM=0.9012
146.png: PSNR=22.42  SSIM=0.8500
179.png: PSNR=11.93  SSIM=0.6207
22.png: PSNR=20.20  SSIM=0.8410
23.png: PSNR=15.14  SSIM=0.7928
493.png: PSNR=17.65  SSIM=0.6240
547.png: PSNR=15.97  SSIM=0.7746
55.png: PSNR=15.76  SSIM=0.8217
665.png: PSNR=21.78  SSIM=0.7820
669.png: PSNR=23.27  SSIM=0.8424
748.png: PSNR=16.34  SSIM=0.7878
778.png: PSNR=19.81  SSIM=0.7502
780.png: PSNR=15.69  SSIM=0.6587
79.png: PSNR=23.24  SSIM=0.8970

Average over 15 images: PSNR=18.75  SSIM=0.7849
