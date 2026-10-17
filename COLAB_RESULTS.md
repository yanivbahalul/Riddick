# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-17 10:25

Command: python train_unsupervised.py --epochs 150 --spatial-weight 1.0 --color-weight 80.0 --tv-weight 40.0 --exposure-target 0.45 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color80"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color80
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_tv40_color80/model_epoch150.pth" --curvenet
Eval result:
1.png: PSNR=16.43  SSIM=0.7095
111.png: PSNR=14.49  SSIM=0.6789
146.png: PSNR=18.86  SSIM=0.7522
179.png: PSNR=15.27  SSIM=0.6096
22.png: PSNR=15.42  SSIM=0.6812
23.png: PSNR=8.71  SSIM=0.3692
493.png: PSNR=16.67  SSIM=0.5477
547.png: PSNR=22.17  SSIM=0.6372
55.png: PSNR=8.35  SSIM=0.3606
665.png: PSNR=10.90  SSIM=0.3691
669.png: PSNR=15.95  SSIM=0.5924
748.png: PSNR=20.16  SSIM=0.6549
778.png: PSNR=13.92  SSIM=0.4784
780.png: PSNR=20.62  SSIM=0.5568
79.png: PSNR=14.45  SSIM=0.7108

Average over 15 images: PSNR=15.49  SSIM=0.5806
