# Colab run results

Transient handoff file — `colab_train.ipynb`'s last cell appends here and pushes.
Claude reads new entries, writes them into `PROGRESS.md` properly, then clears this file.

## Colab run — 2026-10-14 22:04

Command: python train_unsupervised.py --epochs 500 --lr 1e-5 --data-dir data/real_night --init-checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step14_colorweight10/model_epoch150.pth" --spatial-weight 1.0 --color-weight 10.0 --tv-weight 1.0 --checkpoint-dir "/content/drive/MyDrive/Riddick_checkpoints/models_step16_finetuned_500ep"
Checkpoint dir (on Drive, not in git): /content/drive/MyDrive/Riddick_checkpoints/models_step16_finetuned_500ep
Eval command: python evaluate.py --checkpoint "/content/drive/MyDrive/Riddick_checkpoints/models_step16_finetuned_500ep/model_epoch500.pth" --curvenet
Eval result:
1.png: PSNR=10.31  SSIM=0.4882
111.png: PSNR=15.40  SSIM=0.5263
146.png: PSNR=13.80  SSIM=0.6355
179.png: PSNR=9.09  SSIM=0.4507
22.png: PSNR=12.81  SSIM=0.5079
23.png: PSNR=13.04  SSIM=0.3842
493.png: PSNR=10.36  SSIM=0.3149
547.png: PSNR=11.88  SSIM=0.4086
55.png: PSNR=13.08  SSIM=0.3931
665.png: PSNR=14.74  SSIM=0.3452
669.png: PSNR=15.50  SSIM=0.4587
748.png: PSNR=11.41  SSIM=0.4323
778.png: PSNR=11.77  SSIM=0.3930
780.png: PSNR=9.56  SSIM=0.2931
79.png: PSNR=15.57  SSIM=0.5451

Average over 15 images: PSNR=12.55  SSIM=0.4385
