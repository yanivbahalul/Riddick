"""Regenerate progress_chart.png from the results in PROGRESS.md.
Update RUNS below after adding a new row to the evaluation table, then run:
    python chart.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, psnr, ssim) -- keep in sync with the table in PROGRESS.md
RUNS = [
    ("Baseline\n(Step 0)", 17.82, 0.7322),
    ("+epochs\n(Step 1)", 18.38, 0.7573),
    ("+LR\n(Step 2)", 18.54, 0.7442),
    ("+width\n(Step 3)", 18.26, 0.7474),
    ("+depth\n(Step 4)", 18.03, 0.7379),
    ("+BatchNorm\n(Step 5)", 17.77, 0.6822),
    ("+SSIM loss\n(Step 6)", 18.95, 0.7671),
]

PSNR_TARGET = 20
SSIM_TARGET = 0.8

labels = [r[0] for r in RUNS]
psnr = [r[1] for r in RUNS]
ssim = [r[2] for r in RUNS]
x = range(len(RUNS))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.3))

colors = ["#444444"] + ["#2a9d8f" if v >= psnr[0] else "#888888" for v in psnr[1:]]
ax1.bar(x, psnr, color=colors)
ax1.axhline(psnr[0], color="black", linestyle="--", linewidth=1, label="Baseline")
ax1.axhline(PSNR_TARGET, color="red", linestyle=":", linewidth=1, label=f"Target ({PSNR_TARGET})")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("PSNR")
ax1.set_title("PSNR across experiments")
ax1.legend(fontsize=8)
psnr_lo = min(psnr + [PSNR_TARGET]) - 1
psnr_hi = max(psnr + [PSNR_TARGET]) + 1
ax1.set_ylim(psnr_lo, psnr_hi)

colors2 = ["#444444"] + ["#2a9d8f" if v >= ssim[0] else "#888888" for v in ssim[1:]]
ax2.bar(x, ssim, color=colors2)
ax2.axhline(ssim[0], color="black", linestyle="--", linewidth=1, label="Baseline")
ax2.axhline(SSIM_TARGET, color="red", linestyle=":", linewidth=1, label=f"Target ({SSIM_TARGET})")
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("SSIM")
ax2.set_title("SSIM across experiments")
ax2.legend(fontsize=8)
ssim_lo = min(ssim + [SSIM_TARGET]) - 0.03
ssim_hi = max(ssim + [SSIM_TARGET]) + 0.03
ax2.set_ylim(ssim_lo, ssim_hi)

plt.tight_layout()
plt.savefig("progress_chart.png", dpi=120)
print("saved progress_chart.png")
