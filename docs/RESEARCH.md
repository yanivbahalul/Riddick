# Research notes — toward a real-time night camera

Literature and hardware survey done before scoping Phase 3, with a
specific end goal in mind: **a camera that sits outdoors at night and
brightens the scene in real time**, with full control over how many frames
to capture and what to do with them.

This is a different system from what the project has built so far (a
single-image enhancer). The findings below are organized around what
actually changes when you control the capture.

---

## 1. Is real-time achievable? Yes — but not by speeding up our current design

The 30fps budget is **33ms per frame**. Published results comfortably fit:

| Approach | Resolution | Time |
|---|---|---|
| Mobile GPU (TFLite, ARM writeup) | 512x512 | 11 ms |
| LLF-LUT++ (Laplacian pyramid + LUT) | 4K | 13 ms |
| 4K exposure correction, ~8K params | 4K | ~8 ms (125 FPS) |
| Deep Bilateral Learning (HDRnet) | HD | <20 ms |

**The thing that makes these fast is architectural, not hardware.** They
do *not* run a conv net at full resolution. They run a small network on a
**downsampled** image to predict a *transformation* — a bilateral grid, a
3D LUT, or curve coefficients — and then apply that transformation to the
full-resolution image, which is cheap.

**This matters directly for our code.** `CurveNet` is already in the right
*family* (it predicts curve coefficients, not pixels — same idea as
Zero-DCE). But it predicts them at **full resolution**, which is the
expensive part and is mostly wasted: illumination varies smoothly across a
scene, so the coefficient maps don't need per-pixel resolution. Predicting
them at 1/8 scale and upsampling is the single change that would move us
from "runs in seconds" to "runs in milliseconds", and the bilateral-grid
literature suggests the quality cost is small.

Relevant: [Deep Bilateral Learning](https://arxiv.org/pdf/1707.02880),
[LLF-LUT++](https://arxiv.org/abs/2510.11613),
[4K exposure correction @125FPS](https://arxiv.org/pdf/2311.08759),
[ARM on real-time mobile low-light video](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/real-time-low-light-video-neural-networks-mobile)

---

## 2. The biggest win available, and why our use case is better than a phone's

Google's Night Sight / HDR+ is the reference design, and it is **not**
primarily a neural enhancer. It captures **up to 15 short exposures**,
aligns them with sub-pixel precision (FFT-based alignment), and merges
them with a hybrid 2D/3D Wiener filter — **on raw Bayer data**, before
tone mapping. Averaging N frames reduces noise by roughly **sqrt(N)**.
That is real additional signal, not a guess.

**The insight specific to this project:** a phone is limited to ~1-3
seconds of capture because a human is holding it — hand shake sets the
ceiling on how many frames can be usefully aligned.

**A camera on a tripod has no such limit.** With a static mount:
- Alignment becomes nearly trivial (no parallax, no hand shake)
- You can integrate **far more frames** — 30, 60, 100 — rather than 15
- sqrt(100) = **10x noise reduction**, versus a phone's sqrt(15) ≈ 3.9x

This is an area where a cheap static camera can genuinely **beat a
flagship phone's night mode**, purely because it isn't fighting motion.
It costs nothing but capture time and is classical signal processing —
no training required for a first version.

Relevant: [Night Sight](https://research.google/blog/night-sight-seeing-in-the-dark-on-pixel-phones/),
[HDR+ open implementation (IPOL)](https://www.ipol.im/pub/art/2021/336/article_lr.pdf),
[Mobile Computational Photography: A Tour](https://arxiv.org/pdf/2102.09000)

---

## 3. RAW vs 8-bit sRGB — a ceiling we're currently sitting under

Confirmed across the literature: RAW is better for low light because it is
**linear with scene light**, has **12-14 bits** instead of 8, and preserves
**unaltered sensor noise** (which makes it *modellable*).

Two consequences for us:

**a) Information is being destroyed before we ever see it.** In a dark
8-bit JPEG, the shadows are already quantized to near-zero. Brightening
can't recover what quantization deleted. This is a hard ceiling that no
model change fixes.

**b) We are brightening in the wrong space.** sRGB is gamma-encoded
(roughly `V_srgb = V_linear^(1/2.2)`), so file values are *not*
proportional to actual light. Doubling the light means doubling the
*linear* value; applying a curve directly to sRGB values gives non-uniform
gain across the tonal range and **shifts colors**.

This connects to a symptom documented repeatedly in `VISUAL_PROGRESS.md`:
washed-out color and a blue/beige cast. Converting to linear, enhancing,
then converting back is cheap and physically correct, and hands the model
the gamma relationship for free instead of making it learn one implicitly.

Relevant: [Retinex-RAWMamba](https://arxiv.org/pdf/2409.07040),
[CIE XYZ Net (unprocessing)](https://arxiv.org/pdf/2006.12709)

---

## 4. Noise should be modelled, not fought

Sensor noise follows a **Poisson-Gaussian** model: shot noise that scales
with signal, plus read noise fixed per ISO. Two established techniques
follow from that:

- **Train with realistic synthetic noise.** LOL is comparatively clean;
  real night footage is not. Adding Poisson-Gaussian noise to training
  images teaches the model to suppress noise instead of amplifying it.
  This directly addresses the Step 16 breakdown on real photos.
- **Condition on noise level (FFDNet / CBDNet style).** If ISO is known,
  the expected noise level is known — feed it in as an extra input
  channel rather than making the network infer it. Noise-aware denoisers
  consistently beat blind ones.

**Practical blocker found:** the existing `data/real_night/` photos have
**no EXIF at all** (stripped in transfer). Any metadata-driven work needs
footage transferred losslessly — AirDrop original, not messaging apps.

---

## 5. Real paired video datasets exist — we don't have to invent them

Phase 3 doesn't need a self-made dataset to start:

| Dataset | What it is | Why it matters here |
|---|---|---|
| **SMID / DRV** | RAW short/long exposure pairs, static scenes | RAW + static = closest match to a tripod night camera |
| **SDSD** | Dynamic scenes, mechatronically aligned pairs | Real motion with real ground truth |
| **OLiVES** | **Outdoor** low-light video benchmark | Matches the actual target domain (LOL does not) |
| **BVI-RLV** | Fully registered low-light video | Alternative registered benchmark |

`OLiVES` is especially relevant — our entire domain-gap problem is that
LOL is indoor studio footage.

Relevant: [SDSD](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Seeing_Dynamic_Scene_in_the_Dark_A_High-Quality_Video_Dataset_ICCV_2021_paper.pdf),
[OLiVES](https://arxiv.org/html/2610.20501v1),
[Spatial-temporal decomposition for LLVE (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0238.pdf)

---

## 6. Hardware: what a real build costs

### Compute

| Option | Perf | Power | Unit cost |
|---|---|---|---|
| Existing M4 Mac (dev only) | — | — | $0 |
| **Jetson Orin Nano 8GB** | 40 TOPS INT8 | 7-15W | ~$250 (~$300-600 with carrier/SSD/PSU) |
| Jetson Orin NX | 100 TOPS INT8 | 10-25W | ~$700 |
| Jetson AGX Orin | highest | higher | ~$2,000 |

For a single camera stream with an efficient architecture, **Orin Nano is
sufficient** — reference workloads run several 1080p streams in real time
with headroom inside the 33ms budget.

### Sensor — this matters more than the compute

Sony **STARVIS 2** is the relevant family (back-illuminated CMOS built for
low light): the **IMX678** (8.3MP, 4K) outputs usable color at **0.01 lux
or lower**. Available as USB/MIPI modules (Arducam, e-con Systems),
roughly **$150-350**.

A good sensor beats a good model: 0.01-lux-capable silicon starts you with
signal that no amount of post-processing can synthesize from a worse
sensor.

### Not worth it: event cameras

140 dB dynamic range (vs ~60 dB conventional) and excellent low-light
behavior, but **thousands of dollars per unit**. Technically fascinating,
economically out of scope. By comparison "starlight" CMOS cameras run
$110-430 for 1080p.

### Realistic build

```
Jetson Orin Nano 8GB + carrier         ~$300-400
STARVIS 2 IMX678 camera module         ~$200-300
Lens (fast, low f-number)              ~$50-150
Enclosure / mount / PSU                ~$100
---------------------------------------------------
Total                                  ~$650-950
```

Relevant: [Jetson Orin edge deployment costs](https://clusterbid.com/blog/ai-inference-edge-jetson-embedded),
[IMX678 module](https://www.e-consystems.com/camera-modules/4k-sony-starvis2-imx678-low-light-camera-module.asp),
[event camera survey](https://arxiv.org/pdf/2408.13627)

---

## 7. The real economic constraints

Hardware is **not** the bottleneck — under $1,000 buys a capable rig. The
actual constraints:

1. **RAW video data rate.** 4K 12-bit RAW at 30fps is on the order of
   ~1.5 Gbit/s uncompressed. You cannot casually record it; it has to be
   processed on the fly or heavily buffered. This, not compute, is what
   makes RAW video pipelines hard.
2. **Training compute for video models.** Free Colab is already a
   constraint for image models; temporal models on video are heavier.
   Rented A100 time is roughly $1-2/hr — manageable in bursts, not for
   open-ended experimentation.
3. **Capturing paired night video is genuinely expensive.** SDSD needed
   *mechatronic* alignment hardware. Use the public datasets instead of
   trying to make one.
4. **Engineering time is the dominant cost.** An HDR+-class pipeline is
   many well-understood stages; the cost is building and tuning them, not
   buying anything.

---

## 8. Recommended architecture — what the system should actually be

The current design is one-shot image→image. The system described (a
camera, with control over frame count and processing) should instead be an
**ISP-like pipeline**, with the learned model as *one stage*:

```
capture N raw frames (static mount -> N can be large)
        |
   align  (FFT / phase correlation; near-trivial on a tripod)
        |
   merge  (average or Wiener -> SNR improves ~sqrt(N))
        |
  denoise (noise model known from ISO)
        |
  enhance (low-res coefficient prediction, applied in LINEAR space)
        |
 tone map (local, not a single global curve)
        |
      output
```

Each stage is independently measurable — which fits this project's
one-change-at-a-time method exactly.

### Priority order (value per effort)

| # | Change | Effort | Why |
|---|---|---|---|
| 1 | **Multi-frame capture + merge** | days | Only step that adds *real* signal; amplified by static mount |
| 2 | **Work in linear space** | hours | Free, physically correct, matches an observed symptom |
| 3 | **Low-res coefficient prediction** | days | The real-time enabler |
| 4 | **Poisson-Gaussian noise in training** | hours | Closes the Step 16 domain gap |
| 5 | **Switch to a real video dataset (OLiVES/SDSD)** | days | LOL will never teach outdoor night |
| 6 | **ISO conditioning** | days | Needs metadata-preserving capture first |
| 7 | **Retinex decomposition** | days+ | For streetlight-next-to-black dynamic range |

**(2) is the cheapest real experiment available right now** and testable
without any new hardware or data.

---

## 9. Original analysis — derivations done on this project's own data

Everything above is survey. This section is analysis done here, with the
numbers measured on this repo's data rather than taken from a paper.

### 9.1 Hypothesis that was wrong: "the exposure loss destroys contrast"

`exposure_loss` is `mean_patches |mean(patch) - T|`. Its global minimum is
reached when *every* patch mean equals `T` — i.e. when the inter-patch
brightness variance is **zero**, a flat image at large scale. That seemed
like a serious design flaw worth proving.

Measured (16x16 patches, 256x256, luma):

| | patch mean | Var | Std |
|---|---|---|---|
| Real well-exposed (`data/high`) | 0.4874 | 0.02378 | 0.1542 |
| Real dark (`data/low`) | 0.0566 | 0.00140 | 0.0375 |
| **Our model's output** | 0.3408 | 0.02031 | 0.1425 |

**The hypothesis is refuted.** Output variance (0.0203) is close to real
well-exposed photos (0.0238) — the other losses (spatial consistency,
smoothness) evidently hold structure in place against the flattening
pressure. Recording this because a plausible-sounding theoretical
objection turned out not to matter in practice.

**But the measurement found something else:** the model lands at patch
mean **0.3408** when the target is **0.45** — a systematic ~24% shortfall,
and further still from real photos' 0.4874. With `tv-weight=40` the
smoothness term dominates the objective, so the exposure term never gets
fully satisfied. Worth knowing that the "exposure target" is not actually
being hit.

### 9.2 Confirmed: the exposure loss measures brightness in the wrong space

sRGB is gamma-encoded, `s = L^(1/γ)` with γ≈2.2. Since `x^(1/2.2)` is
**concave**, Jensen's inequality gives, for any patch:

```
mean(s) = mean(L^(1/γ))  <=  (mean L)^(1/γ)
```

So **the sRGB patch mean understates the patch's true average light**, and
the gap grows with the variance of light inside the patch. Measured as
`mean(s) - (mean L)^(1/γ)`, per patch:

| Data | overall bias | low-contrast patches | high-contrast patches |
|---|---|---|---|
| LOL dark (`data/low`) | −0.0028 | −0.0002 | −0.0085 |
| LOL bright (`data/high`) | −0.0091 | −0.0001 | −0.0282 |
| **Real night photos** | **−0.0162** | −0.0018 | **−0.0395** |

Two things follow, and both match previously observed behaviour:

1. **Direction of the error.** The loss reads a high-contrast patch
   (streetlight next to shadow) as *darker* than it really is, so it
   pushes to brighten it **more** than it should. That is the
   blown-out-to-near-white failure recorded in Step 16's round-5 LR sweep.
2. **Why it breaks specifically on real photos.** The bias on real night
   footage is **~5.8x larger** than on the LOL images the weights were
   tuned against (−0.0162 vs −0.0028). Every hyperparameter in this
   project was chosen in a regime where this error is small, then deployed
   where it is large. This is a concrete, measured mechanism for the
   domain gap — not just "real photos are different".

**Fix (implemented as `--linear-exposure`):** convert to linear light,
average there, convert back. This removes the bias by construction. The
form `(mean(s^γ))^(1/γ)` was chosen over a linear-space target so the loss
keeps the same numeric range, leaving the existing weight balance
(tv=40, color=1) roughly valid.

**The target has to move with it.** Measured on `data/high` (n=300):
sRGB-space patch mean **0.4701** (the current 0.45 target is right for the
old measurement), linear-corrected **0.4819**. So `--linear-exposure`
should pair with `--exposure-target 0.48`, not 0.45.

**Experiment running:** same recipe, `--seed 1`, so it is directly
comparable to `models_seedcheck_s1` with random initialization held fixed
— a cleaner comparison than any made so far in this project, all of which
were single unseeded runs. Note that two values change together
(measurement space and target), but they are not independent: both encode
"match well-exposed photos under this measurement".

### 9.3 Derivation: how many frames are worth merging?

Averaging N frames reduces shot and read noise as `1/N` in variance, but
**fixed-pattern noise (FPN/PRNU) is systematic and does not average out**:

```
sigma_total(N) = sqrt( (sigma_shot^2 + sigma_read^2)/N + sigma_FPN^2 )
```

Returns diminish once the first term falls below the FPN floor:

```
N* = (sigma_shot^2 + sigma_read^2) / sigma_FPN^2
```

For a dark scene (~40 e- per pixel per frame, read noise 2 e-):

| FPN as % of signal | N* | time @30fps |
|---|---|---|
| 0.3% | ~3056 frames | 102 s |
| 0.5% | ~1100 frames | 37 s |
| 1.0% | ~275 frames | 9 s |

**Conclusion: FPN is not the binding constraint at any practical N.** In
the useful range (8-64 frames) shot noise still dominates, so averaging
keeps paying off at the full `sqrt(N)` rate.

The real constraint is latency, and it is mild:

| N | SNR gain | latency @30fps |
|---|---|---|
| 8 | 2.83x | 0.27 s |
| 16 | 4.00x | 0.53 s |
| 32 | 5.66x | 1.07 s |
| 64 | 8.00x | 2.13 s |

A rolling 8-frame window costs 0.27 s of latency and buys ~2.8x noise
reduction — roughly **1.5 stops of exposure, for free**. For a static
camera producing 30fps output this is very cheap. (Moving objects ghost
under a rolling average, so motion masking is required — the standard
approach, and easier here than on a phone because the camera itself is
still.)

### 9.4 What the experiments actually found — including the part I got wrong

The Jensen analysis in §9.2 was correct about the bias existing. It was
**wrong about the bias mattering**, and chasing it surfaced something
roughly 14x larger that had been sitting unexamined.

**Noise floor first (Step 17).** Three runs of the identical recipe,
changing only `--seed`:

| | mean | std | range | ~95% band |
|---|---|---|---|---|
| PSNR | 18.247 | 0.035 | 0.070 | ±0.07 |
| SSIM | 0.6203 | 0.0002 | 0.0004 | ±0.0004 |

Much tighter than feared, which **vindicates most of the existing log** —
the +0.92, +0.73 and +0.57 PSNR jumps are 8-26σ, unambiguous. Even Run
29's +0.19 and the width retest's −0.16, both of which I expected to
dissolve into noise, are real.

**One caveat that did emerge:** Eyal's Colab run of this exact recipe
scored 18.06, while all three local seeds land 18.21-18.28 — 0.19 below,
far outside the 0.070 local range. So the ±0.07 floor holds *within a
machine*; comparisons **across** Colab and local carry roughly ±0.2. Parts
of the log mix the two, and those cross-machine comparisons are
correspondingly weaker than they look.

**The linear-exposure fix (Step 18) — prediction refuted.** On `eval15`:
18.39 vs the 18.247 baseline mean, about +4σ on PSNR but −0.002 on SSIM.
Then the test that actually mattered, on real night photos where the bias
is 5.8x larger — it predicted *fewer* blown highlights, and delivered
**more**: 4.60% vs 2.88%.

Two reasons, and the first is my own experimental error:

1. **The experiment was confounded.** Measurement space and target changed
   together (0.45 → 0.48). I flagged that in writing but never checked
   which dominated: the +0.03 target bump outweighs the ~0.016 bias
   correction, so the net effect was *more* brightening, in the opposite
   direction from the mechanism being tested.
2. **The effect size was never going to be visible.** §9.2's own numbers
   say the bias on LOL is −0.0028 — testing the fix on `eval15` is
   checking windscreen wipers on a dry day.

**The finding that came out of it.** Comparing target against achieved
brightness, the same model behaves in opposite directions on the two
distributions:

| | target | achieved | |
|---|---|---|---|
| LOL `eval15` | 0.45 | 0.381 | **−15%** |
| Real night | 0.45 | 0.677 | **+50%** |

The overshoot on real photos is **0.22 — about 14x the Jensen bias I was
chasing.** The cause is a plain covariate shift in input brightness:

| | input mean | range |
|---|---|---|
| LOL training | 0.0496 | 0.010-0.138 |
| Real night photos | 0.2071 | 0.112-0.348 |

Real night inputs are **4.2x brighter**, and the two ranges barely
overlap — the real-night *minimum* (0.112) is near the LOL *maximum*
(0.138). LOL's "low" images are studio shots deliberately underexposed to
near-black; a phone night photo has already been auto-exposed by its own
ISP. The model has effectively never seen an input this bright, applies
the aggressive 8-iteration curve it learned for near-black input, and
blows out.

**One line of input normalization fixes it, with no retraining at all:**

| real night photos | blown-out | output mean |
|---|---|---|
| as-is | 2.88% | 0.677 |
| **input scaled to training brightness** | **0.00%** | **0.347** |
| *(its behaviour on `eval15`, for reference)* | *0.00%* | *0.381* |

Scaling the input to the brightness the model trained on puts it back
exactly in its calibrated regime, and visibly removes the washed-out haze
(`docs/comparisons/real_night_input_norm.png`).

**This reframes the whole Step 16 saga.** "Phase 2 breaks down on real
photos" took five rounds of fine-tuning and an LR sweep up to 3e-3 to
partially fix — the model was being retrained to re-calibrate to a new
input distribution. The distribution mismatch itself was the problem, and
it is addressable in preprocessing.

**Methodological note worth keeping:** the derivation in §9.2 is sound and
the bias is real and measured. It simply wasn't the dominant term. Being
right about a mechanism is not the same as it being the one that matters —
the only thing that separated the two here was measuring the effect sizes
against each other instead of just confirming the smaller one existed.

### 9.5 The fix across all 11 real photos, and what it does *not* fix

Blown-out pixels go to **0.00% on every single photo**, worst case
included (08.jpg: 13.61% → 0.00%). But the more telling number is
consistency:

| | output mean range | correlation with input brightness |
|---|---|---|
| as-is | 0.542 - 0.814 (spread **0.272**) | **r = +0.97** |
| input-matched | 0.317 - 0.357 (spread **0.040**) | r = +0.45 |

At `r = +0.97` the model is **not doing exposure correction — it is
applying gain.** A brighter input comes out proportionally brighter, which
is the opposite of what an auto-exposure stage should do. With the input
matched, output lands near 0.35 almost regardless of what came in. That
consistency matters more for video than the blowout number does: it is the
difference between a stable exposure and one that pumps as the scene
changes.

**What it does not fix.** The output still looks hazy, and measuring it
contradicted the obvious explanation:

| | local contrast (16x16 std) | high-frequency energy |
|---|---|---|
| model output (fixed) | 0.0794 | 0.0337 |
| real well-exposed photo | 0.0581 | 0.0143 |

Local contrast is **higher** than a real photo (137%), so detail is not
being lost. But high-frequency energy — adjacent-pixel differences, mostly
noise rather than structure — sits **2.4x above** a clean photo, and the
model amplifies the input's own noise by 1.3x on top of that.

**So the haze is amplified sensor noise, not lost detail**, which is
expected: nothing in this pipeline denoises. Input normalization fixed the
exposure problem; the noise problem is untouched and is now the visible
limit. That is precisely what §2's multi-frame merge addresses, and it is
the strongest argument yet for making it the next real step rather than
further single-image tuning.

### 9.6 Multi-frame merge, implemented and validated

`merge.py` — phase-correlation alignment (a spatial shift is a phase ramp
in the FFT, so normalizing away magnitude leaves a sharp peak at the
shift; robust to the brightness differences between frames in a way plain
cross-correlation is not) plus an averaging merge.

**Scope: this checks that the code does what the maths says, nothing
more.** It is a correctness test of `merge.py`, not a claim about how the
system behaves on real footage — §9.8 measures that, and it comes out
very differently. Synthetic input is used because a real burst has no
ground truth to compute error against. Each capture applies Poisson shot
noise plus Gaussian read noise (full-well 2000e-, read noise 2e-) to a
darkened well-exposed photo, averaged over 6 scenes:

| N | RMSE vs clean | measured gain | sqrt(N) | error |
|---|---|---|---|---|
| 1 | 0.00248 | 1.00x | 1.00x | 0.0% |
| 2 | 0.00176 | 1.41x | 1.41x | 0.2% |
| 4 | 0.00125 | 1.99x | 2.00x | 0.7% |
| 8 | 0.00089 | 2.79x | 2.83x | 1.5% |
| 16 | 0.00064 | 3.88x | 4.00x | 3.0% |
| 32 | 0.00047 | 5.33x | 5.66x | 5.8% |
| 64 | 0.00035 | 7.14x | 8.00x | 10.8% |

**Near-ideal through N=16 (3% off theory)** — which is the range that
matters for 30fps video anyway — then drifting at higher N.

The drift is explainable rather than a bug: each capture is clipped to
[0,1], and in near-black regions a sample that would land below zero gets
clipped to 0, biasing the mean upward. That bias is *systematic*, so it
does not average out — the same shape as the FPN floor derived in §9.3,
arriving from a different source. Real sensors have the same effect at the
black level, which is why they calibrate it out with dark frames.

**For scale against everything else tried in this project:** 16 frames buy
a 3.9x noise reduction, about 4 stops. No architecture or loss change in
Phases 1 or 2 came close to that.

### 9.7 A real camera is already reachable

`ffmpeg -f avfoundation -list_devices true` on this machine lists, among
others, **`[4] iPhone 15 Pro Camera`** (Continuity Camera) and a USB
`Elgato Facecam MK.2`. So real bursts can be captured directly rather than
simulated — real sensor, real noise — without any new hardware. That
closes the gap between the synthetic validation above and a measurement
on actual sensor data.

Worth noting what it does *not* give: Continuity Camera delivers
processed video, not RAW or Log, so the 8-bit/gamma ceiling from §3 still
applies. For RAW the phone would have to record to its own storage
(ProRes Log on a 15 Pro) and be transferred losslessly.

### 9.8 Merge on a real sensor — where the theory breaks, and why

Captured a 45-frame burst from a USB webcam (Elgato Facecam MK.2) via
`ffmpeg -f avfoundation`, locked-off, scene brightness 0.157. Phase
correlation reported **zero shift on all 45 frames**, confirming the
camera was genuinely static — so alignment can be taken out of the
picture and the merge measured on its own.

**It does not reach sqrt(N).** Measuring an N-frame merge's noise as
`std(A - B)/sqrt(2)` between two disjoint N-frame merges:

| N | noise | measured | sqrt(N) | shortfall |
|---|---|---|---|---|
| 1 | 0.02277 | 1.00x | 1.00x | — |
| 2 | 0.01720 | 1.32x | 1.41x | 6% |
| 4 | 0.01397 | 1.63x | 2.00x | 19% |
| 8 | 0.01227 | 1.86x | 2.83x | 34% |
| 16 | 0.01000 | 2.28x | 4.00x | 43% |
| 22 | 0.00925 | 2.46x | 4.69x | 48% |

Fitting `sigma(N) = sqrt(random^2/N + floor^2)` gives a floor at **38% of
single-frame noise**, so **N\* ≈ 6** and the ceiling is ~2.6x. §9.3
predicted N\* = 275-3000 for a clean sensor. **Off by roughly 50-500x.**

**Diagnosis, after two wrong turns.** First guess was fixed-pattern
noise — wrong on its own terms: FPN is identical every frame, so it
cancels in `A - B` and could not show up in this measurement at all.
Second guess was slow scene drift, which would inflate the estimate
because A and B came from different time windows. Both were tested and
ruled out:

- Frame differences grow with temporal gap (0.0193 at gap 1 → 0.0331 at
  gap 40), which looks like drift.
- But the difference is **spatially uniform** — median identical
  (0.02092) at gap 1 and gap 40, same block-level distribution — so it is
  not something moving in the scene.
- And removing each frame's global mean changed the numbers **not at all**
  (identical to five decimals), ruling out auto-exposure drift.

What is left is **temporal correlation in the noise itself**. Solving
`E[(F_i - F_j)^2] = 2σ²(1-ρ)` with gap 40 as the ρ≈0 reference gives
**ρ = 0.62 between adjacent frames**. The camera is doing its own temporal
noise reduction, so consecutive frames are *not independent samples* and
averaging them cannot pay the full sqrt(N).

**A measurement-design trap worth recording.** Trying to "fix" the
original estimate by interleaving (A = even frames, B = odd frames over
the same window) produced **8.14x at N=22 — above the sqrt(N) ceiling of
4.69x**, which is impossible for independent noise. The reason is the
same ρ: interleaving pairs every A-frame with a temporally adjacent
B-frame, so the correlated component cancels in `A - B` and the noise is
underestimated. The separate-window design is the honest one here; the
"correction" was worse than the original.

**The actionable part.** If adjacent frames are correlated, frames spaced
further apart should merge better — and with a correctly disjoint design
they do:

| within-set spacing | frames used | noise |
|---|---|---|
| 1 (consecutive) | 0,1,2,3,4 | 0.01555 |
| 2 | 0,2,4,6,8 | 0.01369 |
| 4 | 0,4,8,12,16 | **0.01140** |

**27% less noise from the same number of frames**, purely by spreading
them out. Costs nothing but capture duration.

**The bigger implication.** None of this is a property of light or
silicon — it is an artifact of a camera that has already processed its
output. RAW would not have ρ=0.62 baked in. §3 argued for RAW from the
literature; this measures what the alternative costs: **a processed 8-bit
stream caps merging at ~2.6x where RAW should deliver sqrt(N).** For the
target system that reorders the hardware priorities — sensor *access*
matters more than frame count, because no amount of frames buys past a
floor the camera put there.

### 9.9 The assembled pipeline, and an image-quality audit

`pipeline.py` chains the three stages: merge -> normalize -> enhance. On
the real 22-frame burst:

```
merge     22 frames, 0 realigned, brightness 0.1570
normalize scaled to 0.0496 (the trained regime)
enhance   output brightness 0.3603, blown-out 0.00%
```

Brightening alone is not the goal — it has to not damage the image on the
way. Three things were checked rather than assumed.

**Merging recovers bit depth, it does not just smooth.** Counting distinct
levels present in the data:

| | distinct levels | effective bits |
|---|---|---|
| single frame | 221 | ~7.8 |
| 22-frame merge | 4081 | **~12.0** |

Averaging pushes real signal *below* the 8-bit step, so detail that was
quantisation-limited becomes recoverable. This also answers the obvious
objection to the normalize stage — scaling down by 3.2x would throw away
precision on a single 8-bit frame, but the merge has already bought that
precision back several times over.

**Merging does not blur.** A naive "edge sharpness" metric (mean gradient
at the strongest 3% of edges) *dropped* 32% after merging, which looks
alarming until you check the actual edge:

| offset from edge | −4 | −2 | 0 | +2 | +4 |
|---|---|---|---|---|---|
| single frame | 0.2810 | 0.2919 | 0.2788 | 0.1516 | 0.1597 |
| 22-frame merge | 0.2838 | 0.2925 | 0.2782 | 0.1520 | 0.1595 |

Step height 0.1413 vs 0.1440, transition width unchanged. The edge is
intact. With zero measured misalignment the merge is a pixel-wise temporal
average, which *cannot* reduce spatial detail — so the metric was counting
noise spikes as edges in the single frame, and lost them when the noise
went. **A sharpness metric measured on a noisy image is not measuring
sharpness.**

**The real quality problem is downstream.** Flat-area noise through the
pipeline:

| stage | flat-area noise |
|---|---|
| single frame | 0.0291 |
| 22-frame merge | 0.0196 |
| merged -> enhanced | **0.0511** |
| single -> enhanced | 0.0616 |

The enhancement **amplifies noise ~2.6x**, so the merge's 1.5x cleanup
collapses to roughly 1.2x by the time it reaches the output. Expected —
brightening multiplies whatever is there, and nothing in the model was
ever trained to suppress noise (§4). The merge is still worth having, but
**most of its benefit is being spent re-amplifying noise the enhancement
reintroduces.**

This makes noise-aware training (Poisson-Gaussian augmentation, Step 25)
the highest-value next change rather than an optional refinement: without
it, further gains in the merge stage keep getting eaten at the end.

### 9.10 The full pipeline on 144 real frames

A longer burst than §9.8's, 144 frames from the same webcam. Merge
scaling, measured the same way:

| N | noise | measured | sqrt(N) | time @30fps |
|---|---|---|---|---|
| 8 | 0.01071 | 1.83x | 2.83x | 0.3 s |
| 32 | 0.00769 | 2.55x | 5.66x | 1.1 s |
| 72 | 0.00552 | **3.55x** | 8.49x | 2.4 s |

Still roughly 58% short of theory — the same ρ=0.62 temporal correlation
from §9.8 — but 3.55x on real data, up from 2.5x on the earlier 22-frame
burst, so more frames keep paying even against that ceiling.

Through the whole pipeline, measuring on the flattest 64x64 block (found
on the *clean* merged image, then measured at that same location in every
version — picking it per-image would just find whichever patch each one
happened to smooth most):

| | brightness | noise |
|---|---|---|
| input, single frame | 0.2104 | 0.01712 |
| input, 72-frame merge | 0.1836 | 0.00797 |
| enhanced from 1 frame | 0.3630 | 0.04379 |
| **enhanced from 72 frames** | 0.3631 | **0.02586** |

Merge alone is 2.15x cleaner; **1.69x survives the enhancement**, which
is the §9.9 noise-amplification tax showing up again on real data.

`docs/comparisons/real_burst_pipeline.png` shows it at 2x zoom. The
painting in the corner is the clearest tell: a colour mush in the
single-frame version, individual grapes resolved in the 72-frame one.
Same brightness (0.3630 vs 0.3631), visibly different quality.

**The open gap, and what it needs.** Every capture so far sits at
0.15-0.21 brightness — a dim room, not the near-total darkness this is
ultimately for. The webcam's auto-exposure and built-in night handling
keep lifting the scene, and nothing on the software side can undo that:
by the time the frames arrive, the camera has already decided the
exposure. **This is a hardware blocker, not an analysis one** — it needs
a camera that exposes manual exposure/ISO control, which is the plan.
Every number in §9.7-9.10 therefore describes a dim room, and the
deep-darkness case is simply untested rather than estimated.

**A failed measurement worth recording.** Attempted to calibrate the
sensor model against reality with a photon transfer curve (temporal
variance against signal level, whose slope gives full-well capacity and
intercept gives read noise) on the real burst. It returned **negative**
values for both — physically impossible. The cause is the camera's own
temporal denoising: it destroys the photon statistics the method depends
on. A processed webcam stream cannot be used to characterise its own
sensor, which is one more concrete argument for RAW.

### 9.11 Running live: what it took, and what it cost

The actual end goal is a **live feed with a button** that switches night
enhancement on. That makes speed a gate rather than a refinement, so it
got measured before anything else — and the existing pipeline did not
pass. Against a 33.3 ms budget for 30fps, at 960x540:

| | time |
|---|---|
| CurveNet at full resolution | 28.0 ms |
| re-averaging an 8-frame window | 17.0 ms |
| **total** | **46 ms** (1080p: 194 ms) |

Two changes fixed it, both already identified in §1.

**Coefficients predicted at reduced resolution**, with the curve still
applied at full resolution (`CurveNet(..., coeff_scale=N)`). The
coefficients describe *illumination*, which varies smoothly across a
scene, so they do not need per-pixel resolution — the output detail comes
from applying the curve at full size, which is unchanged.

| scale | 960x540 | 1920x1080 | PSNR vs full-res | SSIM |
|---|---|---|---|---|
| 1 | 28.0 ms | 115.6 ms | — | — |
| 2 | 10.9 ms | 64.2 ms | 88.92 dB | 1.00000 |
| 4 | 14.7 ms | 49.8 ms | 83.55 dB | 1.00000 |
| 8 | 8.8 ms | 39.0 ms | 79.79 dB | 1.00000 |

**The quality cost is nil, measured rather than assumed**: visually
lossless starts around 45 dB and even scale 8 sits at 79.8 dB with SSIM
1.00000 and a maximum pixel error of 0.004.

**The merge made incremental.** A rolling mean adds the arriving frame and
subtracts the one leaving the window instead of re-averaging: **17 ms ->
0.004 ms**, verified to equal a plain window mean exactly.

Net: **22.1 ms/frame, about 45 fps at 960x540** — inside the budget with
headroom. `live_enhance.py` shows raw and enhanced side by side with
SPACE to toggle.

Worth noting where the remaining time goes: with the convolutions now
cheap, the bottleneck has shifted to the 8 sequential curve iterations at
full resolution. That is the next thing to attack if 1080p live is wanted
(39 ms at scale 8, still just over budget).

### 9.12 Noise-aware training — measured too early to call, and confounded

Step 25 adds Poisson-Gaussian noise to training inputs only, so the model
is asked to suppress noise rather than reproduce it. A first check at
epoch 30 against the real burst:

| model | output noise | amplification |
|---|---|---|
| Phase 2 CurveNet | 0.02586 | 3.25x |
| Phase 1 + noise training (ep 30) | 0.02736 | 3.43x |

**This comparison does not support a conclusion.** The two models are
different architectures (CurveNet vs LowLightEnhanceNet at hc=64), so the
difference cannot be attributed to the noise training — and 30 of 150
epochs is early regardless. Recorded because the temptation to read it as
"noise training didn't help" is exactly the error this project's method
exists to avoid.

A matched baseline is now training — identical config, only `--noise`
removed — which is the comparison that can actually answer it.

### 9.13 Input normalisation is a patch, not a fix — measured

§9.4 showed that scaling a real night photo's brightness into the trained
range eliminates blown-out highlights (2.88% → 0.00%). The obvious next
move is to make it unconditional. Measured on `eval15` first, and it
would have been a mistake:

| | PSNR | SSIM |
|---|---|---|
| as-is | **19.03** | **0.7921** |
| input normalised to 0.0496 | 17.13 | 0.7851 |

**It costs 1.9 dB on the benchmark.** `eval15` inputs already average
0.0594 — close to the trained 0.0496 — so normalising only scales them
down 17% for no reason, and the output lands dimmer than the target.

So the rule is conditional: normalisation helps when the input is far
outside the trained range (real photos at 0.21) and hurts when it is
already inside (eval15 at 0.059). A fixed threshold would work, but it
would be another inherited constant of exactly the kind this project has
twice found to be wrong (the 0.6 exposure target, the 8 curve
iterations).

**The better fix is to remove the dependency rather than correct for
it** — which is Step 28: jitter the input exposure during training so the
model stops having a preferred input brightness at all. Then no
inference-time rule is needed in either direction.

### 9.14 Noise-aware training (Step 26) — refuted, and probably for a knowable reason

Two runs, identical in every respect except `--noise` (Poisson-Gaussian
on inputs only, target left clean): hc=64, SSIM loss, augmentation, 150
epochs.

**On `eval15` it is a wash**, measured with `evaluate.py`'s native
resolution — baseline 19.29/0.7847 against 19.18/0.7851, so −0.11 PSNR
(marginally outside the ±0.07 floor) and an SSIM difference inside the
floor entirely.

*An aside on measurement discipline:* the first pass used a 256×256
resize and reported 19.58/**0.8367** for the baseline, which looks like a
large SSIM jump past the 0.8 target. It was an artefact — every number in
this log comes from `evaluate.py` at native resolution, and mixing the
two protocols invents improvements that are not there.

**On the real noisy burst — which is what it was for — it is clearly
worse:**

| | from a single frame | from the 72-frame merge |
|---|---|---|
| baseline | 3.91x amplification | **5.08x** |
| + noise training | 4.60x | **7.08x** |

So training on synthetic noise did not teach suppression; it made the
model amplify *more*. The most likely reason connects to §9.8's failed
photon transfer curve: **there was never any way to calibrate the
synthetic noise against this camera.** The model trained on independent
Poisson-Gaussian noise, while the camera delivers noise that is
temporally correlated (ρ=0.62), spatially processed, and of unknown
magnitude. Training against the wrong noise distribution is not obviously
better than training against none.

**What this does and does not close.** Step 26 is answered: this
particular approach does not fix the amplification. The underlying
problem — the enhancement stage spending most of what merging buys
(§9.9) — is untouched and still the main quality limit. Retrying it
sensibly needs a sensor whose noise can actually be characterised, which
is the same RAW/manual-control requirement blocking Phase 3.

### 9.15 Exposure jitter (Step 28) — works, at a price worth naming

Same config as the Step 26 baseline, one change: `--exposure-jitter 0.3
5.0` randomly rescales the input's brightness while the target stays
fixed, so the model sees each scene at many exposures and is always asked
for the same well-exposed result.

**1. On `eval15` it costs real accuracy.**

| | PSNR | SSIM |
|---|---|---|
| baseline | **19.29** | **0.7847** |
| + exposure jitter | 16.95 | 0.7462 |

−2.34 PSNR, far outside the ±0.07 floor. Expected: the model now spends
capacity covering a range of input exposures instead of specialising in
the one `eval15` happens to use.

**2. It does what it was built to do — exposure dependence drops 42%.**
Feeding the same scenes at 0.5x to 4x input brightness:

| input scale | baseline output | jitter output |
|---|---|---|
| 0.5x | 0.3512 | 0.3404 |
| 1.0x | 0.4611 | 0.4021 |
| 2.0x | 0.5851 | 0.4636 |
| 4.0x | 0.6898 | 0.5370 |
| **spread over 8x range** | **0.3386** | **0.1966** |

Not fully invariant — a perfect enhancer would be flat — but
substantially less of a gain stage and more of an exposure correction.

**3. On real night photos it removes the failure outright, with no
patch.** Fed raw at 0.21 brightness, no `--match-input-brightness`:

| | output brightness | blown-out |
|---|---|---|
| baseline | 0.6491 | **3.06%** |
| + exposure jitter | 0.5328 | **0.00%** |

**The verdict, and the tension it exposes.** Step 28 succeeded: the
model no longer needs an inference-time correction to survive real
input, which is what §9.13 argued for after showing the normalisation
patch was conditional and brittle.

But it is **worse on the benchmark and better on real footage**, which
is the clearest case yet of the bias documented at the top of §9: a
score tuned on 15 clean in-distribution images is not the same thing as
working. Given the project's actual goal is real night video, the trade
looks correct — but it should be recorded as a trade, not as a win, and
it means `eval15` can no longer be the sole arbiter of a "best"
checkpoint.

### 9.16 Porting the jitter to Phase 2 (Step 29) — fails, for the reason it was predicted to work

Step 28 worked on the supervised model, so the same change was tried on
Phase 2's `CurveNet`, seed-matched with only `--exposure-jitter 0.3 5.0`
added. The stated reasoning was that it would *transfer cleanly because*
Phase 2's losses are zero-reference. That reasoning was backwards.

| | PSNR | SSIM | drift | blown-out |
|---|---|---|---|---|
| Phase 2 baseline | **18.21** | 0.6201 | 0.4438 | 2.88% |
| + exposure jitter | 12.37 | 0.5281 | **0.3902** | **0.14%** |

Set against Phase 1, where the same change cost −2.34 PSNR and bought a
42% drift reduction, here it costs **−5.84 PSNR for only −12%**. Roughly
five times worse value.

**Why the zero-reference property hurts rather than helps.** Supervised
training has a per-scene target, so jitter teaches a useful invariance:
*whatever brightness arrives, produce this particular image*. Phase 2 has
no such reference — its losses only ask that patches average
`exposure_target`, with no notion of what the scene should look like.
Adding jitter on top leaves the model learning little beyond a brightness
normaliser, which is exactly what the numbers show: drift barely moves
while image quality falls off a cliff.

**The general lesson, which is the useful part.** Augmentations are not
portable on their own — they are only meaningful against what the loss
is actually anchored to. "It worked on the other model" was not a
sufficient reason, and the mechanism given for why it *should* transfer
turned out to name the precise thing that broke it.

Kept as an available flag, not recommended for Phase 2. The input
covariate shift there is still better handled by
`--match-input-brightness` at inference (§9.4), despite that being the
patch §9.13 wanted to eliminate.

## 10. Honest expectation setting

Matching Apple/Google in software alone, from 8-bit single frames, is not
achievable — they work on calibrated RAW with dedicated silicon and
sensor-specific noise models.

But the *principles* are public and reproducible, and **the static-camera
case is genuinely easier than the phone case**. A tripod camera that can
integrate 100 frames has an advantage a phone fundamentally cannot use.
Getting into "clearly useful night video" territory is realistic; matching
a flagship ISP pixel-for-pixel is not.
