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

## 10. Honest expectation setting

Matching Apple/Google in software alone, from 8-bit single frames, is not
achievable — they work on calibrated RAW with dedicated silicon and
sensor-specific noise models.

But the *principles* are public and reproducible, and **the static-camera
case is genuinely easier than the phone case**. A tripod camera that can
integrate 100 frames has an advantage a phone fundamentally cannot use.
Getting into "clearly useful night video" territory is realistic; matching
a flagship ISP pixel-for-pixel is not.
