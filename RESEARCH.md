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

## 9. Honest expectation setting

Matching Apple/Google in software alone, from 8-bit single frames, is not
achievable — they work on calibrated RAW with dedicated silicon and
sensor-specific noise models.

But the *principles* are public and reproducible, and **the static-camera
case is genuinely easier than the phone case**. A tripod camera that can
integrate 100 frames has an advantage a phone fundamentally cannot use.
Getting into "clearly useful night video" territory is realistic; matching
a flagship ISP pixel-for-pixel is not.
