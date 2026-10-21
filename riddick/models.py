"""The three architectures, in one place.

LowLightEnhanceNet and UNet are Phase 1 (supervised, trained against
paired bright targets). CurveNet is Phase 2 (zero-reference - it predicts
curve parameters, never an image, and trains with no ground truth).

They live here rather than inside the training scripts so that loading a
checkpoint does not mean importing a trainer.
"""
import torch
import torch.nn as nn


class LowLightEnhanceNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, extra_layer=False, batchnorm=False, skip_connection=False):
        super().__init__()
        self.skip_connection = skip_connection

        def hidden_block():
            block = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)]
            if batchnorm:
                block.append(nn.BatchNorm2d(hidden_channels))
            block.append(nn.ReLU(inplace=True))
            return block

        stem = [nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        if batchnorm:
            stem.append(nn.BatchNorm2d(hidden_channels))
        stem.append(nn.ReLU(inplace=True))

        blocks = hidden_block()
        if extra_layer:
            blocks += hidden_block()

        final_in_channels = hidden_channels * 2 if skip_connection else hidden_channels
        final = [nn.Conv2d(final_in_channels, out_channels, kernel_size=3, padding=1), nn.Sigmoid()]

        if skip_connection:
            # Keep stem/blocks/final as separate pieces so forward() can torch.cat the
            # stem's output back in before the final conv (Step 9) - the final conv then
            # sees both the raw early features and the processed ones, side by side.
            self.stem = nn.Sequential(*stem)
            self.blocks = nn.Sequential(*blocks)
            self.final = nn.Sequential(*final)
        else:
            # Unchanged from Steps 0-8: one flat pipeline, so every existing checkpoint
            # still loads (same net.0, net.1, ... keys).
            self.net = nn.Sequential(*stem, *blocks, *final)

        # input: dark image, 3 matrices (R, G, B)
        # stem: 1st Conv2d, 32 filters -> 32 feature maps (+ optional BatchNorm) -> ReLU
        # blocks: one (or two, extra_layer=True) more Conv2d+ReLU, 32->32
        # skip_connection=True: concat the stem's output with the blocks' output
        #   (32+32=64 channels) before the final conv, instead of only using blocks' output
        # final: last Conv2d back to 3 channels (R, G, B), Sigmoid squashes to valid pixel range

    def forward(self, x):
        if self.skip_connection:
            stem_out = self.stem(x)
            features = torch.cat([stem_out, self.blocks(stem_out)], dim=1)
            return self.final(features)
        return self.net(x)


class UNet(nn.Module):
    """Step 10: a real U-Net - 2 downsample/upsample stages, skip connections
    at each resolution level, not just one (Step 9 showed one skip connection
    does nothing on a shallow network; this gives it actual depth to matter)."""

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32):
        super().__init__()
        C = hidden_channels

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            )

        # Encoder: each stage halves resolution, doubles channels
        self.enc0 = conv_block(in_channels, C)       # full res,  C channels  -> skip0
        self.enc1 = conv_block(C, C * 2)              # 1/2 res,   2C channels -> skip1
        self.bottleneck = conv_block(C * 2, C * 4)    # 1/4 res,   4C channels
        self.pool = nn.MaxPool2d(2)

        # Decoder: each stage upsamples back, concats the matching skip, then convs down
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = conv_block(C * 4 + C * 2, C * 2)  # upsampled(4C) + skip1(2C) -> 2C
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = conv_block(C * 2 + C, C)          # upsampled(2C) + skip0(C) -> C

        self.final = nn.Sequential(nn.Conv2d(C, out_channels, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, x):
        skip0 = self.enc0(x)
        skip1 = self.enc1(self.pool(skip0))
        x = self.bottleneck(self.pool(skip1))

        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))
        x = self.dec0(torch.cat([self.up0(x), skip0], dim=1))
        return self.final(x)


class CurveNet(nn.Module):
    """Outputs per-pixel curve parameters (not a direct image). Same small
    stem+blocks+final shape as Phase 1's LowLightEnhanceNet, but the final
    layer produces n_iterations*3 channels (one RGB alpha map per curve
    iteration) squashed to [-1, 1] with Tanh instead of an image with Sigmoid."""

    def __init__(self, in_channels=3, hidden_channels=32, n_iterations=8):
        super().__init__()
        self.n_iterations = n_iterations

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_iterations * 3, kernel_size=3, padding=1), nn.Tanh(),
        )

    def forward(self, x, return_alphas=False, coeff_scale=1):
        """coeff_scale > 1 predicts the curve coefficients at 1/scale resolution
        and upsamples them, instead of running the convolutions at full size.

        This is the standard real-time trick (bilateral grid / LUT methods hit
        4K in ~13ms with it - docs/RESEARCH.md section 1). It works because the
        coefficients describe *illumination*, which varies smoothly across a
        scene, so they do not need per-pixel resolution - while the curve
        itself is still applied at full resolution, so output detail is
        unaffected. Cost drops roughly with the square of the scale.

        Training always runs at scale 1; this is an inference-time option.
        """
        if coeff_scale > 1:
            small = F.interpolate(x, scale_factor=1 / coeff_scale, mode="bilinear",
                                  align_corners=False, recompute_scale_factor=False)
            alphas = F.interpolate(self.net(small), size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        else:
            alphas = self.net(x)  # (B, n_iterations*3, H, W), each value in [-1, 1]

        enhanced = x
        for i in range(self.n_iterations):
            alpha = alphas[:, i * 3:(i + 1) * 3]
            # the light-enhancement curve: push each pixel up (or down) by an
            # amount that depends on how far it already is from 0 and 1, so
            # pixels near the extremes change less than ones in the middle
            enhanced = enhanced + alpha * (enhanced - enhanced ** 2)
        enhanced = enhanced.clamp(0, 1)
        if return_alphas:
            # only Step 13's illumination smoothness loss needs the raw
            # per-pixel curve parameters - evaluate.py/inference.py never
            # pass this, so they keep getting just the enhanced image back
            return enhanced, alphas
        return enhanced
