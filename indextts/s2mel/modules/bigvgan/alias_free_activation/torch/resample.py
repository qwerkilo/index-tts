# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn as nn
from torch.nn import functional as F
from .filter import LowPassFilter1d
from .filter import kaiser_sinc_filter1d


def _zero_stuff(x, ratio):
    """Interleave ``ratio - 1`` zeros between neighbouring samples.

    ``[B, C, T] -> [B, C, ratio * T]`` where ``out[..., ratio * i] == x[..., i]``
    and every other position is zero. This is the "zero stuffing" step that turns
    a stride-``ratio`` transposed convolution into a plain stride-1 convolution.
    """
    return F.pad(x.unsqueeze(-1), (0, ratio - 1)).flatten(-2)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter)

    # x: [B,C,T]
    def forward(self, x):
        _, C, T = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")

        # indextts4android patch (HTP expressibility): this upsampler used a grouped
        # ConvTranspose1d. It is now a grouped Conv1d over the zero-stuffed signal,
        # built from the SAME FIR taps, with the same edge padding and the same
        # trim. The two are algebraically identical: a stride-s transposed
        # convolution is the adjoint of a stride-s convolution, i.e. the FULL
        # convolution of the s-times zero-stuffed signal with the time-reversed
        # kernel,
        #
        #     conv_transpose1d(z, w, stride=s)[o]
        #         == sum_j w[j] * zs[o - K + 1 + j]
        #         == conv1d(zs, w.flip(-1), padding=K - 1)[o]
        #
        # with zs = zero_stuff(z, s) and K = len(w). ``padding=K-1`` is the "full"
        # convolution padding; the slice below then keeps exactly the
        # ``stride * T`` samples the transposed convolution produced
        # (``pad_left + stride*T == len(conv_transpose output) - pad_right``).
        #
        # Why: on the QNN HTP backend ConvTranspose1d maps to TransposeConv2d,
        # whose datatype matrix has no fp16-activation + int8-weight row (unlike
        # Conv2d), so this resampler could not share the mixed-precision
        # configuration used by the plain convolutions. A fixed convolution is the
        # well-supported form. Nothing about the filter changes: ``self.filter``
        # is used verbatim. The time reversal is a no-op here because the
        # kaiser-sinc taps are symmetric (measured: max |w - w.flip(-1)| == 0),
        # but it is applied so the rewrite stays exact for arbitrary taps.
        zs = _zero_stuff(x, self.stride)
        x = self.ratio * F.conv1d(
            zs,
            self.filter.expand(C, -1, -1).flip(-1),
            # indextts4android patch (per-channel quantization): a (tiny,
            # numerically invisible) bias makes this convolution acceptable to
            # qairt-quantizer's per-channel weight path, which otherwise aborts
            # with "No bias info for op (DepthWiseConv2d)". A plain ZERO bias
            # gets constant-folded away by the ONNX exporter, so 1e-12 is used:
            # below the fp32 resolution of any signal this FIR produces, and it
            # survives export. Host simulation puts per-channel int8 weights
            # ~12.6 dB above per-tensor on the truncated BigVGAN.
            bias=torch.full((C,), 1e-12, dtype=zs.dtype, device=zs.device),
            stride=1,
            padding=self.kernel_size - 1,
            groups=C,
        )
        x = x[..., self.pad_left : self.pad_left + self.ratio * T]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        xx = self.lowpass(x)

        return xx
