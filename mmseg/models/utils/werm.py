import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        device = x.device
        dtype = x.dtype

        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device, dtype=dtype)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]],
                          device=device,
                          dtype=dtype)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]],
                          device=device,
                          dtype=dtype)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]],
                          device=device,
                          dtype=dtype)

        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        filt = filt.repeat(c, 1, 1, 1)

        y = F.conv2d(x, filt, stride=2, groups=c)
        y = y.view(b, c, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3], (h, w)

    def inverse(self, ll, lh, hl, hh, orig_size):
        b, c, h, w = ll.shape
        device = ll.device
        dtype = ll.dtype

        x = torch.stack([ll, lh, hl, hh], dim=2).reshape(b, c * 4, h, w)

        f_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]],
                            device=device,
                            dtype=dtype)
        f_lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]],
                            device=device,
                            dtype=dtype)
        f_hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]],
                            device=device,
                            dtype=dtype)
        f_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]],
                            device=device,
                            dtype=dtype)

        filt = torch.stack([f_ll, f_lh, f_hl, f_hh], dim=0).unsqueeze(1)
        filt = filt.repeat(c, 1, 1, 1)

        out = F.conv_transpose2d(x, filt, stride=2, groups=c)
        h0, w0 = orig_size
        return out[:, :, :h0, :w0]


class WERMBlock(nn.Module):
    """Wavelet-enhanced ERM block for 2D feature refinement."""

    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * dw_expand

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.dwt = HaarDWT2D()

        self.ll_proj = nn.Sequential(
            nn.Conv2d(c, dw_channel // 2, kernel_size=1, bias=True),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                kernel_size=3,
                padding=1,
                groups=dw_channel // 2,
                bias=True,
            ),
            nn.GELU(),
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(c * 3, dw_channel // 2, kernel_size=1, bias=True),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                kernel_size=3,
                padding=1,
                groups=dw_channel // 2,
                bias=True,
            ),
            nn.GELU(),
        )
        self.high_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1,
                      bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(dw_channel, c * 4, kernel_size=1, bias=True)

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.dropout1 = (nn.Dropout(drop_out_rate)
                         if drop_out_rate > 0.0 else nn.Identity())
        self.dropout2 = (nn.Dropout(drop_out_rate)
                         if drop_out_rate > 0.0 else nn.Identity())

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)),
                                  requires_grad=True)

    @staticmethod
    def _apply_layer_norm(x, norm):
        x = x.permute(0, 2, 3, 1)
        x = norm(x)
        return x.permute(0, 3, 1, 2)

    def forward(self, inp):
        x = self._apply_layer_norm(inp, self.norm1)
        ll, lh, hl, hh, orig_size = self.dwt(x)

        ll_feat = self.ll_proj(ll)
        high = torch.cat([lh, hl, hh], dim=1)
        high_feat = self.high_proj(high)
        high_feat = high_feat * self.high_gate(high_feat)

        fused = torch.cat([ll_feat, high_feat], dim=1)
        fused = self.fuse(fused)

        ll_r, lh_r, hl_r, hh_r = torch.chunk(fused, 4, dim=1)
        x = self.dwt.inverse(ll_r, lh_r, hl_r, hh_r, orig_size)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self._apply_layer_norm(y, self.norm2))
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma