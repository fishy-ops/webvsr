"""SPAN-Lite with early exits (multi-exit / truncatable backbone).

Motivation (RESEARCH.md 3a): adaptive depth is the one idea that addresses both
the efficiency ceiling and the busy-scene collapse. Plain SPANLite cannot be
truncated, because `conv_cat` consumes f0, block_1's mid, block_3's mid and
block_4's output by construction -- stopping after block 2 leaves two of its
four inputs undefined.

The fix is not to change the trunk but to give each supported exit its own
small head. Blocks are shared; only `conv_cat` / `conv_last` / `upsampler` are
per-exit, which is a few tens of kB of weights.

The depth-4 head is named exactly as in `model_span.SPANLite` (`conv_cat`,
`conv_last`, `upsampler`), so a shipped checkpoint loads into it unchanged with
`strict=False` and the full-depth path stays bit-for-bit the shipped model.
Early-exit heads carry a `_d{n}` suffix and start untrained.

Routing is meant to be driven by an *exogenous* signal (how compressed the
input is), not by the model's own confidence -- see RESEARCH.md 3a for why that
distinction matters.
"""

import torch
import torch.nn as nn

from model_span import Conv3XC, SPAB, count_params


# Which features each exit concatenates. Order matters: it is the channel
# layout `conv_cat` expects. Entries are ("f0",) | ("mid", block) | ("out", block).
# Depth 4 reproduces model_span.SPANLite exactly.
TAPS = {
    2: (("f0", 0), ("mid", 1), ("mid", 2), ("out", 2)),
    3: (("f0", 0), ("mid", 1), ("mid", 2), ("out", 3)),
    4: (("f0", 0), ("mid", 1), ("mid", 3), ("out", 4)),
}


class SPANLiteME(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, feature_channels=32,
                 upscale=2, img_range=1.0, exit_depths=(2, 4), num_blocks=4):
        super().__init__()
        self.img_range = img_range
        self.upscale = upscale
        self.num_blocks = num_blocks
        self.mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)

        exit_depths = tuple(sorted(set(exit_depths)))
        for d in exit_depths:
            if d not in TAPS:
                raise ValueError(f"no tap spec for exit depth {d}; have {sorted(TAPS)}")
            if d > num_blocks:
                raise ValueError(f"exit depth {d} exceeds num_blocks {num_blocks}")
        self.exit_depths = exit_depths
        self.max_depth = max(exit_depths)

        self.conv_first = Conv3XC(num_in_ch, feature_channels, gain=2)

        self.blocks = nn.ModuleList(
            [SPAB(feature_channels) for _ in range(num_blocks)]
        )

        # Per-exit heads. The deepest exit uses the shipped names so that a
        # SPANLite checkpoint loads straight into it.
        self.conv_cat = nn.ModuleDict()
        self.conv_last = nn.ModuleDict()
        self.upsampler = nn.ModuleDict()
        for d in exit_depths:
            key = self._key(d)
            self.conv_last[key] = Conv3XC(feature_channels, feature_channels, gain=2)
            self.conv_cat[key] = nn.Conv2d(
                feature_channels * 4, feature_channels, 1, bias=True
            )
            self.upsampler[key] = nn.Sequential(
                nn.Conv2d(feature_channels, num_out_ch * (upscale ** 2), 3, padding=1),
                nn.PixelShuffle(upscale),
            )

    def _key(self, d):
        return "full" if d == self.max_depth else f"d{d}"

    def forward(self, x, depth=None):
        """Run to `depth` blocks and exit there. Defaults to the deepest exit.

        Blocks beyond `depth` are never executed -- that is where the saving is.
        """
        if depth is None:
            depth = self.max_depth
        if depth not in self.exit_depths:
            raise ValueError(f"depth {depth} is not a trained exit; have {self.exit_depths}")

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        f0 = self.conv_first(x)

        mids = {}
        outs = {}
        h = f0
        for i in range(depth):
            h, mid, _ = self.blocks[i](h)
            mids[i + 1] = mid
            outs[i + 1] = h

        key = self._key(depth)
        parts = []
        for kind, idx in TAPS[depth]:
            if kind == "f0":
                parts.append(f0)
            elif kind == "mid":
                parts.append(mids[idx])
            else:
                parts.append(self.conv_last[key](outs[idx]))

        out = self.conv_cat[key](torch.cat(parts, dim=1))
        return self.upsampler[key](out)

    def forward_all_exits(self, x):
        """Every exit's output, sharing trunk computation. Used for joint training.

        Runs the trunk once to max_depth rather than once per exit.
        """
        self.mean = self.mean.type_as(x)
        xn = (x - self.mean) * self.img_range

        f0 = self.conv_first(xn)
        mids, outs = {}, {}
        h = f0
        for i in range(self.max_depth):
            h, mid, _ = self.blocks[i](h)
            mids[i + 1] = mid
            outs[i + 1] = h

        results = {}
        for d in self.exit_depths:
            key = self._key(d)
            parts = []
            for kind, idx in TAPS[d]:
                if kind == "f0":
                    parts.append(f0)
                elif kind == "mid":
                    parts.append(mids[idx])
                else:
                    parts.append(self.conv_last[key](outs[idx]))
            out = self.conv_cat[key](torch.cat(parts, dim=1))
            results[d] = self.upsampler[key](out)
        return results

    def load_span_lite(self, state_dict, strict_trunk=True):
        """Load a plain SPANLite checkpoint into the trunk and the deepest exit.

        Returns (loaded, skipped) key counts. Early-exit heads are left at init.
        """
        remap = {}
        for k, v in state_dict.items():
            if k.startswith("block_"):
                # block_1.c1.weight -> blocks.0.c1.weight
                n, rest = k[len("block_"):].split(".", 1)
                remap[f"blocks.{int(n) - 1}.{rest}"] = v
            elif k.startswith("conv_cat."):
                remap[f"conv_cat.full.{k[len('conv_cat.'):]}"] = v
            elif k.startswith("conv_last."):
                remap[f"conv_last.full.{k[len('conv_last.'):]}"] = v
            elif k.startswith("upsampler."):
                remap[f"upsampler.full.{k[len('upsampler.'):]}"] = v
            else:
                remap[k] = v

        missing, unexpected = self.load_state_dict(remap, strict=False)
        if strict_trunk and unexpected:
            raise RuntimeError(f"unexpected keys from SPANLite checkpoint: {unexpected[:5]}")
        return len(remap), len(missing)


if __name__ == "__main__":
    from model_span import SPANLite

    ch = 32
    me = SPANLiteME(upscale=2, feature_channels=ch, exit_depths=(2, 4))
    base = SPANLite(upscale=2, feature_channels=ch)
    print(f"SPANLite      params: {count_params(base):,}")
    print(f"SPANLiteME    params: {count_params(me):,}  (+{count_params(me)-count_params(base):,})")

    # The shipped checkpoint must load, and the full path must match it exactly.
    n, missing = me.load_span_lite(base.state_dict())
    print(f"loaded {n} keys from SPANLite, {missing} left at init (early-exit heads)")

    x = torch.randn(1, 3, 64, 64)
    base.eval(); me.eval()
    with torch.no_grad():
        y_base = base(x)
        y_full = me(x, depth=4)
        y_d2 = me(x, depth=2)
        allx = me.forward_all_exits(x)
    print(f"depth4 vs SPANLite max|diff|: {(y_base - y_full).abs().max().item():.3e}")
    print(f"depth2 out {tuple(y_d2.shape)}, depth4 out {tuple(y_full.shape)}")
    print(f"forward_all_exits keys: {sorted(allx)}  "
          f"matches single-exit: {all(torch.allclose(allx[d], me(x, depth=d), atol=1e-6) for d in (2,4))}")
