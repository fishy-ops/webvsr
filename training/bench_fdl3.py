"""FDL cost vs. which stages contribute to the LOSS.

The VGG stages are sequential, so an early stage cannot be skipped -- stage 2
needs stage 1's 64-channel output. But the expense is not the convolution: it is
the FFT, the random projections and the sort over a full-resolution feature map.
Those can be skipped per stage while the forward chain still runs intact.
"""
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "training")
from model_span import SPANLite
from losses import FDLLoss

dev = "cuda"
B, CROP = 8, 256
model = SPANLite(num_in_ch=3, feature_channels=16, upscale=2).to(dev)
lr = torch.rand(B, 3, CROP//2, CROP//2, device=dev)
hr = torch.rand(B, 3, CROP, CROP, device=dev)
torch.manual_seed(0)
xx = torch.linspace(0, 20, CROP, device=dev)
gt = (0.5 + 0.4*torch.sin(xx.view(1,1,-1,1)*3)*torch.cos(xx.view(1,1,1,-1)*2)).repeat(4,3,1,1)
gt = (gt + 0.05*torch.rand_like(gt)).clamp(0,1)
k = torch.ones(3,1,5,5, device=dev)/25
blur = F.conv2d(gt, k, padding=2, groups=3)
shift = torch.roll(gt, (1,1), (2,3))


def patched(first_loss_stage):
    c = FDLLoss().to(dev)
    orig = c.forward.__wrapped__ if hasattr(c.forward, "__wrapped__") else None

    def fwd(self, pred, target):
        x = (pred.float().clamp(0,1) - self.vgg_mean) / self.vgg_std
        with torch.no_grad():
            y = (target.float().clamp(0,1) - self.vgg_mean) / self.vgg_std
        score = 0.0
        for i, st in enumerate(self.stages):
            x = st(x)
            with torch.no_grad():
                y = st(y)
            if i < first_loss_stage:      # run the stage, skip its loss term
                continue
            fx, fy = torch.fft.fftn(x, dim=(-2,-1)), torch.fft.fftn(y, dim=(-2,-1))
            s = self._sliced_w1(fx.abs(), fy.abs(), i)
            s = s + self.phase_weight*self._sliced_w1(torch.angle(fx), torch.angle(fy), i)
            score = score + self._STAGE_W[i]*s
            del fx, fy
        return score*0.01

    import types
    c._fwd = types.MethodType(fwd, c)
    return c


for first in (0, 1, 2):
    c = patched(first)
    with torch.no_grad():
        b, s_ = c._fwd(blur, gt).item(), c._fwd(shift, gt).item()
    for _ in range(3):
        c._fwd(model(lr), hr).backward()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); N = 6
    for _ in range(N):
        c._fwd(model(lr), hr).backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)/N
    print(f"  loss from stage {first}: {dt*1000:6.1f} ms/step  "
          f"{torch.cuda.max_memory_allocated()/2**20:5.0f} MiB  "
          f"blur/shift {b/max(s_,1e-9):5.2f}")
    del c; torch.cuda.empty_cache()
