"""The one thing that can silently ruin a schedule-free run.

AdamWScheduleFree keeps two weight sets: the iterate it steps from (train mode)
and the average it should be evaluated on (eval mode). If the trainer validates
or saves while the optimiser is in train mode, the DISTS printed describes one
model and the .pth on disk holds another -- a checkpoint scored on numbers it
does not have. This asserts the two modes really do differ, so that the
train()/eval() calls added to train_span.py are load-bearing and not decoration.
"""
import sys, torch, copy
sys.path.insert(0, "training")
import schedulefree
from model_span import SPANLite

torch.manual_seed(0)
m = SPANLite(num_in_ch=3, feature_channels=16, upscale=2).cuda()
opt = schedulefree.AdamWScheduleFree(m.parameters(), lr=1e-3, warmup_steps=5)
x = torch.rand(2, 3, 64, 64).cuda()
y = torch.rand(2, 3, 128, 128).cuda()

opt.train()
for _ in range(20):
    opt.zero_grad()
    torch.nn.functional.l1_loss(m(x), y).backward()
    opt.step()

w_train = copy.deepcopy({k: v.clone() for k, v in m.state_dict().items()})
opt.eval()
w_eval = {k: v.clone() for k, v in m.state_dict().items()}
opt.train()
w_train2 = {k: v.clone() for k, v in m.state_dict().items()}

d_te = max((w_train[k] - w_eval[k]).abs().max().item() for k in w_train)
d_tt = max((w_train[k] - w_train2[k]).abs().max().item() for k in w_train)
print(f"max |train - eval| weights: {d_te:.3e}   (must be > 0: the two differ)")
print(f"max |train - train| weights: {d_tt:.3e}  (must be 0: train() restores)")
assert d_te > 1e-8, "eval() did not change the weights -- the guard is pointless"
# The swap is arithmetic (x <-> z interpolation), not a copy, so the round trip
# carries fp32 round-off. 1e-6 is still ~1000x below the train/eval difference
# above, so this stays a real check that train() puts the iterate back.
assert d_tt < 1e-6, "train() did not restore the iterate"
print("schedule-free mode swap verified: saving in eval mode saves a different, "
      "and correct, set of weights")
