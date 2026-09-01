"""Wire compare_dump into the eval harness so every run leaves visual evidence."""
import io

p = "stratified_eval.py"
s = io.open(p, encoding="utf-8").read()

a = "from model_span import SPANLite  # noqa: E402"
b = """from model_span import SPANLite  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_dump import save_comparison  # noqa: E402"""
assert s.count(a) == 1, f"import anchor {s.count(a)}"
s = s.replace(a, b)

# evaluate() needs to know where to write and which frames to sample
a = "def evaluate(models, hr_paths, lr_paths, lo, hi, perc, scale, device):"
b = ("def evaluate(models, hr_paths, lr_paths, lo, hi, perc, scale, device,\n"
     "             evidence_dir=None, clip_name=\"clip\", crf=0, every=8):")
assert s.count(a) == 1, f"signature {s.count(a)}"
s = s.replace(a, b)

a = """            prev[name] = sr
        prev_hr = hr"""
b = """            prev[name] = sr
            frame_out[name] = sr
        # Save evidence periodically rather than every frame: enough to see
        # the trend across a clip without writing thousands of PNGs.
        if evidence_dir is not None and i % every == 0:
            try:
                save_comparison(evidence_dir, clip_name, i, crf, hr,
                                frame_out, m["texture"])
            except Exception as e:
                print(f"    [evidence] {type(e).__name__}: {e}", file=sys.stderr)
        prev_hr = hr"""
assert s.count(a) == 1, f"loop tail {s.count(a)}"
s = s.replace(a, b)

a = """        m = masks_for(hr, lo, hi)

        for name, fn in models.items():"""
b = """        m = masks_for(hr, lo, hi)
        frame_out = {}

        for name, fn in models.items():"""
assert s.count(a) == 1, f"loop head {s.count(a)}"
s = s.replace(a, b)

a = "            acc = evaluate(models, hr, lr, lo, hi, perc, args.scale, device)"
b = ("            acc = evaluate(models, hr, lr, lo, hi, perc, args.scale, device,\n"
     "                           evidence_dir=args.evidence, clip_name=clip.stem,\n"
     "                           crf=args.crf)")
assert s.count(a) == 1, f"call site {s.count(a)}"
s = s.replace(a, b)

a = '''    ap.add_argument("--json", help="write raw results here")'''
b = '''    ap.add_argument("--json", help="write raw results here")
    ap.add_argument("--evidence", default="/tank/webvsr/evidence",
                    help="directory for side-by-side comparison crops; these are "
                         "the visual record behind the numbers and are never "
                         "deleted. Pass an empty string to disable.")'''
assert s.count(a) == 1, f"argparse {s.count(a)}"
s = s.replace(a, b)

a = "    device = torch.device(args.device)"
b = """    if not args.evidence:
        args.evidence = None
    device = torch.device(args.device)"""
assert s.count(a) == 1, f"device {s.count(a)}"
s = s.replace(a, b)

io.open(p, "w", encoding="utf-8").write(s)
print("stratified_eval.py: evidence saving wired")
