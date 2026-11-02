"""Cut a time window out of a mjlab-format motion npz (every per-frame array sliced, the rest kept), for a short
booster_deploy clip that starts and ends standing.
  python scripts/t2/crop_clip.py IN.npz OUT.npz --start 0 --end 10.75     (seconds)
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("--start", type=float, default=0.0)
parser.add_argument("--end", type=float, required=True)
args = parser.parse_args()

d = np.load(args.src)
fps = float(np.asarray(d["fps"]).reshape(-1)[0])
num_frames = len(d["joint_pos"])
a, b = int(round(args.start * fps)), min(int(round(args.end * fps)), num_frames)
out = {}
for key in d.files:
    v = d[key]
    out[key] = v[a:b] if v.ndim > 0 and len(v) == num_frames else v
out["source"] = np.array(f"{d['source']} [{args.start:g}-{args.end:g} s]")
np.savez(args.dst, **out)
print(f"{args.dst}: frames {a}-{b} of {num_frames} ({(b - a) / fps:.2f} s)")
