"""Export a SuperMimic policy checkpoint to ONNX for booster_deploy.

Inputs  obs_hist (1, 930) float32: paper proprioception, term-major 10-frame history, unscaled (the fixed per-term
                                   scales are buffers inside the graph);
        goal     (1, 680) float32: 10 future reference frames x 68 (+9 with anchor_obs), unscaled.
Output  action   (1, 29)  float32: the policy mean; the deployment target is default + scale * action per joint.
The batch dimension is fixed at 1 on purpose: booster_deploy's CpuOnnxRunner returns its first result forever on
dynamic-batch graphs (onnxruntime 1.23.2); fixed-shape graphs are unaffected.

  uv run --group export python scripts/export_onnx.py --ckpt logs/<run>/ckpt_latest.pt --out models/<run>.onnx
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import tyro

from supermimic.eval.benchmark import load_policy_from_checkpoint


class _Exported(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.policy(obs_hist, goal)


def main(ckpt: Path, out: Path | None = None, atol: float = 1e-5) -> None:
    policy, cfg = load_policy_from_checkpoint(ckpt, "cpu")
    goal_dim = cfg["policy"]["goal_dim"]
    obs_dim = policy.decoder.scale.numel()
    out = out or ckpt.with_suffix(".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _Exported(policy).eval()
    obs = torch.zeros(1, obs_dim)
    goal = torch.zeros(1, goal_dim)
    torch.onnx.export(wrapped, (obs, goal), str(out), input_names=["obs_hist", "goal"], output_names=["action"], opset_version=17, dynamo=False)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        o = rng.normal(size=(1, obs_dim)).astype(np.float32)
        g = rng.normal(size=(1, goal_dim)).astype(np.float32)
        ref = wrapped(torch.as_tensor(o), torch.as_tensor(g)).detach().numpy()
        got = sess.run(["action"], {"obs_hist": o, "goal": g})[0]
        worst = max(worst, float(np.abs(got - ref).max()))
    assert worst < atol, f"onnx vs torch max abs diff {worst}"
    meta = {
        "checkpoint": str(ckpt),
        "iteration": int(torch.load(ckpt, map_location="cpu", weights_only=False)["iteration"]),
        "robot": cfg["env"].get("robot", "g1"),
        "obs_dim": obs_dim,
        "goal_dim": goal_dim,
        "history": cfg["policy"]["history"],
        "future_frames": cfg["env"]["future_frames"],
        "anchor_obs": cfg["env"].get("anchor_obs", False),
        "obs_term_scale": cfg["policy"]["obs_term_scale"],
        "goal_term_scale": cfg["policy"]["goal_term_scale"],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"{out}: obs {obs_dim} goal {goal_dim} -> 29, onnx vs torch {worst:.1e}, meta {out.with_suffix('.json').name}")


if __name__ == "__main__":
    tyro.cli(main)
