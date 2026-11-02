#!/bin/bash
# The kitchen-reference deploy clips the Booster_T2 panel offers (config/deploy_config.py KITCHEN_CLIPS, kre_*):
# t2_kitchen_reference.zip (unpacked at $1) -> SuperMimic npz per piece + full (data/npz/t2_kitchen/kr_*), then eased in /
# out of the standing pose into the deploy motions folder. Run from the repo root.
#   bash scripts/t2/make_kitchen_clips.sh <unpacked t2_kitchen_reference dir>
set -e
RAW=data/npz/t2_kitchen/raw
OUT=data/npz/t2_kitchen
DST=deploy/booster_deploy_supermimic/robots/t2/motions
env -u PYTHONPATH ~/.local/bin/uv run python scripts/t2/convert_kitchen.py "$1" scripts/t2/kitchen_joint_map.json "$RAW"
for f in "$RAW"/*.npz; do
  n=$(basename "$f" .npz)
  cp "$f" "$OUT/kr_${n%%_*}.npz"
  env -u PYTHONPATH ~/.local/bin/uv run python scripts/t2/blend_clip.py "$OUT/kr_${n%%_*}.npz" "$DST/kre_${n%%_*}.npz" --out
done
