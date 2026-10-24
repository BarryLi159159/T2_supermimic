# SuperMimic tracker as a booster_deploy task

Symlink this folder to `booster_deploy/tasks/supermimic`; every `robots/t2/motions/*.npz` registers a task
`t2_supermimic_<clip>`. Run from the `teleopit` env with `PYTHONPATH` including `booster_assets/src`:

    python deploy.py --task t2_supermimic_walk1_subject1 --mujoco   # sim2sim with the reference ghost
    python deploy.py --task t2_supermimic_walk1_subject1            # real robot (X prepare, A start, B stop)

Model: `robots/t2/models/t2_supermimic.onnx` + `.json` from `scripts/export_onnx.py` (fixed batch 1; the json carries
joint order, per-joint action scale and default pose). Clips: mjlab-format npz at 50 Hz, trunk-rooted T2
(`scripts/t2/pkl_to_npz_t2.py` output). `obs_layout.py` is the observation contract; `scripts/t2/check_deploy_obs.py`
asserts it against the mjlab environment (1e-7 on a 30-step rollout).

IMU: the policy root is the trunk, the IMU body, so `robot.data.root_quat_w` / `root_ang_vel_b` enter the observation
directly. Gains are the v3 contract (t2_walk hard table, wrists 18.21 / 1.449, head 20 / 30); the head holds zero.
