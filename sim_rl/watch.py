"""Watch a trained policy in the MuJoCo viewer (real time).  python watch.py [path\\to\\model.zip]"""
import sys, time, glob, os
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import SAC
from so101_lift_env import SO101LiftEnv, CTRL_DT

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("runs", "sac_lift", "best_model.zip")
model = SAC.load(path, device="cpu"); env = SO101LiftEnv(seed=0)
ok = n = 0
with mujoco.viewer.launch_passive(env.model, env.data) as v:
    while v.is_running():
        obs, _ = env.reset(seed=n); done = False; s = 0
        while not done and v.is_running():
            t = time.time()
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a); done = term or trunc; s = max(s, info["is_success"])
            v.sync(); time.sleep(max(0, CTRL_DT - (time.time() - t)))
        n += 1; ok += s; print(f"episode {n}: {'success' if s else 'fail'}   ({ok}/{n})")
        time.sleep(0.5)
