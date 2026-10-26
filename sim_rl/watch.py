"""Watch a trained policy in the MuJoCo viewer (real time).
    python watch.py                      -> pick-and-place best model if it exists, else lift
    python watch.py runs\\sac_lift\\best_model.zip"""
import sys, time, glob, os
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import SAC
from so101_lift_env import SO101LiftEnv, CTRL_DT
from so101_place_env import SO101PlaceEnv

default = os.path.join("runs", "sac_place", "best_model.zip")
if not os.path.exists(default): default = os.path.join("runs", "sac_lift", "best_model.zip")
path = sys.argv[1] if len(sys.argv) > 1 else default
env = SO101PlaceEnv(seed=0) if "place" in path else SO101LiftEnv(seed=0)
model = SAC.load(path, device="cpu"); print("watching", path)
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
