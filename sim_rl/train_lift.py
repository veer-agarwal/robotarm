"""
Train SAC on the SO-101 "pick up the block" simulation (so101_lift_env.py).

    python train_lift.py --hours 30            # full run, stops on its own after 30 h
    python train_lift.py --hours 0.05 --envs 2 # smoke test

Writes to runs/<name>/:
    progress.csv     every eval: timesteps, wall hours, success rate, mean reward, mean lift
    tb/              tensorboard logs  (tensorboard --logdir runs)
    best_model.zip   best eval success so far
    checkpoints/     every 250k steps (resume with --resume)
    videos/          one rollout mp4 per eval (skipped if offscreen rendering is unavailable)
"""
import argparse, csv, glob, os, time
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from so101_lift_env import SO101LiftEnv
from so101_place_env import SO101PlaceEnv
ENVS = {"lift": SO101LiftEnv, "place": SO101PlaceEnv}


def make_env(task, rank, seed):
    def f():
        return ENVS[task](seed=seed + rank)
    return f


class EvalAndStop(BaseCallback):
    """Every `every` steps: deterministic eval on fixed seeds, CSV + tensorboard, best model, video.
    Stops training when the wall-clock budget is used up."""

    def __init__(self, run_dir, every, n_eval, hours, task="lift"):
        super().__init__()
        self.run_dir, self.every, self.n_eval = run_dir, every, n_eval
        self.deadline = time.time() + hours * 3600; self.t0 = time.time()
        self.best = -1.0; self.next_eval = every
        self.env = ENVS[task](seed=12345)
        self.csv = os.path.join(run_dir, "progress.csv")
        if not os.path.exists(self.csv):
            with open(self.csv, "w", newline="") as f:
                csv.writer(f).writerow(["timesteps", "hours", "success_rate", "mean_reward", "mean_max_lift_cm", "grasp_rate"])

    def _eval(self):
        succ, rews, lifts, grasps = [], [], [], []
        frames = []
        for i in range(self.n_eval):
            obs, _ = self.env.reset(seed=10_000 + i)
            done, R, mx, g, s = False, 0.0, 0.0, 0.0, 0.0
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.env.step(a)
                R += r; mx = max(mx, info["lift"]); g = max(g, info["grasped"]); s = max(s, info["is_success"])
                done = term or trunc
                if i == 0 and self.video_ok:
                    try: frames.append(self.env.render())
                    except Exception: self.video_ok = False
            succ.append(s); rews.append(R); lifts.append(mx * 100); grasps.append(g)
        return float(np.mean(succ)), float(np.mean(rews)), float(np.mean(lifts)), float(np.mean(grasps)), frames

    def _on_training_start(self):
        self.video_ok = True

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            sr, mr, ml, gr, frames = self._eval()
            hrs = (time.time() - self.t0) / 3600
            with open(self.csv, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, round(hrs, 3), sr, round(mr, 2), round(ml, 2), gr])
            self.logger.record("eval/success_rate", sr); self.logger.record("eval/mean_reward", mr)
            self.logger.record("eval/max_lift_cm", ml); self.logger.record("eval/grasp_rate", gr)
            print(f"[{hrs:5.2f} h] {self.num_timesteps:>9,d} steps  success {sr:.0%}  grasp {gr:.0%}  lift {ml:4.1f} cm  reward {mr:7.1f}", flush=True)
            if sr >= self.best:
                self.best = sr; self.model.save(os.path.join(self.run_dir, "best_model"))
            if frames:
                try:
                    import imageio
                    os.makedirs(os.path.join(self.run_dir, "videos"), exist_ok=True)
                    imageio.mimsave(os.path.join(self.run_dir, "videos", f"eval_{self.num_timesteps:09d}.mp4"), frames, fps=20)
                except Exception as e:
                    print("video skipped:", e); self.video_ok = False
        return time.time() < self.deadline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(ENVS), default="lift")
    ap.add_argument("--name", default=None)
    ap.add_argument("--hours", type=float, default=30.0)
    ap.add_argument("--envs", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--steps", type=int, default=50_000_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.name = args.name or f"sac_{args.task}"
    run_dir = os.path.join("runs", args.name); os.makedirs(run_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"task {args.task}, device {device}, {args.envs} envs, budget {args.hours} h, run dir {run_dir}", flush=True)

    fns = [make_env(args.task, i, args.seed * 100) for i in range(args.envs)]
    venv = VecMonitor(SubprocVecEnv(fns) if args.envs > 1 else DummyVecEnv(fns))

    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*.zip")), key=os.path.getmtime)
    if args.resume and ckpts:
        print("resuming from", ckpts[-1])
        model = SAC.load(ckpts[-1], env=venv, device=device)
    else:
        model = SAC("MlpPolicy", venv, device=device, seed=args.seed,
                    learning_rate=3e-4, buffer_size=2_000_000, batch_size=512, gamma=0.98, tau=0.005,
                    learning_starts=20_000, train_freq=1, gradient_steps=max(1, args.envs // 2),
                    ent_coef="auto", policy_kwargs=dict(net_arch=[256, 256, 256]),
                    tensorboard_log=os.path.join(run_dir, "tb"), verbose=0)

    cbs = CallbackList([
        CheckpointCallback(save_freq=max(1, 250_000 // args.envs), save_path=os.path.join(run_dir, "checkpoints"), name_prefix="sac"),
        EvalAndStop(run_dir, args.eval_every, args.n_eval, args.hours, args.task),
    ])
    model.learn(total_timesteps=args.steps, callback=cbs, reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(os.path.join(run_dir, "final_model"))
    venv.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
