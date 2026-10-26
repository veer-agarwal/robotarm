"""
SO-101 pick-and-place in MuJoCo: pick the block up, carry it to the green target disc, put it down there,
let go, and move the gripper away.

Builds on so101_lift_env (same arm, finger pads, block, actions, 20 Hz control).
Episode = 160 steps (8 s). Target spawns 6-20 cm from the block, inside the same reachable patch.

Success (10 consecutive steps; the episode keeps running so releasing is never worth less than hovering): block within 2 cm of the target centre, resting on the table, no finger touching it,
grasp point at least 3 cm above the table, and the block was lifted >= 3 cm at some point (so pushing the
block along the table doesn't count).

Reward stages (dense): reach -> grasp -> lift -> carry towards the target -> lower onto it -> release -> clear.
"""
import numpy as np
import mujoco
from gymnasium import spaces
from so101_lift_env import SO101LiftEnv, BLOCK, BLOCK_X, BLOCK_Y, GRIP_OPEN, GRIP_CLOSED, MAX_DELTA, N_SUB

EP_STEPS = 160
PLACE_TOL = 0.02
LIFTED = 0.03


class SO101PlaceEnv(SO101LiftEnv):
    def __init__(self, render_mode=None, seed=None):
        super().__init__(render_mode, seed, target_marker=True)
        self.mocap_id = self.model.body("target").mocapid[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (38,), np.float32)

    def _obs(self):
        base = super()._obs()
        bp = self._block_pos()
        return np.concatenate([base, self.goal, self.goal - bp, [float(self.was_lifted)],
                               [float(len(self._contacts()[0]) > 0)]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.was_lifted = False
        self.goal = np.zeros(3)
        _, info = super().reset(seed=seed, options=options)
        r = self.np_random; bp = self._block_pos()
        for _ in range(100):
            g = np.array([r.uniform(*BLOCK_X), r.uniform(*BLOCK_Y), BLOCK])
            if 0.06 <= np.linalg.norm(g[:2] - bp[:2]) <= 0.20: break
        self.goal = g
        self.data.mocap_pos[self.mocap_id] = [g[0], g[1], 0.0005]
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), info

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        self.target[:5] = np.clip(self.target[:5] + a[:5] * MAX_DELTA, self.lo[:5], self.hi[:5])
        self.target[5] = GRIP_CLOSED + (a[5] + 1) / 2 * (GRIP_OPEN - GRIP_CLOSED)
        self.data.ctrl[self.act_ids] = self.target
        for _ in range(N_SUB): mujoco.mj_step(self.model, self.data)
        self.t += 1

        gp, bp = self._grasp_pos(), self._block_pos()
        touch, on_floor = self._contacts()
        grasped = len(touch) == 2
        lift = bp[2] - self.z0
        if grasped and lift > LIFTED: self.was_lifted = True
        d_grip = np.linalg.norm(bp - gp)
        d_goal_xy = np.linalg.norm(bp[:2] - self.goal[:2])
        resting = abs(bp[2] - self.z0) < 0.004
        at_goal = d_goal_xy < PLACE_TOL

        if not self.was_lifted:
            # phase 1: reach, grasp, lift (same shaping as the lift task)
            reward = (1 - np.tanh(10 * d_grip)) + 0.5 * len(touch) / 2 + (0.5 if grasped else 0.0) \
                     + (2.0 * np.clip(lift / LIFTED, 0, 1) if grasped else 0.0)
        else:
            # phase 2: carry, lower, release, clear. Always worth more than phase 1.
            reward = 4.0 + 3.0 * (1 - np.tanh(8 * d_goal_xy))
            if at_goal:
                reward += 2.0 * (1 - np.tanh(50 * max(0.0, bp[2] - self.z0)))   # lower it
                if resting:
                    reward += 2.0 if not touch else 0.0                           # let go
                    reward += 2.0 * np.clip((gp[2] - (BLOCK + 0.01)) / 0.03, 0, 1) if not touch else 0.0  # clear
            elif grasped:
                reward += 1.0                                                     # keep holding while carrying
        success = self.was_lifted and at_goal and resting and not touch and gp[2] > BLOCK + 0.03
        self.held = self.held + 1 if success else 0
        reward += 5.0 if success else 0.0
        reward -= 0.05 * float(np.sum(a[:5] ** 2))
        off_table = bp[2] < -0.05 or np.linalg.norm(bp[:2]) > 0.6
        # no early termination on success: the placed-and-released state simply pays the most per step
        # (18 vs <= 9 for hovering with the block), so the policy is never better off holding on.
        terminated = bool(off_table)
        truncated = self.t >= EP_STEPS
        info = {"is_success": float(self.held >= 10), "lift": float(lift), "dist": float(d_grip),
                "grasped": float(grasped), "goal_dist": float(d_goal_xy), "was_lifted": float(self.was_lifted)}
        return self._obs(), float(reward), terminated, bool(truncated), info
