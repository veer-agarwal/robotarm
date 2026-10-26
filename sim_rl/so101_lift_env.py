"""
SO-101 "pick up the block" task in MuJoCo, as a Gymnasium environment.

Arm model: TheRobotStudio SO-ARM100 so101_new_calib.xml (Apache-2.0), unchanged except:
  - the fixed-jaw and moving-jaw meshes stop colliding (MuJoCo collides meshes as convex hulls, and the
    fixed jaw's hull fills the gap between the fingers), replaced by box finger pads placed on the real
    finger surfaces (measured from the meshes);
  - a 25 mm, 20 g block on a free joint, a floor and a light are added.

Action (6): per-step change of the joint targets, like the gamepad teleop. Arm joints move up to
MAX_DELTA rad per step, the gripper target is set directly from the last action entry.
Control at 20 Hz (25 physics steps of 2 ms). Episode = 100 steps (5 s).

Observation (state, not pixels): joint positions + velocities, grasp point, block position / quaternion /
velocity, block - grasp vector, block height above its start.

Reward (dense): reach + grasp + lift + hold. Success = block lifted 4 cm above its start while grasped.
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_XML = os.path.join(HERE, "so101", "so101_new_calib.xml")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME = np.array([0.0, -0.9, 0.9, 0.9, 0.0, 0.3])      # arm up and back, gripper slightly open
BLOCK = 0.0125                                       # half size (25 mm cube)
BLOCK_X, BLOCK_Y = (0.17, 0.27), (-0.12, 0.12)       # spawn area in front of the base
CTRL_DT, N_SUB = 0.05, 25
MAX_DELTA = 0.05
EP_STEPS = 100
LIFT_SUCCESS = 0.04
GRIP_OPEN, GRIP_CLOSED = 1.0, -0.17


def build_model(target_marker=False):
    spec = mujoco.MjSpec.from_file(ARM_XML)
    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], rgba=[0.35, 0.4, 0.45, 1])

    grip = spec.body("gripper"); jaw = spec.body("moving_jaw_so101_v1")
    for body in (grip, jaw):
        for g in body.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1") \
                    and g.contype:
                g.contype = 0; g.conaffinity = 0
    pad = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0035, 0.008, 0.022], friction=[2.0, 0.02, 0.002],
               condim=4, solref=[0.004, 1], solimp=[0.99, 0.999, 0.0005, 0.5, 2], rgba=[0.9, 0.3, 0.2, 1])
    grip.add_geom(name="pad_fixed", pos=[-0.0115, 0, -0.080], **pad)
    # moving pad: mirror of the fixed pad in the gripper frame at q_gripper = 0, expressed in the jaw frame
    pj, qj = _moving_pad_pose()
    jaw.add_geom(name="pad_moving", pos=pj, quat=qj, **pad)
    # grasp point: block centre when it sits against the fixed pad, 10 mm above the pad bottom
    grip.add_site(name="grasp", pos=[0.0055, 0.0, -0.092], size=[0.004, 0, 0], rgba=[0, 1, 0, 1])

    blk = wb.add_body(name="block", pos=[0.22, 0, BLOCK])
    blk.add_freejoint(name="block_free")
    blk.add_geom(name="block_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK] * 3, mass=0.02,
                 friction=[1.5, 0.02, 0.002], condim=4, solref=[0.004, 1], solimp=[0.99, 0.999, 0.0005, 0.5, 2],
                 rgba=[0.2, 0.5, 0.9, 1])
    if target_marker:   # place-task goal: a flat green disc on a mocap body, no collisions
        tgt = wb.add_body(name="target", pos=[0.22, 0.05, 0.0005], mocap=True)
        tgt.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.0005, 0],
                     contype=0, conaffinity=0, rgba=[0.1, 0.9, 0.2, 0.6])
    return spec.compile()


def _moving_pad_pose():
    """Gripper-frame pose (0.0115, 0, -0.080) at q_gripper = 0, converted to the moving-jaw body frame."""
    m = mujoco.MjModel.from_xml_path(ARM_XML); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    gb, jb = m.body("gripper").id, m.body("moving_jaw_so101_v1").id
    Rg, Rj = d.xmat[gb].reshape(3, 3), d.xmat[jb].reshape(3, 3)
    pw = d.xpos[gb] + Rg @ np.array([0.0115, 0, -0.080])
    pj = Rj.T @ (pw - d.xpos[jb])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, (Rj.T @ Rg).flatten())
    return pj.tolist(), q.tolist()


class SO101LiftEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CTRL_DT)}

    def __init__(self, render_mode=None, seed=None, target_marker=False):
        self.model = build_model(target_marker)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        self.act_ids = np.array([m.actuator(j).id for j in JOINTS])
        self.lo = m.actuator_ctrlrange[self.act_ids, 0].copy(); self.hi = m.actuator_ctrlrange[self.act_ids, 1].copy()
        self.lo[5], self.hi[5] = GRIP_CLOSED, GRIP_OPEN
        self.block_q = m.joint("block_free").qposadr[0]; self.block_d = m.joint("block_free").dofadr[0]
        self.grasp_site = m.site("grasp").id
        self.block_geom = m.geom("block_geom").id
        self.pads = {m.geom("pad_fixed").id, m.geom("pad_moving").id}
        self.floor = m.geom("floor").id
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (30,), np.float32)
        self.render_mode = render_mode
        self._renderer = None
        self.np_random = np.random.default_rng(seed)

    # ---------------------------------------------------------------- helpers
    def _grasp_pos(self): return self.data.site_xpos[self.grasp_site].copy()
    def _block_pos(self): return self.data.qpos[self.block_q:self.block_q + 3].copy()

    def _contacts(self):
        touch = set(); block_floor = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]; pair = {c.geom1, c.geom2}
            if self.block_geom in pair:
                other = (pair - {self.block_geom}).pop() if len(pair) > 1 else None
                if other in self.pads: touch.add(other)
                if other == self.floor: block_floor = True
        return touch, block_floor

    def _obs(self):
        d = self.data
        gp, bp = self._grasp_pos(), self._block_pos()
        bq = d.qpos[self.block_q + 3:self.block_q + 7]
        bv = d.qvel[self.block_d:self.block_d + 3]
        return np.concatenate([d.qpos[self.qadr], np.clip(d.qvel[self.dadr], -10, 10) * 0.1, gp, bp, bq, bv,
                               bp - gp, [bp[2] - self.z0], [float(len(self._contacts()[0]) == 2)]]).astype(np.float32)

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        if seed is not None: self.np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        d = self.data; r = self.np_random
        q = HOME + r.uniform(-0.05, 0.05, 6) * np.array([1, 1, 1, 1, 1, 0])
        d.qpos[self.qadr] = q; d.ctrl[self.act_ids] = q
        bx, by = r.uniform(*BLOCK_X), r.uniform(*BLOCK_Y); yaw = r.uniform(-np.pi, np.pi)
        d.qpos[self.block_q:self.block_q + 7] = [bx, by, BLOCK + 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, d)
        for _ in range(20): mujoco.mj_step(self.model, d)      # settle
        self.z0 = self._block_pos()[2]
        self.target = d.ctrl[self.act_ids].copy()
        self.t = 0; self.held = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        self.target[:5] = np.clip(self.target[:5] + a[:5] * MAX_DELTA, self.lo[:5], self.hi[:5])
        self.target[5] = GRIP_CLOSED + (a[5] + 1) / 2 * (GRIP_OPEN - GRIP_CLOSED)
        self.data.ctrl[self.act_ids] = self.target
        for _ in range(N_SUB): mujoco.mj_step(self.model, self.data)
        self.t += 1

        gp, bp = self._grasp_pos(), self._block_pos()
        dist = np.linalg.norm(bp - gp)
        touch, _ = self._contacts()
        grasped = len(touch) == 2
        lift = bp[2] - self.z0
        r_reach = 1 - np.tanh(10 * dist)
        r_grasp = 0.5 * len(touch) / 2 + (0.5 if grasped else 0.0)
        r_lift = (4.0 * np.clip(lift / LIFT_SUCCESS, 0, 1)) if grasped else 0.0
        success = grasped and lift > LIFT_SUCCESS
        self.held = self.held + 1 if success else 0
        reward = r_reach + r_grasp + r_lift + (2.0 if success else 0.0) - 0.05 * float(np.sum(a[:5] ** 2))
        off_table = bp[2] < -0.05 or np.linalg.norm(bp[:2]) > 0.6
        terminated = bool(self.held >= 10 or off_table)
        truncated = self.t >= EP_STEPS
        info = {"is_success": float(self.held >= 10 or success), "lift": float(lift), "dist": float(dist),
                "grasped": float(grasped)}
        if self.held >= 10: reward += 20.0
        return self._obs(), float(reward), terminated, bool(truncated), info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 360, 480)
            self._cam = mujoco.MjvCamera(); self._cam.lookat[:] = [0.2, 0, 0.08]
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.65, 150, -25
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None: self._renderer.close(); self._renderer = None
