"""Feasibility check: scripted IK grasp must lift the block, and the sim must run fast enough."""
import time, numpy as np, mujoco
from so101_lift_env import SO101LiftEnv, MAX_DELTA

env = SO101LiftEnv(); m, d = env.model, env.data
gb = m.body("gripper").id

def ik_targets(goal, q, iters=60):
    """damped least squares on joints 0..3: grasp site -> goal, gripper z axis -> world +z (fingers down)."""
    q = q.copy(); dd = mujoco.MjData(m); dd.qpos[:] = d.qpos
    for _ in range(iters):
        dd.qpos[env.qadr] = q; mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
        p = dd.site_xpos[env.grasp_site]; zax = dd.xmat[gb].reshape(3, 3)[:, 2]
        e = np.concatenate([goal - p, 0.05 * np.cross(zax, [0, 0, 1])])
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, dd, jp, jr, env.grasp_site)
        J = np.vstack([jp, 0.05 * jr])[:, env.dadr[:4]]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q[:4] = np.clip(q[:4] + dq, env.lo[:4], env.hi[:4])
    return q

def roll_for_block(q):
    """wrist roll that lines the jaw axis (gripper x) up with a block face normal."""
    bq = d.qpos[env.block_q + 3:env.block_q + 7]; R = np.zeros(9); mujoco.mju_quat2Mat(R, bq); bx = R.reshape(3, 3)[:, 0]
    best, bestv = 0.0, -1
    dd = mujoco.MjData(m); dd.qpos[:] = d.qpos
    for r in np.linspace(-np.pi / 2, np.pi / 2, 73):
        qq = q.copy(); qq[4] = r; dd.qpos[env.qadr] = qq; mujoco.mj_kinematics(m, dd)
        gx = dd.xmat[gb].reshape(3, 3)[:, 0]; v = abs(gx @ bx) + 0 * r
        v = max(v, abs(gx @ R.reshape(3, 3)[:, 1]))
        if v > bestv: best, bestv = r, v
    return best

def drive(qt, grip, steps):
    for _ in range(steps):
        a = np.zeros(6); a[:5] = np.clip((qt[:5] - env.target[:5]) / MAX_DELTA, -1, 1)
        a[5] = grip
        obs, r, term, trunc, info = env.step(a)
    return info

succ = 0; N = 20; t0 = time.time(); steps = 0
for ep in range(N):
    env.reset(seed=ep); env.t = -10**6
    bp = env._block_pos()
    q = env.target.copy()
    q[4] = roll_for_block(ik_targets(bp + [0, 0, 0.06], q))
    q1 = ik_targets(bp + [0, 0, 0.06], q)
    drive(q1, 1.0, 25)
    q2 = ik_targets(bp + [0, 0, 0.003], q1); q2[4] = q1[4]
    drive(q2, 1.0, 20)
    drive(q2, -1.0, 10)                       # close
    q3 = ik_targets(bp + [0, 0, 0.09], q2); q3[4] = q1[4]
    info = drive(q3, -1.0, 25)
    steps += 80
    lifted = env._block_pos()[2] - env.z0
    print(f"ep {ep}: block at {bp[:2].round(3)}  lift {lifted*100:.1f} cm  grasped {info['grasped']:.0f}  grasp-block dist {info['dist']*100:.1f} cm")
    succ += lifted > 0.04
dt = time.time() - t0
print(f"scripted success {succ}/{N};  sim speed {steps/dt:.0f} control steps/s ({steps*25/dt:.0f} physics steps/s) on 1 core")
import sys
sys.exit(0 if succ >= 0.9 * N else 1)
