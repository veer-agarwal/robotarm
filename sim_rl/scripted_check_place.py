"""Feasibility check for pick-and-place: scripted IK pick, carry, place, release, clear must succeed."""
import sys, time, numpy as np, mujoco
from so101_place_env import SO101PlaceEnv
from so101_lift_env import MAX_DELTA

env = SO101PlaceEnv(); m, d = env.model, env.data
gb = m.body("gripper").id

def ik(goal, q, iters=60):
    q = q.copy(); dd = mujoco.MjData(m); dd.qpos[:] = d.qpos
    for _ in range(iters):
        dd.qpos[env.qadr] = q; mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
        p = dd.site_xpos[env.grasp_site]; zax = dd.xmat[gb].reshape(3, 3)[:, 2]
        e = np.concatenate([goal - p, 0.05 * np.cross(zax, [0, 0, 1])])
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, dd, jp, jr, env.grasp_site)
        J = np.vstack([jp, 0.05 * jr])[:, env.dadr[:4]]
        q[:4] = np.clip(q[:4] + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), env.lo[:4], env.hi[:4])
    return q

def roll_for_block(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[env.block_q + 3:env.block_q + 7]); R = R.reshape(3, 3)
    dd = mujoco.MjData(m); dd.qpos[:] = d.qpos; best, bv = 0.0, -1
    for r in np.linspace(-np.pi / 2, np.pi / 2, 73):
        qq = q.copy(); qq[4] = r; dd.qpos[env.qadr] = qq; mujoco.mj_kinematics(m, dd)
        gx = dd.xmat[gb].reshape(3, 3)[:, 0]; v = max(abs(gx @ R[:, 0]), abs(gx @ R[:, 1]))
        if v > bv: best, bv = r, v
    return best

def drive(qt, grip, steps):
    info = None
    for _ in range(steps):
        a = np.zeros(6); a[:5] = np.clip((qt[:5] - env.target[:5]) / MAX_DELTA, -1, 1); a[5] = grip
        _, _, term, trunc, info = env.step(a)
        if term: break
    return info

N, ok, t0 = 20, 0, time.time()
for ep in range(N):
    env.reset(seed=ep); env.t = -10**6
    bp, g = env._block_pos(), env.goal.copy(); q = env.target.copy()
    q[4] = roll_for_block(ik(bp + [0, 0, 0.06], q))
    q1 = ik(bp + [0, 0, 0.06], q); drive(q1, 1.0, 25)
    q2 = ik(bp + [0, 0, 0.003], q1); drive(q2, 1.0, 20)
    drive(q2, -1.0, 10)
    q3 = ik(bp + [0, 0, 0.08], q2); drive(q3, -1.0, 20)
    q4 = ik(g + [0, 0, 0.08], q3); drive(q4, -1.0, 30)
    q5 = ik(g + [0, 0, 0.004], q4); drive(q5, -1.0, 20)
    drive(q5, 1.0, 10)                                   # release
    gx = d.xmat[gb].reshape(3, 3)[:, 0]; gx[2] = 0; gx /= np.linalg.norm(gx)
    q6 = ik(env._grasp_pos() - 0.015 * gx, q5); drive(q6, 1.0, 10)   # slide the fixed finger off the block face
    q7 = ik(env._grasp_pos() + [0, 0, 0.07], q6); info = drive(q7, 1.0, 40)
    s = info["is_success"]; ok += s
    print(f"ep {ep}: goal dist {info['goal_dist']*100:4.1f} cm  lifted {info['was_lifted']:.0f}  success {s:.0f}")
print(f"scripted pick-and-place success {ok}/{N}  ({time.time()-t0:.0f} s)")
sys.exit(0 if ok >= 0.9 * N else 1)
