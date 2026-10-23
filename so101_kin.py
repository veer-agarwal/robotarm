"""Minimal SO-101 forward kinematics from the official URDF (so101_new_calib), no dependencies but numpy.
Joint angles in DEGREES, in LeRobot's calibrated frame (0 = calibration middle pose)."""
import numpy as np

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def _T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T

def _Rz(t):
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; return T

# (name, origin xyz, origin rpy) in chain order, from so101_new_calib.urdf; all joints rotate about local z
CHAIN = [
    ("shoulder_pan",  (0.0388353, 0.0, 0.0624),          (3.14159, 0.0, -3.14159)),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    ("elbow_flex",    (-0.11257, -0.028, 0.0),           (0.0, 0.0, 1.5708)),
    ("wrist_flex",    (-0.1349, 0.0052, 0.0),            (0.0, 0.0, -1.5708)),
    ("wrist_roll",    (0.0, -0.0611, 0.0181),            (1.5708, 0.0486795, 3.14159)),
]
GRIPPER_FRAME = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))  # fixed, from gripper_link
JOINTS = [c[0] for c in CHAIN]

def fk(q_deg):
    """q_deg: dict or sequence of 5 angles (pan, lift, elbow, wrist_flex, wrist_roll) in degrees.
    Returns 4x4 pose of gripper_frame_link in base_link frame (metres)."""
    if isinstance(q_deg, dict):
        q_deg = [q_deg[j] for j in JOINTS]
    T = np.eye(4)
    for (name, xyz, rpy), q in zip(CHAIN, q_deg):
        T = T @ _T(xyz, rpy) @ _Rz(np.deg2rad(q))
    return T @ _T(*GRIPPER_FRAME)

def pos_pitch(q_deg):
    """Gripper position (x,y,z m) and approach-axis pitch (deg, 0 = horizontal, -90 = pointing straight down)."""
    T = fk(q_deg)
    p = T[:3, 3]
    a = T[:3, 2]                       # gripper frame z axis = approach direction
    pitch = np.degrees(np.arctan2(a[2], np.hypot(a[0], a[1])))
    return p, pitch

def ik(target_xyz, target_pitch, q0_deg, wrist_roll_deg, iters=30, damping=0.05, step_limit=8.0):
    """Damped least squares IK for pan, lift, elbow, wrist_flex (wrist_roll held). Angles in degrees."""
    q = np.array([q0_deg[j] for j in JOINTS[:4]], dtype=float)
    def f(qq):
        p, pitch = pos_pitch([*qq, wrist_roll_deg])
        return np.array([*p, np.deg2rad(pitch)])
    goal = np.array([*target_xyz, np.deg2rad(target_pitch)])
    for _ in range(iters):
        cur = f(q); err = goal - cur
        if np.linalg.norm(err[:3]) < 5e-4 and abs(err[3]) < np.deg2rad(0.5):
            break
        J = np.zeros((4, 4)); h = 1e-3
        for i in range(4):
            dq = q.copy(); dq[i] += h
            J[:, i] = (f(dq) - cur) / h
        W = np.diag([1, 1, 1, 0.04])     # metres vs radians weighting
        JW = W @ J; eW = W @ err
        dq = np.linalg.solve(JW.T @ JW + damping**2 * np.eye(4), JW.T @ eW)
        dq = np.clip(dq, -step_limit, step_limit)
        q += dq
    return dict(zip(JOINTS[:4], q))
