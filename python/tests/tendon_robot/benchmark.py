import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp


def hat(v):
    vx, vy, vz = v
    return np.array([[0, -vz,  vy],
                     [vz,  0, -vx],
                     [-vy, vx,  0]])


def pack_state(p, R, n, m):
    return np.hstack([p, R.reshape(-1), n, m])


def unpack_state(y):
    p = y[0:3]
    R = y[3:12].reshape(3,3)
    n = y[12:15]
    m = y[15:18]
    return p, R, n, m


def segment_dynamics(s, y, K_se_inv, K_bt_inv):
    p, R, n, m = unpack_state(y)

    u = K_bt_inv @ (R.T @ m)
    v = K_se_inv @ (R.T @ n) + np.array([0, 0, 1])

    p_dot = R @ v
    R_dot = R @ hat(u)

    n_dot = np.zeros(3)
    m_dot = -hat(p_dot) @ n

    return pack_state(p_dot, R_dot, n_dot, m_dot)


def integrate_robot(x, s_discs, K_se_inv, K_bt_inv):
    nm0 = x[:6]
    fl = x[6:].reshape(-1, 6)
    f = fl[:, :3]
    l = fl[:, 3:]

    p0 = np.zeros(3)
    R0 = np.array([[-1, 0, 0],
                   [ 0, 0, 1],
                   [ 0, 1, 0]])
    n0 = nm0[:3]
    m0 = nm0[3:]
    y0 = pack_state(p0, R0, n0, m0)

    s, p, R, n, m = [], [], [], [], []

    for k in range(len(s_discs) - 1):
        a, b = s_discs[k], s_discs[k + 1]

        sol = solve_ivp(segment_dynamics, [a, b], y0, args=(K_se_inv, K_bt_inv), method="DOP853")

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        
        for (s_i, state) in zip(sol.t, sol.y.T):
            p_i, R_i, n_i, m_i = unpack_state(state)
            s.append(s_i); p.append(p_i); R.append(R_i); n.append(n_i); m.append(m_i)

        y_end = sol.y[:, -1]
        p_end, R_end, n_end, m_end = unpack_state(y_end)
        
        U, _, Vt = np.linalg.svd(R_end)
        R_orth = U @ Vt
        if np.linalg.det(R_orth) < 0:
            U[:, -1] *= -1
            R_orth = U @ Vt

        y0 = pack_state(p_end, R_orth, n_end - f[k], m_end - l[k])

    return np.array(s), np.array(p), np.array(R), np.array(n), np.array(m)


def compute_residual(x, s_discs, K_se_inv, K_bt_inv, tensions, tip_force, holes):
    s, p, R, n, m = integrate_robot(x, s_discs, K_se_inv, K_bt_inv)
    f_pred, l_pred = compute_backbone_loads(s, p, R, s_discs, tensions, tip_force, holes)

    fl = x[6:].reshape(-1, 6)
    f  = fl[:, :3]
    l  = fl[:, 3:]

    e_n = n[-1] - f[-1]
    e_m = m[-1] - l[-1]
    e_f = (f - f_pred).reshape(-1)
    e_l = (l - l_pred).reshape(-1)

    return np.hstack([e_n, e_m, e_f, e_l])


def compute_backbone_loads(s, p, R, s_discs, tensions, tip_force, holes):
    idxs = np.abs(s[:, None] - s_discs[None, :]).argmin(axis=0)
    p_discs, R_discs = p[idxs], R[idxs]

    f, l = [], []

    for i in range(1, len(p_discs)):
        f_i = np.zeros(3)
        l_i = np.zeros(3)

        for j in range(len(tensions)):
            hole = R_discs[i] @ holes[i][j] + p_discs[i]
            hole_prev = R_discs[i - 1] @ holes[i - 1][j] + p_discs[i - 1]
            hole_diff_prev = hole_prev - hole
            f_j = tensions[j] * hole_diff_prev / np.linalg.norm(hole_diff_prev)
            
            if i + 1 < len(p_discs):
                hole_next = R_discs[i + 1] @ holes[i + 1][j] + p_discs[i + 1]
                hole_diff_next = hole_next - hole
                f_j += tensions[j] * hole_diff_next / np.linalg.norm(hole_diff_next)

            l_j = hat(hole - p_discs[i]) @ f_j
            f_i += f_j; l_i += l_j

        f.append(f_i); l.append(l_i)
    
    f[-1] += tip_force

    return f, l


def solve_kinematics_bvp(tensions, tip_force, config, holes, x0_guess=None):
    K_se_inv = config.K_inv[3:,3:]
    K_bt_inv = config.K_inv[:3,:3]

    robot_length = config.rod_length
    num_discs = config.num_discs
    s_discs = np.linspace(0, robot_length, num_discs)

    x0 = np.zeros(6 + 6*(config.num_discs - 1)) if x0_guess is None else x0_guess

    def get_res(x):
        return compute_residual(x, s_discs, K_se_inv, K_bt_inv, tensions, tip_force, holes)

    sol = root(get_res, x0, tol=1e-6, method='hybr')

    if not sol.success:
        raise RuntimeError(f"Root finding failed: {sol.message}")
    
    s, p, R, n, m = integrate_robot(sol.x, s_discs, K_se_inv, K_bt_inv)
    
    return p, sol.x
