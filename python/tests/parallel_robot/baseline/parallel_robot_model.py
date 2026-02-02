import numpy as np

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import logm


def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def hat(v):
    return np.array([
        [ 0.0, -v[2],  v[1]],
        [ v[2],  0.0, -v[0]],
        [-v[1],  v[0], 0.0]
    ])


def vee(M):
    return np.array([M[2,1], M[0,2], M[1,0]])


class ParallelRobotSolver:
    def __init__(self, K_se_inv, K_bt_inv):
        self.K_se_inv = K_se_inv
        self.K_bt_inv = K_bt_inv
        self.guess_init = np.zeros(6 * 6)

    def single_rod_deriv(self, s, y):
        R = y[3:12].reshape(3, 3)
        n = y[12:15]
        m = y[15:18]
            
        v = self.K_se_inv @ R.T @ n + np.array([0, 0, 1])
        u = self.K_bt_inv @ R.T @ m
            
        p_dot = R @ v
        R_dot = R @ hat(u)
        n_dot = np.zeros(3)
        m_dot = -np.cross(p_dot, n)
            
        y_dot = np.hstack((p_dot, R_dot.flatten(), n_dot, m_dot))
        
        return y_dot

    def solve(self, rod_lengths, tip_force=np.zeros(3), tip_moment=np.zeros(3)):
        end_conditions = root(self.total_shooting_residual, self.guess_init, args=(rod_lengths, tip_force, tip_moment))
        self.guess_init = end_conditions.x
        print("solved")

    def total_shooting_residual(self, x, rod_lengths, tip_force, tip_moment):

        def unpack_block(b):
            n = b[:3]
            m = np.array([b[3], b[4], 0])
            theta = b[5]
            R = rotz(theta).reshape(9, 1)
            return R, n, m

        x = x.reshape(6, 6)

        R1, n1, m1 = unpack_block(x[0])
        R2, n2, m2 = unpack_block(x[1])
        R3, n3, m3 = unpack_block(x[2])
        R4, n4, m4 = unpack_block(x[3])
        R5, n5, m5 = unpack_block(x[4])
        R6, n6, m6 = unpack_block(x[5])

        def p(angle_deg, r=0.087):
            a = np.deg2rad(angle_deg)
            return np.array([r*np.cos(a), r*np.sin(a), 0.0])

        p_init = [
            p(0 - 10), p(0 + 10),
            p(120 - 10), p(120 + 10),
            p(240 - 10), p(240 + 10)
        ]

        p_final = [
            p(0 + 10), p(120 - 10), p(120 + 10),
            p(240 - 10), p(240 + 10), p(0 - 10)
        ]

        y0 = [
            np.hstack([p_init[i], Ri.flatten(), ni, mi])
            for i, (Ri, ni, mi) in enumerate([
                (R1, n1, m1), (R2, n2, m2), (R3, n3, m3),
                (R4, n4, m4), (R5, n5, m5), (R6, n6, m6)
            ])
        ]

        results = []
        for i in range(6):
            sol = solve_ivp(
                self.single_rod_deriv,
                (0.0, rod_lengths[i]),
                y0[i],
                t_eval=[rod_lengths[i]]
            )
            results.append(sol.y[:, -1])

        def unpack_end(y):
            p = y[0:3]
            R = y[3:12].reshape(3, 3)
            n = y[12:15]
            m = y[15:18]
            return p, R, n, m

        ends = [unpack_end(y) for y in results]

        p_ends = [e[0] for e in ends]
        R_ends = [e[1] for e in ends]
        n_ends = [e[2] for e in ends]
        m_ends = [e[3] for e in ends]

        p_c = sum(p_ends) / 6.0

        res_force = sum(n_ends) - tip_force
        res_moment = (
            sum(hat(p_ends[i] - p_c) @ n_ends[i] for i in range(6)) +
            sum(m_ends) - tip_moment
        )

        res_eq = np.hstack([res_force, res_moment])

        p1_end, R1_end = p_ends[0], R_ends[0]

        res_p = []
        for i in range(1, 6):
            res_p.append(
                (p1_end - R1_end @ p_final[0]) -
                (p_ends[i] - R_ends[i] @ p_final[i])
            )

        res_p = 100.0 * np.hstack(res_p)

        res_R = []
        for i in range(1, 6):
            dR = R1_end.T @ R_ends[i]
            res_R.append(vee(logm(dR)).real)

        res_R = np.hstack(res_R)

        res = np.hstack([
            res_eq,
            res_p,
            res_R
        ])

        print(np.linalg.norm(res))
        return res


def main():
    ro = 0.0013/2
    ri = 0.00
    I = 0.25 * np.pi *(ro**4 - ri**4)
    A = np.pi * (ro**2 - ri**2)
    J = 2 * I
    E = 207.0e9
    G = 79.3e9
    
    K_bt_inv = np.array([
        [1 / (E * I), 0, 0], 
        [0, 1 / (E * I), 0],
        [0, 0, 1 / (J * G)]
    ])

    K_se_inv = np.array([
        [1 / (G * A), 0, 0],
        [0, 1 / (G * A), 0],
        [0, 0, 1 / (E * A)]
    ])

    solver = ParallelRobotSolver(K_se_inv, K_bt_inv)

    phase = np.radians(10)
    a = 0.001 * (24*25.4-13-33-400+240)
    A = 0.1
    for i in range(100):
        wt = i / 100 * 2 * np.pi
        L1 = a + A * np.sin(wt - phase)
        L2 = a + A * np.sin(wt + phase); 
        L3 = a + A * np.sin(wt + np.radians(120) - phase)
        L4 = a + A * np.sin(wt + np.radians(120) + phase)
        L5 = a + A * np.sin(wt + np.radians(240) - phase)
        L6 = a + A * np.sin(wt + np.radians(240) + phase)
        rod_lengths = np.array([L1, L2, L3, L4, L5, L6])

        solution = solver.solve(rod_lengths)


if __name__ == "__main__":
    main()