import numpy as np


class MPCController:
    """
    一个适合当前项目的最小可用线性运动学 MPC。

    状态：
        x = 末端 6 维位姿误差（世界系）
    控制：
        u = 关节速度 q_dot

    预测模型：
        x_{k+1} = x_k - dt * J * u_k

    说明：
    - 这里的 J 使用“当前时刻”的 Jacobian，在整个预测域内保持不变
    - 这是一个局部线性化的 MPC
    - 第一版先不依赖 CasADi / QP 求解器，只用 numpy 解一个无约束二次优化
    """

    def __init__(
        self,
        horizon=12,
        dt=0.01,
        q_weights=None,
        r_weights=None,
        du_weights=None,
        qdot_limit=1.2,
    ):
        # 预测步长
        self.horizon = int(horizon)

        # 离散时间步长
        self.dt = float(dt)

        # 关节速度限幅
        self.qdot_limit = float(qdot_limit)

        # 位姿误差权重：前三个是位置，后三个是姿态
        self.q_weights = np.asarray(
            q_weights if q_weights is not None else [8.0, 8.0, 8.0, 4.0, 4.0, 4.0],
            dtype=np.float64,
        )

        # 控制输入权重：惩罚 q_dot 太大
        self.r_weights = np.asarray(
            r_weights if r_weights is not None else [0.08, 0.08, 0.08, 0.05, 0.05, 0.05],
            dtype=np.float64,
        )

        # 控制增量权重：惩罚相邻时刻 q_dot 变化太猛
        self.du_weights = np.asarray(
            du_weights if du_weights is not None else [0.6, 0.6, 0.6, 0.3, 0.3, 0.3],
            dtype=np.float64,
        )

    def _build_prediction_matrices(self, A, B):
        """
        构造预测矩阵，使整个预测域可以写成：

            X = Sx * x0 + Su * U

        其中：
        - X: 所有未来状态堆叠起来的列向量
        - x0: 当前状态
        - U: 所有未来控制量堆叠起来的列向量
        """
        n = A.shape[0]   # 状态维度，这里是 6
        m = B.shape[1]   # 控制维度，这里一般是 6
        N = self.horizon

        Sx = np.zeros((N * n, n), dtype=np.float64)
        Su = np.zeros((N * n, N * m), dtype=np.float64)

        A_power = np.eye(n, dtype=np.float64)

        for i in range(N):
            # 当前步的 A^(i+1)
            A_power = A_power @ A
            Sx[i * n:(i + 1) * n, :] = A_power

            # 当前预测步对历史控制量的响应
            for j in range(i + 1):
                A_ij = np.eye(n, dtype=np.float64)
                for _ in range(i - j):
                    A_ij = A_ij @ A

                Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = A_ij @ B

        return Sx, Su

    def _build_difference_matrix(self, m):
        """
        构造控制增量矩阵 D，使得：

            D @ U - d

        表示：
            [u0 - u_last,
             u1 - u0,
             u2 - u1,
             ...]

        这样就能在代价函数里惩罚控制变化过快。
        """
        N = self.horizon
        D = np.zeros((N * m, N * m), dtype=np.float64)

        for i in range(N):
            row = i * m
            col = i * m

            # 当前控制项系数 +I
            D[row:row + m, col:col + m] = np.eye(m)

            # 和前一个控制项做差
            if i > 0:
                prev = (i - 1) * m
                D[row:row + m, prev:prev + m] = -np.eye(m)

        return D

    def solve(self, error_state_world, jacobian, last_q_dot=None, reference_trajectory=None):
        """
        求解当前时刻最优的第一步关节速度。

        输入：
        - error_state_world: 当前 6 维末端误差（世界系）
        - jacobian: 当前 6xm Jacobian
        - last_q_dot: 上一时刻的关节速度，用于平滑
        - reference_trajectory:
          未来 N 步的参考误差轨迹，shape = (N, n) 或 (N*n,)
          如果不给，则默认每一步的参考都是 0

        输出：
        - q_dot_cmd: 当前时刻应该执行的关节速度
        """
        x0 = np.asarray(error_state_world, dtype=np.float64).reshape(-1)
        J = np.asarray(jacobian, dtype=np.float64)

        n = x0.shape[0]
        m = J.shape[1]
        N = self.horizon

        if last_q_dot is None:
            last_q_dot = np.zeros(m, dtype=np.float64)
        else:
            last_q_dot = np.asarray(last_q_dot, dtype=np.float64).reshape(-1)

        # 线性预测模型：
        # x_{k+1} = x_k - dt * J * u_k
        A = np.eye(n, dtype=np.float64)
        B = -self.dt * J

        Sx, Su = self._build_prediction_matrices(A, B)

        if reference_trajectory is None:
            x_ref = np.zeros(N * n, dtype=np.float64)
        else:
            x_ref = np.asarray(reference_trajectory, dtype=np.float64).reshape(-1)
            if x_ref.shape[0] != N * n:
                raise ValueError(
                    f"reference_trajectory has invalid size {x_ref.shape[0]}, expected {N * n}"
                )

        # 构造代价矩阵
        Q = np.diag(self.q_weights)

        # 如果给的权重维度和控制维度不一致，就裁剪到 m
        R_vec = self.r_weights[:m] if self.r_weights.shape[0] != m else self.r_weights
        Rd_vec = self.du_weights[:m] if self.du_weights.shape[0] != m else self.du_weights

        R = np.diag(R_vec)
        Rd = np.diag(Rd_vec)

        # 扩展到整个预测域
        Qbar = np.kron(np.eye(N), Q)
        Rbar = np.kron(np.eye(N), R)
        Rdbar = np.kron(np.eye(N), Rd)

        # 控制增量项
        D = self._build_difference_matrix(m)

        # d 里面只在第一段放上一时刻控制，表示 u0 - last_q_dot
        d = np.zeros(N * m, dtype=np.float64)
        d[:m] = last_q_dot

        # 二次型代价：
        # ||Sx x0 + Su U - Xref||_Q^2 + ||U||_R^2 + ||D U - d||_Rd^2
        H = Su.T @ Qbar @ Su + Rbar + D.T @ Rdbar @ D
        g = Su.T @ Qbar @ (Sx @ x0 - x_ref) - D.T @ Rdbar @ d

        # 给 Hessian 加一点正则，避免数值太差
        H = H + 1e-8 * np.eye(H.shape[0], dtype=np.float64)

        # 无约束最优解：
        # U* = - H^{-1} g
        U_star = -np.linalg.solve(H, g)

        # 展开成 N 个控制步
        U_star = U_star.reshape(N, m)

        # 第一版先用简单限幅，后面如果你要更严格约束再换 QP
        U_star = np.clip(U_star, -self.qdot_limit, self.qdot_limit)

        # 只执行第一步，这就是 receding horizon control
        return U_star[0]
