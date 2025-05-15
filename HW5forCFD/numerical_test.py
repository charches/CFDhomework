import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#定义基本参数
Re = 1000 #对应运动粘度为0.001
N = 401
N_refined = 4001
h = 1 / (N - 1)
dt = 0.001
iteration = 0

def utop(x):
    return np.sin(np.pi * x) ** 2

#初始化涡量，流函数
omega = np.zeros((N, N))
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
psi = np.zeros((N, N))
u = np.zeros((N, N)) 
v = np.zeros((N, N))

u[:, -1] = utop(x)

#插值后数据
x_refined = np.linspace(0, 1, N_refined)
y_refined = np.linspace(0, 1, N_refined)
psi_refined = np.zeros((N_refined, N_refined))
u_refined = np.zeros((N_refined, N_refined))
v_refined = np.zeros((N_refined, N_refined))

#利用Thom公式计算涡量的边界值
def set_boundary():
    omega[1:-1, -1] = -2 * (psi[1:-1, -2] / (h ** 2) + u[1:-1, -1] / h)
    omega[1:-1, 0] = -2 * psi[1:-1, 1] / (h ** 2)
    omega[-1, 1:-1] = -2 * psi[-2, 1:-1] / (h ** 2)
    omega[0, 1:-1] = -2 * psi[1, 1:-1] / (h ** 2)

#SOR迭代求解poisson方程，松弛因子设为最优
def solve_poisson_sor():
    max_itr = 10000
    itr = 0
    rho = 0.5 * (np.cos(np.pi / N) + np.cos(np.pi / N)) #Jacobi迭代矩阵谱半径
    o = 2 / (1 + (1 - rho ** 2) ** 0.5)
    while itr < max_itr:
        itr += 1
        maxi = 0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                old = psi[i, j]
                new = (1 - o) * old + o / 4 * (psi[i + 1, j] + psi[i, j + 1] + psi[i - 1, j] + psi[i, j-1] + omega[i, j] * h ** 2) #sor迭代
                psi[i, j] = new
                maxi = max(maxi, abs(new - old))
        if maxi < 1e-2:
            break

#求解涡量输运方程
def vorticity_solve():
    omega_old = omega.copy()
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            u[i, j] = (psi[i, j + 1] - psi[i, j - 1]) / (2 * h)
            v[i, j] = - (psi[i + 1, j] - psi[i - 1, j]) / (2 * h)
            conv = u[i, j] * (omega_old[i + 1, j] - omega_old[i - 1, j]) / (2 * h) + v[i, j] * (omega_old[i, j + 1] - omega_old[i, j - 1]) / (2 * h)
            diff = (1 / Re) * (omega_old[i + 1, j] + omega_old[i - 1, j] + omega_old[i, j + 1] + omega_old[i, j - 1] - 4 * omega_old[i, j]) / (h ** 2)
            omega[i, j] = dt * (diff - conv) + omega_old[i, j]
    return np.max(np.abs(omega - omega_old))

#网格插值
def refine():
    interp_fn = interpolate.RectBivariateSpline(x, y, psi)
    psi_refined = interp_fn(x_refined, y_refined)
    u_refined[:, -1] = utop(x_refined)
    u_refined[1:-1, 1:-1] = (psi_refined[1:-1, 2:] - psi_refined[1:-1, :-2]) / (2 * h)
    v_refined[1:-1, 1:-1] = -(psi_refined[2:, 1:-1] - psi_refined[:-2, 1:-1]) / (2 * h)

#计算所得数据存储
def save():
    np.savez(
        './data/flow_data.npz',
        x = x, y = y, psi = psi, u = u, v = v, N = np.array(N)
    )
    np.savez(
        './data/flow_refined_data.npz',
        x_refined = x_refined, y_refined = y_refined, psi_refined = psi_refined, u_refined = u_refined, v_refined = v_refined, N_refined = np.array(N_refined)
    )

#求解总流程
def solve_lid_driven_cavity_flow():
    global iteration
    max_itr = 50000
    while iteration < max_itr:
        iteration += 1
        tolerence = vorticity_solve()
        solve_poisson_sor()
        set_boundary()
        print(iteration)
        if iteration > 1000 and tolerence < 1e-5:
            break

# 调用求解
solve_lid_driven_cavity_flow()
refine()
save()

