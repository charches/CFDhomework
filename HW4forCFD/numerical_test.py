import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_sor(h, omega):
    epsilon = 1e-5
    max_itr = 1000000
    itr = 0
    nc, nr = int(15 / h), int(12 / h)
    T = np.zeros((nr + 1, nc + 1))
    T[-1, :] = 100 #顶部边界条件
    T[0, :] = 20 #底部边界条件
    T[:, -1] = 20 #右侧边界条件
    T[:, 0] = 20 #左侧边界条件

    while itr < max_itr:
        itr += 1
        maxi = 0
        for i in range(1, nr):
            for j in range(1, nc):
                old = T[i, j]
                new = (1 - omega) * old + omega / 4 * (T[i + 1, j] + T[i, j + 1] + T[i - 1, j] + T[i, j-1]) #sor迭代
                T[i, j] = new
                maxi = max(maxi, abs(new - old))
        if maxi < epsilon:
            break
    return T, itr

def plot_solution():
    h, omega = 0.3, 1.5 #步长0.3，松弛因子1.5
    T, _ = solve_poisson_sor(h, omega)
    #绘图，ai生成
    plt.figure()
    plt.contourf(T.T, levels=20, cmap='hot')
    plt.colorbar()
    plt.title(f"Temperature Distribution (h={h} cm, ω={omega})")
    filename = f'./pictures/Temperature Distribution.png'
    plt.savefig(filename)

def relaxation_iterations():
    h = 1.5
    omegas = np.linspace(0.5, 1.9, 141)
    itrs = []
    for omega in omegas:
        _, itr = solve_poisson_sor(h, omega)
        itrs.append(itr)
    #绘图，ai生成
    plt.figure()
    plt.plot(omegas, itrs)
    plt.xlabel("Relaxation factor ω")
    plt.ylabel("Iterations to converge")
    plt.title(f"Convergence vs. ω (h={h} cm)")
    filename = f'./pictures/Convergence vs. ω.png'
    plt.savefig(filename)

def best_relaxation():
    hs = [1.5, 1.0, 0.5]
    omega_opts = []
    plt.figure()
    #计算理论最佳松弛因子
    for h in hs:
        N, M = int(15 / h), int(12 / h)
        rho = 0.5 * (np.cos(np.pi / N) + np.cos(np.pi / M)) #Jacobi迭代矩阵谱半径
        omega_opt = 2 / (1 + (1 - rho ** 2) ** 0.5)
        omega_opts.append(omega_opt)
    #数值计算最佳松弛因子并绘图
    omegas = np.linspace(0.5, 1.9, 141)
    for i, h in enumerate(hs):
        itrs = []
        for omega in omegas:
            _, itr = solve_poisson_sor(h, omega)
            itrs.append(itr)
        opt_idx = np.argmin(itrs)
        #绘图，ai生成，手工修正
        plt.plot(omegas, itrs, label=f'h={h} cm')
        plt.axvline(x = omega_opts[i], color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=0.5, label=f'Theoretical ω_opt={omega_opts[i]:.2f} (h={h} cm)')#标注理论最佳松弛因子
        plt.scatter([omegas[opt_idx]], [itrs[opt_idx]], color=plt.gca().lines[-1].get_color(), s=100, label=f'Numerical ω_opt={omegas[opt_idx]:.2f}')#标注数值最佳松弛因子
        plt.xlabel("Relaxation factor ω")
        plt.ylabel("Iterations to converge")
        plt.title("Theoretical vs Numerical Optimal Relaxation Factors")
        plt.legend()
        plt.grid(True)
        filename = './pictures/Theoretical vs Numerical Optimal Relaxation Factors.png'
        plt.savefig(filename)

plot_solution()
relaxation_iterations()
best_relaxation()