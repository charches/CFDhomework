import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from abc import ABC, abstractmethod

u_exact = lambda x, t: np.sin(2 * np.pi * (x - t))
u_initial = lambda x: np.sin((2 * x) * np.pi)

class scheme(ABC):
    @abstractmethod
    def numerical_solution(self, Nx, Nt, CFL): #分别为空间格点数， 时间格点数，CFL数
        pass
    
    @abstractmethod
    def name(self):
        pass

    def stablity_analysis(self, Nx = 301):
        for T in [1.0, 1.25, 2.0]:#默认空间格点数为301，考察CFL数为0.9，1.0，1.1时推进到T=1.0，1.25和2.0时的情况
            CFLs = np.linspace(0.9, 1.1, 3)
            plt.figure(figsize=(10, 6))
            dx = 3.0 / (Nx - 1)
            x = np.linspace(0, 3, Nx - 1, endpoint = False)
            plt.plot(x, u_exact(x, T), label='u_exact')
            for CFL in CFLs:
                dt = dx * CFL
                Nt = int(T / dt) + 1
                x, u, _ = self.numerical_solution(Nx, Nt, CFL)
                #绘图部分，ai生成
                plt.plot(x, u[-1, :], label=f'CFL = {CFL:.1f}')
            plt.title(f"Numerical solution with varying CFL number ({self.name()}) at T={T}")
            plt.xlabel("x")
            plt.ylabel("u")
            plt.legend()
            plt.grid()
            filename = f'./pictures/Stablity_of_{self.name()}_at_{T}.png'
            plt.savefig(filename)

    def order_analysis(self, CFL = 0.8, T = 1.0):
        Ns = [60 * (2 ** i) for i in range(11)]#默认CFL数为0.8，推进到T=1.0，逐渐分半加密空间网格
        L2_errors = []
        h = []
        orders = []
        for N in Ns:
            Nx = N + 1
            dx = 3.0 / N
            h.append(dx)
            dt = dx * CFL
            Nt = int(T / dt) + 1
            _, _, L2_error = self.numerical_solution(Nx, Nt, CFL)
            L2_errors.append(L2_error)
        for i in range(len(h) - 1):
            orders.append(np.log(L2_errors[i] / L2_errors[i + 1]) / np.log(2))
        #绘图，ai生成
        plt.figure(figsize=(10, 6))
        plt.plot(np.log10(h[:-1]), orders, 'o-', label='Error Convergence Order')
        plt.xlabel('Grid Spacing (log10(h))')
        plt.ylabel('Convergence Order')
        plt.title(f'Convergence Order vs Grid Spacing ({self.name()})')
        plt.grid(True)
        plt.legend()
        filename = f'./pictures/Convergence_Order_of_{self.name()}.png'
        plt.savefig(filename)
    
    def evolution(self, CFL = 0.8, T = 100.0):#生成解的演化动画，时间推进到T=100.0，CFL数=0.8，时间步长固定为0.01
        dt = 0.01
        dx = dt / CFL
        Nt = int(T / dt) + 1
        Nx = int(3.0 / dx) + 1
        x, u, _ = self.numerical_solution(Nx, Nt, CFL)
        filenames = []
        for t in range(Nt):
            if t % 10 == 0:
                #绘图部分，ai生成
                plt.clf()
                plt.plot(x, u[t, :], 'r-', label='Numerical Solution')
                plt.plot(x, u_exact(x, dt * t), 'b--', label='Exact Solution')
                plt.title(f'Solution Comparison at t = {t * dt:.2f}')
                plt.xlabel('x')
                plt.ylabel('u')
                plt.legend()
                plt.grid(True)
                filename = f'./pictures/{self.name()}_{t}.png'
                plt.savefig(filename)
                filenames.append(filename)
        with imageio.get_writer(f'./pictures/solution_evolution_of_{self.name()}.mp4', fps=10) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in filenames:
            os.remove(filename)



class Upwind(scheme):
    def name(self):
        return "Upwind_Scheme"
    
    def numerical_solution(self, Nx, Nt, CFL):
        x = np.linspace(0, 3, Nx - 1, endpoint = False)
        dx = 3 / (Nx - 1)
        dt = dx * CFL
        u = np.zeros((Nt, Nx - 1))
        u[0, : ] = u_initial(x)
        for t in range(1, Nt):
            u[t, :] = u[t - 1, :] - CFL * (u[t - 1, :] - np.roll(u[t - 1, :], 1))
        L2_error = np.sqrt(dx * np.sum((u[-1, :] - u_exact(x, (Nt - 1) * dt)) ** 2))
        return x, u, L2_error

class LeapFrog(scheme):
    def name(self):
        return "LeapFrog_Scheme"
    
    def numerical_solution(self, Nx, Nt, CFL):
        x = np.linspace(0, 3, Nx - 1, endpoint = False)
        dx = 3 / (Nx - 1)
        dt = dx * CFL
        u = np.zeros((Nt, Nx - 1))
        u[0, : ] = u_initial(x)
        #Lax_Wendroff计算第一层
        u[1, : ] = u[0, : ] - 0.5 * CFL * (np.roll(u[0, : ], -1) - np.roll(u[0, : ], 1)) + 0.5 * (CFL ** 2) * (np.roll(u[0, : ], -1) + np.roll(u[0, : ], 1) - 2 * u[0, : ])
        for t in range(2, Nt):
            u[t, :] = u[t - 2, :] - CFL * (np.roll(u[t - 1, :], -1) - np.roll(u[t - 1, :], 1))
        L2_error = np.sqrt((dx * np.sum((u[-1, :] - u_exact(x, (Nt - 1) * dt)) ** 2)))
        return x, u, L2_error

class LaxWendroff(scheme):
    def name(self):
        return "LaxWendroff_Scheme"
    
    def numerical_solution(self, Nx, Nt, CFL):
        x = np.linspace(0, 3, Nx - 1, endpoint = False)
        dx = 3 / (Nx - 1)
        dt = dx * CFL
        u = np.zeros((Nt, Nx - 1))
        u[0, : ] = u_initial(x)
        for t in range(1, Nt):
            u[t, : ] = u[t - 1, : ] - 0.5 * CFL * (np.roll(u[t - 1, : ], -1) - np.roll(u[t - 1, : ], 1)) + 0.5 * (CFL ** 2) * (np.roll(u[t - 1, : ], -1) + np.roll(u[t - 1, : ], 1) - 2 * u[t - 1, : ])
        L2_error = np.sqrt((dx * np.sum((u[-1, :] - u_exact(x, (Nt - 1) * dt)) ** 2)))
        return x, u, L2_error

schemes = [Upwind(), LeapFrog(), LaxWendroff()]
for scheme in schemes:
    #scheme.stablity_analysis()
    #scheme.order_analysis()
    scheme.evolution()