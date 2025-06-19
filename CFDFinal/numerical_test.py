import numpy as np
import matplotlib.pyplot as plt
from shocktubecalc import sod

GAMMA = 1.4
EPSILON = 1e-8

def primitive_to_conserved(rho, u, p):
    U = np.zeros(3)
    U[0] = rho
    U[1] = rho * u
    U[2] = p / (GAMMA - 1) + 0.5 * rho * u ** 2
    return U

def conserved_to_primitive(U):
    rho = U[0]
    u = U[1] / U[0]
    p = (GAMMA - 1) * (U[2] - 0.5 * rho * u ** 2)
    return rho, u, p

def steger_warming(eigen_values, rho, u, c):
    F = np.zeros(3)
    F[0] = 2 * (GAMMA - 1) * eigen_values[0] + eigen_values[1] + eigen_values[2]
    F[1] = 2 * (GAMMA - 1) * eigen_values[0] * u + (u + c) * eigen_values[1] + (u - c) * eigen_values[2]
    F[2] = (GAMMA - 1) * eigen_values[0] * u ** 2 + 0.5 * (3 - GAMMA) / (GAMMA - 1) * (eigen_values[1] + eigen_values[2]) * c ** 2 + 0.5 * eigen_values[1] * (u + c) ** 2 + 0.5 * eigen_values[2] * (u - c) ** 2
    F *= rho / (2 * GAMMA)
    return F

def Fp(U):#利用steger-warming分裂通量
    rho, u, p = conserved_to_primitive(U)
    c = (GAMMA * p / rho) ** 0.5
    eigen_values = [u, u + c, u - c]
    eigen_values_p = []

    for v in eigen_values:
        v_p = (v + (v ** 2 + EPSILON ** 2) ** 0.5) / 2
        eigen_values_p.append(v_p)
    
    Fp = steger_warming(eigen_values_p, rho, u, c)

    return Fp

def Fn(U):
    rho, u, p = conserved_to_primitive(U)
    c = (GAMMA * p / rho) ** 0.5
    eigen_values = [u, u + c, u - c]
    eigen_values_n = []

    for v in eigen_values:
        v_n = (v - (v ** 2 + EPSILON ** 2) ** 0.5) / 2
        eigen_values_n.append(v_n)
    
    Fn = steger_warming(eigen_values_n, rho, u, c)

    return Fn

def compute_R_inv(U):
    rho, u, p = conserved_to_primitive(U)
    c = np.sqrt(GAMMA * p / rho)  # 声速
    c2 = c**2
    
    # 构造矩阵元素（直接按公式计算）
    m11 = u * (2*c - u + GAMMA*u) / (4*c2)
    m12 = -(c - u + GAMMA*u) / (2*c2)
    m13 = (GAMMA - 1) / (2*c2)
    
    m21 = (2*c2 - GAMMA*u**2 + u**2) / (2*c2)
    m22 = u * (GAMMA - 1) / c2
    m23 = -(GAMMA - 1) / c2
    
    m31 = -u * (2*c + u - GAMMA*u) / (4*c2)
    m32 = (c + u - GAMMA*u) / (2*c2)
    m33 = (GAMMA - 1) / (2*c2)
    
    return np.array([
        [m11, m12, m13],
        [m21, m22, m23],
        [m31, m32, m33]
    ])

def compute_R(U):
    rho, u, p = conserved_to_primitive(U)
    c = np.sqrt(GAMMA * p / rho)  # 声速
    
    # 构造矩阵元素
    row1 = np.array([1, 1, 1])
    row2 = np.array([u - c, u, u + c])
    
    # 第三行各项
    term1 = (c**2)/(GAMMA - 1) - c*u + 0.5*u**2
    term2 = 0.5*u**2
    term3 = (c**2)/(GAMMA - 1) + c*u + 0.5*u**2
    row3 = np.array([term1, term2, term3])
    
    return np.vstack([row1, row2, row3])

#计算roe矩阵，ai生成
def compute_roe_matrix(u, c, lambdas):
    
    lambda1, lambda2, lambda3 = lambdas
    A = np.zeros((3, 3), dtype=np.float64)
    
    # 第一行
    A[0, 0] = (lambda2 * (2*c**2 - GAMMA*u**2 + u**2) / (2*c**2) +
               lambda1 * u * (2*c - u + GAMMA*u) / (4*c**2) -
               lambda3 * u * (2*c + u - GAMMA*u) / (4*c**2))
    
    A[0, 1] = (lambda3 * (c + u - GAMMA*u) / (2*c**2) -
                lambda1 * (c - u + GAMMA*u) / (2*c**2) +
                lambda2 * u * (GAMMA - 1) / c**2)
    
    A[0, 2] = (GAMMA - 1) * (lambda1 - 2*lambda2 + lambda3) / (2*c**2)
    
    # 第二行
    A[1, 0] = (lambda2 * u * (2*c**2 - GAMMA*u**2 + u**2) / (2*c**2) -
               lambda3 * u * (c + u) * (2*c + u - GAMMA*u) / (4*c**2) -
               lambda1 * u * (c - u) * (2*c - u + GAMMA*u) / (4*c**2))
    
    A[1, 1] = (lambda2 * u**2 * (GAMMA - 1) / c**2 +
               lambda3 * (c + u) * (c + u - GAMMA*u) / (2*c**2) +
               lambda1 * (c - u) * (c - u + GAMMA*u) / (2*c**2))
    
    A[1, 2] = (GAMMA - 1) * (c*lambda3 - c*lambda1 + lambda1*u - 2*lambda2*u + lambda3*u) / (2*c**2)
    
    # 第三行
    A[2, 0] = (lambda2 * u**2 * (2*c**2 - GAMMA*u**2 + u**2) / (4*c**2) +
                lambda1 * u * (c**2/(GAMMA-1) - c*u + u**2/2) * (2*c - u + GAMMA*u) / (4*c**2) -
                lambda3 * u * (c*u + c**2/(GAMMA-1) + u**2/2) * (2*c + u - GAMMA*u) / (4*c**2))
    
    A[2, 1] = (lambda3 * (c*u + c**2/(GAMMA-1) + u**2/2) * (c + u - GAMMA*u) / (2*c**2) +
                lambda2 * u**3 * (GAMMA - 1) / (2*c**2) -
                lambda1 * (c**2/(GAMMA-1) - c*u + u**2/2) * (c - u + GAMMA*u) / (2*c**2))
    
    A[2, 2] = (lambda1 * (GAMMA - 1) * (c**2/(GAMMA-1) - c*u + u**2/2) / (2*c**2) -
               lambda2 * u**2 * (GAMMA - 1) / (2*c**2) +
               lambda3 * (GAMMA - 1) * (c*u + c**2/(GAMMA-1) + u**2/2) / (2*c**2))
    
    return A

def compute_dt(U, CFL):
        N = U.shape[0]
        dt = 1e9
        dx = 1 / (N - 1)
        for i in range(N):
            rho, u, p = conserved_to_primitive(U[i])
            c = (GAMMA * p / rho) ** 0.5
            dt = min(dt, CFL * dx / (np.abs(u) + c))
        return dt

def RK3(U, dt, RHS): #三阶Runge-Kutta（heun格式）
    U1 = U + 1 / 3 * dt * RHS(U)
    U2 = U + 2 / 3 * dt * RHS(U1)
    return 1 / 4 * (U + 3 * U1) + 3 / 4 * dt * RHS(U2)

def calculate_l2_error(num, ref, N):
        return np.sqrt(np.mean((num - ref) ** 2)) / (N - 1)

def F(U):
    rho, u, p = conserved_to_primitive(U)
    F = np.zeros(3)
    F[0] = rho * u
    F[1] = rho * u ** 2 + p
    F[2] = u * (U[2] + p)
    return F

class solver:
    def __init__(self, CFL, T):
        self.CFL = CFL
        self.T = T
    
    def solve(self, U):#完整求解器
        N = U.shape[0]

        #初始化
        x = np.linspace(-0.5, 0.5, N)
        UL = primitive_to_conserved(1, 0, 1)
        UR = primitive_to_conserved(0.125, 0, 0.1)
        for i in range(N):
            if x[i] < 0:
                U[i] = UL
            else:
                U[i] = UR
        
        #时间推进
        t = 0
        cnt = 0
        while t <= self.T:
            cnt += 1
            dt = compute_dt(U, self.CFL)
            U = RK3(U, dt, self.RHS)
            print(t)
            t += dt
        
        rho = np.zeros(N)
        u = np.zeros(N)
        p = np.zeros(N)
        for i in range(N):
            rho[i], u[i], p[i] = conserved_to_primitive(U[i])

        return rho, u, p

    def RHS(self, U):
        pass

class TVD(solver):
    def __init__(self, CFL, T, type):
        super().__init__(CFL, T)
        self.type = type
    
    def limiter(self, r): #minmod限制器
        phi = np.zeros(3)
        if self.type == "superbee":
            for i in range(3):
                if r[i] < 0:
                    phi[i] = 0
                elif 0 <= r[i] and r[i] < 0.5:
                    phi[i] = 2 * r[i]
                elif 0.5 <= r[i] and r[i] < 1:
                    phi[i] = 1
                elif 1 <= r[i] and r[i] < 2:
                    phi[i] = r[i]
                else:
                    phi[i] = 2
        elif self.type == "minmod":
            for i in range(3):
                if r[i] < 0:
                    phi[i] = 0 
                elif 0 <= r[i] and r[i] < 1:
                    phi[i] = r[i]
                else:
                    phi[i] = 1
        elif self.type == "vanleer":
            for i in range(3):
                phi[i] = (r[i] + np.abs(r[i])) / (1 + np.abs(r[i]))
        return phi

    def RHS(self, U):#计算半离散后右侧表达式 
        N = U.shape[0]
        Fp = np.apply_along_axis(globals()['Fp'], axis = 1, arr = U)
        Fn = np.apply_along_axis(globals()['Fn'], axis = 1, arr = U)

        ng = 2 #TVD格式两侧各有2个ghost cell
        #计算正负数值通量
        Fp_hat = np.zeros((N - 1, 3))
        Fn_hat = np.zeros((N - 1, 3))
        for i in range(ng - 1, N - ng):
            rp = (Fp[i] - Fp[i - 1]) / (Fp[i + 1] - Fp[i] + EPSILON)
            Fp_hat[i] = Fp[i] + 0.5 * self.limiter(rp) * (Fp[i + 1] - Fp[i])

            rn = (Fn[i + 2] - Fn[i + 1]) / (Fn[i + 1] - Fn[i] + EPSILON)
            Fn_hat[i] = Fn[i + 1] - 0.5 * self.limiter(rn) * (Fn[i + 1] - Fn[i])

        #计算总体右侧表达式
        dx = 1 / (N - 1)
        RHS = np.zeros((N, 3))
        for i in range(ng, N - ng):
            RHS[i] = -(Fp_hat[i] - Fp_hat[i - 1] + Fn_hat[i] - Fn_hat[i - 1]) / dx

        return RHS

class NND(solver):
    def minmod(self, a, b):
        phi = np.zeros(3)
        for i in range(3):
            if a[i] * b[i] < 0:
                phi[i] = 0
            elif abs(a[i]) > abs(b[i]):
                phi[i] = b[i]
            else:
                phi[i] = a[i]
        return phi
    
    def RHS(self, U):#计算半离散后右侧表达式 
        N = U.shape[0]
        Fp = np.apply_along_axis(globals()['Fp'], axis = 1, arr = U)
        Fn = np.apply_along_axis(globals()['Fn'], axis = 1, arr = U)

        ng = 2 #NND格式两侧各有2个ghost cell
        #计算正负数值通量
        Fp_hat = np.zeros((N - 1, 3))
        Fn_hat = np.zeros((N - 1, 3))
        for i in range(ng - 1, N - ng):
            Fp_hat[i] = Fp[i] + 0.5 * self.minmod(Fp[i + 1] - Fp[i], Fp[i] - Fp[i - 1])

            Fn_hat[i] = Fn[i + 1] - 0.5 * self.minmod(Fn[i + 1] - Fn[i], Fn[i + 2] - Fn[i + 1])

        #计算总体右侧表达式
        dx = 1 / (N - 1)
        RHS = np.zeros((N, 3))
        for i in range(ng, N - ng):
            RHS[i] = -(Fp_hat[i] - Fp_hat[i - 1] + Fn_hat[i] - Fn_hat[i - 1]) / dx

        return RHS

class WENO(solver):
    def __init__(self, CFL, T, type):
        super().__init__(CFL, T)
        self.type = type
    
    def RHS(self, U):#计算半离散后右侧表达式 
        N = U.shape[0]
        Fp = np.apply_along_axis(globals()['Fp'], axis = 1, arr = U)
        Fn = np.apply_along_axis(globals()['Fn'], axis = 1, arr = U)

        ng = 3 #WENO格式两侧各有3个ghost cell
        #计算正负数值通量
        Fp_hat = np.zeros((N - 1, 3))
        Fn_hat = np.zeros((N - 1, 3))
        for i in range(ng - 1, N - ng):
            c = np.array([0.1, 0.6, 0.3])

            if (self.type == "My Ver"):
                fp = np.array([1 / 3 * Fp[i - 2] - 7 / 6 * Fp[i - 1] + 11 / 6 * Fp[i],
                            1 / 3 * Fp[i + 1] + 5 / 6 * Fp[i] - 1 / 6 * Fp[i - 1],
                            -1 / 6 * Fp[i + 2] + 5 / 6 * Fp[i + 1] + 1 / 3 * Fp[i]])
                betap = np.array([(Fp[i] - 2 * Fp[i - 1] + Fp[i - 2]) ** 2 + 1 / 4 * ((Fp[i - 2] - 4 * Fp[i - 1] + 3 * Fp[i]) ** 2 + (Fp[i] - Fp[i - 2]) ** 2 + (Fp[i] - 4 * Fp[i - 1] + 3 * Fp[i - 2]) ** 2),
                                (Fp[i + 1] - 2 * Fp[i] + Fp[i - 1]) ** 2 + 1 / 4 * ((Fp[i - 1] - 4 * Fp[i] + 3 * Fp[i + 1]) ** 2 + (Fp[i + 1] - Fp[i - 1]) ** 2 + (Fp[i + 1] - 4 * Fp[i] + 3 * Fp[i - 1]) ** 2),
                                (Fp[i + 2] - 2 * Fp[i + 1] + Fp[i]) ** 2 + 1 / 4 * ((Fp[i] - 4 * Fp[i + 1] + 3 * Fp[i + 2]) ** 2 + (Fp[i + 2] - Fp[i]) ** 2 + (Fp[i + 2] - 4 * Fp[i + 1] + 3 * Fp[i]) ** 2)])
                alphap = c / (EPSILON + betap) ** 2
                alphap /= np.sum(alphap, axis = 0)
                Fp_hat[i] = np.sum(fp * alphap, axis = 0)

                fn = np.array([1 / 3 * Fn[i + 3] - 7 / 6 * Fn[i + 2] + 11 / 6 * Fn[i + 1],
                            -1 / 6 * Fn[i + 2] + 5 / 6 * Fn[i + 1] + 1 / 3 * Fn[i],
                            1 / 3 * Fn[i + 1] + 5 / 6 * Fn[i] - 1 / 6 * Fn[i - 1]
                            ])
                betan = np.array([(Fn[i + 1] - 2 * Fn[i + 2] + Fn[i + 3]) ** 2 + 1 / 4 * ((Fn[i + 3] - 4 * Fn[i + 2] + 3 * Fn[i + 1]) ** 2 + (Fn[i + 3] - Fn[i + 1]) ** 2 + (Fn[i + 1] - 4 * Fn[i + 2] + 3 * Fn[i + 3]) ** 2),
                                (Fn[i] - 2 * Fn[i + 1] + Fn[i + 2]) ** 2 + 1 / 4 * ((Fn[i + 2] - 4 * Fn[i + 1] + 3 * Fn[i]) ** 2 + (Fn[i + 2] - Fn[i]) ** 2 + (Fn[i] - 4 * Fn[i + 1] + 3 * Fn[i + 2]) ** 2),
                                (Fn[i - 1] - 2 * Fn[i] + Fn[i + 1]) ** 2 + 1 / 4 * ((Fn[i + 1] - 4 * Fn[i] + 3 * Fn[i - 1]) ** 2 + (Fn[i + 1] - Fn[i - 1]) ** 2 + (Fn[i] - 4 * Fn[i] + 3 * Fn[i + 1]) ** 2)])
                alphan = c / (EPSILON + betan) ** 2
                alphan /= np.sum(alphan, axis = 0)
                Fn_hat[i] = np.sum(fn * alphan, axis = 0)

            if (self.type == "Official"):
                fp = np.array([1 / 3 * Fp[i - 2] - 7 / 6 * Fp[i - 1] + 11 / 6 * Fp[i],
                            1 / 3 * Fp[i + 1] + 5 / 6 * Fp[i] - 1 / 6 * Fp[i - 1],
                            -1 / 6 * Fp[i + 2] + 5 / 6 * Fp[i + 1] + 1 / 3 * Fp[i]])
                betap = np.array([13 / 12 * (Fp[i] - 2 * Fp[i - 1] + Fp[i - 2]) ** 2 + 1 / 4 * (Fp[i - 2] - 4 * Fp[i - 1] + 3 * Fp[i]) ** 2,
                                13 / 12 * (Fp[i + 1] - 2 * Fp[i] + Fp[i - 1]) ** 2 + 1 / 4 * (Fp[i + 1] - Fp[i - 1]) ** 2,
                                13 / 12 * (Fp[i + 2] - 2 * Fp[i + 1] + Fp[i]) ** 2 + 1 / 4 * (Fp[i + 2] - 4 * Fp[i + 1] + 3 * Fp[i]) ** 2])
                alphap = c / (EPSILON + betap) ** 2
                alphap /= np.sum(alphap, axis = 0)
                Fp_hat[i] = np.sum(fp * alphap, axis = 0)

                fn = np.array([1 / 3 * Fn[i + 3] - 7 / 6 * Fn[i + 2] + 11 / 6 * Fn[i + 1],
                                -1 / 6 * Fn[i + 2] + 5 / 6 * Fn[i + 1] + 1 / 3 * Fn[i],
                                1 / 3 * Fn[i + 1] + 5 / 6 * Fn[i] - 1 / 6 * Fn[i - 1]
                                ])
                betan = np.array([13 / 12 * (Fn[i + 1] - 2 * Fn[i + 2] + Fn[i + 3]) ** 2 + 1 / 4 * (Fn[i + 3] - 4 * Fn[i + 2] + 3 * Fn[i + 1]) ** 2,
                                13 / 12 * (Fn[i] - 2 * Fn[i + 1] + Fn[i + 2]) ** 2 + 1 / 4 * (Fn[i + 2] - Fn[i]) ** 2 ,
                                13 / 12 * (Fn[i - 1] - 2 * Fn[i] + Fn[i + 1]) ** 2 + 1 / 4 * (Fn[i - 1] - 4 * Fn[i] + 3 * Fn[i + 1]) ** 2])
                alphan = c / (EPSILON + betan) ** 2
                alphan /= np.sum(alphan, axis = 0)
                Fn_hat[i] = np.sum(fn * alphan, axis = 0)

        dx = 1 / (N - 1)
        RHS = np.zeros((N, 3))
        for i in range(ng, N - ng):
            RHS[i] = -(Fp_hat[i] - Fp_hat[i - 1] + Fn_hat[i] - Fn_hat[i - 1]) / dx
        return RHS

class Roe(solver):
    def roe(self, UL, UR):
        rhol, ul, pl = conserved_to_primitive(UL)
        rhor, ur, pr = conserved_to_primitive(UR)
        Hl = GAMMA * pl / ((GAMMA - 1) * rhol) + 0.5 * ul ** 2
        Hr = GAMMA * pr / ((GAMMA - 1) * rhor) + 0.5 * ur ** 2

        u = (rhol ** 0.5 * ul + rhor ** 0.5 * ur) / (rhol ** 0.5 + rhor ** 0.5)
        H = (rhol ** 0.5 * Hl + rhor ** 0.5 * Hr) / (rhol ** 0.5 + rhor ** 0.5)
        c = ((GAMMA - 1) * (H - u ** 2 / 2)) ** 0.5

        lambdas = [u - c, u, u + c]
        lambdas = np.abs(lambdas)

        for i in range(3):
            if lambdas[i] < EPSILON:
                lambdas[i] = (lambdas[i] ** 2 + EPSILON ** 2) / (2 * EPSILON)


        return compute_roe_matrix(u, c, lambdas)

    def RHS(self, U):#计算半离散后右侧表达式 
        N = U.shape[0]
        F = np.apply_along_axis(globals()['F'], axis = 1, arr = U)

        ng = 1 #Roe格式两侧各有1个ghost cell
        #计算正负数值通量
        F_hat = np.zeros((N - 1, 3))
        for i in range(ng - 1, N - ng):
            F_hat[i] = 0.5 * (F[i] + F[i + 1]) - 0.5 * self.roe(U[i], U[i + 1]) @ (U[i + 1] - U[i])

        #计算总体右侧表达式
        dx = 1 / (N - 1)
        RHS = np.zeros((N, 3))
        for i in range(ng, N - ng):
            RHS[i] = -(F_hat[i] - F_hat[i - 1]) / dx

        return RHS

class WENO_Charastic_Reconstruction(solver):
    def RHS(self, U):#计算半离散后右侧表达式 
        N = U.shape[0]
        Fp = np.apply_along_axis(globals()['Fp'], axis = 1, arr = U)
        Fn = np.apply_along_axis(globals()['Fn'], axis = 1, arr = U)

        ng = 3 #WENO格式两侧各有3个ghost cell
        #计算正负数值通量
        Fp_hat = np.zeros((N - 1, 3))
        Fn_hat = np.zeros((N - 1, 3))
        for i in range(ng - 1, N - ng):
            R = compute_R((U[i] + U[i + 1]) / 2)
            R_inv = compute_R_inv((U[i] + U[i + 1]) / 2)
            c = np.array([0.1, 0.6, 0.3])
            
            
            fp = np.array([R_inv @ (1 / 3 * Fp[i - 2] - 7 / 6 * Fp[i - 1] + 11 / 6 * Fp[i]),
                        R_inv @ (1 / 3 * Fp[i + 1] + 5 / 6 * Fp[i] - 1 / 6 * Fp[i - 1]),
                        R_inv @ (-1 / 6 * Fp[i + 2] + 5 / 6 * Fp[i + 1] + 1 / 3 * Fp[i])])
            betap = np.array([13 / 12 * (R_inv @ (Fp[i] - 2 * Fp[i - 1] + Fp[i - 2])) ** 2 + 1 / 4 * (R_inv @ (Fp[i - 2] - 4 * Fp[i - 1] + 3 * Fp[i])) ** 2,
                            13 / 12 * (R_inv @ (Fp[i + 1] - 2 * Fp[i] + Fp[i - 1])) ** 2 + 1 / 4 * (R_inv @ (Fp[i + 1] - Fp[i - 1])) ** 2,
                            13 / 12 * (R_inv @ (Fp[i + 2] - 2 * Fp[i + 1] + Fp[i])) ** 2 + 1 / 4 * (R_inv @ (Fp[i + 2] - 4 * Fp[i + 1] + 3 * Fp[i])) ** 2])
            alphap = c / (EPSILON + betap) ** 2
            alphap /= np.sum(alphap, axis = 0)
            Fp_hat[i] = R @ np.sum(fp * alphap, axis = 0)

            fn = np.array([R_inv @ (1 / 3 * Fn[i + 3] - 7 / 6 * Fn[i + 2] + 11 / 6 * Fn[i + 1]),
                        R_inv @ (-1 / 6 * Fn[i + 2] + 5 / 6 * Fn[i + 1] + 1 / 3 * Fn[i]),
                        R_inv @ (1 / 3 * Fn[i + 1] + 5 / 6 * Fn[i] - 1 / 6 * Fn[i - 1])])
            betan = np.array([13 / 12 * (R_inv @ (Fn[i + 1] - 2 * Fn[i + 2] + Fn[i + 3])) ** 2 + 1 / 4 * (R_inv @ (Fn[i + 3] - 4 * Fn[i + 2] + 3 * Fn[i + 1])) ** 2,
                            13 / 12 * (R_inv @ (Fn[i] - 2 * Fn[i + 1] + Fn[i + 2])) ** 2 + 1 / 4 * (R_inv @ (Fn[i + 2] - Fn[i])) ** 2 ,
                            13 / 12 * (R_inv @ (Fn[i - 1] - 2 * Fn[i] + Fn[i + 1])) ** 2 + 1 / 4 * (R_inv @ (Fn[i - 1] - 4 * Fn[i] + 3 * Fn[i + 1])) ** 2])
            alphan = c / (EPSILON + betan) ** 2
            alphan /= np.sum(alphan, axis = 0)
            Fn_hat[i] = R @ np.sum(fn * alphan, axis = 0)

        dx = 1 / (N - 1)
        RHS = np.zeros((N, 3))
        for i in range(ng, N - ng):
            RHS[i] = -(Fp_hat[i] - Fp_hat[i - 1] + Fn_hat[i] - Fn_hat[i - 1]) / dx
        return RHS
    
def plot_comparison(rho, u, p, ref_data, title_suffix = ""):
    plt.figure(figsize=(12, 8))

    N = len(rho)
    x = np.linspace(-0.5, 0.5, N)
    x_ref = ref_data['x']  # 参考解的网格（10001个点）

    # 找到数值解网格点在参考解中的最近邻索引
    indices = np.abs(x_ref[:, None] - x).argmin(axis=0)
    # 提取匹配的参考解值
    ref_rho_matched = ref_data['rho'][indices]
    ref_u_matched = ref_data['u'][indices]
    ref_p_matched = ref_data['p'][indices]

    l2_rho = calculate_l2_error(rho, ref_rho_matched, N)
    l2_u = calculate_l2_error(u, ref_u_matched, N)
    l2_p = calculate_l2_error(p, ref_p_matched, N)

    # 设置颜色和线型
    num_style = {'color': 'red', 'linestyle': '-', 'linewidth': 2, 'label': 'Numerical'}
    ref_style = {'color': 'blue', 'linestyle': '--', 'linewidth': 1.5, 'label': 'Reference'}
    error_text = f'$L^2$ error: ρ={l2_rho:.3e}, u={l2_u:.3e}, p={l2_p:.3e}'
    
    # 1. 密度对比
    plt.subplot(3, 1, 1)
    plt.plot(x, rho, **num_style)
    plt.plot(ref_data['x'], ref_data['rho'], **ref_style)
    plt.ylabel('Density (ρ)')
    plt.title(f'Solution Comparison {title_suffix}\n{error_text}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis([-0.5, 0.5, 0, 1.1])
    
    # 2. 速度对比
    plt.subplot(3, 1, 2)
    plt.plot(x, u, **num_style)
    plt.plot(ref_data['x'], ref_data['u'], **ref_style)
    plt.ylabel('Velocity (u)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis([-0.5, 0.5, -0.1, 1.0])
    
    # 3. 压强对比
    plt.subplot(3, 1, 3)
    plt.plot(x, p, **num_style)
    plt.plot(ref_data['x'], ref_data['p'], **ref_style)
    plt.ylabel('Pressure (p)')
    plt.xlabel('Position (x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis([-0.5, 0.5, 0, 1.1])
    
    plt.tight_layout()
    plt.savefig(f"./pictures/Solution Comparison {title_suffix}.png", dpi = 300, bbox_inches = 'tight')

#基本参数设置
CFL = 0.4
T = 0.2

#参考解
_, _, values = sod.solve(left_state = (1, 1, 0), right_state = (0.1, 0.125, 0.), geometry = (-0.5, 0.5, 0), t = T, GAMMA = 1.4, npts = 1001)


'''#TVD格式（N = 1001）使用minmod限制器
U = np.zeros((1001, 3))
TVDsolver = TVD(CFL, T, "minmod")
rho, u, p = TVDsolver.solve(U)
plot_comparison(rho, u, p, values, "(TVD Scheme, N=1001, minmod limiter)")

#TVD格式（N = 1001）使用superbee限制器
U = np.zeros((1001, 3))
TVDsolver = TVD(CFL, T, "superbee")
rho, u, p = TVDsolver.solve(U)
plot_comparison(rho, u, p, values, "(TVD Scheme, N=1001, superbee limiter)")

#TVD格式（N = 1001）使用vanleer限制器
U = np.zeros((1001, 3))
TVDsolver = TVD(CFL, T, "vanleer")
rho, u, p = TVDsolver.solve(U)
plot_comparison(rho, u, p, values, "(TVD Scheme, N=1001, vanleer limiter)")

#NND格式（N = 1001）
U = np.zeros((1001, 3))
NNDsolver = NND(CFL, T)
rho, u, p = NNDsolver.solve(U)
plot_comparison(rho, u, p, values, "(NND Scheme, N=1001)")

#WENO格式（N = 1001），我的版本
U = np.zeros((1001, 3))
WENOsolver = WENO(CFL, T, "My Ver")
rho, u, p = WENOsolver.solve(U)
plot_comparison(rho, u, p, values, "(WENO Scheme, N=1001, My version)")

#WENO格式（N = 1001）,论文中版本
U = np.zeros((1001, 3))
WENOsolver = WENO(CFL, T, "Official")
rho, u, p = WENOsolver.solve(U)
plot_comparison(rho, u, p, values, "(WENO Scheme, N=1001, Jiang and Shu et al.)")

#Roe格式（N = 201）
U = np.zeros((201, 3))
Roesolver = Roe(CFL, T)
rho, u, p = Roesolver.solve(U)
plot_comparison(rho, u, p, values, "(Roe Scheme, N=201)")

#Roe格式（N = 501）
U = np.zeros((501, 3))
Roesolver = Roe(CFL, T)
rho, u, p = Roesolver.solve(U)
plot_comparison(rho, u, p, values, "(Roe Scheme, N=501)")

#Roe格式（N = 1001）
U = np.zeros((1001, 3))
Roesolver = Roe(CFL, T)
rho, u, p = Roesolver.solve(U)
plot_comparison(rho, u, p, values, "(Roe Scheme, N=1001)")

#Roe格式（N = 2001）
U = np.zeros((2001, 3))
Roesolver = Roe(CFL, T)
rho, u, p = Roesolver.solve(U)
plot_comparison(rho, u, p, values, "(Roe Scheme, N=2001)")'''

#WENO格式（N = 1001）,论文中版本
U = np.zeros((1001, 3))
WENOsolver = WENO_Charastic_Reconstruction(CFL, T)
rho, u, p = WENOsolver.solve(U)
plot_comparison(rho, u, p, values, "(WENO Scheme with Charastic Reconstruction, N=1001, Jiang and Shu et al.)")