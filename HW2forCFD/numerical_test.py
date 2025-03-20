import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class finite_diff(ABC):
    def _prepare_inputs(self, h, x, dtype=np.float64): #保证浮点类型
        h = np.asarray(h, dtype=dtype)
        x = np.asarray(x, dtype=dtype)
        return h, x
    
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def compute(self, f, h, x, dtype = np.float64):
        pass
    
    @abstractmethod
    def order(self):#返回导数阶数
        pass

    #精度计算,浮点精度采用默认单精度
    def convergence_order_compute(self): #diff_method是采取的差分格式
        h_values = np.logspace(-3, 0, 100) #1到1e-3的100个等比步长
        u = lambda x: np.sin(x) #以sinx作为测试函数, 目标点为1
        u_h = []
        u_2h = []
        for h in h_values:
            u_h.append(self.compute(u, h, 1))
            u_2h.append(self.compute(u, 2 * h, 1))
        u_h = np.array(u_h)
        u_2h = np.array(u_2h)
        if self.order() == 1:
            u_e = np.cos(1)
        else:
            u_e = -np.sin(1)
        ratios = np.abs(u_e - u_2h) / (np.abs(u_e - u_h) + 1e-12)#见数理算法原理推导
        log_values = np.log2(ratios)
        return h_values, log_values
    
    def convergence_order_paint(self):#精度阶计算结果绘制，ai生成
        h_values, log_values = self.convergence_order_compute()
        plt.figure(figsize=(8, 6))
        plt.plot(h_values, log_values, label=r'$p = \lim_{h \to 0^+}\log_2\frac{\vert{u_e-u_{2h}\vert}}{\vert{u_e-u_h}\vert}$', color='b')
        plt.xscale('log')  # 步长为 log scale
        plt.xlabel('Step size $h$ (log scale)', fontsize=12)
        plt.ylabel(r'$\log_2\frac{\vert{u_e-u_{2h}\vert}}{\vert{u_e-u_h}\vert}$', fontsize=12)
        plt.title('Convergence Order for ' + self.name(), fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.show()


#一阶导数的前向差分
class forward_diff_1st(finite_diff):
    def compute(self, f, h, x, dtype = np.float64): #f是函数，h是步长，x是目标点，dtype是精度
        h, x = self._prepare_inputs(h, x, dtype)
        derivative = (f(x + h) - f(x)) / h
        return np.asarray(derivative, dtype=dtype)
    
    def order(self):
        return 1
    
    def name(self):
        return "Forward_Diff_1st"


#一阶导数的中心差分 
class central_diff_1st(finite_diff):
    def compute(self, f, h, x, dtype = np.float64):
        h, x = self._prepare_inputs(h, x, dtype)
        derivative = (f(x + h) - f(x - h)) / (2 * h)
        return np.asarray(derivative, dtype=dtype)
    
    def order(self):
        return 1
    
    def name(self):
        return "Central_Diff_1st"

#二阶导数的前向差分 
class forward_diff_2st(finite_diff):
    def compute(self, f, h, x, dtype = np.float64): 
        h, x = self._prepare_inputs(h, x, dtype)
        derivative = (f(x + 2 * h) - 2 * f(x + h) + f(x)) / (h ** 2)
        return np.asarray(derivative, dtype=dtype)
    
    def order(self):
        return 2
    
    def name(self):
        return "Forward_Diff_2st"

#二阶导数的中心差分 
class central_diff_2st(finite_diff):
    def compute(self, f, h, x, dtype = np.float64): 
        h, x = self._prepare_inputs(h, x, dtype)
        derivative = (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
        return np.asarray(derivative, dtype=dtype)
    
    def order(self):
        return 2
    
    def name(self):
        return "Central_Diff_2st"

def convergence_order():
    diffs = [forward_diff_1st(), forward_diff_2st(), central_diff_1st(), central_diff_2st()]
    for diff in diffs:
        diff.convergence_order_paint()

convergence_order()