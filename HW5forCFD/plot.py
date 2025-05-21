import numpy as np
import matplotlib.pyplot as plt

data = np.load('./data/flow_data.npz', allow_pickle = True)
vortex = []

def find_vortex(x, y, psi, N):#寻找流函数极值点确定涡心
    for j in range(N - 2, 0, -1):
        for i in range(1, N - 1):
            if psi[i, j] > max(psi[i - 1, j], psi[i + 1, j], psi[i, j - 1], psi[i, j + 1], psi[i + 1, j + 1], psi[i - 1, j + 1], psi[i - 1, j - 1], psi[i + 1, j - 1]):
                vortex.append([x[i], y[j]])
            elif psi[i, j] < min(psi[i - 1, j], psi[i + 1, j], psi[i, j - 1], psi[i, j + 1], psi[i + 1, j + 1], psi[i - 1, j + 1], psi[i - 1, j - 1], psi[i + 1, j - 1]):
                vortex.append([x[i], y[j]])

#绘图，ai生成
def plot_streamline(x, y, psi, u, v, N, density, x_start, x_end, y_start, y_end, num):#order代表着重展示的涡的编号
    # 创建网格
    X, Y = np.meshgrid(x, y)
    
    plt.figure()
    # 使用更精细的contourf显示
    cf = plt.contourf(X, Y, psi.T, levels=100, cmap='jet', alpha=0.7)
    plt.colorbar(cf, label='Stream Function (ψ)')

    # 大幅增加流线密度和优化显示
    plt.streamplot(
        X, Y, u.T, v.T,
        density = density,                  # 可调整到5-15
        color = 'k',                  
        linewidth = 0.8,             
        arrowsize = 0.5,              
        arrowstyle = '->',            
        minlength = 0.01,              
        maxlength = 1000,             
        integration_direction='both',
        zorder=3                    
    )
    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)

    plt.scatter(vortex[num - 1][0], vortex[num - 1][1], s=100, c='white', edgecolors='black', linewidths=1.5, zorder=4)
    plt.text(vortex[num - 1][0] + (x_end - x_start) / 50, vortex[num - 1][1] + (y_end - y_start) / 50, f'({vortex[num - 1][0]:.3f}, {vortex[num - 1][1]:.3f})', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.title(f'Lid-Driven Cavity Flow (Re={1000}, Nx=Ny={N}, vortex={num})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'./pictures/streamline_vortex_{num}')
    
def plot_v(N, y, v):
    #x=0.5处的v速度剖面
    plt.figure(figsize=(6, 5))
    x_idx = int(N / 2)
    plt.plot(y, v[:, x_idx], 'b-', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.title('Vertical Velocity (v) at x=0.5')
    plt.xlabel('y')
    plt.ylabel('v velocity')
    plt.grid(True)
    plt.savefig('./pictures/v_at_x_m')
    
def plot_u(N, x, u):
    # y=0.5处的u速度剖面
    plt.figure(figsize=(6, 5))
    y_idx = int(N / 2)
    plt.plot(x, u[y_idx, :], 'r-', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.title('Horizontal Velocity (u) at y=0.5')
    plt.xlabel('x')
    plt.ylabel('u velocity')
    plt.grid(True)
    plt.savefig('./pictures/u_at_y_m')

find_vortex(data['x'], data['y'], data['psi'], data['N'])
plot_streamline(data['x'], data['y'], data['psi'], data['u'], data['v'], data['N'], 2, 0, 1, 0, 1, 1)#主涡
plot_streamline(data['x'], data['y'], data['psi'], data['u'], data['v'], data['N'], 8, 0.6, 1, 0, 0.5, 2)#右侧二次涡
plot_streamline(data['x'], data['y'], data['psi'], data['u'], data['v'], data['N'], 8, 0, 0.3, 0, 0.3, 3)#左侧二次涡
plot_streamline(data['x'], data['y'], data['psi'], data['u'], data['v'], data['N'], 15, 0.97, 1, 0, 0.03, 4)#右侧三次涡
plot_streamline(data['x'], data['y'], data['psi'], data['u'], data['v'], data['N'], 15, 0, 0.02, 0, 0.03, 5)#左侧三次涡
plot_u(data['N'], data['x'], data['u'])
plot_v(data['N'], data['y'], data['v'])