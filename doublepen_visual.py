from doublepen import simulate_double_pendulum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def random_show():
    # 模拟参数
    theta1 = np.random.uniform(0, 360)  # 初始角度，以弧度表示
    theta2 = np.random.uniform(0, 360)


    num_steps = 1000  # 模拟步数
    time_step = 0.01  # 时间步长
    l1=1.0
    l2=1.0

    # 进行模拟
    angles1, angles2  = simulate_double_pendulum(theta1, theta2, num_steps, time_step)

    x1=l1*np.sin(angles1)
    y1=-l1*np.cos(angles1)
    x2=x1+l2*np.sin(angles2)
    y2=y1-l2*np.cos(angles2)


    # 绘制模拟结果
    idx = np.arange(len(x1))
    time_values = np.arange(0, num_steps * time_step, time_step)
    plt.scatter(x1, y1, label='mass1',c=idx, cmap='viridis', edgecolors='black')
    plt.scatter(x2, y2, label='mass2',c=idx, cmap='viridis', edgecolors='black')

    plt.colorbar(label='Index')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Double Pendulum')
    plt.legend()
    plt.show()
    
    
def animate_dp(theta1,theta2, num_steps = 1000 ,time_step = 0.01):

    # 创建画布和轴
    fig = plt.figure()
    ax = plt.subplot(111, aspect = 'equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # 创建两个小球的点
    points, = ax.plot([], [], 'o', markersize=10)
    line, = ax.plot([], [], '-', lw=2,color="orange")
    line_origin, = ax.plot([], [], '-', lw=2,color="orange")  # 新增的原点到第一个小球的连线


    l1=1.0
    l2=1.0

    # 进行模拟
    angles1, angles2  = simulate_double_pendulum(theta1, theta2, num_steps, time_step)

    x1=l1*np.sin(angles1)
    y1=-l1*np.cos(angles1)
    x2=x1+l2*np.sin(angles2)
    y2=y1-l2*np.cos(angles2)

    coords = np.stack([np.stack([x1,y1],axis=1),np.stack([x2,y2],axis=1)],axis=1)
    

    # 初始化函数
    def init():
        
        points.set_data([], [])
        line.set_data([], [])
        line_origin.set_data([], [])

        return points,line ,line_origin

    # 更新函数，每个时间步更新小球的位置
    def update(frame):

        x_values = coords[frame, :, 0]
        y_values = coords[frame, :, 1]
        points.set_data(x_values, y_values)
        line.set_data(x_values, y_values)
        line_origin.set_data([0, x_values[0]], [0, y_values[0]])
    
        return points,line,line_origin


    # 创建动画
    num_frames = coords.shape[0]
    anim = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True ,interval = 10)

    # 显示动画
    plt.show()

if __name__ == "__main__":
    
    animate_dp(130,150,1000,0.01)