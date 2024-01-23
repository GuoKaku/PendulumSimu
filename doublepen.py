import numpy as np
import matplotlib.pyplot as plt



def double_pendulum_dynamics(theta1, theta2, omega1, omega2):
    g = 9.8  # 重力加速度
    l1 = 1.0  # 绳长
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0
    
    #中间量，降低计算复杂度
    massratio=m1/m2+1
    cosdiff=np.cos(theta1-theta2)
    sindiff=np.sin(theta1-theta2)
    sin1=np.sin(theta1)
    sin2=np.sin(theta2)
    omega1_2=omega1**2*l1
    omega2_2=omega2**2*l2
    denorm=massratio-cosdiff**2
    
    
    # 动力学方程
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    domega1_dt = ((cosdiff*sin2-massratio*sin1)*g-(cosdiff*omega1_2+omega2_2)*sindiff)/(denorm*l1)
    domega2_dt = ((cosdiff*sin1-sin2)*g*massratio+(massratio*omega1_2+cosdiff*omega2_2)*sindiff)/(denorm*l2)
    return dtheta1_dt, dtheta2_dt, domega1_dt, domega2_dt

def euler_dp(theta1, theta2, omega1, omega2, h):
    dtheta1_dt, dtheta2_dt, domega1_dt, domega2_dt = double_pendulum_dynamics(theta1, theta2, omega1, omega2)

    omega1_new = omega1 + domega1_dt * h
    omega2_new = omega2 + domega2_dt * h
    theta1_new = theta1 + omega1_new * h
    theta2_new = theta2 + omega2_new * h
    return theta1_new, theta2_new, omega1_new, omega2_new


def simulate_double_pendulum(theta10, theta20, num_steps, time_step):


    # 初始化数组
    theta1 = np.zeros(num_steps)
    theta2 = np.zeros(num_steps)
    angular_velocities1 = np.zeros(num_steps)
    angular_velocities2 = np.zeros(num_steps)

    # 设置初始条件
    theta1[0] = theta10
    theta2[0] = theta20
    angular_velocities1[0] = 0.0
    angular_velocities2[0] = 0.0
    
    
    
    for step in range(1, num_steps):
        theta1[step], theta2[step], angular_velocities1[step], angular_velocities2[step] = euler_dp( theta1[step-1], theta2[step-1], angular_velocities1[step-1], angular_velocities2[step-1], time_step)

    return  theta1 ,  theta2



if __name__ == "__main__":
    # 模拟参数
    theta1 = np.radians(-20)  # 初始角度，以弧度表示
    theta2 = np.radians(90)

    num_steps = 1000  # 模拟步数
    time_step = 0.01  # 时间步长

    # 进行模拟
    angles1, angles2  = simulate_double_pendulum(theta1, theta2, num_steps, time_step)

    # 绘制模拟结果
    time_values = np.arange(0, num_steps * time_step, time_step)
    plt.plot(time_values, np.degrees(angles1), label='mass1')
    plt.plot(time_values, np.degrees(angles2), label='mass2')

    plt.xlabel('time(s)')
    plt.ylabel(r'$\theta (\circ)$')
    plt.title('Pendulum Simulation')
    plt.legend()
    plt.show()