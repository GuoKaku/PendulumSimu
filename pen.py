import numpy as np
import matplotlib.pyplot as plt


def pendulum_dynamics(theta, omega, damping_factor):
    g = 9.8  # 重力加速度
    length = 1.0  # 绳长

    # 动力学方程
    dtheta_dt = omega
    domega_dt = -(g / length) * np.sin(theta)- damping_factor * omega

    return dtheta_dt, domega_dt

def runge_kutta4(theta, omega, damping_factor, h):
    # RK4 方法
    k1_theta, k1_omega = pendulum_dynamics(theta, omega, damping_factor)
    k2_theta, k2_omega = pendulum_dynamics(theta + 0.5 * h * k1_theta, omega + 0.5 * h * k1_omega, damping_factor)
    k3_theta, k3_omega = pendulum_dynamics(theta + 0.5 * h * k2_theta, omega + 0.5 * h * k2_omega, damping_factor)
    k4_theta, k4_omega = pendulum_dynamics(theta + h * k3_theta, omega + h * k3_omega, damping_factor)

    # 更新角度和角速度
    theta_new = theta + (h / 6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    omega_new = omega + (h / 6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

    return theta_new, omega_new

def euler(theta, omega, damping_factor, h):
    dtheta_dt, domega_dt = pendulum_dynamics(theta, omega, damping_factor)

    omega_new = omega + domega_dt * h
    theta_new = theta + omega_new * h
    return theta_new, omega_new

SimulateMethod = {
    "euler": euler,
    "rk4": runge_kutta4,
}

def simple_harmonic_motion(initial_angle, num_steps, time_step, damping_factor = 0.0):
    g = 9.8  # 重力加速度
    length = 1.0  # 绳长
    omega = np.sqrt(g / length)
    m=1
    
    time_values = np.arange(0, num_steps * time_step, time_step)
    
        # 计算阻尼角频率
    omega_d = np.sqrt((g / length) - (damping_factor / (2 * m))**2)

    # 计算简谐振动的角度随时间的变化
    theta = initial_angle * np.exp(-damping_factor / (2 * m) * time_values) * np.cos(omega_d * time_values)

    return theta

def simulate_pendulum(initial_angle, num_steps, time_step, method="euler", damping_factor = 0.0):


    # 初始化数组
    angles = np.zeros(num_steps)
    angular_velocities = np.zeros(num_steps)

    # 设置初始条件
    angles[0] = initial_angle
    angular_velocities[0] = 0.0
    
    update_func=SimulateMethod[method]
    
    for step in range(1, num_steps):
        angles[step], angular_velocities[step] = update_func(angles[step-1], angular_velocities[step-1], damping_factor, time_step)

    return angles



# 模拟参数
initial_angle = np.radians(45)  # 初始角度，以弧度表示
damping_factor = 0.1
num_steps = 100  # 模拟步数
time_step = 0.1  # 时间步长

# 进行模拟
angles_euler = simulate_pendulum(initial_angle, num_steps, time_step, method="euler",damping_factor=damping_factor)
angles_rk4 = simulate_pendulum(initial_angle, num_steps, time_step, method="rk4",damping_factor=damping_factor)
angles_harmony = simple_harmonic_motion(initial_angle, num_steps, time_step,damping_factor=damping_factor)

# 绘制模拟结果
time_values = np.arange(0, num_steps * time_step, time_step)
plt.plot(time_values, np.degrees(angles_euler), label='Euler')
plt.plot(time_values, np.degrees(angles_rk4), label='RK4')
plt.plot(time_values, np.degrees(angles_harmony),label='harmony')

plt.xlabel('time(s)')
plt.ylabel(r'$\theta (\circ)$')
plt.title('Pendulum Simulation')
plt.legend()
plt.show()