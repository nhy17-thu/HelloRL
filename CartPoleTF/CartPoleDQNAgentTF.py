"""
Deep Q network,
Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import gym
import matplotlib.pyplot as plt

from CartPoleTF.DQNCoreTF import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

print("env.action_space:           ", env.action_space)  # Discrete(2)
print("env.action_space.n         :", env.action_space.n)  # 2
print("env.observation_space:      ", env.observation_space)  # Box(4,)
print("env.observation_space.shape:", env.observation_space.shape)  # (4,)
print("env.observation_space.high: ",
      env.observation_space.high)  # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print("env.observation_space.low:  ",
      env.observation_space.low)  # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

RL = DeepQNetwork(n_actions=env.action_space.n,  # 2
                  n_features=env.observation_space.shape[0],  # 4
                  learning_rate=0.01,
                  e_greedy=0.9,
                  replace_target_iter=100,
                  memory_size=2000,
                  e_greedy_increment=0.001, )

START_POINT = 1000  # 从START_POINT次互动开始训练网络
total_steps = 0  # 记录总的互动次数（从第一个episode开始计数，跨域episode也可以）
epi_r_record = []  # 记录每个episode的总奖励值
epi_step_record = []  # 记录每个episode的总步数

for i_episode in range(300):  # 100个episode

    observation = env.reset()  # observation是一个np.ndarray,如：[-0.02081334 -0.02731451  0.02365333 -0.04925076]
    epi_r = 0  # 每个episode初始奖励为0
    epi_step = 0  # episode运行的步数计数器

    # 下面的每次循环：
    # 1.根据初始ob选择action
    # 2.action与环境互动，产生数据，并存储数据
    # 3.做两个判断：①超过START_POINT步后，开始训练；②如果episode结束，print出信息，并结束该episode
    while True:
        # env.render()  # 渲染输出画面，注释掉则不显示画面，但是可以大大加快仿真速度

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        ######################################################
        #### x,          / *cart position, meters * /    #####
        #### x_dot,      / *cart velocity * /            #####
        #### theta,      / *pole angle, radians * /      #####
        #### theta_dot;  / *pole angular velocity * /    #####
        ######################################################
        # x_threshold = 2.4
        # theta_threshold_radians = 12 * 2 * math.pi / 360 即：12°对应的弧度0.209
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # 0.8是一个偏置，abs(x)=2.4时，r1=-0.8,abs(x)=0时，r1=0.2
        r2 = (env.theta_threshold_radians - abs(
            theta)) / env.theta_threshold_radians - 0.5  # 同上，# 0.5是一个偏置，r2=(-0.5,0.5)
        reward = r1 + r2  # 若出现redeclared 'xxxxx' defined above without usage，可能是误报

        RL.store_transition(observation, action, reward, observation_)

        epi_r += reward
        if total_steps > START_POINT:
            RL.learn()

        if done:
            print('episode:    ', i_episode,
                  'epi_r:    ', round(epi_r, 2),  # round(x,y)浮点数x保留y位小数
                  ' epsilon:    ', round(RL.epsilon, 2))
            epi_r_record.append(epi_r)  # 记录每个episode奖励值
            epi_step_record.append(epi_step)  # 记录每个回合能够运行的step数
            epi_step = 0  # 每个episode结束后，计数器置0
            break

        observation = observation_
        epi_step += 1
        total_steps += 1

plt.figure(1)
plt.plot(epi_r_record)
plt.figure(2)
print(epi_step_record)
plt.plot(epi_step_record)
plt.show()
# RL.plot_cost()
