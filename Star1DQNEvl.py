#!/usr/bin/env python
# coding: utf-8

# # Step1 安装依赖

# In[1]:


# !pip uninstall -y parl  # 说明：AIStudio预装的parl版本太老，容易跟其他库产生兼容性冲突，建议先卸载
# !pip uninstall -y pandas scikit-learn # 提示：在AIStudio中卸载这两个库再import parl可避免warning提示，不卸载也不影响parl的使用

# !pip install gym
# !pip install paddlepaddle==1.6.3
# !pip install parl==1.3.1
# 说明：安装日志中出现两条红色的关于 paddlehub 和 visualdl 的 ERROR 与parl无关，可以忽略，不影响使用


# # Step2  导入依赖

# In[2]:


import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
from Paddle.paddle2 import Paddle

# # Step3 设置超参数

# In[3]:


LEARN_FREQ = 10  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = int(1e4)  # replay memory的大小，越大越占用内存 20000
MEMORY_WARMUP_SIZE = 1000  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 128  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.9  # reward 的衰减因子，一般取 0.9 到 0.999 不等，越小表示注重眼前利益，否则注重长远利益

E_GREED = 0.5  # 学习时随机动作的概率，就是探索
E_GREED_D = 1e-7  # 学习时随着时间，每一步的随机性的下降
######################################################################
######################################################################
#
# 1. 请设定 learning rate，可以从 0.001 起调，尝试增减
#
######################################################################
######################################################################
LEARNING_RATE = 0.001  # 学习率


# # Step4 搭建Model、Algorithm、Agent架构
# * `Agent`把产生的数据传给`algorithm`，`algorithm`根据`model`的模型结构计算出`Loss`，使用`SGD`或者其他优化器不断的优化，`PARL`架构可以很方便的应用在各类深度强化学习问题中。
#
# ## Model
# * `Model`用来定义前向(`Forward`)网络，用户可以自由的定制自己的网络结构。

# In[4]:


class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 100
        hid2_size = 20
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q


# ## Algorithm
# * `Algorithm`定义了具体的算法来更新前向网络(`Model`)，也就是通过定义损失函数来更新`Model`，和算法相关的计算都放在`algorithm`中。

# In[5]:


from parl.algorithms import DQN  # 直接从parl库中导入DQN算法，无需自己重写算法


# ## Agent
# * `Agent`负责算法与环境的交互，在交互过程中把生成的数据提供给`Algorithm`来更新模型(`Model`)，数据的预处理流程也一般定义在这里。

# In[6]:


class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作

        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost


# # Step5 ReplayMemory
# * 经验池：用于存储多条经验，实现 经验回放。

# In[7]:


# replay_memory.py
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), np.array(action_batch).astype('float32'), np.array(
            reward_batch).astype('float32'), np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype(
            'float32')

    def __len__(self):
        return len(self.buffer)


# # Step6 Training && Test（训练&&测试）

# In[8]:


# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        reward, next_obs, done = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        evl_step = 0
        while (evl_step <1000):
            #print(evl_step)
            action = agent.predict(obs)  # 预测动作，只选最优动作
            reward, next_obs, done = env.step(action)
            episode_reward += reward

            next_obs = np.reshape(next_obs, (1, 5))
            obs = next_obs
            if done:
                break
            evl_step += 1
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


# # Step7 创建环境和Agent，创建经验池，启动训练，保存模型

# In[9]:


# 创建环境
env = Paddle()
act_dim = 3  # 0 move left  1 do nothing  2 move right
obs_dim = 5

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=act_dim)
algorithm = DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_dim,
    act_dim=act_dim,
    e_greed=E_GREED,
    e_greed_decrement=E_GREED_D
)

# 加载模型

save_path = './dqn_model_2100(7.92).ckpt'
agent.restore(save_path)







eval_reward_all = []
for i in range(0, 100):

    # test part
    eval_reward = evaluate(env, agent)  # render=True 查看显示效果
    eval_reward_all.append(eval_reward)
    logger.info('step:{}    eval_reward:{}  '.format(
        i, eval_reward))
eval_reward_mean = np.mean(eval_reward_all)
eval_reward_all = np.sum(eval_reward_all)

logger.info('test_reward_mean:{}    test_reward_sum:{}  '.format(
    eval_reward_mean, eval_reward_all))



