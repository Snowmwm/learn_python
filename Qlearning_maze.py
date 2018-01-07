#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from maze_env import Maze

MAX_EPISODES = 100
EPSILON = 0.9 #选择最优动作的概率
ALPHA = 0.1 #learning rate
GAMMA = 0.9 #discont factor


class QLearningTable:
    def __init__(self, actions, learning_rate=ALPHA,
                reward_decay=GAMMA, e_greedy=EPSILON):
        self.actions = actions  # a list: [0,1,2,3]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        
        #初始化生成空的q_table
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
    def choose_action(self, observation):
        self.check_state_exist(observation) #检测当前state是否在q_table中存在
        
        if np.random.uniform() < self.epsilon:
            # 选择最优动作
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index)) 
            #随机打乱q_table中action的顺序,以免当q值相等时只取第一个值
            action = state_action.idxmax()
        else:
            # 随机选择动作
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_) #检测 s_ 是否在 q_table 中存在
        q_predict = self.q_table.ix[s, a] #从Q表中获得Q估计值
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
            #Q现实 = R + γ*max(Q(s',a1), Q(s',a2),……)
        else:
            q_target = r
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict) #q_table更新
        #新q值 = 老q值 + α(Q现实 - Q估计)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            #每当出现新的state,就将其加入q_table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
def update():
    for episode in range(MAX_EPISODES):
        observation = env.reset() #初始化(state的)观测值
        while True:
            #刷新环境
            env.render()

            #根据 state 的观测值挑选 action
            action = RL.choose_action(str(observation)) #传入string作为索引

            #获得环境对动作的反馈
            observation_, reward, done = env.step(action)

            #从这个动作中学习(更新Q值)
            RL.learn(str(observation), action, reward, str(observation_))

            #移动到下一个state
            observation = observation_

            #如果掉入hell或到达终点就结束这一回合
            if done:
                #print(RL.q_table)
                break

    print('finish')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()