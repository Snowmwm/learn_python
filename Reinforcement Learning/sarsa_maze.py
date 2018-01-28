#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from maze_env import Maze
from RL_brain import SarsaTable, SarsaLambdaTable

MAX_EPISODES = 100

def update_sarsa():
    for episode in range(MAX_EPISODES):
        #初始化(state的)观测值
        observation = env.reset()

        #根据 state 的观测值挑选 action
        action = RL.choose_action(str(observation)) #传入string作为索引

        while True:
            #刷新环境
            env.render()

            #获得环境对动作的反馈
            observation_, reward, done = env.step(action)

            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            #从这个动作中学习(更新Q值)
            RL.learn(str(observation), action, reward,
                    str(observation_), action_)

            #移动到下一个state
            observation = observation_
            action = action_

            #如果掉入hell或到达终点就结束这一回合
            if done:
                print(RL.q_table)
                break

    print('finish')
    env.destroy()


def update_sarsalambda():
    for episode in range(MAX_EPISODES):
        #初始化(state的)观测值
        observation = env.reset()

        #根据 state 的观测值挑选 action
        action = RL.choose_action(str(observation)) #传入string作为索引

        RL.eligibility_trace *= 0

        while True:
            #刷新环境
            env.render()
 
            #获得环境对动作的反馈
            observation_, reward, done = env.step(action)

            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            #从这个动作中学习(更新Q值)
            RL.learn(str(observation), action, reward,
                    str(observation_), action_)

            #移动到下一个state
            observation = observation_
            action = action_
            
            #如果掉入hell或到达终点就结束这一回合
            if done:
                print(RL.q_table)
                break

    print('finish')
    env.destroy()

if __name__ == "__main__":
    env = Maze()

    #RL = SarsaTable(actions=list(range(env.n_actions)))
    #env.after(100, update_sarsa)

    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))
    env.after(100, update_sarsalambda)

    env.mainloop()