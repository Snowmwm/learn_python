#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 10
ACTIONS = ['left', 'right'] #available
EPSILON = 0.9 #选择最优动作的概率
ALPHA = 0.1 #learning rate
LAMBDA = 0.9 #discont factor
MAX_EPISODES = 12
FRESH_TIME = 0.02

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), 
        columns = actions,
    )
    #print(table)
    return table
    
def choose_action(state, q_table):
    #如何选择一个动作
    state_actions = q_table.iloc[state, :] 
    #将当前state的各动作Q值复制到state_actions
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name
    
def get_env_feedback(S, A):
    """
    环境对行为的反应
    输入当前环境状态S和采取的动作A
    输出下一个状态S_和得分奖励R
    """
    if A =='right':
        if S == N_STATES - 2: #因为是从S=0开始的
            S_ = 'terminal' #到达终点
            R = 1
        else:
            S_ = S + 1
            R = 0
    else: #A == 'left'
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R
    
def update_env(S, episode, step_counter):
    #建立环境
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        #到达终点后停顿2秒，输出当前回合数及移动步数
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                 ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
    
def rl():
    #主循环
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        update_env(S, episode, step_counter) #建立新环境
        
        is_terminated = False
        while not is_terminated:
        
            A = choose_action(S, q_table) #根据state和Q表选择动作
            S_, R = get_env_feedback(S, A) #获得环境对动作的反馈
            
            q_predict = q_table.ix[S, A] #从Q表中获得Q估计值
            
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
                #Q现实 = R + γ*max(Q(s',a1), Q(s',a2),……)
                #即当前动作从环境实际获得的回报 + (衰减γ x 预测的下回合回报)
            else:
                q_target = R
                is_terminated = True
                
            q_table.ix[S, A] += ALPHA *(q_target - q_predict) #q_table更新
            #新q值 = 老q值 + α(Q现实 - Q估计)
            
            S = S_ #移动到下一个state
            
            step_counter += 1
            update_env(S, episode, step_counter) #更新环境
            
    return q_table
    
if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ_table:\n')
    print(q_table)
    
    
    
    
    
    
    
    
    