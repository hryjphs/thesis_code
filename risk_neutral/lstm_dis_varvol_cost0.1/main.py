# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:31:00 2021

@author: 洪睿
"""
import numpy as np
import gym
#from utils import plotLearning
import tensorflow as tf


tf.compat.v1.enable_eager_execution()
    #tf.compat.v1.disable_eager_execution()
num_simulation = 10000
env = TradingEnv(num_sim = num_simulation, continuous_action_flag=False,sabr_flag = True) # see tradingenv.py for more info 
lr = 0.001
agent = Agent(gamma=1, epsilon=1.0, lr=lr, n_actions = 101,
                input_dims=env.num_state,env = env,
                mem_size=1000, batch_size=128,
                 prioritized_replay= True)
agent.load_model()

scores = []
    #eps_history = []


for i in range(num_simulation):
    done = False
    score = 0
    observation = env.reset()  #[price, position, ttm], price=S, position=0, ttm=init_ttm
    j=0
    while not done:
        action = agent.choose_action(tf.convert_to_tensor(np.expand_dims(observation,-1)))  #action is tensor
        #action = action.numpy()[0]           #change to numpy
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
        #eps_history.append(agent.epsilon)
    scores.append(score)  # score for every episode
    avg_score = np.mean(scores[-100:])
    if i % 100 == 0:
        print('episode %.2f' % i, 'score %.2f' % score, 'average_score %.2f' % avg_score)
        #        'epsilon %.2f' % agent.epsilon)

filename = 'dddqn_tf2_lstm_dv01.png'
x = [i+1 for i in range(num_simulation)]
plot_learning_curve(x, scores, filename)
agent.save_model()



total_episode_test = 3000
env_test2 = TradingEnv(continuous_action_flag=False, sabr_flag=True, dg_random_seed=2, num_sim=total_episode_test)

delta_u = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='delta_dv01', delta_flag=True, bartlett_flag=False)
barlette_u = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='barlette_dv01', delta_flag=False, bartlett_flag=True)

rl_u = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='rl_dv01', delta_flag=False, bartlett_flag=False)
    
plot_obj(delta_u, figure_file='delta_u_dv01')
plot_obj(rl_u, figure_file='rl_u_dv01')
plot_obj(barlette_u, figure_file='barlette_u_dv01')











