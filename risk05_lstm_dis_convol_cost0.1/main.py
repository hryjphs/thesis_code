# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:16:53 2021

@author: 洪睿
"""


import numpy as np
import gym
#from utils import plotLearning
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
num_simulation = 10000
env = TradingEnv(num_sim = num_simulation, continuous_action_flag=False) # see tradingenv.py for more info 
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
    while not done:
        action = agent.choose_action(tf.convert_to_tensor(np.expand_dims(observation,-1)))  #action is tensor
        #action = action.numpy()[0]           #change to numpy
        observation_, reward, done, info,_ = env.step(action)
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

filename = 'dddqn_tf2_lstm_dc01_risk05.png'
x = [i+1 for i in range(num_simulation)]
plot_learning_curve(x, scores, filename)
agent.save_model()



total_episode_test = 3000
env_test2 = TradingEnv(continuous_action_flag=False, sabr_flag=False, dg_random_seed=2, num_sim=total_episode_test)

delta_u,delta_r, delta_pl, delta_cpl = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='delta_dc01_risk05.csv', delta_flag=True, bartlett_flag=False)
rl_u, rl_r,rl_pl, rl_cpl = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='rl_dc01_risk05.csv', delta_flag=False, bartlett_flag=False)
       
#plot_obj(delta_u, figure_file='delta_u_dc0')
#plot_obj(rl_u, figure_file='rl_u_dc0')

########### time step reward
a = []
for i in range(len(delta_r)):
    for j in delta_r[i]:
        a.append(j)



upperbound = total_episode_test*25 + 1
epi = np.arange(1, upperbound, 1)  
history = dict(zip(epi, a))
#name = os.path.join('history', name)
df1 = pd.DataFrame(history,index=[0])
df1.to_csv('delta_dc01_r_risk05.csv', index=False, encoding='utf-8')


b = []
for i in range(len(rl_r)):
    for j in rl_r[i]:
        b.append(j)



history = dict(zip(epi, b))
#name = os.path.join('history', name)
df2 = pd.DataFrame(history,index=[0])
df2.to_csv('rl_dc01_r_risk05.csv', index=False, encoding='utf-8')

############# episode p&l

history = dict(zip(epi, rl_cpl))
#name = os.path.join('history', name)
df3 = pd.DataFrame(history,index=[0])
df3.to_csv('rl_dc01_cpl_risk05.csv', index=False, encoding='utf-8')    

history = dict(zip(epi, delta_cpl))
#name = os.path.join('history', name)
df4 = pd.DataFrame(history,index=[0])
df4.to_csv('delta_dc01_cpl_risk05.csv', index=False, encoding='utf-8')    

############# time step p&l
e = []
for i in range(len(rl_pl)):
    for j in rl_pl[i]:
        e.append(j)




history = dict(zip(epi, e))
#name = os.path.join('history', name)
df5 = pd.DataFrame(history,index=[0])
df5.to_csv('rl_dc01_pl_risk05.csv', index=False, encoding='utf-8')        
        
f = []
for i in range(len(delta_pl)):
    for j in delta_pl[i]:
        f.append(j)



        
history = dict(zip(epi, f))
#name = os.path.join('history', name)
df6 = pd.DataFrame(history,index=[0])
df6.to_csv('delta_dc01_pl_risk05.csv', index=False, encoding='utf-8')    


print('Fishined')
        

