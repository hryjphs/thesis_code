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

filename = 'dddqn_tf2_lstm_dv0_risk05.png'
x = [i+1 for i in range(num_simulation)]
plot_learning_curve(x, scores, filename)
agent.save_model()



total_episode_test = 3000
env_test2 = TradingEnv(continuous_action_flag=False, sabr_flag=True, dg_random_seed=2, num_sim=total_episode_test)

delta_u,delta_r,delta_pl,delta_cpl = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='delta_dv0_risk05', delta_flag=True, bartlett_flag=False)
barlette_u,barlette_r,barlette_pl,barlette_cpl = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='barlette_dv0_risk05', delta_flag=False, bartlett_flag=True)

rl_u,rl_r,rl_pl,rl_cpl = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='rl_dv0_risk05', delta_flag=False, bartlett_flag=False)
    
#plot_obj(delta_u, figure_file='delta_u_dv0')
#plot_obj(rl_u, figure_file='rl_u_dv0')
#plot_obj(barlette_u, figure_file='barlette_u_dv0')

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
df1.to_csv('delta_dv0_r_risk05.csv', index=False, encoding='utf-8')


b = []
for i in range(len(rl_r)):
    for j in rl_r[i]:
        b.append(j)




history = dict(zip(epi, b))
#name = os.path.join('history', name)
df2 = pd.DataFrame(history,index=[0])
df2.to_csv('rl_dv0_r_risk05.csv', index=False, encoding='utf-8')

c = []
for i in range(len(barlette_r)):
    for j in barlette_r[i]:
        c.append(j)




history = dict(zip(epi, c))
#name = os.path.join('history', name)
df7 = pd.DataFrame(history,index=[0])
df7.to_csv('barlette_dv0_r_risk05.csv', index=False, encoding='utf-8')        
        
############# episode p&l

history = dict(zip(epi, rl_cpl))
#name = os.path.join('history', name)
df3 = pd.DataFrame(history,index=[0])
df3.to_csv('rl_dv0_cpl_risk05.csv', index=False, encoding='utf-8')    

history = dict(zip(epi, delta_cpl))
#name = os.path.join('history', name)
df4 = pd.DataFrame(history,index=[0])
df4.to_csv('delta_dv0_cpl_risk05.csv', index=False, encoding='utf-8')    

history = dict(zip(epi, barlette_cpl))
#name = os.path.join('history', name)
df8 = pd.DataFrame(history,index=[0])
df8.to_csv('barlette_dv0_cpl_risk05.csv', index=False, encoding='utf-8')        
        

############# time step p&l

d = []
for i in range(len(barlette_pl)):
    for j in barlette_pl[i]:
        d.append(j)




history = dict(zip(epi, d))
#name = os.path.join('history', name)
df9 = pd.DataFrame(history,index=[0])
df9.to_csv('barlette_dv0_pl_risk05.csv', index=False, encoding='utf-8')  

e = []
for i in range(len(rl_pl)):
    for j in rl_pl[i]:
        e.append(j)




history = dict(zip(epi, e))
#name = os.path.join('history', name)
df5 = pd.DataFrame(history,index=[0])
df5.to_csv('rl_dv0_pl_risk05.csv', index=False, encoding='utf-8')        
        
f = []
for i in range(len(delta_pl)):
    for j in delta_pl[i]:
        f.append(j)



        
history = dict(zip(epi, f))
#name = os.path.join('history', name)
df6 = pd.DataFrame(history,index=[0])
df6.to_csv('delta_dv0_pl_risk05.csv', index=False, encoding='utf-8')    


print('Fishined')
        











