from __future__ import division
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math
import mdptoolbox
import random
import gym


row_size = 4
num_rows = 4
col_size = row_size
num_states = row_size * row_size
num_actions = 4
num_episodes = 500
num_rewards = 3
num_instances = 20




trans_prior = np.zeros((num_actions, num_states, num_states))
reward_prior = np.zeros((num_actions, num_states, num_rewards))


trans_samp = np.zeros((num_actions, num_states, num_states))
reward_samp = np.zeros((num_states, num_actions))


#Initialize Priors

# Transition Prior
for a in range(num_actions):
    for s in range(num_states):
        for new in range(num_states):
            trans_prior[a][s][new]=2/num_states

# Reward Prior  
for a in range(num_actions):
    for s in range(num_states):
        for r in range(num_rewards):
            reward_prior[a][s][r] = 1



all_results = np.zeros((num_instances, num_episodes))
all_regrets = np.zeros((num_instances, num_episodes))



for instance in range(num_instances):
    
    reward_obtained = np.empty(num_episodes)

    trans_post = trans_prior
    reward_post = reward_prior

    obs_rewards = np.zeros((num_actions, num_states , num_rewards))
    obs_trans = np.zeros((num_actions, num_states , num_states))
     
    
    env = gym.make('FrozenLake-v0')
        
    for episode in range(num_episodes):
        done = False
        current_state = env.reset()

        #update posterior
        trans_post = trans_post + obs_trans
        reward_post = reward_post + obs_rewards
        
        obs_rewards = np.zeros((num_actions, num_states , num_rewards))
        obs_trans = np.zeros((num_actions, num_states , num_states))
        


        # sample reward and transitions
     
        for a in range(num_actions):
            for s in range(num_states):
                temp = np.random.dirichlet(trans_post[a][s])
                for next_place in range(num_states):

                    trans_samp[a][s][next_place] = temp[next_place]
                  
                temp2 = np.random.dirichlet(reward_post[a][s])
                reward_samp[s][a] = -1 * temp2[0] + 1 * temp2[2]

        
        # Calculate optimal policy for sampled transition and reward matrices
        pi = mdptoolbox.mdp.PolicyIteration(trans_samp, reward_samp, .99)
        pi.run()
        opt_strat = pi.policy
        
        
        # Carry out optimal action at each stage
        while done == False:
            action = opt_strat[int(current_state)]
            
            new_state, reward, done, _ = env.step(action)
            
            obs_trans[action][current_state][new_state] += 1
            
            current_state = new_state
            if done == False:
                obs_rewards[action][current_state][1] += 1
        
        if reward == 0:
            obs_rewards[action][current_state][0] += 1
        else:
            obs_rewards[action][current_state][2] += 1
                    
        reward_obtained[episode] = reward
        all_results[instance][episode] = reward
        print "episode: ", episode, reward
        
    print "instance: ", instance


avg_rewards = np.average(all_results, axis=0)

plt.plot(avg_rewards)
plt.yticks(np.arange(0, 1, .05))
plt.ylim(0,1)
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Reward (Averaged over 20 Instances)')
plt.title('Frozen Lake PSRL')

#plt.savefig('PSRL_N' + str(num_rows) + '.png')

plt.show()










