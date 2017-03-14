#Q-Learning
#http://mnemstudio.org/path-finding-q-learning-tutorial.htm

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

num_episodes = 2000

def run_episode(env,Q,learning_rate,discount,episode,render=False):
	observation = env.reset()
	done = False
	t_reward = 0
	max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
	for i in xrange(max_steps):		
		if done:
			break

		if render:
			env.render()

		curr_state = observation

		action = np.argmax(Q[curr_state,:]  + np.random.randn(1, env.action_space.n)*(1./(episode+1)))
			
		observation, reward, done, info = env.step(action)

		t_reward += reward

		#Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
		Q[curr_state,action] += learning_rate * (reward+ discount*np.max(Q[observation,:])-Q[curr_state,action])
	
	return Q, t_reward

def train():
	env = gym.make('FrozenLake8x8-v0')
	# env = wrappers.Monitor(env, '/tmp/FrozenLake-experiment-6',force=True)
	learning_rate = 0.81
	discount = 0.97

	reward_ep = list()
	reward_per_ep = list()
	Q = np.zeros((env.observation_space.n, env.action_space.n))
	for i in xrange(num_episodes):
		Q,reward = run_episode(env,Q,learning_rate,discount,i)
		reward_ep.append(reward)
		reward_per_ep.append(np.mean(reward_ep[np.max(len(reward_ep)-100,0):-1]))
		#print "----------Next Episode---------"
		#print i
	plt.plot(reward_per_ep)
	plt.xlabel('period')
	plt.ylabel('mean reward')
	plt.savefig('test', bbox_inches='tight')
	plt.show()

	return Q

q = train()
