# Frozen Lake RL
# This is for continuous state
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

class FunctionFamily(object):

	def __init__(self):
		self.type = 'linear'
		self.numa = 1 # number of action
		self.dim = (8,4)

	def map(self,s,a,theta):
		# maybe a if base on self.type
		if self.type == 'linear':
			# theta is a matrix, whose row index are the 
			return np.dot(s,theta[:,a])

	def partial(self,s,a):
		# gradient of the Q function
		g = np.zeros(self.dim)
		g[:,a] = s
		return g

	def maxQ_a(self,s,theta):
		return max([self.map(s,a,theta) for a in range(self.numa)])


def cache(Buffer,sars,finite = False, NBuffer = 600):
	if Buffer == None:
		return []
	else:
		Buffer.append(sars)
		if finite:
			if len(Buffer) >= NBuffer:
				Buffer.pop(0)
	return Buffer

def act(J,A,l,method = 'epsgreedy'):
	# if method == 'epsgreedy':
	eps = 1000/(l+1)**2
	# eps = 0.05
	p = np.random.rand()
	if p<eps:
		a = np.random.choice(range(A))
	else:
		Reward = [J(a) for a in range(A)]
		maxa = np.argwhere(Reward == np.max(Reward)).flatten().tolist()
		# print Reward, maxa
		a = np.random.choice(maxa)
	return a

def learn(Q,Buffer,thetatilde,method = 'lsvi'):
	if np.sum(thetatilde) == None:
		return np.zeros(Q.dim)

	'''
	lsvi + linear function with full dimension + theta_hat = 0
	'''
	# if method == 'lsvi':
	# 	H = 3
	# 	v = 1
	# 	NData = len(Buffer)
	# 	Lambda = v/1.
	# 	theta = np.zeros(Q.dim) # initialization of theta
	# 	# theta = thetatilde
	# 	thetaLen = Q.dim[0]*Q.dim[1]
	# 	for h in range(H):
	# 		y = np.zeros(NData)
	# 		X = np.zeros((NData,thetaLen))
	# 		# construct the data (y,X)
	# 		for n in xrange(NData):
	# 			y[n] = Buffer[n][2] + 0.99*max(theta[Buffer[n][3]])
	# 			x = np.zeros(Q.dim)
	# 			x[Buffer[n][0],Buffer[n][1]] = 1
	# 			X[n,:] = x.reshape(thetaLen)
	# 		# do linear regression update
	# 		theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X) + \
	# 		 Lambda/1000*np.identity(X.shape[1])),X.T),y).reshape(Q.dim)

	if method == 'lsvi_td':
		alpha = 0.00001 # learning rate
		gamma = 1 # discount factor
		Nbatch = min(500,len(Buffer))
		batchindex = np.random.randint(len(Buffer),size = Nbatch)
		Buffer = np.array(Buffer)
		batch = Buffer[batchindex,:]
		theta = thetatilde
		# calculateing the gradient:
		gradient = np.zeros(Q.dim)
		for i in range(Nbatch):
			gradient += (batch[i][2] + gamma*Q.maxQ_a(batch[i][0],theta) - \
			 Q.map(batch[i][0],batch[i][1],theta)) * Q.partial(batch[i][0],batch[i][1])
		theta += alpha*gradient
		# for i in range(Nbatch):
		# 	theta[Buffer[i][0],Buffer[i][1]] += alpha*(Buffer[i][2] + \
		# 		gamma*np.max(theta[Buffer[i][3],:]) - theta[Buffer[i][0],Buffer[i][1]])
	return theta

def live():
	NEpisodes = 3000 # Number of episodes

	env = gym.make('LunarLander-v2')
	reward_ep = list()
	Q = FunctionFamily()
	Q.type = 'linear'
	# Q.dim = (env.observation_space.shape[0],env.action_space.n)

	Buffer = cache(None,None)
	theta = learn(Q,Buffer,None) # Initialization of parameter theta
	reward_per_ep = list()
	for l in range(NEpisodes):
		# print l
		s = env.reset()
		done = False
		rewared_thisep = 0
		while not done:
			# env.render()
			a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
			sprime, r, done, _ = env.step(a)
			Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
			s = sprime
			rewared_thisep += r
		theta = learn(Q,Buffer,theta,method = 'lsvi_td')
		reward_ep.append(rewared_thisep)
		reward_per_ep.append(np.mean(reward_ep[np.max(len(reward_ep)-100,0):-1]))
		print l,rewared_thisep

	plt.plot(reward_per_ep)
	plt.show()

	return


if __name__ == "__main__":
	live()


