# Frozen Lake RL
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

class FunctionFamily(object):

	def __init__(self):
		self.type = 'linear'
		self.dim = (1,1)

	def map(self,s,a,theta):
		# maybe a if base on self.type
		if self.type == 'linear':
			# theta is a matrix, whose row index are the 
			return theta[s,a]


def cache(Buffer,sars,finite = False, NBuffer = 5000):
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
	eps = 1000000/(l+1)**2
	p = np.random.rand()
	if p<eps:
		a = np.random.choice(range(A))
	else:
		Reward = [J(a) for a in range(A)]
		maxa = np.argwhere(Reward == np.max(Reward)).flatten().tolist()
		a = np.random.choice(maxa)
	return a

def learn(Q,Buffer,thetatilde,method = 'lsvi'):
	if np.sum(thetatilde) == None:
		return np.zeros(Q.dim)

	'''
	lsvi + linear function with full dimension + theta_hat = 0
	'''
	if method == 'lsvi':
		H = 3
		v = 1
		NData = len(Buffer)
		Lambda = v/1.
		# theta = np.zeros(Q.dim) # initialization of theta
		theta = thetatilde
		thetaLen = Q.dim[0]*Q.dim[1]
		for h in range(H):
			y = np.zeros(NData)
			X = np.zeros((NData,thetaLen))
			# construct the data (y,X)
			for n in xrange(NData):
				y[n] = Buffer[n][2] + 0.99*max(theta[Buffer[n][3]])
				x = np.zeros(Q.dim)
				x[Buffer[n][0],Buffer[n][1]] = 1
				X[n,:] = x.reshape(thetaLen)
			# do linear regression update
			theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X) + \
			 Lambda/1000*np.identity(X.shape[1])),X.T),y).reshape(Q.dim)

	if method == 'lsvi_td':
		alpha = 0.1 # learning rate
		gamma = 0.99 # discount factor
		Nbatch = min(400,len(Buffer))
		batchindex = np.random.randint(len(Buffer),size = Nbatch)
		Buffer = np.array(Buffer,dtype = int)
		batch = Buffer[batchindex,:]
		theta = thetatilde
		for i in range(Nbatch):
			theta[Buffer[i][0],Buffer[i][1]] += alpha*(Buffer[i][2] + \
				gamma*np.max(theta[Buffer[i][3],:]) - theta[Buffer[i][0],Buffer[i][1]])
	return theta


def live():
	NEpisodes = 20000 # Number of episodes

	env = gym.make('FrozenLake8x8-v0')
	reward_ep = list()
	Q = FunctionFamily()
	Q.type = 'linear'
	Q.dim = (env.observation_space.n,env.action_space.n)

	Buffer = cache(None,None)
	theta = learn(Q,Buffer,None) # Initialization of parameter theta
	reward_per_ep = list()
	for l in xrange(NEpisodes):
		# print l
		s = env.reset()
		done = False
		rewared_thisep = 0
		while not done:
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


