# Frozen Lake RL
# This is for continuous state

# MLP backward propagation: https://en.wikipedia.org/wiki/Backpropagation

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from scipy.special import expit

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

class MLP(FunctionFamily):
	# A three-layer MLP (one hidden layer)

	def __init__(self):
		self.type = 'MLP'
		self.numa = 4 # cardinality of action space (discrete)
		self.Nin = 12 # number of input node
		self.Nhidden = 3 # number of nodes in the hidden layer
		self.Nout = 1 # number of output node, when learning Q function, thus is always 1
		self.outputrange = [-300,300]

	def W_initial(self):
		# initialization of all the parameters in the MLP
		W_in = np.random.rand(self.Nin,self.Nhidden) # the weight of the input layer
		W_hidden = np.random.rand(self.Nhidden,self.Nout) # weight from hidden layer to output
		return [W_in,W_hidden]

	def getinput(self,s,a):
		# translating the state-action pair to input
		invec = np.zeros(self.Nin)
		invec[:len(s)] = s
		invec[len(s)+a] = 1
		return invec

	def forwardprop(self,s,a,theta):
		# forward propagation with logistic function
		# return: O[O_hidden,O_output], is a combination of the out put of hidden and out node
		W_in, W_hidden = theta
		invec = self.getinput(s,a)
		O_hidden = np.zeros(self.Nhidden)
		O_output = np.zeros(self.Nout)
		for i in range(self.Nhidden):
			O_hidden[i] = expit(np.dot(invec,W_in[:,i]))
		for j in range(self.Nout):
			O_output[j] = expit(np.dot(O_hidden,W_hidden[:,j]))
		return O_hidden, O_output

	def map(self,s,a,theta):
		# perform a step of forward propagation, and then scale the output into the output range
		_, O_output = self.forwardprop(s,a,theta)
		Q = O_output[0]
		return self.outputrange[0] + Q*(self.outputrange[1] - self.outputrange[0])

	def partial(self,s,a,t,theta):
		# gradient of the Q function
		# input: state-action s,a; objective value t

		# perform a step of the f-propagation, and then b-propagation
		W_in, W_hidden = theta
		invec = self.getinput(s,a)
		O_hidden, O_output = self.forwardprop(s,a,theta)
		GW_in = np.zeros((self.Nin,self.Nhidden)) # gradient of the weight of the input layer
		GW_hidden = np.zeros((self.Nhidden,self.Nout)) # gradient of weight from hidden layer to output
		delta = np.zeros(self.Nout) # delta of the output layer
		for j in range(self.Nout):
			delta[j] = (O_output[j] - t) * O_output[j] * (1 - O_output[j])
		for i in range(self.Nhidden):
			for j in range(self.Nout):
				GW_hidden[i,j] = delta[j] * O_hidden[i]
		for i in range(self.Nin):
			for j in range(self.Nhidden):
				GW_in[i,j] = invec[i] * O_hidden[j] * (1 - O_hidden[j]) * \
				 sum([delta[l]*W_hidden[j,l] for l in range(self.Nout)])
		return GW_in, GW_hidden

	def maxQ_a(self,s,theta):
		return max([self.map(s,a,theta) for a in range(self.numa)])



def cache(Buffer,sars,finite = False, NBuffer = 2000):
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
	eps = 300.0/(l+1)
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

def learn(Q,Buffer,thetatilde,l,method = 'lsvi'):
	if l+1 == 0:
		if Q.type == 'MLP':
			return Q.W_initial()
		return np.random.rand(Q.dim[0],Q.dim[1])

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
	# 	for h in range(H):g
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
		alpha = 0.01 # learning rate
		gamma = 1 # discount factor
		Nbatch = min(500,len(Buffer))
		batchindex = np.random.randint(len(Buffer),size = Nbatch)
		Buffer = np.array(Buffer)
		batch = Buffer[batchindex,:]
		# calculateing the gradient:
		if Q.type == 'linear':
			theta = thetatilde
			gradient = np.zeros(Q.dim)
			for i in range(Nbatch):
				gradient += (batch[i][2] + gamma*Q.maxQ_a(batch[i][0],theta) - \
				 Q.map(batch[i][0],batch[i][1],theta)) * Q.partial(batch[i][0],batch[i][1])
			theta += alpha*gradient

		if Q.type == 'MLP':
			W_in, W_hidden = thetatilde
			g_in = np.zeros(W_in.shape)
			g_hidden = np.zeros(W_hidden.shape)
			for i in range(Nbatch):
				d1,d2 = Q.partial(batch[i][0],batch[i][1], \
					batch[i][2] + gamma*Q.maxQ_a(batch[i][0],thetatilde),thetatilde)
				g_in += d1
				g_hidden += d2
			W_in -= alpha * g_in
			W_hidden -= alpha * g_hidden
			# print np.linalg.norm(W_in) + np.linalg.norm(W_in)
			theta = [W_in,W_hidden]

		# for i in range(Nbatch):
		# 	theta[Buffer[i][0],Buffer[i][1]] += alpha*(Buffer[i][2] + \
		# 		gamma*np.max(theta[Buffer[i][3],:]) - theta[Buffer[i][0],Buffer[i][1]])
	return theta

def live():
	NEpisodes = 3000 # Number of episodes

	env = gym.make('LunarLander-v2')
	reward_ep = list()

	# Q = FunctionFamily()
	# Q.type = 'linear'
	# Q.dim = (env.observation_space.shape[0],env.action_space.n)

	# Q as a MLP
	Q = MLP()

	Buffer = cache(None,None)
	theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
	reward_per_ep = list()
	for l in range(NEpisodes):
		# print l
		s = env.reset()
		done = False
		rewared_thisep = 0
		while not done:
			env.render()
			a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
			sprime, r, done, _ = env.step(a)
			Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
			s = sprime
			rewared_thisep += r
		theta = learn(Q,Buffer,theta,l+1,method = 'lsvi_td')
		reward_ep.append(rewared_thisep)
		reward_per_ep.append(np.mean(reward_ep[np.max(len(reward_ep)-100,0):-1]))
		print l,rewared_thisep

	plt.plot(reward_per_ep)
	plt.show()

	return


if __name__ == "__main__":
	live()


