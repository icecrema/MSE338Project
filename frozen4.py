# Frozen Lake RL
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from scipy.special import expit

global gNbuffer, gNbatch, gNEpisodes, gNtrials

gNEpisodes = 2000 # Number of episodes
gNtrials = 1 # number of trials
gNbuffer = 100
gNbatch = 30

class FunctionFamily(object):

	def __init__(self):
		self.type = 'linear'
		self.dim = (1,1)

	def map(self,s,a,theta):
		# maybe a if base on self.type
		if self.type == 'linear':
			# theta is a matrix, whose row index are the 
			return theta[s,a]

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

def cache(Buffer,sars,finite = False, NBuffer = gNbuffer):
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
	eps = 100000/(l+1)**2
	eps = 0.01
	p = np.random.rand()
	if p<eps:
		a = np.random.choice(range(A))
	else:
		Reward = [J(a) for a in range(A)]
		maxa = np.argwhere(Reward == np.max(Reward)).flatten().tolist()
		a = np.random.choice(maxa)
		# a = maxa[0]
	return a

def learn(Q,Buffer,thetatilde,l,method = 'lsvi',batchtype = 'random'):
	if np.sum(thetatilde) == None:
		return np.zeros(Q.dim)

	'''
	lsvi + linear function with full dimension + theta_hat = 0
	'''
	if method == 'lsvi':
		H = 15
		v = 1
		NData = len(Buffer)
		Lambda = v/0.5
		# theta = np.zeros(Q.dim) # initialization of theta
		theta = thetatilde
		thetaLen = Q.dim[0]*Q.dim[1]
		for h in range(H):
			y = np.zeros(NData)
			X = np.zeros((NData,thetaLen))
			# construct the data (y,X)
			for n in xrange(NData):
				y[n] = expit(Buffer[n][2]/1)*(Buffer[n][2] + max(theta[Buffer[n][3]]))
				x = np.zeros(Q.dim)
				x[Buffer[n][0],Buffer[n][1]] = expit(Buffer[n][2]/1)*1
				X[n,:] = x.reshape(thetaLen)
			# do linear regression update
			theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X) + \
			 Lambda*np.identity(X.shape[1])),X.T),y).reshape(Q.dim)

	if method == 'lsvi_td':
		alpha = 0.2 # learning rate
		gamma = 0.95 # discount factor
		temperature = 100000 # control the probability of weighted sampling
		Nbatch = min(gNbatch,len(Buffer)) # batch number
		Buffer = np.array(Buffer,dtype = int)
		if batchtype == 'random':
			Nbatch = min(NB,len(Buffer))
			batchindex = np.random.randint(len(Buffer),size = Nbatch)
			batch = Buffer[batchindex,:]
		if batchtype == 'weighted_random':
			# generate the probability according to the reward
			R = Buffer[:,2]
			p = expit(R/temperature)
			p = p/np.sum(p)
			batchindex = np.random.choice(range(len(Buffer)),size = Nbatch, replace = False, p =p)
			batch = Buffer[batchindex,:]
		if Q.type == 'linear':
			theta = np.copy(thetatilde)
			for i in range(Nbatch):
				theta[batch[i][0],batch[i][1]] += alpha*(batch[i][2] + \
					gamma*np.max(theta[batch[i][3],:]) - theta[batch[i][0],batch[i][1]])
			# for i in range(Nbatch):
			# 	theta[batch[i][0],batch[i][1]] += alpha*(batch[i][2] + \
			# 		gamma*np.max(thetatilde[batch[i][3],:]) - thetatilde[batch[i][0],batch[i][1]])
		if Q.type == 'MLP':
			pass
	return theta

def Q_leaning():
	pass

def live():

	env = gym.make('FrozenLake-v0')
	reward_ep = list()
	Q = FunctionFamily()
	Q.type = 'linear'
	Q.dim = (env.observation_space.n,env.action_space.n)

	rewared_per_ep_ave = np.zeros(gNEpisodes)
	for trial in range(gNtrials): 
		Buffer = cache(None,None)
		theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
		reward_per_ep = list()
		for l in xrange(gNEpisodes):
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
			theta = learn(Q,Buffer,theta,l,method = 'lsvi',batchtype = 'weighted_random')
			reward_ep.append(rewared_thisep)
			reward_per_ep.append(np.mean(reward_ep[np.max(len(reward_ep)-100,0):]))
			print l,rewared_thisep
		print trial
		rewared_per_ep_ave += np.array(reward_per_ep)
	rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
	plt.plot(rewared_per_ep_ave)
	plt.show()

	return

def main():

	live()

if __name__ == "__main__":
	# global parameters, for tunning
	main()

