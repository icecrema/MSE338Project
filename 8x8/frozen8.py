# Frozen Lake RL
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from scipy.special import expit

global gNbuffer, gNbatch, gNEpisodes, gNtrials, gepsilon, gH, gLam, ggamma, galpha

gNEpisodes = 2000 # Number of episodes
gNtrials = 30 # number of trials
gNbuffer = 20
gNbatch = 10
gepsilon = 0.001
gH = 30
gLam = 1
ggamma = 0.95
galpha = 0.1

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

def act(J,A,l,epsilon = gepsilon, method = 'epsgreedy',adaptive = False):
	# if method == 'epsgreedy':
	eps = epsilon
	if adaptive:
		eps = 100000/(l+1)**2
		# if l >=300:
		# 	eps = eps/l
	p = np.random.rand()
	if p<eps:
		a = np.random.choice(range(A))
	else:
		Reward = [J(a) for a in range(A)]
		maxa = np.argwhere(Reward == np.max(Reward)).flatten().tolist()
		a = np.random.choice(maxa)
		# a = maxa[0]
	return a

def learn(Q,Buffer,thetatilde,l,tH = gH,Lam = gLam, Nbatch = gNbatch, \
gamma = ggamma, alpha = galpha, method = 'lsvi',batchtype = 'random'):
	if np.sum(thetatilde) == None:
		return np.zeros(Q.dim)

	'''
	lsvi + linear function with full dimension + theta_hat = 0
	'''
	if method == 'lsvi':
		H = tH
		# v = 1
		NData = len(Buffer)
		Lambda = Lam
		theta = np.zeros(Q.dim) # initialization of theta
		# theta = thetatilde
		thetaLen = Q.dim[0]*Q.dim[1]
		for h in range(H):
			y = np.zeros(NData)
			X = np.zeros((NData,thetaLen))
			# construct the data (y,X)
			for n in xrange(NData):
				y[n] = Buffer[n][2] + max(theta[Buffer[n][3]])
				x = np.zeros(Q.dim)
				x[Buffer[n][0],Buffer[n][1]] = 1
				X[n,:] = x.reshape(thetaLen)
			# do linear regression update
			theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X) + \
			 Lambda*np.identity(X.shape[1])),X.T),y).reshape(Q.dim)

	if method == 'grlsvi':
		H = tH
		# v = 1
		NData = len(Buffer)
		Lambda = Lam
		theta = np.zeros(Q.dim) # initialization of theta
		# theta = thetatilde
		thetaLen = Q.dim[0]*Q.dim[1]
		for h in range(H):
			y = np.zeros(NData)
			X = np.zeros((NData,thetaLen))
			# construct the data (y,X)
			for n in xrange(NData):
				y[n] = Buffer[n][2] + max(theta[Buffer[n][3]]) + np.random.randn()
				x = np.zeros(Q.dim)
				x[Buffer[n][0],Buffer[n][1]] = 1
				X[n,:] = x.reshape(thetaLen)
			# do linear regression update
			theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X) + \
			 Lambda*np.identity(X.shape[1])),X.T),y).reshape(Q.dim)


	if method == 'lsvi_td':
		# alpha = 0.2 # learning rate
		# gamma = 0.95 # discount factor
		temperature = 100000 # control the probability of weighted sampling
		Nbatch = min(Nbatch,len(Buffer)) # batch number
		Buffer = np.array(Buffer,dtype = int)
		if batchtype == 'random':
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


def live_1_scale(env,Q):

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 2)])

	labels = []
	# unscaled
	rewared_per_ep_ave = np.zeros(gNEpisodes)
	for trial in range(gNtrials): 
		Buffer = cache(None,None)
		theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
		reward_ep = list()
		reward_per_ep = list()
		for l in xrange(gNEpisodes):
			# print l
			s = env.reset()
			done = False
			rewared_thisep = 0
			while not done:
				a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
				sprime, r, done, _ = env.step(a)
				if done and r ==0:
					Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
				else:
					Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
				s = sprime
				rewared_thisep += r
			theta = learn(Q,Buffer,theta,l,method = 'lsvi',batchtype = 'random')
			reward_ep.append(rewared_thisep)
			reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
			print l,rewared_thisep
		print trial
		rewared_per_ep_ave += np.array(reward_per_ep)
	rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
	plt.plot(rewared_per_ep_ave)
	labels.append('lsvi, unscaled')

	# scaled data
	rewared_per_ep_ave = np.zeros(gNEpisodes)
	for trial in range(gNtrials): 
		Buffer = cache(None,None)
		theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
		reward_ep = list()
		reward_per_ep = list()
		for l in xrange(gNEpisodes):
			# print l
			s = env.reset()
			done = False
			rewared_thisep = 0
			while not done:
				a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
				sprime, r, done, _ = env.step(a)
				if done and r ==0:
					Buffer = cache(Buffer,(s,a,-1,sprime),finite = True)
				else:
					Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
				s = sprime
				rewared_thisep += r
			theta = learn(Q,Buffer,theta,l,method = 'lsvi',batchtype = 'random')
			reward_ep.append(rewared_thisep)
			reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
			# print l,rewared_thisep
		print trial
		rewared_per_ep_ave += np.array(reward_per_ep)
	rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
	plt.plot(rewared_per_ep_ave)
	labels.append('lsvi, scaled')
	plt.xlabel('episode')
	plt.ylabel('averaged reward')
	plt.legend(labels,loc=2)
	plt.savefig('1_lsvi_scaling', bbox_inches='tight')

	return

def live_2_lsvi_H(env,Q):

	disc_H = [5,10,20,50]

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(disc_H))])
	labels = []

	# plus the terminating time of each episode
	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()
	for H in disc_H:
		rewared_per_ep_ave = np.zeros(gNEpisodes)
		tau_per_ep_ave = np.zeros(gNEpisodes)
		for trial in xrange(gNtrials): 
			Buffer = cache(None,None)
			theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
			reward_ep = list()
			tau_ep = list()
			reward_per_ep = list()
			tau_per_ep = list()
			for l in xrange(gNEpisodes):
				# print l
				s = env.reset()
				done = False
				rewared_thisep = 0
				tau = 0
				while not done:
					a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
					sprime, r, done, _ = env.step(a)
					if done and r == 0:
						Buffer = cache(Buffer,(s,a,-1,sprime),finite = True)
					else:
						Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
					s = sprime
					rewared_thisep += r
					tau += 1
				theta = learn(Q,Buffer,theta,l,tH = H,method = 'lsvi',batchtype = 'random')
				reward_ep.append(rewared_thisep)
				tau_ep.append(tau)
				reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
				tau_per_ep.append(np.mean(tau_ep[np.max([len(tau_ep)-100,0]):]))
				# print l,rewared_thisep
			print H, trial
			rewared_per_ep_ave += np.array(reward_per_ep)
			tau_per_ep_ave += np.array(tau_per_ep)
		rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
		tau_per_ep_ave = tau_per_ep_ave*1.0/gNtrials
		ax1.plot(rewared_per_ep_ave)
		ax2.plot(tau_per_ep_ave)
		labels.append(r'lsvi, H=%d'%H)
	ax1.set_xlabel('episode')
	ax2.set_xlabel('episode')
	ax1.set_ylabel('averaged reward')
	ax2.set_ylabel('average termination time')
	ax1.legend(labels,loc=4)
	ax2.legend(labels,loc=4)
	fig1.savefig('2_lsvi_H_r', bbox_inches='tight')
	fig2.savefig('2_lsvi_H_tau', bbox_inches='tight')

	return

def live_3_lsvi_buffersize(env,Q):

	disc_Nbuffer = [200,500,1000,5000]

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(disc_Nbuffer))])
	labels = []

	for Nb in disc_Nbuffer:
		rewared_per_ep_ave = np.zeros(gNEpisodes)
		for trial in range(gNtrials): 
			Buffer = cache(None,None)
			theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
			reward_per_ep = list()
			reward_ep = list()
			for l in range(gNEpisodes):
				# print l
				s = env.reset()
				done = False
				rewared_thisep = 0
				while not done:
					a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
					sprime, r, done, _ = env.step(a)
					if done and r ==0:
						Buffer = cache(Buffer,(s,a,-1,sprime),finite = True,NBuffer =Nb)
					else:
						Buffer = cache(Buffer,(s,a,r,sprime),finite = True,NBuffer =Nb)
					s = sprime
					rewared_thisep += r
				theta = learn(Q,Buffer,theta,l,method = 'lsvi',batchtype = 'random')
				reward_ep.append(rewared_thisep)
				reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
				# print l,rewared_thi
			print Nb, trial
			rewared_per_ep_ave += np.array(reward_per_ep)
		rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
		plt.plot(rewared_per_ep_ave)
		labels.append(r'lsvi, Nbuffer=%d'%Nb)
	plt.xlabel('episode')
	plt.ylabel('averaged reward')
	plt.legend(labels,loc=2)
	plt.savefig('3_lsvi_buffersize', bbox_inches='tight')
	# plt.show()
	return

def live_4_lsvi_lambda(env,Q):

	disc_lam = [0.01,0.1,1,10,100,1000]

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(disc_lam))])
	labels = []

	for lam in disc_lam:
		rewared_per_ep_ave = np.zeros(gNEpisodes)
		for trial in xrange(gNtrials): 
			Buffer = cache(None,None)
			theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
			reward_per_ep = list()
			reward_ep = list()
			for l in range(gNEpisodes):
				# print l
				s = env.reset()
				done = False
				rewared_thisep = 0
				while not done:
					a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
					sprime, r, done, _ = env.step(a)
					if done and r ==0:
						Buffer = cache(Buffer,(s,a,-1,sprime),finite = True)
					else:
						Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
					s = sprime
					rewared_thisep += r
				theta = learn(Q,Buffer,theta,l,Lam = lam,method = 'lsvi',batchtype = 'random')
				reward_ep.append(rewared_thisep)
				reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
				# print l,rewared_thi
			print lam, trial
			rewared_per_ep_ave += np.array(reward_per_ep)
		rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
		plt.plot(rewared_per_ep_ave)
		labels.append(r'lsvi, lambda=%0.2f'%lam)
	plt.xlabel('episode')
	plt.ylabel('averaged reward')
	plt.legend(labels,loc=2)
	plt.savefig('4_lsvi_lambda', bbox_inches='tight')
	# plt.show()
	return

def live_5_lsvitd_batchsize(env,Q):

	# fix batchsize
	# disc_buffer = [10,50,500,1000]
	# disc_batch = [5,5,5,5]

	# fix buffersize
	disc_buffer = [1000,1000,1000,1000]
	disc_batch = [5,50,300,800]

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(disc_buffer))])
	labels = []

	for nbuffer,nbatch in zip(disc_buffer,disc_batch):
		rewared_per_ep_ave = np.zeros(gNEpisodes)
		for trial in xrange(gNtrials): 
			Buffer = cache(None,None)
			theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
			reward_per_ep = list()
			reward_ep = list()
			for l in range(gNEpisodes):
				# print l
				s = env.reset()
				done = False
				rewared_thisep = 0
				while not done:
					a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
					sprime, r, done, _ = env.step(a)
					if done and r ==0:
						Buffer = cache(Buffer,(s,a,-1,sprime),finite = True,NBuffer = nbuffer)
					else:
						Buffer = cache(Buffer,(s,a,r,sprime),finite = True,NBuffer = nbuffer)
					s = sprime
					rewared_thisep += r
				theta = learn(Q,Buffer,theta,l,Nbatch = nbatch, method = 'lsvi_td',batchtype = 'random')
				reward_ep.append(rewared_thisep)
				reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
				# print l,rewared_thisep
			print nbuffer,nbatch, trial
			rewared_per_ep_ave += np.array(reward_per_ep)
		rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
		plt.plot(rewared_per_ep_ave)
		labels.append(r'lsvi_td, %d Buffer, %d Batch'%(nbuffer,nbatch))
	plt.xlabel('episode')
	plt.ylabel('averaged reward')
	plt.legend(labels,loc=4)
	plt.savefig('5_lsvi_td_BBsize_2', bbox_inches='tight')
	# plt.show()
	return

def live_6_lsvitd_gamma(env,Q):

	disc_gamma = [1,0.95,0.5,0.1]

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(disc_gamma))])
	labels = []

	for ga in disc_gamma:
		rewared_per_ep_ave = np.zeros(gNEpisodes)
		for trial in xrange(gNtrials): 
			Buffer = cache(None,None)
			theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
			reward_per_ep = list()
			reward_ep = list()
			for l in range(gNEpisodes):
				# print l
				s = env.reset()
				done = False
				rewared_thisep = 0
				while not done:
					a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
					sprime, r, done, _ = env.step(a)
					if done and r ==0:
						Buffer = cache(Buffer,(s,a,-1,sprime),finite = True)
					else:
						Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
					s = sprime
					rewared_thisep += r
				theta = learn(Q,Buffer,theta,l,gamma = ga, method = 'lsvi_td',batchtype = 'random')
				reward_ep.append(rewared_thisep)
				reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
				# print l,rewared_thisep
			print ga, trial
			rewared_per_ep_ave += np.array(reward_per_ep)
		rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
		plt.plot(rewared_per_ep_ave)
		labels.append(r'lsvi_td, gamma = %.2f'%ga)
	plt.xlabel('episode')
	plt.ylabel('averaged reward')
	plt.legend(labels,loc=2)
	plt.savefig('6_lsvi_td_gamma', bbox_inches='tight')
	# plt.show()
	return

def live_7_lsvitd_alpha(env,Q):

	disc_alpha = [1,0.5,0.1,0.01]

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(disc_alpha))])
	labels = []

	for al in disc_alpha:
		rewared_per_ep_ave = np.zeros(gNEpisodes)
		for trial in xrange(gNtrials): 
			Buffer = cache(None,None)
			theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
			reward_per_ep = list()
			reward_ep = list()
			for l in range(gNEpisodes):
				# print l
				s = env.reset()
				done = False
				rewared_thisep = 0
				while not done:
					a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
					sprime, r, done, _ = env.step(a)
					if done and r ==0:
						Buffer = cache(Buffer,(s,a,-1,sprime),finite = True)
					else:
						Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
					s = sprime
					rewared_thisep += r
				theta = learn(Q,Buffer,theta,l,alpha = al, method = 'lsvi_td',batchtype = 'random')
				reward_ep.append(rewared_thisep)
				reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
				# print l,rewared_thisep
			print al, trial
			rewared_per_ep_ave += np.array(reward_per_ep)
		rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
		plt.plot(rewared_per_ep_ave)
		labels.append(r'lsvi_td, alpha = %.2f'%al)
	plt.xlabel('episode')
	plt.ylabel('averaged reward')
	plt.legend(labels,loc=2)
	plt.savefig('7_lsvi_td_alpha_2', bbox_inches='tight')
	# plt.show()
	return

def live_101_act_eps(env,Q):

	disc_eps = [0.001,0.01,0.1,1]

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(disc_eps))])
	labels = []

	for eps in disc_eps:
		rewared_per_ep_ave = np.zeros(gNEpisodes)
		for trial in xrange(gNtrials): 
			Buffer = cache(None,None)
			theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
			reward_per_ep = list()
			reward_ep = list()
			for l in range(gNEpisodes):
				# print l
				s = env.reset()
				done = False
				rewared_thisep = 0
				while not done:
					a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l,epsilon = eps)
					sprime, r, done, _ = env.step(a)
					if done and r ==0:
						Buffer = cache(Buffer,(s,a,-1,sprime),finite = True)
					else:
						Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
					s = sprime
					rewared_thisep += r
				theta = learn(Q,Buffer,theta,l, method = 'lsvi_td',batchtype = 'random')
				reward_ep.append(rewared_thisep)
				reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
				# print l,rewared_thisep
			print eps, trial
			rewared_per_ep_ave += np.array(reward_per_ep)
		rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
		plt.plot(rewared_per_ep_ave)
		labels.append(r'lsvi_td, alpha = %.3f'%eps)
	plt.xlabel('episode')
	plt.ylabel('averaged reward')
	plt.legend(labels,loc=2)
	plt.savefig('101_act_lsvi_td_eps', bbox_inches='tight')
	plt.close()

	colormap = plt.cm.gist_ncar
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(disc_eps))])
	labels = []

	for eps in disc_eps:
		rewared_per_ep_ave = np.zeros(gNEpisodes)
		for trial in xrange(gNtrials): 
			Buffer = cache(None,None)
			theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
			reward_per_ep = list()
			reward_ep = list()
			for l in range(gNEpisodes):
				# print l
				s = env.reset()
				done = False
				rewared_thisep = 0
				while not done:
					a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l,epsilon = eps)
					sprime, r, done, _ = env.step(a)
					if done and r ==0:
						Buffer = cache(Buffer,(s,a,-1,sprime),finite = True,NBuffer = 300)
					else:
						Buffer = cache(Buffer,(s,a,r,sprime),finite = True,NBuffer = 300)
					s = sprime
					rewared_thisep += r
				theta = learn(Q,Buffer,theta,l, method = 'lsvi',batchtype = 'random')
				reward_ep.append(rewared_thisep)
				reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
				# print l,rewared_thisep
			print eps, trial
			rewared_per_ep_ave += np.array(reward_per_ep)
		rewared_per_ep_ave = rewared_per_ep_ave/gNtrials
		plt.plot(rewared_per_ep_ave)
		labels.append(r'lsvi_td, alpha = %.3f'%eps)
	plt.xlabel('episode')
	plt.ylabel('averaged reward')
	plt.legend(labels,loc=2)
	plt.savefig('101_act_lsvi_eps', bbox_inches='tight')
	plt.close()

	return


def live(env,Q):
	rewared_per_ep_ave = np.zeros(gNEpisodes)
	for trial in range(gNtrials): 
		Buffer = cache(None,None)
		theta = learn(Q,Buffer,None,-1) # Initialization of parameter theta
		reward_per_ep = list()
		reward_ep = list()
		for l in xrange(gNEpisodes):
			# print l
			s = env.reset()
			done = False
			rewared_thisep = 0
			while not done:
				a = act(lambda a:Q.map(s,a,theta),env.action_space.n,l)
				sprime, r, done, _ = env.step(a)
				if done and r ==0:
					Buffer = cache(Buffer,(s,a,-1,sprime),finite = True)
				else:
					Buffer = cache(Buffer,(s,a,r,sprime),finite = True)
				s = sprime
				rewared_thisep += r
			theta = learn(Q,Buffer,theta,l, method = 'lsvi_td',batchtype = 'random')
			reward_ep.append(rewared_thisep)
			reward_per_ep.append(np.mean(reward_ep[np.max([len(reward_ep)-100,0]):]))
			print l,rewared_thisep
		print trial
		rewared_per_ep_ave += np.array(reward_per_ep)
	rewared_per_ep_ave = rewared_per_ep_ave/gNtrials

	plt.plot(rewared_per_ep_ave)
	plt.show()

	return

env = gym.make('FrozenLake8x8-v0')
Q = FunctionFamily()
Q.type = 'linear'
Q.dim = (env.observation_space.n,env.action_space.n)

# live(env,Q)

# live_1_scale(env,Q)

# live_2_lsvi_H(env,Q)

# live_3_lsvi_buffersize(env,Q)

# live_4_lsvi_lambda(env,Q)

# live_5_lsvitd_batchsize(env,Q)

# live_6_lsvitd_gamma(env,Q)

# live_7_lsvitd_alpha(env,Q)

live_101_act_eps(env,Q)








# if __name__ == "__main__":
# 	# global parameters, for tunning

# 	main()

