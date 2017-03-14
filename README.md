# MSE338Project

frozen4x4: lsvi_td,SGD, alpha = 0.2, gamma = 0.95, NBuffer = 2000, Nbatch = 200, eps = 10000/(l+1)**2
	lsvi_td with minibatch, Nbatch = 3, less stable









Insights:

SGD, eps = 10000/(l+1)**2, alpha = 0.2, gamma = 0.95
- the size of the batch do plays a very important roll.
	
	
	eg. Nbatch = 30, Nbuffer = 30   reward = 0.7 when l = 1000
	    Nbatch = 30,Nbuffer = 500  reward = 0.6 when l = 1000
	    Nbatch = 500,Nbuffer = 500   reward = 0.6 when l = 1000
	    


- how to choose action and how to achieve 
	

	eps = 10000/(l+1)**2 
	eps = 0.01

	does help for td


doing




About lsvi:
- rescaling the reward
	
	make 
	
- tunning H for lsvi

- Buffer size fot lsvi

- penalty lam




- prsl v.s. lsvi v.s. lsvi_td (randomized)










