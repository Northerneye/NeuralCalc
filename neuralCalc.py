import numpy as np
def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))
X = np.array([[0,1,0,1],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,0,1,1],[1,1,1,1],[0,0,0,1],[0,1,0,0]]) #input array. Feel free to change this.  Right now it is set so that anything with a one in the first column gets output as a 1 [1,*,*], the wanted outputs are in the next column
y = np.array([[0,1,0],[0,1,1],[1,0,0],[1,1,0],[1,0,1],[1,1,1],[0,0,1],[0,0,1]]) #output array, tells the computer what should be output as 1 or as a 0.  Corresponding to the arrays above, anything with a 1 in the first coumn, [1,*,*] gets output as a 1
np.random.seed(1) 
syn0 = 2*np.random.random((4,3)) - 1
#print(syn0)
for iter in range(1000000):
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l1_error = y - l1
	l1_delta = l1_error * nonlin(l1,True)
	syn0 += np.dot(l0.T,l1_delta)
#l0 = [[1,1]]
l1 = nonlin(np.dot(l0,syn0))
#print(l1)
print(np.around(l1))