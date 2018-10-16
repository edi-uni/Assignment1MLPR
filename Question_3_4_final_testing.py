import numpy as np
import matplotlib.pyplot as plt
import scipy  
import scipy.io 
import math

#######################Question-1##############################
amp_data=scipy.io.loadmat('amp_data.mat')
amp_data_keys=amp_data['amp_data']
#line_graph=plt.plot(amp_data_keys,'b-')
#plt.show()
#histogram=plt.hist(amp_data_keys,bins=20)
#plt.show()
################################################################
#################Cleansing data#################################
c=math.floor(len(amp_data_keys)/21)
array_gen=np.reshape(amp_data_keys[:21*c],(c,21))
#np.random.shuffle(array_gen)
#print(array_gen[0])

def data_split(array_gen):
	i=math.floor(len(array_gen)*0.7)
	j=math.floor(len(array_gen)*0.15)
	X_shuf_train=array_gen[:i,:-1]
	Y_shuf_train=array_gen[:i,-1]
	X_shuf_valid=array_gen[i:i+j,:-1]
	Y_shuf_valid=array_gen[i:i+j,-1]
	X_shuf_test=array_gen[i+j:i+2*j,:-1]
	Y_shuf_test=array_gen[i+j:i+2*j,-1]
	return X_shuf_train,Y_shuf_train,X_shuf_valid,Y_shuf_valid,X_shuf_test,Y_shuf_test

#data_split(X_shuf_train)
###pending- randomising data#######################################
##########################Question 3b (i)##################################
'''
This code creates a design matrix phi(c,k) which constructs the matrix in the form of 
phi = [1 t t**2...]
''' 
def Phi(c,k):
	x=np.arange(0,1,0.05)
	phi_values=x[-c:]
	phi_matrix=phi_values.reshape(1,c)
	for i in range(k):
		if i>1:
			phi_values=(phi_values**i).reshape(1,c)
			phi_matrix=np.concatenate([phi_matrix,phi_values],axis=0)
	phi_matrix=np.matrix.transpose(phi_matrix)		
	
	phi=np.concatenate([np.ones((phi_matrix.shape[0],1)),phi_matrix],axis=1)
	return(phi)
########################## Question 3b (ii)#####################################

def make_vv(c, k):
	for i in range(1,c+1,1):
		for j in range(1,k+1,1):
			x=Phi(c=i,k=j)
			x_transpose=np.matrix.transpose(x)
			v=np.linalg.pinv(x_transpose.dot(x)).dot(x_transpose) #least square solution for v matrix
			
	return v

######################## Question 3b (iii)#####################################

def vv_least_squares(c,k):
	v=make_vv(c, k)
	X_shuf_train,Y_shuf_train,X_shuf_valid,Y_shuf_valid,X_shuf_test,Y_shuf_test=data_split(array_gen)
	y_initial=X_shuf_train[0]
	quartic_vv=np.dot(v,y_initial[-c:])
	quartic_lstsq=np.linalg.lstsq(Phi(c=c,k=k),y_initial[-c:],rcond=None)[0]
	print("Weight matrix with least squares:",quartic_lstsq)
	print("Weight matrix with with vv:",quartic_vv)


#vv_least_squares(c=1,k=4)
'''
Result for linear: 
Weight matrix with least squares: [-3.20815539e-05 -3.04774762e-05]
Weight matrix with with vv: [-3.20815539e-05 -3.04774762e-05]

Result for quartic: 
Weight matrix with least squares: [-1.87375781e-05 -1.78006992e-05 -1.69106643e-05 -1.37738417e-05]
Weight matrix with with vv: [-1.87375781e-05 -1.78006992e-05 -1.69106643e-05 -1.37738417e-05]

'''

############################### Question 3c (i) ####################################
def min_leastsquare_inv(c,k):
	X_shuf_train,Y_shuf_train,X_shuf_valid,Y_shuf_valid,X_shuf_test,Y_shuf_test=data_split(array_gen)
	y_expected=Y_shuf_train[1]
	y_predicted=0
	min_squareerror=1000 #A random value for error comparison to get the least value. Can be any value. Should be large
	y_initial=X_shuf_train[1]
	c3=0
	k3=0
	for i in range(1,c+1,1):
		for j in range(1,k+1,1):
			v=make_vv(c=i, k=j)
			w=np.dot(v,y_initial[-i:])
			y_predicted=np.dot([np.ones((w.shape[0]))],w)
			min_error=y_expected-y_predicted
			if min_error<min_squareerror:
				min_squareerror=min_error
				c3=i
				k3=j
	print("Minimum square error with v:",min_squareerror,"Value for k:",k3,"Value for c:",c3)
#min_leastsquare_inv(c=1,k=10)
'''
Result: Minimum square error with v: [-0.0001014] Value for k: 6 Value for c: 1	
'''
################################ Question 3c (ii)####################################
def min_leastsquare_data(c,k):
	X_shuf_train,Y_shuf_train,X_shuf_valid,Y_shuf_valid,X_shuf_test,Y_shuf_test=data_split(array_gen)
	y_predicted=0
	mean_square_error=0
	min_error=0
	for i in range(len(Y_shuf_test)):
		y_expected=Y_shuf_test[i]
		y_initial=X_shuf_test[i]
		for j in range(1,c+1,1):
			y=y_initial[-j:]
			for r in range(1,k+1,1):
				v=make_vv(c=j, k=r)
				w=np.dot(v,y)
				y_predicted=np.dot([np.ones((w.shape[0]))],w)
				min_error+=y_expected-y_predicted
	mean_square_error=min_error/len(Y_shuf_test)

	print("Mean square error with inverse operation:",mean_square_error)
#min_leastsquare_data(c=1,k=4)
'''
Training data for C*k=1*4:
Result : Mean square error with inverse operation: 4.952014950010723e-10 

Validation data for C*k=1*4:
Mean square error with inverse operation: -6.6993145829630385e-09

Test data for C*k=1*4:
Mean square error with inverse operation: 6.721897075015231e-10
'''

####################### Question 4 (a) ##################################################

#######################Question-4:Design minimum least square############################
def min_leastsquare(c,k):
	X_shuf_train,Y_shuf_train,X_shuf_valid,Y_shuf_valid,X_shuf_test,Y_shuf_test=data_split(array_gen)
	y_predicted=0
	min_squareerror=1000
	c2=0
	k2=0
	for i in range(len(Y_shuf_valid)):
		y_initial=X_shuf_valid[i]
		y_expected=Y_shuf_valid[i]
		for j in range(1,c+1,1):
			y=y_initial[-j:]
			for r in range(1,k+1,1):
				x=Phi(c=j,k=r)
				w=np.linalg.lstsq(x,y,rcond=None)[0]
				y_predicted=np.matrix.transpose(np.dot([np.ones((w.shape[0]))],w))
				min_error=y_expected-y_predicted
				if min_error<min_squareerror:
					min_squareerror=min_error
					c2=j
					k2=r
	print("y_expected:",y_expected,"y-predicted:",y_predicted,"Value of C:",c2,", Value of K",k2)
	print("Least square error:",min_squareerror,"at Value of C:",c2,", Value of K",k2)
	return (c2,k2)
	#print("Mean square error with least squares:",mean_square_error)

#min_leastsquare(c=10,k=20)

'''
Result:
Result on validation data:
y_expected: -0.01226806640625 y-predicted: [-2.95059646] Value of C: 5 , Value of K 7
Least square error: [-19.53224823] at Value of C: 5 , Value of K 7
'''

##################################Question 4b  ####################################

def test_data_meanerror(c,k):
	X_shuf_train,Y_shuf_train,X_shuf_valid,Y_shuf_valid,X_shuf_test,Y_shuf_test=data_split(array_gen)
	min_error=0
	mean_square_error=0
	for i in range(len(Y_shuf_test)):
		y_initial=X_shuf_test[i]
		y_expected=Y_shuf_test[i]
		x=Phi(c=c,k=k)
		y=y_initial[-c:]
		w=np.linalg.lstsq(x,y,rcond=None)[0]
		y_predicted=np.matrix.transpose(np.dot([np.ones((w.shape[0]))],w))
		min_error+=(y_expected-y_predicted)
	mean_square_error=min_error/len(Y_shuf_test)
	print("Mean square error for the best validation fit on test data with least squares:",mean_square_error)
	return y_predicted
#test_data_meanerror()

'''

Mean square error for the best validation fit on test data with least squares: [-0.00068315]
'''
		
################################# Question 4c Plot Histogram #########################################

def plot_histogram():
	X_shuf_train,Y_shuf_train,X_shuf_valid,Y_shuf_valid,X_shuf_test,Y_shuf_test=data_split(array_gen)
	c,k=min_leastsquare(c=5,k=10)
	y_predicted=test_data_meanerror(c=c,k=k)
	plt.hist(y_predicted,bins=10)
	plt.show()
plot_histogram()	