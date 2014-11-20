#!/usr/bin/env python

"""
Conditional Optimization with Complex Numbers / M-QAM Symboles Input using PyOpenCL.
By Avinash Sridhar (as4626@columbia.edu) & Sareena Abdul Razak (sta2378@columbia.edu)

Note: Open README.md to see brief details of algorithms
"""

import time
#import pyopencl as cl
import numpy as np
from numpy import linalg as LA

# Maximum range for M-QAM is M i.e. 0,1,2,.....,M-1
M = 4

# Number of received input symbols is N
N = 20

# These are the symbols transmitted on Tx1 and Tx2
# Format of symbols is :
# [x11 x12 x21 x22 x31 x32 ........; x13 x14 x23 x24 x33 x34] (Similiar to MATLAB notation for column separation)
#x = np.array([[0 + 0j,0 + 0j],[0 + 0j,0 + 0j]])
x = np.random.randint(0, M, size = (2,N*2)).astype(np.complex64)
x_mod = np.zeros_like(x)

# Silver Code Matrix (Constant throughout code)
V = (1/7**(0.5))*np.array([[1-1j, 1-2j],[1+2j, -1-1j]])
#testprint
#print "Silver Code Matrix V is ", V

# Generate array constellation for M-QAM, in our case M = 4
constellation = np.array([-1-1j, -1+1j, 1+1j, 1-1j])

# We will do modulation of the symbols, stored in x_mod
for i in range(2) :
	for j in range(N*2) :
		if x[i,j] == 0 :
			x_mod[i,j] = -1 - 1j
		elif x[i,j] == 1 :
			x_mod[i,j] = -1 + 1j
		elif x[i,j] == 2 :
			x_mod[i,j] = 1 + 1j
		elif x[i,j] == 3 :
			x_mod[i,j] = 1 - 1j
		else: pass

#testprint
#print M, " -QAM Modulated Input is ", x_mod

# Generate channel matrix h = [h1 h2]
h = np.array([[0.2254 - 0.9247*j, -0.3066 + 0.2423*j]]) #(1/(2**(0.5)))*np.random.random_sample((1,2)).astype(np.complex64) + (1/(2**(0.5)))*1j*np.random.random_sample((1,2)).astype(np.complex64)
# Generate channel matrix g = [g1 g2]
g =  np.array([[2.5303 + 1.9583*j, -0.9545 + 2.1460*j]]) # (1/(2**(0.5)))*np.random.random_sample((1,2)).astype(np.complex64) + (1/(2**(0.5)))*1j*np.random.random_sample((1,2)).astype(np.complex64)
# Generate h_bar = hV
h_bar = np.dot(h,V)
# Generate g_bar = gV
g_bar = np.dot(g, V)
# Generate noise matrix n1 = [n11, n12]
n1 = 0.3802 + 1.2968*j #(1/(2**(0.5)))*np.random.random_sample((1,2)).astype(np.complex64) + (1/(2**(0.5)))*1j*np.random.random_sample((1,2)).astype(np.complex64)
# Generate noise matrix n1 = [n11, n12]
n2 = -1.5972 + 0.6096*j #(1/(2**(0.5)))*np.random.random_sample((1,2)).astype(np.complex64) + (1/(2**(0.5)))*1j*np.random.random_sample((1,2)).astype(np.complex64)
n = np.array([[0+0j, 0+0j]])
#testprint
print "h is ", h
print "g is ", g
print "h bar is ", h_bar
print "g bar is ", g_bar
print "Noise matrix n1 is ", n1
print "Noise matrix n2 is ", n2

rx_sym = np.zeros((2,N)).astype(np.complex64)

#testprint
#print "Initialization of rx_sym = ", rx_sym

# Calculate the received symbols rx_sym on receiver 1 and 2 after input symbols passed through above channel matrices with noise
"""
for i in range(N):
	HG = np.array([[h[0,0], h[0,1]], [np.conjugate(h[0,1]), -np.conjugate(h[0,0])]])
	HG_bar = np.array([[h_bar[0,0], h_bar[0,1]], [np.conjugate(h_bar[0,1]), -np.conjugate(h_bar[0,1])]])
	x1x2 = np.array([[x_mod[0, 2*i]], [x_mod[0, (2*i)+1]]])
	x3x4 = np.array([[x_mod[1, 2*i]], [x_mod[1, (2*i)+1]]])
	r_temp = np.dot(HG,x1x2) + np.dot(HG_bar, x3x4) + n
	rx_sym[0,i] = r_temp[0,0]
	rx_sym[1,i] = r_temp[1,0]
"""

for i in range(N) :
	# Store matrix [x1 -x2*, x2 x1*] in x1_temp
	# Store matrix [x3 -x4*, x4 x3*] in x2_temp
	x1_temp = np.zeros((2,2)).astype(np.complex64)
	x2_temp = np.zeros((2,2)).astype(np.complex64)
	x1_temp[0,0] = x_mod[0,2*i]
	x1_temp[0,1] = -np.conjugate(x_mod[0,(2*i)+1])
	x1_temp[1,0] = x_mod[0,(2*i)+1]
	x1_temp[1,1] = np.conjugate(x_mod[0,2*i])
	x2_temp[0,0] = x_mod[1,2*i]
	x2_temp[0,1] = -np.conjugate(x_mod[1,(2*i)+1])
	x2_temp[1,0] = x_mod[1,(2*i)+1]
	x2_temp[1,1] = np.conjugate(x_mod[1,2*i])
	#testprint
	#print "x1_temp = ", x1_temp
	#print "x2_temp = ", x2_temp

	# r1 = [r11 r12], r2 = [r21 r22]
	r1 = np.dot(h, x1_temp) + np.dot(h_bar, x2_temp) + n
	r2 = np.dot(g, x1_temp) + np.dot(g_bar, x2_temp) + n

	#testprint
	#print "r1 is ", r1
	#print "r2 is ", r2

	# rx_sym will store r1 and r2 as [rx1_1, rx1_2, rx1_3,.....; rx2_1, rx2_2 rx2_3,......] rx1 is receiver 1
	# and rx2 is receiver 2. rx1_1 is total received 1st symbol on rx1 and rx2_1 is total received 1st symbol on rx2,
	# and so on
	# Combine [r11, r12] and [r21, r22] as [r1;r2], note that ; implies next row (Similar tor MATLAB manipulations)
	rx_sym[0,i] = r1[0,0] + r1[0,1]
	rx_sym[1,i] = r2[0,0] + r2[0,1]

#testprint
print "Received matrix of 10 symbols is ", rx_sym

# Generate P and Q matrices, P = tranpose([h;g]) and Q = transpose([h_bar;g_bar])
P = np.array([[h[0,0], h[0,1]], [g[0,0], g[0,1]]])
#Q = np.array([[h_bar[0,0], h_bar[0,1]], [g_bar[0,0], g_bar[0,1]]])

#P = np.array([[h[0,0], h[0,1]], [h_conj[0,1], -h_conj[0,0]]])
Q = np.array([[h_bar[0,0], h_bar[0,1]], [g_bar[0,0], g_bar[0,1]]])
#test print

print "P matrix is ", P
print "Q matrix is ", Q


#Calculate (|H|f)^2 + (|G|f)^2 where |H|f and |G|f are frobenius norms of h and g matrices
HfGf_sqr = (LA.norm(h, 'fro'))**2 + (LA.norm(g, 'fro'))**2
print LA.norm(h, 'fro')**2
print LA.norm(g, 'fro')**2
#testprint
print "(|H|f)^2 + (|G|f)^2 =  ", HfGf_sqr
print "hbar square + gbar square = ", (LA.norm(h_bar, 'fro'))**2 + (LA.norm(g_bar, 'fro'))**2

# Calculate adjoint(P). I could not find a function for this in numpy and hence manually performing it
P_adj = np.array([[g[0,1], -h[0,1]],[-g[0,0], h[0,0]]])
#P_adj = np.array([[-np.conjugate(h[0,0]), -np.conjugate(h[0,1])],[-h[0,1], h[0,0]]])
Q_adj = np.array([[g_bar[0,1], -h_bar[0,1]], [-g_bar[0,0], h_bar[0,0]]])
#testprint
print "adjoint(P) = ", P_adj
print "adjoint(Q) = ", Q_adj

print np.dot(P_adj, P) # as verification for P_adj.P = 1/2(HfGf_sqr).I
print np.dot(Q_adj, Q)
# Initialize decoded matrix with all 0s (2x2 matrix)
result_decode = np.zeros((2,N*2)).astype(np.complex64)

# Perform Conditional Optimization here algorithm here and decode the received symbols correctly
for i in range(N) :
	r = np.array([[rx_sym[0,i]],[rx_sym[1,i]]])
	mean_sqr = 0
	temp= 100000 + 100000j
	temp_index = 0
	temp_s_bar = np.zeros((2,1)).astype(np.complex64)
	temp_c_bar = np.zeros((2,1)).astype(np.complex64)
	# Iterate through every combination of x3' and x4' from constellation of M symbols
	for j in range(M) :
		for k in range(M):
			# s_bar is tranpose([j k])
			s_bar = np.array([[constellation[j]],[constellation[k]]], np.complex64)
			Cs = np.dot(((2*P_adj)/(HfGf_sqr)),(r - np.dot(Q,s_bar)))
			# Take ceiling of x3'
			print Cs,' ',((Cs[0,0].real)/(np.absolute(Cs[0,0].real)))*np.ceil(np.absolute(Cs[0,0].real)),' ', ((Cs[0,0].imag)/(np.absolute(Cs[0,0].imag)))*np.ceil(np.absolute(Cs[0,0].imag))
			x3_real = ((Cs[0,0].real)/(np.absolute(Cs[0,0].real)))*np.ceil(np.absolute(Cs[0,0].real))
			x3_complex = ((Cs[0,0].imag)/(np.absolute(Cs[0,0].imag)))*np.ceil(np.absolute(Cs[0,0].imag))
			# Take floor of x4'
			x4_real = ((Cs[1,0].real)/(np.absolute(Cs[1,0].real)))*np.floor(np.absolute(Cs[1,0].real))
			x4_complex = ((Cs[1,0].imag)/(np.absolute(Cs[1,0].imag)))*np.floor(np.absolute(Cs[1,0].imag))
			Cs[0,0] = x3_real + x3_complex*j
			Cs[1,0] = x4_real + x4_complex*j
			print Cs
			mean_sqr = (LA.norm(r - np.dot(P,Cs) - np.dot(Q,s_bar), 'fro'))**2
			#print mean_sqr
			if mean_sqr < temp :
				temp = mean_sqr
				temp_s_bar = s_bar
				temp_c_bar = Cs
	#print "Least Sqr Metric = ", temp
	result_decode[0,2*i] = temp_c_bar[0,0]
	result_decode[0,(2*i)+1] = temp_c_bar[1,0]
	result_decode[1,2*i] = temp_s_bar[0,0]
	result_decode[1, (2*i)+1] = temp_s_bar[1,0]

#print Cs_array
print 'Input = ', x_mod
print 'Decoded Result = ', result_decode

print -1.1090213/np.absolute(-1.1090213)




