#!/usr/bin/env python

"""
4-QAM
Conditional Optimization with Complex Numbers / M-QAM Symboles Input using PyOpenCL.
By Avinash Sridhar (as4626@columbia.edu) & Sareena Abdul Razak (sta2378@columbia.edu)
Reference for symbol information: http://www.ece.ubc.ca/~edc/7860/lectures/lec8.pdf
NOTE: The ratio N/40 under channel gain parameter generation for h & g might need to be varied based on size of N
"""

import time
import pyopencl as cl
import numpy as np
from numpy import linalg as LA


# Selecting OpenCL platform.
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
        if platform.name == NAME:
                devs = platform.get_devices()

# Setting Command Queue.
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

# You need to set the flags of the buffers you create properly; otherwise,
# you might not be able to read or write them as needed:
mf = cl.mem_flags


# Maximum range for M-QAM is M i.e. 0,1,2,.....,M-1
M = 4
# Number of received input symbols is N
N = 120
kernel = """
#include <pyopencl-complex.h>   
__kernel void ml_decoder(__global cfloat_t* decoded, __global cfloat_t* r, __global cfloat_t* MatP, __global cfloat_t* MatQ, __global cfloat_t* MatAdjP, __global cfloat_t* QAMconstell, const int N,const int sizeQAM, const float frobNorm) {
        /* Pick any two symbols from the constellation. this is x3 and x4 */

        cfloat_t s_bar[2][1];
        cfloat_t c_bar[2][1];
        cfloat_t Pc[2][1];
        cfloat_t Qs[2][1];
        cfloat_t rQs[2][1];
        cfloat_t cbar_temp[2][1];
        cfloat_t tempMs[2][1];
        cfloat_t matq[2][2];
        cfloat_t decoded_sbar[2][1], decoded_cbar[2][1];
        float ceil1, ceil2, floor1, floor2;
        float temp = 10000.0 ;
        float Ms;
        cfloat_t Cs[2][1];
        for (int i = 0; i< 2; i++)
        {
                for (int j = 0; j < 2; j++)
                matq[i][j] = MatQ[i+j];
        }
        unsigned int indexr = get_global_id(0);
        for (int i = 0 ; i <  sizeQAM ; i++)
        {
                for(int j = 0 ; j < sizeQAM ; j++)
                {
           s_bar[0][0] = QAMconstell[i];
           s_bar[1][0] = QAMconstell[j]; 

           /* Calculating c_bar from s_bar */

           /* Multiplying Q and s_bar matrices */
           Qs[0][0] = cfloat_add(cfloat_mul(MatQ[0 + 0] , s_bar[0][0]), cfloat_mul( MatQ[0 + 1] , s_bar[1][0]));
           Qs[1][0] = cfloat_add(cfloat_mul(MatQ[0 + 2] , s_bar[0][0]), cfloat_mul( MatQ[0 + 3] , s_bar[1][0]));

           /* r - Qs */
           rQs[0][0] = cfloat_add(r[indexr + 0] ,-Qs[0][0]);
           rQs[1][0] = cfloat_add(r[indexr + N] , -Qs[1][0]);

           /* Calculate c_bar */
           cbar_temp[0][0] =  cfloat_add(cfloat_mul(MatAdjP[0 + 0] , rQs[0][0]) , cfloat_mul(MatAdjP[0 + 1] , rQs[1][0])) ;
           cbar_temp[1][0] =  cfloat_add(cfloat_mul(MatAdjP[0 + 2] , rQs[0][0]) , cfloat_mul(MatAdjP[1 + 3] , rQs[1][0])) ;
                   c_bar[0][0] =  cfloat_mul((2/frobNorm) , cbar_temp[0][0]);
                   c_bar[1][0] = cfloat_mul((2/frobNorm) , cbar_temp[1][0]);

                   

                   if((cfloat_real(c_bar[0][0]) < 0.0) && (cfloat_imag(c_bar[0][0]) < 0.0))
                   {
 
                    Cs[0][0] = -1-j;           
                   }
                   if((cfloat_real(c_bar[0][0]) < 0.0) && (cfloat_imag(c_bar[0][0]) >0.0))
                   {
 
                     Cs[0][0] = -1+j;           
                   }

                   if((cfloat_real(c_bar[0][0]) > 0.0) && (cfloat_imag(c_bar[0][0]) < 0.0))
                   {
 
                     Cs[0][0] = 1-j;           
                   }

                   if((cfloat_real(c_bar[0][0]) > 0.0) && (cfloat_imag(c_bar[0][0]) > 0.0))
                   {
 
                     Cs[0][0] = 1+j;           
                   }
                   if((cfloat_real(c_bar[1][0]) < 0.0) && (cfloat_imag(c_bar[1][0]) < 0.0))
                   {
 
                     Cs[0][0] = -1-j;           
                   }
                   if((cfloat_real(c_bar[1][0]) < 0.0) && (cfloat_imag(c_bar[1][0]) >0.0))
                   {
 
                     Cs[0][0] = -1+j;           
                   }

                   if((cfloat_real(c_bar[1][0]) > 0.0) && (cfloat_imag(c_bar[1][0]) < 0.0))
                   {
 
                     Cs[0][0] = 1-j;           
                   }

                   if((cfloat_real(c_bar[1][0]) > 0.0) && (cfloat_imag(c_bar[1][0]) > 0.0))
                   {
 
                     Cs[0][0] = 1+j;           
                   }

                   /* Multiplying P and c matrices */
                   Pc[0][0] = cfloat_add(cfloat_mul(MatP[0 + 0], Cs[0][0]), cfloat_mul(MatP[0 + 1], Cs[1][0]));
                   Pc[1][0] = cfloat_add(cfloat_mul(MatP[0 + 2], Cs[0][0]), cfloat_mul(MatP[0 + 3], Cs[1][0]));


                   /* Calculate Ms */
                   /* First, we calculate the complex numbers' abs and then proceed with over all ||Ms|| calculation */
                   tempMs[0][0] = cfloat_add(r[indexr + 0], cfloat_add(-Pc[0][0], -Qs[0][0]));
                   tempMs[1][0] = cfloat_add(r[indexr + N], cfloat_add(-Pc[1][0], -Qs[1][0]));
                   Ms = pow(sqrt(pow((cfloat_real(tempMs[0][0])),2) + pow((cfloat_imag(tempMs[0][0])),2)) + sqrt(pow((cfloat_real(tempMs[1][0])),2) + pow((cfloat_imag(tempMs[1][0])),2)),2);
                   
                   /* Check if Ms < temp, if TRUE, then store Ms in temp and store decoded_sbar and decoded_cbar with their respective values */
                   if (Ms < temp)
                   {
                        decoded_cbar[0][0] = Cs[0][0];
                        decoded_cbar[1][0] = Cs[1][0];
                        decoded_sbar[0][0] = s_bar[0][0];
                        decoded_sbar[1][0] = s_bar[1][0];
                        temp = Ms;
                   }
                }
        }
        decoded[2*indexr] = decoded_cbar[0][0];
        decoded[2*indexr+1] = decoded_cbar[1][0];
        decoded[2*indexr + 2*N] = decoded_sbar[0][0];
        decoded[2*indexr + 2*N +1] = decoded_sbar[1][0];
}
"""

# These are the symbols transmitted on Tx1 and Tx2
# Format of symbols is :
# [x11 x12 x21 x22 x31 x32 ........; x13 x14 x23 x24 x33 x34] (Similiar to MATLAB notation for column separation)
#x = np.array([[0 + 0j,0 + 0j],[0 + 0j,0 + 0j]])
x = np.random.randint(0, M, size = (2,N*2)).astype(np.complex64)
x_mod = np.zeros_like(x)
#print x
# Silver Code Matrix (Constant throughout code)
V = (1/7**(0.5))*np.array([[1-1j, 1-2j],[1+2j, -1-1j]])
#testprint
#print "Silver Code Matrix V is ", V

# Generate array constellation for M-QAM, in our case M = 4
constellation = np.array([-1-1j, -1+1j, 1+1j, 1-1j]).astype(np.complex64)

#We need mapping of every complex number to integer
mapper = {constellation[0] : 0, constellation[1] : 1, constellation[2] : 2, constellation[3] : 3}

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
#h = np.array([[0.2254 - 0.9247j, -0.3066 + 0.2423j]]) 
h =(1/(2**(0.5)))*np.random.random_integers(1, N/40, size=(1,2)).astype(np.complex64) + (1/(2**(0.5)))*1j*np.random.random_integers(1, N/40, size=(1,2)).astype(np.complex64)
#h = np.array([[1.0001+0j, 1.00002+0j]]).astype(np.complex64)
# Generate channel matrix g = [g1 g2]
#g =  np.array([[2.5303 + 1.9583j, -0.9545 + 2.1460j]]) g = 
g = (1/(2**(0.5)))*np.random.random_integers(1, N/40, size=(1,2)).astype(np.complex64) + (1/(2**(0.5)))*1j*np.random.random_integers(1, N/40, size=(1,2)).astype(np.complex64)
#g = np.array([[1.00002+0j, 1.00001+0j]]).astype(np.complex64)
# Generate h_bar = hV
h_bar = np.dot(h,V)
# Generate g_bar = gV
g_bar = np.dot(g, V)
# Generate noise matrix n1 = [n11, n12]
#n1 = 0.003802 + 0.002968j #(1/(2**(0.5)))*np.random.random_sample((1,2)).astype(np.complex64) + (1/(2**(0.5)))*1j*np.random.random_sample((1,2)).astype(np.complex64)
n1 = np.random.normal(0,1)
#n1 = 0 + 0j
# Generate noise matrix n1 = [n11, n12]
#n2 = 0.005972 + 0.006096j #(1/(2**(0.5)))*np.random.random_sample((1,2)).astype(np.complex64) + (1/(2**(0.5)))*1j*np.random.random_sample((1,2)).astype(np.complex64)
n2 = np.random.normal(0,1)
n = np.array([[n1, n2]])
#testprint
#print "h is ", h
#print "g is ", g
#print "h bar is ", h_bar
#print "g bar is ", g_bar
#print "Noise matrix n1 is ", n1
#print "Noise matrix n2 is ", n2

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
#print "Received matrix of 10 symbols is ", rx_sym

# Generate P and Q matrices, P = tranpose([h;g]) and Q = transpose([h_bar;g_bar])
P = np.array([[h[0,0], h[0,1]], [g[0,0], g[0,1]]]).astype(np.complex64)

det_P = (h[0,0]*g[0,1]) - (h[0,1]*g[0,0])
#print "det(P) = ", det_P

Q = np.array([[h_bar[0,0], h_bar[0,1]], [g_bar[0,0], g_bar[0,1]]]).astype(np.complex64)
#test print

#print "P matrix is ", P
#print "Q matrix is ", Q


#Calculate (|H|f)^2 + (|G|f)^2 where |H|f and |G|f are frobenius norms of h and g matrices
HfGf_sqr = (LA.norm(h, 'fro'))**2 + (LA.norm(g, 'fro'))**2
#print LA.norm(h, 'fro')**2
#print LA.norm(g, 'fro')**2
#testprint
#print "(|H|f)^2 + (|G|f)^2 =  ", HfGf_sqr
#print "hbar square + gbar square = ", (LA.norm(h_bar, 'fro'))**2 + (LA.norm(g_bar, 'fro'))**2

# Calculate adjoint(P). I could not find a function for this in numpy and hence manually performing it
P_adj = np.array([[g[0,1], -h[0,1]],[-g[0,0], h[0,0]]]).astype(np.complex64)
inv_P = (1/det_P)*P_adj

Q_adj = np.array([[g_bar[0,1], -h_bar[0,1]], [-g_bar[0,0], h_bar[0,0]]]).astype(np.complex64)
#testprint
#print "adjoint(P) = ", P_adj
#print "adjoint(Q) = ", Q_adj
#print "Inverse(P) = ", inv_P
#print "Verification of inv(P).P = I ", np.dot(inv_P, P)
#print np.dot(P_adj, P) # as verification for P_adj.P = 1/2(HfGf_sqr).I
#print np.dot(Q_adj, Q)
# Initialize decoded matrix with all 0s (2x2 matrix)
result_decode = np.zeros((2,N*2)).astype(np.complex64)
opencl_recvsymbols = np.zeros_like(result_decode)

# Perform Conditional Optimization here algorithm here and decode the received symbols correctly
for i in range(N) :
	r = np.array([[rx_sym[0,i]],[rx_sym[1,i]]])
	mean_sqr = 0.0
	temp= 10000
	temp_index = 0
	temp_s_bar = np.zeros((2,1)).astype(np.complex64)
	temp_c_bar = np.zeros((2,1)).astype(np.complex64)
	# Iterate through every combination of x3' and x4' from constellation of M symbols

	for j in range(M) :
		for k in range(M):
			# s_bar is tranpose([j k])
			Cs = np.zeros((2,1)).astype(np.complex64)
			s_bar = np.array([[constellation[j]],[constellation[k]]], np.complex64)
			#Cs = np.dot(((2*P_adj)/(HfGf_sqr)),(r - np.dot(Q,s_bar)))
			Cs = np.dot(inv_P,(r - np.dot(Q,s_bar)))
			x1_real = 0
			x1_complex = 0
			x2_real = 0
			x2_complex = 0
			if Cs[0.0].real < 0 and Cs[0,0].imag < 0 :
				x1_real = -1
				x1_complex = -1
			elif Cs[0.0].real < 0 and Cs[0,0].imag > 0 :
				x1_real = -1
				x1_complex = 1
			elif Cs[0.0].real > 0 and Cs[0,0].imag < 0 :
				x1_real = 1
				x1_complex = -1
			elif Cs[0.0].real > 0 and Cs[0,0].imag > 0 :
				x1_real = 1
				x1_complex = 1
			if Cs[1.0].real < 0 and Cs[1,0].imag < 0 :
				x2_real = -1
				x2_complex = -1
			elif Cs[1.0].real < 0 and Cs[1,0].imag > 0 :
				x2_real = -1
				x2_complex = 1
			elif Cs[1.0].real > 0 and Cs[1,0].imag < 0 :
				x2_real = 1
				x2_complex = -1
			elif Cs[1.0].real > 0 and Cs[1,0].imag > 0 :
				x2_real = 1
				x2_complex = 1

			Cs[0,0] = x1_real + 1j*x1_complex
			Cs[1,0] = x2_real + 1j*x2_complex
			mean_sqr = (LA.norm(r - np.dot(P,Cs) - np.dot(Q,s_bar), 2))**2
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
#print x
#print 'Input = ', x_mod
#print 'Decoded Result = ', result_decode

symbol_err = 0
for i in range(2) :
	for j in range(N*2) :
		if x_mod[i][j] != result_decode[i][j] :
			symbol_err+=1

#print "Number of incorrect symbols received = ", symbol_err

bit_err = 0.0
for i in range(2) :
	for j in range(N*2) :
		print x_mod[i][j]
		if x_mod[i][j] in mapper :
			xor_val = mapper.get(x_mod[i][j])^mapper.get(result_decode[i][j])
			bit_err += bin(xor_val).count("1")
#print "Number of incorrect bits received = ", bit_err
#print "Total number of bits transmitted = ", 2 * 4 * N
#print "Bit error rate = ", bit_err/(2*4*N)

opencl_recvsymbols = np.zeros_like(result_decode)

# Create buffers
r_buf =cl.Buffer(ctx, mf.WRITE_ONLY, opencl_recvsymbols.nbytes)
input_received_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rx_sym)
P_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
Q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
P_adj_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P_adj)
constellation_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=constellation)

# Launch Kernel
# Build kernel for decoding
prg = cl.Program(ctx, kernel).build()

prg.ml_decoder(queue, (N,) , None, r_buf, input_received_buf, P_buf, Q_buf, P_adj_buf, constellation_buf, np.int32(N), np.int32(M), np.float32(HfGf_sqr))

# Copy output back from buffer
cl.enqueue_copy(queue, opencl_recvsymbols, r_buf)

#print opencl_recvsymbols
print 'equal:    ', np.allclose(opencl_recvsymbols, result_decode)
