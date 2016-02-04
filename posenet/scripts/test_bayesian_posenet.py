import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D

# Make sure that caffe is on the python path:
caffe_root = '.../caffe-posenet/'  # Change to your directory to caffe-posenet
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

results = np.zeros((args.iter,4))

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

for i in range(0, args.iter):

	net.forward()

	pose_q= net.blobs['label_wpqr'].data
	pose_x= net.blobs['label_xyz'].data
	predicted_q = net.blobs['cls3_fc_wpqr'].data 
	predicted_x = net.blobs['cls3_fc_xyz'].data 

	pose_q = np.squeeze(pose_q)
	pose_x = np.squeeze(pose_x)
	predicted_q = np.squeeze(predicted_q)
	predicted_x = np.squeeze(predicted_x)
	predictions = np.concatenate((predicted_x,predicted_q),axis=1)

	#Compute Individual Sample Error
	q1 = pose_q / np.linalg.norm(pose_q,axis=1)[:,None]
	q2 = predicted_q / np.linalg.norm(predicted_q,axis=1)[:,None]
	d = abs(np.sum(np.multiply(q1,q2),axis=1))
	theta_individual = np.arccos(d) * 180/math.pi
	error_x_individual = np.linalg.norm(pose_x-predicted_x,axis=1)

	#Total error
	predicted_x_mean = np.mean(predicted_x,axis=0)
	predicted_q_mean = np.mean(predicted_q,axis=0)
	pose_x = np.mean(pose_x,axis=0)
	pose_q = np.mean(pose_q,axis=0)

	q1 = pose_q / np.linalg.norm(pose_q)
	q2 = predicted_q_mean / np.linalg.norm(predicted_q_mean)
	d = abs(np.sum(np.multiply(q1,q2)))
	theta = np.arccos(d) * 180/math.pi
	error_xyz = pose_x-predicted_x_mean
	error_x = np.linalg.norm(error_xyz)

	#Variance
	covariance_x = np.cov(predicted_x,rowvar=0)
	uncertainty_x = covariance_x[0,0] + covariance_x[1,1] + covariance_x[2,2]

	covariance_q = np.cov(predicted_q,rowvar=0)
	uncertainty_q = math.sqrt(covariance_q[0,0] + covariance_q[1,1] + covariance_q[2,2] + covariance_q[3,3])

	results[i,:] = [error_x, math.sqrt(uncertainty_x), theta, math.sqrt(uncertainty_q)]

	print 'Iteration:  ', i
	print 'Error XYZ (m):  ', error_x
	print 'Uncertainty XYZ:   ', uncertainty_x
	print 'Error Q (degrees):  ', theta
	print 'Uncertainty Q:   ', uncertainty_q

median_result = np.median(results,axis=0)
print 'Median error ', median_result[0], 'm  and ', median_result[2], 'degrees.'

plt.scatter(results[:,0], results[:,1], alpha=0.5)
plt.title('Distance error [m] vs uncertainty (trace)')
plt.figure()
plt.scatter(results[:,2], results[:,3], alpha=0.5)
plt.title('Rotation error [deg] vs uncertainty (trace)')
plt.show()

# Save the rotational and translational errors and uncertainties
np.savetxt('results.txt', results, delimiter=' ')

print 'Success!'

