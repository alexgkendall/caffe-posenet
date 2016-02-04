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

results = np.zeros((args.iter,2))

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

	#Compute Individual Sample Error
	q1 = pose_q / np.linalg.norm(pose_q)
	q2 = predicted_q / np.linalg.norm(predicted_q)
	d = abs(np.sum(np.multiply(q1,q2)))
	theta = np.arccos(d) * 180/math.pi
	error_x = np.linalg.norm(pose_x-predicted_x)

	results[i,:] = [error_x,theta]

	print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

median_result = np.median(results,axis=0)
print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'

np.savetxt('results.txt', results, delimiter=' ')

print 'Success!'

