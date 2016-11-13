import multiprocessing
import cPickle as pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss


# parameters

npixels = 4015
niter = 5
numFolds = 10

# prepare data

datafolder = os.path.join(os.path.sep, 'scratch','ansuini','repositories',
       'machine_learning','sparse_logreg','data','tidy','full')
X = loadmat(os.path.join(os.path.sep, datafolder, 'X1.mat') )
y = loadmat(os.path.join(os.path.sep, datafolder, 'y1.mat') )
X = X['X']
y = y['y']
X = StandardScaler().fit_transform(X)
y = y.ravel()

# hyperpars space

nsamples=10
a = np.linspace(0.001, 0.1, nsamples)
b = np.linspace(0.01, 10, nsamples)

# cross validation    

kf = KFold(len(X), numFolds, shuffle=True)

# init matrices

loss = np.zeros((nsamples, nsamples))
W = np.zeros((npixels, nsamples, nsamples))



def worker(i):
    """thread worker function"""
   
    tin = time.time()

    for j in range(len(b)):
        A = a[i] + b[j]
        L = a[i] / (a[i] + b[j])

        params ={"loss": "log", "penalty": "elasticnet",
                'n_iter':niter, "alpha":A, "l1_ratio":L }
        
        total = 0
        for train_indices, test_indices in kf:

            train_X = X[train_indices, :]; train_y = y[train_indices]
            test_X = X[test_indices, :]; test_y = y[test_indices]
            clf = SGDClassifier(**params)
            clf.fit(train_X, train_y)
            predictions = clf.predict(test_X)
            total += log_loss(test_y, predictions)
            W[:,i,j] = clf.coef_.ravel()

        loss[i,j] = total / (numFolds)
 
    print 'Worker : ' + str(i) + ' done in time : ', + "{0:.2f}".format(round(time.time() - t1,4))
  

    return

# main

if __name__ == '__main__':

    jobs = [] 
   
    # loop

    for i in range(len(a)):
        
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()
