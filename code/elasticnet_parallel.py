"""

"""

from __future__ import print_function

from time import time
import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Loading and preprocessing

datafolder = os.path.join(os.path.sep, 'scratch','ansuini','repositories',
    'machine_learning','sparse_logreg','data','tidy','full')
X = loadmat(os.path.join(os.path.sep, datafolder, 'X1.mat') )
y = loadmat(os.path.join(os.path.sep, datafolder, 'y1.mat') )
X = X['X']
y = y['y']
X = StandardScaler().fit_transform(X)
y = y.ravel()

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0)

# Set the parameters by cross-validation

niterations = 5
cvfolds     = 10
niter       = 1000
nsamples    = 10
alphas      = np.linspace(1e-6, 0.1, nsamples)
l1_ratios   = np.linspace(1e-6,   1, nsamples)
tuned_parameters = [dict(alpha=alphas), dict(l1_ratio=l1_ratios) ]

score = 'neg_log_loss'

for nprocs in [1,2,4,8,10,16,20]:

    for k in range(niterations):
        tin = time()
        clf = GridSearchCV(SGDClassifier(loss='log', penalty='elasticnet', n_iter=niter), tuned_parameters, cv=cvfolds, n_jobs=nprocs, scoring=score)
        clf.fit(X_train, y_train)
        print("Num procs : %d --- elapsed time : %g" % (nprocs, time() - tin) )
