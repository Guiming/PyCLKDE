# Author: Guiming Zhang
# Last update: Oct. 23 2021

import os, time
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import conf

def accuracy_score(y_hat, y):
    ''' compute classification accuracy given predictions in y_hat and ground truth in y
    '''
    try:
        return sm.accuracy_score(y, y_hat)
    except Exception as e:
        raise

def classification_report(y_hat, y):
    ''' compute classification report given predictions in y_hat and ground truth in y
    '''
    try:
        return sm.classification_report(y, y_hat)
    except Exception as e:
        raise

def cohen_kappa_score(y_hat, y):
    ''' compute classification report given predictions in y_hat and ground truth in y
    '''
    try:
        return sm.cohen_kappa_score(y, y_hat)
    except Exception as e:
        raise

def roc_curve(y_true, y_score, title = 'roc_curve'):
    ''' ROC curve
        return fpr, tpr, thresholds
    '''
    return sm.roc_curve(y_true, y_score)

def roc_auc_score(y_true, y_score):
    ''' aur score
    '''
    return sm.roc_auc_score(y_true, y_score)
