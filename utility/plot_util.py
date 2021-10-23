# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: Oct. 23 2021

import os, time, sys, random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import conf, evalmetric

font_size = 8
n_col = 2

def plot_roc_curve(y_tr = None, y_hat_tr = None, y_te = None, y_hat_te = None, title = 'roc curver'):
    ''' plot ROC cuver based on y, y_hat
    '''
    fig = plt.figure()

    # plot training roc
    if y_tr is not None and y_hat_tr is not None:
        tr_fpr, tr_tpr, tr_thresholds = evalmetric.roc_curve(y_tr, y_hat_tr)
        auc_tr = evalmetric.roc_auc_score(y_tr, y_hat_tr)
        tr_lbl = 'training (auc=' + str(int(auc_tr*1000)/1000.0) + ')'
        plt.plot(tr_fpr, tr_tpr, '-', color = 'r', label = tr_lbl)

    # plot test roc
    if y_te is not None and y_hat_te is not None:
        te_fpr, te_tpr, te_thresholds = evalmetric.roc_curve(y_te, y_hat_te)
        auc = evalmetric.roc_auc_score(y_te, y_hat_te)
        te_lbl = 'test (auc=' + str(int(auc*1000)/1000.0) + ')'
        plt.plot(te_fpr, te_tpr, '-', color = 'g', label = te_lbl)

    # plot random prediction
    x = np.arange(0, 1.2, 0.2)
    plt.plot(x, x, '-.', color = 'black', label = 'random prediction (auc=0.5)')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.legend(loc = 'best', frameon = False, ncol = 1)
    plt.title(title)
    plt.savefig('figs' + os.sep + title + '.png', dpi = 300)
