import numpy as np
import os
from glob import glob

from sklearn.metrics import roc_auc_score

from scipy.optimize import minimize

from plt import *

import sys

from uqdm import uqdm

dex=0

tasks=glob("res/fair_baseline_*.npz")
tasks.sort()

ids=[task[task.rfind("_")+1:task.rfind(".")] for task in tasks]


def merge(preds, weigths):
    anos=np.dot(weigths,(preds**9))
    return anos

def individual_fairness(preds, protected):
    fairness=[]
    for pred in preds:
        fairness.append(np.abs(roc_auc_score(protected,pred)-0.5))
    return np.array(fairness)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def trafo_weights(weights):
    return sigmoid(weights)+0.5

def gen_fair_by_weights(preds,protected):
    def fair_by_weights(weights):
        merged=merge(preds,weights)
        return np.abs(roc_auc_score(protected,merged)-0.5)
    return fair_by_weights

def simple_optimizer(func,w0,steps=3000):
    minima=func(w0)
    bestw=w0

    for i,loggr in uqdm(range(steps),total=steps):
        dex=np.random.randint(0,len(w0))
        w0=np.copy(bestw)
        w0[dex]=np.random.normal(w0[dex],10**(-np.random.randint(1,4)))
        res=func(w0)
        if res<minima:
            loggr(str(res))
            minima=res
            bestw=w0
            if minima<0.00000001:
                break
    return bestw


for task,idd in zip(tasks,ids):
    f=np.load(task)
    p=f["preds"]
    ty=f["ty"]

    protected=f["tx"][:,dex]

    toopt=gen_fair_by_weights(p,protected)

    w0=individual_fairness(p,protected)
    w1=w0.copy()
    w1[w0<0.2]=0.5
    w1[w0<0.1]=1.0
    w1[w0>0.2]=0.2
    weights=w1
    weights=simple_optimizer(toopt,w1)

    anos=merge(p,weights)



    np.savez_compressed(f"res/fair_weight_{idd}.npz",anos=anos,ty=ty,tx=f["tx"],weights=weights,w0=w0,x=f["x"],preds=f["preds"])



