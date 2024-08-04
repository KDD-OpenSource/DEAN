import numpy as np
import os
from glob import glob

from sklearn.metrics import roc_auc_score

from plt import *

import sys

dex=0

tasks=glob("res/fair_baseline_*.npz")
tasks.sort()

ids=[task[task.rfind("_")+1:task.rfind(".")] for task in tasks]

fractions=[0.1,0.05,0.01]


def merge(preds, weigths):
    anos=np.dot(weigths,(preds**9))
    return anos

def individual_fairness(preds, protected):
    fairness=[]
    for pred in preds:
        fairness.append(np.abs(roc_auc_score(protected,pred)-0.5))
    return np.array(fairness)

for fraction in fractions:
    for task,idd in zip(tasks,ids):
        f=np.load(task)
        p=f["preds"]
        ty=f["ty"]
    
        protected=f["tx"][:,dex]
    
        fairs=individual_fairness(p,protected)
        fairs=fairs.argsort()
        fairs=fairs[::-1]
    
    
        weights=np.ones(len(fairs))
        weights[fairs[:int(fraction*len(fairs))]]=0
    
        anos=merge(p,weights)
    
    
        np.savez_compressed(f"res/fair_prune_{fraction}_{idd}.npz",anos=anos,ty=ty,tx=f["tx"],weights=weights,x=f["x"],preds=f["preds"],fraction=fraction)
    
    

