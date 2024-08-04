#generates a table of the results
import numpy as np
import os
from glob import glob

from sklearn.metrics import roc_auc_score

from plt import *

import sys

from tabulate import tabulate


dex=0

tasks=glob("res/fair_*.npz")
tasks.sort()

ids=[task[task.rfind("_")+1:task.rfind(".")] for task in tasks]

aucs={}
fairs={}

for task,idd in zip(tasks,ids):
    f=np.load(task)

    task=task[9:]
    rep=task[task.rfind("_")+1:task.rfind(".")]
    task=task[:task.rfind("_")]
    if not task in aucs:
        aucs[task]=[]
    if not task in fairs:
        fairs[task]=[]


    p=f["anos"]
    ty=f["ty"]

    protected=f["tx"][:,dex]

    auc=roc_auc_score(ty,p)
    fairness=roc_auc_score(protected,p)

    fairness=abs(fairness-0.5)
    
    aucs[task].append(auc)
    fairs[task].append(fairness)

def to_str(lis):
    return f"${np.mean(lis):.3f} \pm {np.std(lis)/np.sqrt(len(lis)):.3f}$"


matrix=[]


for task in sorted(aucs.keys()):
    auc,fairness=aucs[task],fairs[task]
    print(task)
    print("AUC: ",np.mean(auc),np.std(auc)/np.sqrt(len(auc)))
    print("Fairness: ",np.mean(fairness),np.std(fairness)/np.sqrt(len(fairness)))
    print()
    matrix.append([task,to_str(auc),to_str(fairness)])


print(tabulate(matrix,headers=["Task","AUC","Bias"],tablefmt="latex_raw"))







