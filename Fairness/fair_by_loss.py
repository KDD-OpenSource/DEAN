from deanl import DEAN
import numpy as np

from sklearn.metrics import roc_auc_score


repetitions=5



for i in range(repetitions):
    
    f=np.load("data.npz")
    
    x,tx,ty=f['x'],f['tx'],f['ty']
    
    
    model=DEAN(modelcount=1000)
    model.fit(x)
    anos,preds=model.decision_function(tx,getpreds=True)
    
    
    np.savez_compressed(f"res/fair_loss_{i}.npz",anos=anos,preds=preds,x=x,tx=tx,ty=ty)
    
