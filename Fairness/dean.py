import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from multiprocessing import Pool
paralize=1

class DEAN():
    def __init__(self,learning_rate=0.0001,batch_size=512,epochs=50,modelcount=100,normalise=1,bag=5,neurons=[256,256,256],dropout_rate=0,activation="relu",patience=10,restore_best_weights=False,power=1,bias=True,output_bias=False,output_activation="selu",qstrat=True,ensemble_power=9):
        self.lr=learning_rate
        self.batch_size=batch_size
        self.normalise=normalise
        self.epochs=epochs
        self.modelcount=modelcount
        self.bag=bag
        self.neurons=neurons
        self.dropout=dropout_rate
        self.activation=activation
        self.patience=patience
        self.restore_best_weights=restore_best_weights
        self.power=power
        self.bias=bias
        self.output_bias=output_bias
        self.output_activation=output_activation
        self.qstrat=qstrat
        self.ensemble_power=ensemble_power

    def normalize(self, x):
        if not self.normalise:return x
        if self.normalize==1:return (x-self.mn)/(self.mx-self.mn)
        if self.normalize==2:return (x-self.avg)/(0.000000001+self.std)
        return x
    
    def choose_features(self,count):
        if count<=self.bag:
            return np.arange(count)
        return np.random.choice(count,self.bag,replace=False)

    def select_features(self,x,which):
        return np.stack([x[:,w] for w in which],axis=1)

    def _train_one(self,x,dex):

        feats=self.choose_features(x.shape[1])
        x=self.select_features(x,feats)

        inp=k=keras.Input(shape=(x.shape[1],))
        q=inp
        for neuron in self.neurons:
            q=keras.layers.Dense(neuron,activation=self.activation,use_bias=self.bias)(q)
            if self.dropout>0:
                q=keras.layers.Dropout(self.dropout)(q)
        q=keras.layers.Dense(1,activation=self.output_activation,use_bias=self.output_bias)(q)
        model=keras.Model(inputs=inp,outputs=q)


        #loss=K.mean(K.abs(q-1)**self.power,axis=-1)
        #loss=K.mean(K.square(q-1),axis=-1)
        #model.add_loss(loss)
        def loss(y_true,y_pred):
            return K.mean(K.abs(y_true-y_pred)**self.power,axis=-1)
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss=loss)

        model.fit(x,np.ones(len(x)),epochs=self.epochs,batch_size=self.batch_size,verbose=1,validation_split=0.1,callbacks=[keras.callbacks.EarlyStopping(patience=self.patience,restore_best_weights=self.restore_best_weights)])

        q=np.mean(model.predict(x))

        return model,feats,q

    def _train_arg(self,args):
        return self._train_one(*args)



    def fit(self,x):

        self.mn,self.mx=np.min(x,axis=0),np.max(x,axis=0)
        self.avg,self.std=np.mean(x,axis=0),np.std(x,axis=0)
        x=self.normalize(x)
        self.feats=[]
        self.models=[]
        self.qs=[]
        if paralize>1:
            with Pool(paralize) as pool:
                lis=pool.map(self._train_arg,[(x,i) for i in range(self.modelcount)])
        else:
            lis=map(self._train_arg,[(x,i) for i in range(self.modelcount)])
        for model,feat,q in lis:
            self.feats.append(feat)
            self.models.append(model)
            self.qs.append(q)
        #for i in tqdm(range(self.modelcount),total=self.modelcount):
        #    model,feat,q=self._train_one(x,i)
        #    self.feats.append(feat)
        #    self.models.append(model)
        #    self.qs.append(q)
        self.qs=np.array(self.qs)
        return

    def decision_function(self,tx,getpreds=False):
        #print(tx.shape)
        tx=self.normalize(tx)
        #print(tx.shape)
        #print([model.input_shape for model in self.models])
        #exit()
        preds=[model.predict(self.select_features(tx,feat)) for model,feat in zip(self.models,self.feats)]
        if self.qstrat:
            preds=[(np.mean(np.abs(pred-q)**self.power,axis=-1))**(1/self.power) for pred,q in zip(preds,self.qs)]
        else:
            preds=[np.mean(np.abs(pred-1)**self.power,axis=-1)**(1/self.power) for pred in preds]
        preds=np.array(preds)
        errors=np.mean(preds**self.ensemble_power,axis=0)
        if getpreds:
            return errors,preds
        return errors


if __name__=="__main__":
    f=np.load("../adbench/Acardio.npz")
    x,tx,ty=f["x"],f["tx"],f["ty"]
    model=DEAN()
    model.fit(x)
    preds=model.decision_function(tx)
    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(ty,preds))










