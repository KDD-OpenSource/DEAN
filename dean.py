import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from multiprocessing import Pool

class DEAN():
    def __init__(self,learning_rate=0.0001,batch_size=512,epochs=50,modelcount=100,normalise=1,bag=200,neurons=[256,256,256],dropout_rate=0,activation="relu",patience=10,restore_best_weights=False,power=1,bias=True,output_bias=False,output_activation="selu",qstrat=True,ensemble_power=9,paralelize=1):
        """DEAN: Deep Ensemble ANomaly detection
        class to train a DEAN ensemble and to predict new samples. Takes a few hyperparameters:
        learning_rate=0.0001: learning rate for the neural networks
        batch_size=512: batch size for the neural networks
        epochs=50: number of epochs to train the neural networks
        modelcount=100: number of submodels in the ensemble
        normalise=1: 0 for no normalisation, 1 for min-max normalisation, 2 for mean-std normalisation
        bag=200: number of features to use in each submodel. If bag is larger than the number of features, all features are used
        neurons=[256,256,256]: list of integers, number of neurons in each hidden layer
        dropout_rate=0: dropout rate in the hidden layers
        activation="relu": activation function in the hidden layers
        patience=10: patience for the early stopping callback
        restore_best_weights=False: whether to restore the best weights after training
        power=1: power of the loss function
        bias=True: whether to use bias in the hidden layers
        output_bias=False: whether to use bias in the output layer
        output_activation="selu": activation function in the output layer
        qstrat=True: whether to compare the prediction against the mean of the training data (True) or against the learning goal of 1 (False)
        ensemble_power=9: power of the ensemble combination
        paralelize=1: number of threads to use for training the ensemble
        """
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
        self.paralelize=paralelize

    def _normalize(self, x):
        """handles the normalisation of the data"""
        if not self.normalise:return x
        if self.normalise==1:return (x-self.mn)/(self.mx-self.mn)
        if self.normalise==2:return (x-self.avg)/(0.000000001+self.std)
        return x
    
    def _choose_features(self,count):
        """chooses the features to use in a submodels"""
        if count<=self.bag:
            return np.arange(count)
        return np.random.choice(count,self.bag,replace=False)

    def _select_features(self,x,which):
        """selects the features to use in a submodel"""
        return np.stack([x[:,w] for w in which],axis=1)

    def _train_one(self,x,dex):
        """trains a single submodel"""

        feats=self._choose_features(x.shape[1])
        x=self._select_features(x,feats)

        inp=k=keras.Input(shape=(x.shape[1],))
        q=inp
        for neuron in self.neurons:
            q=keras.layers.Dense(neuron,activation=self.activation,use_bias=self.bias)(q)
            if self.dropout>0:
                q=keras.layers.Dropout(self.dropout)(q)
        q=keras.layers.Dense(1,activation=self.output_activation,use_bias=self.output_bias)(q)
        model=keras.Model(inputs=inp,outputs=q)


        def loss(y_true,y_pred):
            return K.mean(K.abs(y_true-y_pred)**self.power,axis=-1)
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss=loss)

        model.fit(x,np.ones(len(x)),epochs=self.epochs,batch_size=self.batch_size,verbose=1,validation_split=0.1,callbacks=[keras.callbacks.EarlyStopping(patience=self.patience,restore_best_weights=self.restore_best_weights)])

        q=np.mean(model.predict(x))

        return model,feats,q

    def _train_arg(self,args):
        """wrapper for the training function to allow for parallelisation"""
        return self._train_one(*args)



    def fit(self,x):
        """trains the DEAN ensemble on the unlabeled data x"""

        self.mn,self.mx=np.min(x,axis=0),np.max(x,axis=0)
        self.avg,self.std=np.mean(x,axis=0),np.std(x,axis=0)
        x=self._normalize(x)
        self.feats=[]
        self.models=[]
        self.qs=[]
        if self.paralelize>1:
            with Pool(self.paralelize) as pool:
                #paralelisation is buggy on certain systems, so it is disabled by default
                lis=pool.map(self._train_arg,[(x,i) for i in range(self.modelcount)])
        else:
            lis=map(self._train_arg,[(x,i) for i in range(self.modelcount)])
        for model,feat,q in lis:
            self.feats.append(feat)
            self.models.append(model)
            self.qs.append(q)
        self.qs=np.array(self.qs)
        return

    def decision_function(self,tx,getpreds=False):
        tx=self._normalize(tx)
        preds=[model.predict(self._select_features(tx,feat)) for model,feat in zip(self.models,self.feats)]
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
    f=np.load("cardio.npz")
    x,tx,ty=f["x"],f["tx"],f["ty"]
    model=DEAN()
    model.fit(x)
    preds=model.decision_function(tx)
    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(ty,preds))










