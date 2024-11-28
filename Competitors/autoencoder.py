import numpy as np
import tensorflow as tf
import keras as keras
from keras import backend as K
from keras.optimizers import Adam
from outlierdetector import OutlierDetector



class Autoencoder(OutlierDetector):
    """Uses an Autoencoder to detect anomalies"""


    def __init__(self,data,latent_size,encoder_neurons,decoder_neurons,layers,activation,decoder_activation,encoder_activation,\
                 output_activation,loss,epochs,batch_size,learning_rate,\
                dropout_rate,encoder_dropout,decoder_dropout,regularizer,regularisation,evaluation_metric):
        """
        data: Training data
        latent_size: if no neurons provided, used to construct the autoencoder. Number of neurons in the middle layer. If float: relative to input dim
        encoder/decoder_neurons: List of neuron counts. If float: relative to input dim
        layers: If no neurons provided, used to construct the autoencoder. Number of layers in both the encoder and decoder
        activation: Activation for the neural network (except for the last layer). Can be overwritten using decoder/encoder activation
        output_activation: Activation for the Autoencoder output. sigmoid or linear suggested (depending on normalization scheme
        loss: Loss used to train the Autoencoder. Either a loss understood by keras, or a custom function
        # optimizer: Optimizer used to train the Autoencoder. Either a optimizer understood by keras or a custom class
        epochs: Maximum number of training epochs
        batch_size: Number of samples in each training batch
        learning_rate: Autoencoder learning rate
        dropout_rate: Dropout rate for the Autoencoder. Can be overwritten by encoder/decoder_dropout
        regularizer: Regularization factor. Can be overwritten by encoder/decoder_regularizer
        regularisation: Type of regularization. Either l1, l2 or l1_l2 (only l2)
        # validation_size: Fraction of the training data to use for validation
        # validation_data: Explicit validation data
        # early_stopping: Whether to use early stopping
        # patience: Patience for early stopping
        # restore_best_weights: Whether to restore the best weights after early stopping
        evaluation_metric: Power of the distance metric used to calculate the reconstruction error. Alternative use callable.
        # verbose: Verbosity level
        # random_state: Random state
        """

        self.data=data
        self.latent_size = latent_size
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.layers = layers
        self.activation = activation
        self.decoder_activation = decoder_activation
        self.encoder_activation = encoder_activation
        self.output_activation = output_activation
        self.loss = loss
        # self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.regularizer = regularizer
        # self.encoder_regularizer = encoder_regularizer
        # self.decoder_regularizer = decoder_regularizer
        self.regularisation = regularisation
        # self.validation_size = validation_size
        # self.validation_data = validation_data
        # self.early_stopping = early_stopping
        # self.patience = patience
        # self.restore_best_weights = restore_best_weights
        self.evaluation_metric = evaluation_metric
        # self.verbose = verbose

        # super().__init__(data=data,random_state=random_state,**kwargs)

    def checkup(self,data):
        assert len(data.shape) == 2, "Data must be 2D"
        assert self.latent_size > 0, "Latent size must be positive"
        assert self.layers > 0, "Layers must be positive"
        assert self.epochs >= 0, "Epochs must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.dropout_rate >= 0, "Dropout rate must be positive"
        assert self.regularizer >= 0, "Regularizer must be positive"
        # assert self.validation_size >= 0, "Validation size must be positive"
        # assert self.patience >= 0, "Patience must be positive"

    def gen_encoder_neurons(self,input_dim):
        if not self.encoder_neurons is None:return self.encoder_neurons
        hidden_dim=self.latent_size
        if type(hidden_dim) == float:
            hidden_dim = int(input_dim * hidden_dim)
        powers=np.linspace(0,1,self.layers)
        layers=[int(input_dim*(hidden_dim/input_dim)**p) for p in powers]
        return layers

    def gen_decoder_neurons(self,input_dim):
        if not self.decoder_neurons is None:return self.decoder_neurons
        hidden_dim=self.latent_size
        if type(hidden_dim) == float:
            hidden_dim = int(input_dim * hidden_dim)
        powers=np.linspace(0,1,self.layers)
        layers=[int(hidden_dim*(input_dim/hidden_dim)**p) for p in powers]
        return layers

    def get_regularizer(self,encoder=True):
        strength = self.regularizer
        # if encoder:
        #     if not self.encoder_regularizer is None:
        #         strength = self.encoder_regularizer
        # else:
        #     if not self.decoder_regularizer is None:
        #         strength = self.decoder_regularizer
        match self.regularisation:
            case "l1":
                return keras.regularizers.l1(strength)
            case "l2":
                return keras.regularizers.l2(strength)
            case "l1_l2":
                return keras.regularizers.l1_l2(*strength)
            case _:
                raise ValueError("Regularisation must be l1, l2 or l1_l2")

    def one_encoder_layer(self,neurons):
        activation = self.activation
        if not self.encoder_activation is None:
            activation = self.encoder_activation
        regularizer = self.get_regularizer(encoder=True)
        return keras.layers.Dense(neurons,activation=activation,kernel_regularizer=regularizer)

    def one_decoder_layer(self,neurons):
        activation = self.activation
        if not self.decoder_activation is None:
            activation = self.decoder_activation
        regularizer = self.get_regularizer(encoder=False)
        return keras.layers.Dense(neurons,activation=activation,kernel_regularizer=regularizer)

    def final_layer(self,neurons):
        activation = self.output_activation
        if activation is None:
            activation = self.activation
        regularizer = self.get_regularizer(encoder=False)
        return keras.layers.Dense(neurons,activation=activation,kernel_regularizer=regularizer)


    def fit(self,*args,**kwargs):
        return self.train(*args,**kwargs)

    def train(self,data):
        # self.set_seed()
        self.checkup(data)

        samples,input_dim = data.shape

        inp=keras.layers.Input(shape=(input_dim,))
        q=inp

        encoder_dropout = self.dropout_rate
        if not self.encoder_dropout is None:
            encoder_dropout = self.encoder_dropout

        #print("!!!",encoder_dropout)

        for neurons in self.gen_encoder_neurons(input_dim):
            q = self.one_encoder_layer(neurons)(q)
            if encoder_dropout > 0:
                q = keras.layers.Dropout(rate=encoder_dropout)(q)
        self.encoder=keras.Model(inp,q)
        mid=q

        decoder_dropout = self.dropout_rate
        if not self.decoder_dropout is None:
            decoder_dropout = self.decoder_dropout

        #print("!!!",decoder_dropout)

        for neurons in self.gen_decoder_neurons(input_dim)[:-1]:
            q = self.one_decoder_layer(neurons)(q)
            if decoder_dropout > 0:
                q = keras.layers.Dropout(rate=decoder_dropout)(q)
        q = self.final_layer(input_dim)(q)
        self.decoder=keras.Model(mid,q)

        self.model=keras.Model(inp,q)
        self.model.compile(loss=self.loss,optimizer=Adam(learning_rate=self.learning_rate, beta_1=0.5))
        # K.set_value(self.model.optimizer.lr, self.learning_rate)

        # if self.verbose:
        #     self.model.summary()

        # vali={}
        # if not self.validation_data is None:
        #     vali["validation_data"] = self.validation_data
        # elif self.validation_size > 0:
        #     vali["validation_split"] = self.validation_size


        # callbacks=[]
        # if self.early_stopping:
        #     callbacks.append(keras.callbacks.EarlyStopping(patience=self.patience,restore_best_weights=self.restore_best_weights,monitor="val_loss"))

        self.model.fit(data,data,epochs=self.epochs,batch_size=self.batch_size)
                    #    ,verbose=self.verbose,callbacks=callbacks,**vali)

        self.trained=True

    def anomaly_score(self, data):
        self.checkup(data)
        pred=self.model.predict(data,batch_size=self.batch_size)
        if callable(self.evaluation_metric):
            return self.evaluation_metric(data,pred)
        delta=np.abs(data-pred)
        delta=delta**self.evaluation_metric
        return np.mean(delta,axis=1)**(1/self.evaluation_metric)

    def decision_function(self,*args,**kwargs):
        return self.anomaly_score(*args,**kwargs)






