"""
DEAN: Deep Ensemble Anomaly detection

This module implements the DEAN anomaly detection method. It supports configuration
via an external YAML file, which specifies available options for the model, and
the dataset to use.
"""


import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from multiprocessing import Pool


class DEAN:
    """
    DEAN: Deep Ensemble Anomaly detection

    This class trains a DEAN ensemble and predicts anomaly scores for new samples.
    """

    def __init__(
        self,
        learning_rate: float = 0.0001,
        batch_size: int = 512,
        epochs: int = 50,
        model_count: int = 100,
        normalize: int = 1,
        bag: int = 200,
        neurons: list = None,
        dropout_rate: float = 0,
        activation: str = "relu",
        patience: int = 10,
        restore_best_weights: bool = False,
        power: int = 1,
        bias: bool = True,
        output_bias: bool = False,
        output_activation: str = "selu",
        q_strat: bool = True,
        ensemble_power: int = 9,
        parallelize: int = 1
    ):
        """
        Initialize DEAN with specified hyperparameters.

        Parameters:
            learning_rate (float): Learning rate for the neural networks.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            model_count (int): Number of submodels in the ensemble.
            normalize (int): Normalization method (0: none, 1: min-max, 2: mean-std).
            bag (int): Number of features to use in each submodel.
            neurons (list): List of integers defining neurons in each hidden layer.
                            Defaults to [256, 256, 256] if not provided.
            dropout_rate (float): Dropout rate for hidden layers.
            activation (str): Activation function for hidden layers.
            patience (int): Early stopping patience.
            restore_best_weights (bool): Whether to restore best weights after training.
            power (int): Power parameter for the loss calculation.
            bias (bool): Whether to use bias in hidden layers.
            output_bias (bool): Whether to use bias in the output layer.
            output_activation (str): Activation function for the output layer.
            q_strat (bool): Strategy for computing prediction scores.
            ensemble_power (int): Power for ensemble combination.
            parallelize (int): Number of threads for parallel training.
        """
        self.lr = learning_rate
        self.batch_size = batch_size
        self.normalize = normalize
        self.epochs = epochs
        self.model_count = model_count
        self.bag = bag
        self.neurons = neurons if neurons is not None else [256, 256, 256]
        self.dropout = dropout_rate
        self.activation = activation
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.power = power
        self.bias = bias
        self.output_bias = output_bias
        self.output_activation = output_activation
        self.q_strat = q_strat
        self.ensemble_power = ensemble_power
        self.parallelize = parallelize

    def _normalize_data(self, x: np.ndarray, epsilon: float = 0.000000001) -> np.ndarray:
        """
        Normalize the data based on the chosen normalization method.

        Parameters:
            x (np.ndarray): Input data.
            epsilon (float): Small value to avoid division by zero.
        Returns:
            np.ndarray: Normalized data.
        """
        if not self.normalize:
            return x
        if self.normalize == 1:
            return (x - self.mn) / (epsilon + (self.mx - self.mn))
        if self.normalize == 2:
            return (x - self.avg) / (epsilon + self.std)
        return x

    def _choose_features(self, feature_count: int) -> np.ndarray:
        """
        Randomly select features for a submodel.

        Parameters:
            feature_count (int): Total number of features.

        Returns:
            np.ndarray: Indices of selected features.
        """
        if feature_count <= self.bag:
            return np.arange(feature_count)
        return np.random.choice(feature_count, self.bag, replace=False)

    def _select_features(self, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Select features from the input data for a submodel.

        Parameters:
            x (np.ndarray): Input data.
            indices (np.ndarray): Indices of features to select.

        Returns:
            np.ndarray: Data with selected features.
        """
        return np.stack([x[:, w] for w in indices], axis=1)

    def _train_one(self, x: np.ndarray, idx: int):
        """
        Train a single submodel on a subset of features.

        Parameters:
            x (np.ndarray): Training data.
            idx (int): Index of the submodel (unused).

        Returns:
            tuple: (trained model, selected feature indices, mean prediction value)
        """
        feats = self._choose_features(x.shape[1])
        x_sub = self._select_features(x, feats)

        inputs = keras.Input(shape=(x_sub.shape[1],))
        x_model = inputs
        for neuron in self.neurons:
            x_model = keras.layers.Dense(neuron,
                                         activation=self.activation,
                                         use_bias=self.bias)(x_model)
            if self.dropout > 0:
                x_model = keras.layers.Dropout(self.dropout)(x_model)
        outputs = keras.layers.Dense(1,
                                     activation=self.output_activation,
                                     use_bias=self.output_bias)(x_model)
        model = keras.Model(inputs=inputs, outputs=outputs)

        def loss(y_true, y_pred):
            return K.mean(K.abs(y_true - y_pred) ** self.power, axis=-1)

        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss=loss)

        model.fit(x_sub,
                  np.ones(len(x_sub)),
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  verbose=1,
                  validation_split=0.1,
                  callbacks=[keras.callbacks.EarlyStopping(patience=self.patience,
                                                             restore_best_weights=self.restore_best_weights)])
        # Compute average prediction on the training data
        q_val = np.mean(model.predict(x_sub))
        return model, feats, q_val

    def _train_arg(self, args):
        """
        Wrapper to train a submodel (used for parallelization).

        Parameters:
            args (tuple): Arguments for _train_one.

        Returns:
            tuple: Result from _train_one.
        """
        return self._train_one(*args)

    def fit(self, x: np.ndarray):
        """
        Train the DEAN ensemble on the unlabeled data.

        Parameters:
            x (np.ndarray): Unlabeled training data.
        """
        self.mn, self.mx = np.min(x, axis=0), np.max(x, axis=0)
        self.avg, self.std = np.mean(x, axis=0), np.std(x, axis=0)
        x_normalized = self._normalize_data(x)
        self.feats = []
        self.models = []
        self.qs = []
        args = [(x_normalized, i) for i in range(self.model_count)]
        # Parallelization does not work on certain systems, so it is disabled by default
        if self.parallelize > 1:
            with Pool(self.parallelize) as pool:
                results = pool.map(self._train_arg, args)
        else:
            results = list(map(self._train_arg, args))
        for model, feat, q_val in results:
            self.feats.append(feat)
            self.models.append(model)
            self.qs.append(q_val)
        self.qs = np.array(self.qs)

    def decision_function(self, tx: np.ndarray, get_preds: bool = False):
        """
        Compute anomaly scores for the input data.

        Parameters:
            tx (np.ndarray): Test data.
            get_preds (bool): If True, also return individual submodel predictions.

        Returns:
            np.ndarray or tuple: Anomaly scores, and optionally the individual predictions.
        """
        tx_normalized = self._normalize_data(tx)
        preds = [model.predict(self._select_features(tx_normalized, feat))
                 for model, feat in zip(self.models, self.feats)]
        if self.q_strat:
            preds = [(np.mean(np.abs(pred - q) ** self.power, axis=-1)) ** (1 / self.power)
                     for pred, q in zip(preds, self.qs)]
        else:
            preds = [np.mean(np.abs(pred - 1) ** self.power, axis=-1) ** (1 / self.power)
                     for pred in preds]
        preds = np.array(preds)
        errors = np.mean(preds ** self.ensemble_power, axis=0)
        errors = errors/np.mean(errors)
        if get_preds:
            return errors, preds
        return errors
