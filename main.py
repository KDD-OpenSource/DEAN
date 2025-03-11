"""
Main entry point for training and evaluating the DEAN anomaly detection model.
"""


import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all but errors (0 = all logs, 3 = errors only)

import argparse
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score

from dean import DEAN


def str2bool(v):
    """
    Convert a string to a boolean value.
    
    Parameters:
        v (str or bool): The value to convert.
    
    Returns:
        bool: Converted boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Parameters:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(dataset_path: str):
    """
    Load dataset from a .npz file.

    Parameters:
        dataset_path (str): Path to the dataset file.

    Returns:
        tuple: (training data, test data, test labels)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    data = np.load(dataset_path)
    x = data["x"]
    tx = data["tx"]
    ty = data["ty"]
    return x, tx, ty


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the DEAN anomaly detection model."
    )
    parser.add_argument(
        '--config',
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file."
    )
    # Dataset file path override:
    parser.add_argument("--dataset", type=str, help="Path to the dataset file (.npz).")
    # Additional DEAN parameter overrides:
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--model_count", type=int, help="Number of submodels in the ensemble.")
    parser.add_argument("--normalize", type=int, choices=[0, 1, 2],
                        help="Normalization method (0: none, 1: min-max, 2: mean-std).")
    parser.add_argument("--bag", type=int, help="Number of features to use per submodel.")
    parser.add_argument("--neurons", type=lambda s: list(map(int, s.split(','))),
                        help="Comma-separated list of neurons per hidden layer.")
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate for hidden layers.")
    parser.add_argument("--activation", type=str, help="Activation function for hidden layers.")
    parser.add_argument("--patience", type=int, help="Early stopping patience.")
    parser.add_argument("--restore_best_weights", type=str2bool,
                        help="Whether to restore best weights after training.")
    parser.add_argument("--power", type=int, help="Power parameter for the loss calculation.")
    parser.add_argument("--bias", type=str2bool, help="Whether to use bias in hidden layers.")
    parser.add_argument("--output_bias", type=str2bool, help="Whether to use bias in the output layer.")
    parser.add_argument("--output_activation", type=str, help="Activation function for the output layer.")
    parser.add_argument("--q_strat", type=str2bool, help="Prediction strategy for computing anomaly scores.")
    parser.add_argument("--ensemble_power", type=int, help="Power for ensemble combination.")
    parser.add_argument("--parallelize", type=int, help="Number of threads for parallel training.")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Ensure the 'dean_params' key exists in config
    if "dean_params" not in config:
        config["dean_params"] = {}

    # List of DEAN parameter names to override
    dean_param_names = [
        "learning_rate", "batch_size", "epochs", "model_count", "normalize", "bag",
        "neurons", "dropout_rate", "activation", "patience", "restore_best_weights",
        "power", "bias", "output_bias", "output_activation", "q_strat", "ensemble_power", "parallelize"
    ]
    for param in dean_param_names:
        arg_value = getattr(args, param)
        if arg_value is not None:
            config["dean_params"][param] = arg_value

    # Load dataset
    if args.dataset is not None:
        dataset_path = args.dataset
    else:
        dataset_path = config.get("dataset", "data/Aglass.npz")
    logging.info(f"Loading dataset from {dataset_path}")
    x, tx, ty = load_data(dataset_path)

    # Instantiate and train DEAN using parameters from the config
    dean_params = config.get("dean_params", {})
    model = DEAN(**dean_params)
    logging.info("Training DEAN ensemble...")
    model.fit(x)

    # Evaluate the model
    logging.info("Evaluating model performance...")
    preds = model.decision_function(tx)
    auc = roc_auc_score(ty, preds)
    print(f"\nTest AUC-ROC on {dataset_path}: {auc}")


if __name__ == "__main__":
    main()
