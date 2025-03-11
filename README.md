## Towards Optimal Surrogate Models for Unsupervised Anomaly Detection

This repository contains the supplementary material as well as the implementation of our paper *Towards Optimal Surrogate Models for Unsupervised Anomaly Detection*.

### Supplementary

In `supplementary.pdf` we provide the equivalents of the AUC-ROC performance plots from our experimental evaluation for the AUC-PR metric, as well as the detailed results for the specific datasets both for AUC-ROC and AUC-PR.

The supplementary also cover additional analysis results for DEAN with regard to runtime, the effect from incorporation of a learnable shift (bias term) in the network architecture, as well as the impact of ensemble size both on DEAN as well as the other surrogate methods. Furthermore we give more information on the modifiability of DEAN, also demonstrated by example for the inclusion of fairness criteria into the predictions.

### Implementation

`dean.py` implements the DEAN method.
`main.py` allows to conduct the training and evaluation of a DEAN ensemble as specified in the configuration file `config.yaml` for a given dataset as provided in the `data` folder.


Alternatively, the configuration parameters may also be overwritten using command line arguments, e.g.:


```
python main.py --dataset data/Cardio.npz  --model_count 5
```

A suitable conda environment based on the requirements specified in `env.yaml` may be created via:
```
conda env create -f env.yaml
```


The folder `competitors` contains information regarding the implementation and parametrization of the competitor algorithms used during the evaluation.

The folder `fairness` contains modifications of DEAN to for a proof-of-concept to include fairness criteria in the predictions, as referred to in the supplementary.

