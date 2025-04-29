## Unsupervised Surrogate Anomaly Detection

This repository contains the supplementary material as well as the implementation of our paper *Unsupervised Surrogate Anomaly Detection*.

### Supplementary

In `supplementary.pdf` we provide the equivalents of the AUC-ROC performance plots from our experimental evaluation for the AUC-PR metric, as well as the detailed results for the specific datasets both for AUC-ROC and AUC-PR.

The supplementary also covers additional analysis results for DEAN with regard to the effect from incorporation of a learnable shift (bias term) in the network architecture. We further demonstrate the adaptability of DEAN based on the inclusion of fairness criteria into the predictions.

### Implementation

`dean.py` implements the DEAN method.
`main.py` allows to directly conduct the training and evaluation of a DEAN ensemble as specified in the configuration file `config.yaml` for a given dataset as provided in the `data` folder.


Alternatively, the configuration parameters may also be overwritten using command line arguments, e.g.:


```
python main.py --dataset data/Cardio.npz  --model_count 5
```

A suitable conda environment based on the requirements specified in `env.yaml` may be created via:
```
conda env create -f env.yaml
```


The folder `competitors` contains information regarding the implementation and parametrization of the competitor algorithms used during the experimental evaluation.

The folder `fairness` contains modifications of DEAN to for a proof-of-concept to include fairness criteria in the predictions, as referred to in the supplementary.

