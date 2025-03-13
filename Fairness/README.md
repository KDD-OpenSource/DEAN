## DEAN-Fair

Code for various approaches to fair anomaly detection with DEAN.

- Loss based modification: `fair_by_loss.py`
- Pruning based modification: `fair_by_pruning.py`
- Submodel weighting based modification: `fair_by_weight.py`

`baseline.py` generates the baseline results for DEAN on the dataset. These results are used to compare the performance of the fair anomaly detection approaches to the performance of DEAN without any modifications. This is done by `gen_table.py`.
