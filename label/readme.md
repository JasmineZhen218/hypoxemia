# Labelling

### Hypoxemia definition
* A continuous 5-minute low SpO2 
* Low spo2 threshold: 92%

### Raw labels

* Check every time point whether at least 1 hypoxemia event occurred in the next 30 minutes. If so, this time point is labeled as positive; Otherwise, negative.
* If the next 30-minute window contains NA>20%, this point is labeled as NA

### Label exclusion criteria

* Exclude time point when **hypoxemia event** happens----> Only hypoxemia-free interval left
* Exclude hypoxemia-free interval shorter than 30 minutes

To see example labels:
1) Get access to MIMIC time-series dataset and save time-series data in dictionary format. Name: Series
2) `import label`
3) run `summary(name,series)`
(See examples in labelling.ipynb)
