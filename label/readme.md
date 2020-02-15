# Labelling

### Hypoxemia Labelling criteria
1)	First, define hypoxemia event: a continuous 5-minute low SpO2 (<92)
2)	Then, check every time point whether at least 1 hypoxemia event occurred in the next 30 minutes. If so, this time point is labeled as positive; Otherwise, negative.
3)  If the next 30-minute window contains NA>20%, this point is labeled as NA

To see example labels:
1) Get access to MIMIC time-series dataset and save time-series data in dictionary format. Name: Series
2) import label
3) run "summary(name,series)"
