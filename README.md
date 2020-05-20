# Hypoxemia prediction

This is a computational physiology project which aims at predicting near-feature hypoxemia for patients in the ICU

### Introduction

Hypoxemia is defined as an abnormally low level of oxygen in arterial blood. It is commonly diagnosed among general Intensive Care Unit patients.  Real-time prediction of hypoxemia would enable clinicians enough time to prepare proper interventions that minimize patient mortality.

This project used patient vital signs(such as SpO2 and Heart rate), demographic data( such as age, gender and weight) to generate a hypoxemia likelihood prediction.

### Data source

9618 patients extracted from MIMIC-III Database.  All the patients are from 16 years old to 89 years old. Most of the patient have multiple vital signs measurement , including SpO2, heart rate, pulse, respiratory rate, non-invasive blood pressure.

### Preprocessing

Step1: Extract vital signs(at the sample time resampling) from MIMIC-III database

Step2: Clean (remove physiological implausible data) and interpolate(fill in gaps with length below 10 minutes using last observation ) data 

Codes, documents and demo: 

### Labeling and sampling

Step1:  label hypoxemia events from SpO2 series, which is defined as SpO2<=92% for at least 5 minutes.

Step2:  Extract positive and negative samples from time points based on their distance to hypoxemia events.

Codes, documents and demo: 

### Statistic analysis

This section includes codes to conduct statistical analysis on positive and negative samples.

You will see vital signs trend preceding hypoxemia event and non-hypoxemia data in this section.

You will see vital signs distribution at specific time points which are close to a hypoxemia event or far away from it. You will also see T-test results and wether they are statistically significant different.

Codes, documents and demo: 

### Feature generation

This section is aims at extracting features from real-time data to quantify the time-varying structure and filter noise.

* Smooth: Exponentially weighted moving average, Linear least-squares

* Trend: Derivative approximation of linear regression

* Variance and noise: Exponentially weighted moving variance, Spectral entropy.

Codes, documents and demo: 

### Prediction algorithms

This section focuses on fit various models, including GLMs, SVM, random forest, neural networks to distinguish positive and negative samples

Codes, documents and demo: 

### Results 

You will see the comparision of various algorithms in this section.

You will see how model performance changes given various restrictions on positive samples(the distance to events) and restrictions on events(the duration of events)

You will see real-time prediction demo in real-world clinical sceneroi where vital signs are continuously monitoring and alarm is given by a constant threshold of risk score.

Codes, documents and demo: 