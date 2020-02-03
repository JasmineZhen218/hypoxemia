# hypoxemia

This is a hypoxemia prediction projects. It use unidentified MIMIC III Data to predict hypoxemia 

### Target Subjects and records
`target_IDs.txt` includes the IDs of target patients, 6643 in all. Target patients satisfy:
* Patients who have minute-to-minute SPO2 records >= 5 minutes
* Patients who are older than 16

`meta_records.csv`includes information of usable time-series records. Usable records satisfy:
* Records of target patients
* Records whose SPO2 channel>= 5 minutes

Information in 'meta_records.csv' include:
* record name. such as 'p000020-2183-04-28-17-04n'
* record id. such as 20_0
* subject_id. such as 20
* start time: such as 2183/4/28 17:04
* end time: such as 2183/4/29 14:54:00
* length: such as 1310(This is the length of SpO2 record)

### Time-series data
Time-series data includes:
* SpO2
* PULSE
* ABP
* RESP
* HR
* NBP
* PAP
* CO
* CVP

`clean_and_fill.py` is the python code used to handel minute-to-minute time-series data.
* The implausible(out of possible range) data are removed. Data higher than upper threshold is set to upper threshold. Data lower than lower threshold is set to Nan 
* The gaps are interpolated.
  - Interpolation method: last observation
  - Interpolation threshold: 10 minutes (Any gap larger than 10-minute is preserved Nan)
  
 
### Sparse data
Sparse data includes:
* FiO2
* O2 flow
* PEEP
* Tidal volume
* PIP
* BMI

How should we deal with Sparse data
* 1. Extract them from MIMIC III Clinical Database using specific `itemids`
* 2. Only preserve data of target patients, using `target_IDs.txt`
* 3. Align Sparse data with time-series data. 
  - Align reference: SpO2 time-series data `meta_records.csv`
  - Align method: The start of each records is set to 0, each sparse reading get a coordinate via comparation with record start time. Sparse readings out of the range of spo2 records are excluded.
  - Code: `Align.py`
  - Aligned examples: '/assets/FiO2_aligned.csv', `/assets/O2_flow_aligned.csv`
