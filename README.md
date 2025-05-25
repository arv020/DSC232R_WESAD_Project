# DSC232R_WESAD_Project
WESAD Wearable Stress and Affect Detection Analysis Project for DSC 232R
https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html 

## Environment Setup Instructions 
To set-up our SDSC Jupyter Job and environment we used the following configurations and steps:

<u> Compute Resources </u> 
We requested the following in our job submission:
1. Number of cores: 50
2. Memory per node: 25 GB
   
We used the following singularity image provided by professor for each job session for our singularity container: 
'~/esolares/spark_py_latest_jupyter_dsc232r.sif '

**Environment modules to be loaded:**

'singularitypro'

**Working directory:**

'home'

**Type:**

'jupyterLab'


To retrieve the WESAD data we retrieved a zipped version of the file found here: 
<br>'https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download' </br>

We used the following command and put it in our home directory 
'wget https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download . '

Once retrieving we unzipped the data as it comes in a zipped format 
'unzip download '

## Abstract
Wearable Stress and Affect Detection Dataset (17GB)
The dataset is collected from a controlled lab study that focuses on classifying participants’ emotional states, specifically neutral, stress, and amusement, based on physiological and motion data ultimately collecting the WESAD (Wearable Stress and Affect Detection) dataset. Data is collected from both wrist and chest-worn devices with 15 subjects reported from both medical professionals as well as self reports from the subjects. The following sensor modalities included are: blood volume pulse, electrocardiogram, electrodermal activity, electromyogram, respiration, body temperature, and acceleration. With these sensors we are able to explore the important features and trends between the stress classifications of: baseline vs. stress vs. amusement, and compare both devices.

## Data Exploration
Citation: Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven, "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection", ICMI 2018, Boulder, USA, 2018.
The majority of our data is formatted as timeseries data with synchronization points captured by wearable chest and wrist sensors. The sensors were worn while participants were asked to do tasks to induce stress, amusement, meditation and a baseline was collected. For preprocessing we cosolidated self-report questionnaires into one dataframe and assessed sensor data seperately by subject. Additionally, we removed some of the labels we did not need identified by the data structure to be undefined or unneccescary for processing. We visualized participant S2s data to gain and understanding of the distribution for the chest and wrist sensor data associated with the study. We evaluated the data to understand the structure of the files, the sensor descriptions and the number of observations and null values. Because of the nature of collection using body worn sensors, there is no missingness in sensor data. As part of our model building we intend to explore the connection between the self report surveys and the sensor data. See below for a higher level view of data:


## Dataset Overview
## Name: WESAD (Wearable Stress and Affect Detection)

Link: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

Devices:

RespiBAN (worn on chest): ECG, EDA, EMG, RESP, TEMP, ACC
Empatica E4 (worn on wrist): BVP, EDA, TEMP, ACC
Subjects and Observations
Subjects: 15 total (S2 to S17, excluding S12)

Observation Count:

Each subject has ~2–3 million time points
Total labeled observations across dataset: 60,807,600

## Label Definitions

The `data['label']` field is a 1D NumPy array. It contains integer values that represent the experimental condition the subject was in at each time point. These are the meanings of each label:

- 0: Undefined or transitional time
- 1: Baseline
- 2: Stress (TSST)
- 3: Amusement
- 4: Meditation
- 5, 6, 7: Other/control conditions

---

## Data Structure per Subject

Each subject's `.pkl` file contains a dictionary with three keys:

- `data['signal']['chest']`: contains the time-series signals from chest-worn sensors. Each type of sensor (like ECG or EDA) can be accessed individually.
- `data['signal']['wrist']`: contains the time-series signals from wrist-worn sensors (Empatica E4 device).
- `data['label']`: a 1D array of integer labels corresponding to each time point.
- `data['subject']`: a string indicating the subject ID (e.g., 'S2').

Signals can be either 1D (e.g., EDA, ECG, TEMP) or 3D (e.g., ACC — accelerometer data with X, Y, Z axes). These signals are synchronized with the labels, so each row in a signal corresponds to the same time point in `data['label']`.

Example access:
- `data['signal']['chest']['EDA']` gives the chest EDA signal
- `data['signal']['chest']['ACC']` gives the chest accelerometer signal (X, Y, Z)
- `data['signal']['wrist']['BVP']` gives the blood volume pulse from the wrist


### Sensor Descriptions
The WESAD dataset contains data from two devices:

Chest-worn RespiBAN Pro
Wrist-worn Empatica E4
Below is a description of each sensor stream:

Chest Sensor: ACC
3-axis accelerometer (X, Y, Z)
Captures body posture and motion
Units likely in g (gravitational acceleration)


Chest Sensor: ECG
1D electrocardiogram signal
Measures electrical heart activity (voltage)
Useful for detecting heart rate variability (HRV), stress


Chest Sensor: EMG
Electromyography: records muscle tension/activity
High values indicate movement or tension
Useful for stress response analysis


Chest Sensor: EDA
Electrodermal Activity (skin conductance)
Reflects sympathetic nervous system activity (stress/arousal)
High EDA = increased stress


Chest Sensor: Temp
Skin temperature (in °C)
Subtle changes may correlate with stress


Chest Sensor: Resp
Respiration signal (breathing pattern)
Cyclical waveform with inhalation/exhalation
Can extract breathing rate, variability


Wrist Sensor: ACC
3-axis accelerometer from wrist (noisier than chest)
Captures wrist movement (e.g., fidgeting, gestures)


Wrist Sensor: BVP
Blood Volume Pulse
Derived from optical sensors (like PPG)
Used to estimate heart rate and HRV


Wrist Sensor: EDA and Temp
Redundant versions of chest EDA and Temp
Can be used for cross-validation or fallback

Folder & File Descriptions
Each subject folder (e.g., S2, S3, ...) contains the following files:

[SubjectID].pkl
Main preprocessed data file
Contains a dictionary with:
signal → Chest & wrist sensor data
label → 1D array of condition labels
subject → Subject ID

[SubjectID]_quest.csv
Questionnaire and experiment protocol file
Contains:
Condition order
Start/End times (in minutes)
PANAS (affect) responses

[SubjectID]_readme.txt
Notes about subject data

[SubjectID]_E4_Data.zip
Raw data from Empatica E4 wrist device
Included in .pkl

[SubjectID]_respiban.txt
Raw chest sensor data
Included in .pkl



## Self-Reporting Questionnaires Insights and Reformating
The questionaires found as 'SX_quest.csv' were reformatted and organized into a single CSV. The motivation for this was to centralize these metrics and self responses for each subject in a single place. 


**Response Scales**
The following are response scales, and their interpretation according to the WESAD dataset. 

* **PANAS** (24 items): 1 = Not at all, 2 = A little bit, 3 = Somewhat, 4 = Very much, 5 = Extremely

* **STAI** (6 items): 1 = Not at all, 2 = Moderately so, 3 = Somewhat, 4 = Very much so

* **SAM** (DIM (tag), 2 items: Valence & Arousal): 1 = Low (valence/arousal), 9 = High (valence/arousal)

* **SSSQ** (6 items): 1 = Not al all, 2 = A little bit, 3 = Somewhat, 4 = Very much, 5 = Extremely

**Self-Report Questionnaire Notes**

* **PANAS, STAI, SAM** were administered after each of the four study conditions (Baseline, Stress, Amusement, Meditation), so every row has values for those 36 items (24+6+6).

* **SSSQ** was administered only *once*, immediately after the **Stress** condition, so for Baseline, Amusement, and Meditation rows those six columns are intentionally empty (NA).

**Condition Order & Tag Mapping**
* Every'SX_quest.csv' contains a line:
  
    *'# ORDER;Base;TTST;Medi 1;Fun;Medi 2;...'*

* We interpret the remap as:
  
| Raw Token | Condition Name | condition_id |
|:---------:|:--------------:|:------------:|
| Base      | Baseline       | 1            |
| TSST      | Stress         | 2            |
| Medi 1    | Meditation     | 4            |
| Fun       | Amusement      | 3            |
| Medi 2    |*(ignored)*     | -            |

* We only keep the first four conditions (Baseline, Stress, Meditation, Amusement), and ignore anything from 'Medi 2' and beyond (including 'sRead', 'fRead').

* Each questionnaire block is tagged:

|          Tag          | Questionnaire                        |
|:---------------------:|:------------------------------------:|
| '# PANAS'             | PANAS (24 questions)                 |
| '# STAI'              | STAI (6 questions)                   |
| '# DIM'               | SAM (Valence & Arousal; 2 questions) |
| '# SSSQ'              | SSSQ (6 questions; only after Stress)|





* All responses are parsed, lowercased, and written to:
  **all_questionnaires.csv**




