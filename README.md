# DSC232R_WESAD_Project
WESAD Wearable Stress and Affect Detection Analysis Project for DSC 232R
https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html 

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


