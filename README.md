# DSC232R_WESAD_Project
WESAD Wearable Stress and Affect Detection Analysis Project for DSC 232R
https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html 

## Environment Setup Instructions 
To set-up our SDSC Jupyter Job and environment we used the following configurations and steps:

<u> Compute Resources </u> 
We requested the following in our job submission:
1. Number of cores: 50
2. Memory per node: 25 GB
3. Working Directory: Home Directory 
   
We used the following singularity image provided by professor for each job session for our singularity container: 
```~/esolares/spark_py_latest_jupyter_dsc232r.sif ```

**Environment modules to be loaded:**
'singularitypro'

**Working directory:**
'home'

**Type:**
'jupyterLab'


To retrieve the WESAD data we retrieved a zipped version of the file found here: 
<br>'https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download' </br>

We used the following command and put it in our home directory 
<br>```wget https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download . ``` </br>

Once retrieving we unzipped the data as it comes in a zipped format 
<br>```unzip download ``` </br>

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

## Model Building 

The model we decided to use in Milestone 3 was a Random Forest Classifier, as it handles high dimensional and noisy data. We have 2 attempts in our notebook: 1st attempt with a dataframe that includes the self report PANAS questionnaire, and a 2nd attempt with only sensor data.

<u>Pre-processing step: </u>
We began by organizing the data needed for modeling. All .pkl files were grouped by modality—chest (ECG, EMG, EDA, Temp, Resp, ACC) and wrist (BVP, EDA, Temp, ACC)—with each modality stored in its own table. These were then exported as CSVs and stored in a folder named combined_pkl_csv for streamlined access in the next steps.

To reduce noise and prepare the data for time-series modeling, we chunked each signal into segments of 1,000 data points. For each chunk, we computed summary statistics (e.g., mean, std, min, max), transforming raw signals into interpretable features for the model.

The target variable represents four emotional states:
1: Baseline, 2: Stress, 3: Amusement, and 4: Meditation.

All processed chunks across subjects and modalities were then merged into a single DataFrame named final_df. From here, we proceeded with two distinct preprocessing and modeling approaches.

<b><u>1st attempt:Sensor Data + PANAS </u> </b>
To complete the pre-processing, we added the PANAS self-report questionnaire to the final_df and dropped any columns with a significant amount of null values. 

The parameters we used for our RandomForestClassifier are the following: 

rf = RandomForestClassifier(
    n_estimators=75,
    random_state=45,
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=5
)
We adjusted the class_weight to 'Balanced' to ensure our data is proportionate and ensuring underrepresetned classes arent ignored. We chose a max_depth at 10, as when testing out a vartiety of max_depth, 10 was the most reasonable depth. We used a min_samples_leaf of 5 to help regularize the model and reducing overfit. 

We used GroupShuffleSplit to split our data while preserving subject-level groupings. However, despite this, we observed overfitting: the model achieved 100% accuracy on the test set. We suspect this is due to data leakage from the PANAS features, which may allow the model to "memorize" outcomes due to the limited number of questionnaire responses relative to the sensor data.

We also explored feature importance using a correlation bar plot, which gave us insights for future modeling directions.

<b><u>2nd attempt: Sensor Data Only </u> </b>
To complete the pre-processing, we only combined the pkl file time chunks to give us the following shape: (160518, 13) and the following columns: Index(['chunk_id', 'subject', 'modality', 'label', 'mean', 'std', 'min', 'max', 'range', 'skewness', 'iqr', 'mode', 'median'], dtype='object'). We took a look at some summary statistics and made sure that we had no missing values.

The parameters we used for our RandomForestClassifier are the following: 
rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=45, 
    class_weight='balanced',
    max_depth=20, 
    min_samples_leaf=3
)

Again, we adjusted the class_weight to 'Balanced' to ensure our data is proportionateand we chose a max_depth at 20, because it was the most reasonable. We used a min_samples_leaf of 3 to help regularize the model and reducing overfit.
As discussed in the conclusion, removing the PANAS questionnaire improved model generalization and reduced overfitting.



## Next Steps 
For our next Milestone we aim to further improve our model's accruacy through the following steps:

* The first thing we will do to further improve our results, is to enhance our current Random Forest model by exploring additional features and incorporating written survey data, which were not included in the current model. By expanding the feature set, we aim to boost model performance. 

* In addition to improving our baseline, we plan to experiment with different modeling appraches, including gradient (e.g. XGBoost, LightGBM) and support vector machines. By comparing these methods, we hope to identify the approach that yields the msot accurate and reliable results for out dataset, along with finding the most important features that predict our four labels. 

* Moreover, we are currently segmenting the time series sensor data into events of 1000 consecutive readings. However, the best chunk size for optimal model performance will continue being explored in the future milestone as we continue adding different data to our segment.

* As for another approach since our current classification model involves predicting all four label states (Baseline, Stress, Amusement, Meditation). However, this multi-class approach may compromise performance if some states are difficult to distinguish. As a next step, we will explore alternative strategies, such as binary classification (e.g., Stress vs. No Stress), to assess whether simplifying the task can achieve greater accuracy and reliability in our models.



## Conclusion 
For our First RandomForest Model which includes the PANAS dataset our results consist of the following: 
<b>Training Accuracy:</b> 1.0
<b>Validation Accuracy:</b> 0.46539238539238537
<b>Test Accuracy:</b> 0.6128201947528376

For our second RandomFortest Model which does NOT include the PANAS dataset and only the time series data, our results consist of the following: 
<b>Training Accuracy: </b>0.869951199252414
<b>Validation Accuracy: </b> 0.6696361824071767
<b>Test Accuracy: </b> 0.6746511338151009

When the PANAS dataset was included our model was training perfectly, a big sign of overfitting and data leakage. When removing the dataset and only including the WRIST/CHEST device data, our model was fitting the training data well, but with a slight performance gap between the training and testing accuracy, which means there is some level of overfitting. Our performance at 0.67 is an improvement, and does show some reasonable generalization. 

We evaluated its performance on the test set and generated a classification report for deeper insight. It shows that label 1 has the highest precision and recall around 73% - 75%. In comparison to label 3 which is the weakest performer ~51% . Both label 2 and 4 are balanced but can improve. The confusion matrix shows that instances correclty and incorrectly classified, showing that misclassifications were more common between adjacent and similar classes. 

Classification Report (Test):
              precision    recall  f1-score   support

           1       0.75      0.73      0.74     12606
           2       0.65      0.64      0.65      7129
           3       0.51      0.51      0.51      3984
           4       0.67      0.70      0.68      8385

    accuracy                           0.67     32104
   macro avg       0.64      0.64      0.64     32104
weighted avg       0.68      0.67      0.67     32104

Confusion Matrix (Test):
[[9196 1257  885 1268]
 [1111 4571  459  988]
 [ 857  433 2021  673]
[1145  734  635 5871]]

Removing PANAS dataset reduced overfitting and improved generaizability, while the model performs decently label 3 remains challening, and there is lots of room for refinement, feature engineering, balancing trainign data and using a variety of techniques mentioned in our next steps. 
 




