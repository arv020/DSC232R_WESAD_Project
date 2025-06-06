# DSC232R_WESAD_Project
WESAD Wearable Stress and Affect Detection Analysis Project for DSC 232R
https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html 

## Introduction 

We chose the Wearable Stress and Affect Detection (WESAD) dataset because of our mutual interest in utilizing a unique dataset that has real-world applications outside of this course. WESAD offers real-world physiological time-series data in conjunction with robust self-report data.

The dataset is collected from a controlled lab study that focuses on classifying participants’ emotional states. The states include a neutral baseline, stress, amusement and meditative state based on physiological and motion sensor data. What makes this dataset particularly compelling is its combination of physiological and behavioral signals, allowing us to explore complex patterns in human affect.
Building accurate predictive models in this domain could have significant implications for mental health monitoring, early stress detection, and personalized health interventions. In an era where mental well-being is increasingly prioritized, developing models that can detect stress or emotional shifts in real time using wearable devices could transform how individuals and clinicians approach psychological health.
By working with this dataset, we also gain interdisciplinary insights at the intersection of machine learning, psychology, and health technology, equipping us with a more holistic view of how data science can contribute to human well-being. This would also allow the team to widen their knowledge on psychological research paradigms that pair physiological information to emotional states as a hard metric to help inform mental health care.
The following sensor modalities are included (worn either via chest or wrist): blood volume pulse, electrocardiogram, electrodermal activity, electromyogram, respiration, body temperature, and acceleration. With these sensors we are able to explore the important features and trends between the classifications of: baseline vs. stress vs. amusement vs. meditation.

## Figures ADD IN VIA SDSC 



## <u>Methods </u>

**Environment set up instructions:**
<br>To set-up our SDSC Jupyter Job and environment we used the following configurations and steps:</br>

**Compute Resources:**
<br>We requested the following in our job submission: </br>

Number of cores: 50
Memory per node: 25 GB
Working Directory: Home Directory
We used the following singularity image provided by professor for each job session for our singularity container: ~/esolares/spark_py_latest_jupyter_dsc232r.sif 

Environment modules to be loaded: 'singularitypro'

Working directory: 'home'

Type: 'jupyterLab'

**Data Retrieval**
1) To retrieve the WESAD data we retrieved a zipped version of the file found here:
'https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download'

2) We used the following command and put it in our home directory
‘wget https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download . ‘ 

3) Once retrieving we unzipped the data as it comes in a zipped format
‘ unzip download ‘

**Data Exploration:**

Citation: Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven, "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection", ICMI 2018, Boulder, USA, 2018.

The majority of our data is formatted as time series data with synchronization points captured by wearable chest and wrist sensors. The sensors were worn while participants were asked to do tasks to induce stress, amusement, meditation and a baseline was collected. For preprocessing we consolidated self-report questionnaires into one dataframe and assessed sensor data separately by subject. Additionally, we removed some of the labels we did not need identified by the data structure to be undefined or unnecessary for processing. 

We visualized participant S2s data to gain an understanding of the distribution for the chest and wrist sensor data associated with the study. We evaluated the data to understand the structure of the files, the sensor descriptions and the number of observations and null values. Because of the nature of collection using body worn sensors, there is no missingness in sensor data. 
As part of our model building we intend to explore the connection between the self report surveys and the sensor data. See below in ‘results’ for a higher level view of data.


**Preprocessing:**
Signal Aggregation and Chunking: We grouped .pkl data by modality (e.g., ECG, EMG, EDA, TEMP) and exported them as CSVs. Each signal was segmented into chunks of 1,000 consecutive readings. For every chunk, we computed summary statistics: mean, std, min, max, mode, median. These transformed time-series data into tabular features.
Feature Consolidation: Each subject’s processed signal chunks were combined into a unified DataFrame named final_df. Each row corresponded to a chunk.
Label Filtering: Labels outside the main study conditions (i.e., 0, 5, 6, 7) were excluded. Only chunks labeled with 1 (Baseline), 2 (Stress), 3 (Amusement), or 4 (Meditation) were retained.
Train-Test Splitting: To preserve subject-level independence, we used GroupShuffleSplit with subject IDs as groups for model 1. The data was split into training, validation, and test sets.

In parallel, survey data for each subject was extracted from individual ‘_readme.txt’ files located in their respective folders. Key variables (age, height, weight, gender, handedness, coffee consumption, sports, smoking, and illness status) were parsed, encoded as binary or numeric features, and aggregated into a unified table (‘wesad_survey.csv’), excluding subjects S1 and S12. This survey table was then merged with the processed chunk-level feature table (‘final_df’) using the subject identifier, resulting in ‘wesad_merged_df.csv’, which contained both physiological and survey features for each chunk. 

**Model Building: Model 1 - Random Forest Classifier (Sensor Data + PANAS)**
<br> This model used both sensor-derived features and PANAS questionnaire responses. </br>
<br>
```
    rf = RandomForestClassifier(
    n_estimators=75,
    random_state=45,
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=5)
 ```
</br>
The PANAS features were merged with the final_df. Columns with significant missing values were dropped. A GroupShuffleSplit was applied for training and evaluation. The goal of our models is to predict the label 1, 2, 3 or 4.


**Model Building: Model 2 - Random Forest Classifier (Sensor Data Only)**
Model 2 uses a Random Forest Classifier from the sklearn.ensemble module, which is appropriate for handling high dimensionality and noisy datasets. The goal of this model is to predict the label 1, 2, 3 or 4 of each sample and output a predicted class for each sample, allowing us to identify patterns in the features and distinguish the strongest feature statistic that predicts a label.  

This model excluded all questionnaire data and used only sensor-derived statistics. The model was instantiated using the following parameters: 

```
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=45,
    class_weight='balanced',
    max_depth=20,
    min_samples_leaf=3)
``` 

<br>n_estimators = 100: The model builds 100 decision trees.</br>
<br>random_state = 45: Ensures reproducibility.</br>
<br>class_weight = 'balanced': Automatically adjusts weights inversely proportional to class frequencies.</br>
<br>max_depth = 20: Limits the depth of each decision tree to control overfitting.</br>
<br>min_samples_leaf = 3: Setting the minimum number of samples required to be at a leaf node, helping reduce variance</br>
The model was trained using the training set and evaluated on both the validation and test set, we used 70% training , 20 % for test and 10% validation. 

The merged dataset (‘wesad_merged_df.csv’) was used as input for the Random Forest model. Prior to training, features with zero importance (identified previously in an importance ranking) were removed. The final feature set included both physiological summary statistics and survey variables for each data chunk. 


**Model Building: Model 3 - Random Forest Classifier (Sensor Data + Survey Data)**
To select the optimal hyperparameters for the Random Forest model, several values for `n_estimators`, `max_depth`, and `min_samples_leaf` were systematically evaluated. Each combination was assessed based on training and test accuracy, and the values that balanced high test accuracy with reduced overfitting were chosen (see Results section/Model 3, Figure 2).

A Random Forest classifier was trained with the following hyperparameters:
<br>`n_estimators=100` (number of trees in forest)</br>
<br>`max_depth=23` (maximum depth of each tree; set to prevent overfitting)</br>
<br>`min_samples_leaf=2`’ (minimum samples required at a leaf node; helps reduce overfitting)</br>
<br>`random_state=42` (ensures reproducibility)</br>

```
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=23,           
    min_samples_leaf=2,    
    random_state=42 
)

rf.fit(X_train, y_train)
```

**Model Building: Model 4 - Random Forest Classifier(Sensor Data + Survey Data, Binary)**
For binary stress classification, a new target variable `stress_binary` was created, where label 2 (stress) was mapped to 1 and all the other labels to 0. Features with zero importance (e.g., `coffee_last_hour`, `smoked_last_hour`) and columns unrelated to prediction were dropped. The data was split using an 80/20 stratified split to preserve class balance. 

A Random Forest classifier was trained with:
`n_estimators=100` (number of trees, chosen for good performance and efficiency)
`max_depth=23` (limits tree depth to control overfitting)
`min_samples_leaf=2` (minimum samples per leaf, helps reduce overfitting)
`class_weight=’balanced’` (for class imbalance by weighting classes inversely to frequency)
`random_state=42` (ensures reproducibility)
`n_jobs=-1` (enables parallel computation for speed)

```
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=23,          
    min_samples_leaf=2,   
    class_weight=’balanced’, 
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
```

**Model Building: Model 5 - Random Forest Classifier (Sensor + Survey + PANAS Questionnaire Data)**
Model 5 utilized a comprehensive dataset combining sensor-derived features, demographic survey variables, and detailed self-report questionnaire data (PANAS, STAI, SAM, and SSSQ). This model was designed to evaluate the performance of a fully integrated feature set for multiclass emotional state classification (labels 1–4: Baseline, Stress, Amusement, Meditation).

The input data was loaded from wesad_merged_with_questionnaires.csv, which contained 773,832 samples and 62 columns. These columns included:

Sensor-derived features: Summary statistics for each chunk (mean, std, min, max, mode, median)

Demographic survey data: Age, height, weight, gender, dominant hand, lifestyle indicators (e.g., caffeine, smoking, illness)

Self-report data: PANAS (24 items), STAI (6 items), SAM (2 items), and SSSQ (6 items for stress condition)

Non-predictive and identifier columns were excluded from modeling. Specifically, the following columns were dropped:
['subject', 'modality', 'label', 'stress_binary', 'condition_id', 'condition_name'].
Only numeric columns were retained for modeling. Any missing values were imputed using the median of each feature.

The cleaned dataset was split into training and test sets using an 80/20 split. Stratification by label was applied to ensure class balance in both sets.

```
 X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
Model Training
A Random Forest classifier was trained using the sklearn.ensemble.RandomForestClassifier with the following hyperparameters:
```
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=25,    
    min_samples_leaf=2,  
    random_state=42,  
    n_jobs=-1      
)
rf.fit(X_train, y_train)
```

## Results 

**Data Exploration**

Our raw data exploration showed us that each subject has ~2–3 million time points. The total labeled observations across the dataset: 60,807,600. No missing data was observed in any sensor modality. Participant sample size consisted of n = 15 subjects (S2 to S17, excluding S12).

The WESAD dataset contained a series of emotional states induced by laboratory based tasks and were defined as follows:

0: Undefined or transitional time
1: Baseline
2: Stress (TSST)
3: Amusement
4: Meditation
5, 6, 7: Other/control conditions

For our classification tasks we opted to only assess the labels: Baseline (label 1), Stress (2), Amusement (3), and Meditation (4).

WESAD also included a robust self-report battery including: PANAS (24 items), STAI (6 items), SAM (2 items), and SSSQ (6 items for Stress only) were parsed from individual CSV files into a unified DataFrame (all_questionnaires.csv).

The dataset comprises time-series physiological signals using two synchronized wearable devices, a chest and wrist sensor:

RespiBAN (worn on chest): ECG, EDA, EMG, RESP, TEMP, ACC 
Empatica E4 (worn on wrist): BVP, EDA, TEMP, ACC 

The sensors captured a multidimensional perspective of overall physiological and motion data including but not limited to: accelerometer, gyroscope, electrocardiogram,skin conductance, temperature, respiration rate and more.


**Visual exploration figures of raw sensor data:**
The following plots display raw physiological signals collected from the chest- and wrist-worn sensors in the WESAD dataset. For each emotional condition (Baseline, Stress, Amusement, Meditation), we extracted a single 1,000-sample segment from the first occurrence of that label to visualize representative patterns. Multichannel signals (e.g., accelerometer) show each axis separately, while univariate signals (e.g., ECG, EMG) are plotted as single traces. These visualizations provide insight into the temporal behavior and variability of different sensor modalities prior to feature extraction. Emotional states appear to influence sensor modalities in different ways. No significant data quality issues were encountered.

<< ADD IN FIGURE 3 -9 >> 


## Preprocessing
The raw time series data and questionnaire responses were systematically preprocessed to prepare for modeling.

Chunking: Each subject’s signal data was combined into files split up by modality. These results were then segmented into 1000 sample chunks. These chunks for each segment were combined to create final_df which was used in our models. For each chunk, summary statistics such as mean, median, mode were computed. This resulted in a structured data frame that can be used in our models. After processing all subjects, a total of 160,518 chunks were extracted.

Label Filtering: The original labels in the datasets included 0 through 7. It was suggested in the original dataset ReadMe to only focus on 1-4, as 0 is unidentified and 5-7 are control/transitional labels. We filtered the data to retain labels 1-4 only.

Questionnaires: Each subject’s PANAS, STAI, and SAM questionnaire data was extracted from their respective SX_quest.csv files and merged into a master file all_questionnaires.csv. This was combined with final_df in Model 1. 

## Model 1: Random Forest Classifier (Sensor Data + PANAS):
Our approach was to combine the self-report surveys and sensor data to predict labels 1-4. This initial attempt unveiled that the prediction model did not perform well when assessing all features (the sensor summary stats and the self-report). There was also an overfitting issue with a training accuracy of 1. We modified the model to run with features that were highly correlated with the label prediction but found little, to no improvement. Overall, it was decided to take another approach.

Training Accuracy: 1.0
Validation Accuracy: 0.465
Test Accuracy: 0.613

<<ADD IN FIGURE 10/11>>


## Model 2: Random Forest Classifier (Sensor Data Only):

For our second Random Forest Model which does NOT include the PANAS dataset and only the time series data, our results consist of the following: 

<<< ADD IN TABLE FOR ACCURACY RESULTS >>

We created the following Classification Report on the Test set to better understand how our model is predicting each label through calculating Precision, Recall, F1-Score and Support. 

<< ADD IN CLASSIFICATION REPORT >> 

A confusion matrix was also added as part of the results of this project please refer to Figure 3.


## Model 3: Random Forest Classifier (Multiclass classification: Labels - 1,2,3,4):
A Random Forest classifier was trained for multiclass (baseline, stress, meditation, amusement) classification (labels 1-4) using sensors and survey data. Multiple hyperparameter configurations were evaluated, with the optimal parameters chosen based on a balance of training and test accuracy. 

<<ADD IN PERFORMANCE METRICS>>

Figure 12. Confusion Matrix for Model 3 (Sensor + Survey Data)

<<ADD IN CONFUSION MATRIX>>

Figure 13. Hyperparameter search results for Model 3: explores values of `max_depth`, ,`n_estimators`, and `min_samples_leaf` with corresponding training and test accuracy.


<<< ADD IN FIGURE 13 >>> 

## Model 4: Random Forest Classifier (Sensor Data + Survey Data, Binary)
A Random Forest classifier was trained for binary stress classification as described above. 

<<< PERFORMANCE METRICS >>>
Performance Metrics:

Classification Report (Test Set):

<< ADD IN CLASSIFICATION REPORT >>> 

Figure 14. Sensor + Survey Data (Binary Classification) Performance Scores (Test Data)

<< ADD IN SENSOR + SURVEY DATA >>

## Model 5: Random Forest Classifier (Sensor + Survey + PANAS Questionnaire Data)

Performance Metrics:

<< performance metrics table>>
Figure 15. Confusion Matrix for Model 5 (Sensor+Survey+PANAS Questionnaires, Multiclass Classification).

<<< ADD IN 15 Figures>> 
<< ADD IN  FIGURES 16  >>
Figure 16. Top 15 Importances identified by Random Forest Model.


Figure 17. Classification report for Model 5 (Sensor+Survey+PANAS Questionnaires, Multiclass Classification).

<<ADD IN FIGURE 17 and classification >>

**Final Model and Summary**

The final model selected was a Random Forest classifier trained on the dataset including the Sensor, Survey, and Questionnaire (PANAS) data for multiclass affective state classification (labels 1-4). It achieved the strongest test accuracy (within multiclass classification: 0.78), and robust performance across all the classes, with especially high recall for class 1. Feature importance analysis highlighted the value of aggregate sensor statistics and select questionnaire responses. These results indicate that a combination of physiological, questionnaires and self-report data enables effective multiclass label detection

**Discussion**

Our analysis of the WESAD dataset was driven by the goal of building interpretable, generalizable models capable of detecting human physiological and emotional states defined by wearable sensor data and an extensive self-report battery. With this dataset we were able to leverage a well-documented multimodal dataset that captured real-world physiological signals in a controlled laboratory setting. 

In our initial data exploration, the raw signals revealed an expected associated pattern with each of the four labels, i.e. a lower respiratory rate during a meditative state. Assessing the visualizations of each sensor modality helped us gain an understanding and form our hypotheses on how to proceed with feature engineering our models.

The label distribution, although not perfectly distributed, was sufficient to train our models after applying things like class weighting. One strength of the data set for model building was the variety of sensor types (ECG, ACC, etc.) and the variety of self-report that allowed us to explore multiple prediction models and created a dimensional data set for feature selection. 

During our preprocessing phase, we opted to chunking the time-series data into event spaces consisting of 1000 consecutive sample segments. We determined that the volume of information from each sensor, using one singular timepoint, would not yield enough information to make a solid prediction so we had to create event spaces that were more representative of each physiological state. We also computed summary statistics which reshaped the data into a tabular format more well-suited for our machine learning model. Although these choices made a more digestible model, it is likely that we lost some of the more fine-grained temporal patterns by doing so. During this phase we also excluded ambiguous and undefined labels and focused on building our models on the 4 classes: Baseline, Stress, Amusement and Meditation.

**Model 1: Sensor Data + PANAS**
The integration of self-reported PANAS questionnaire data into Model 1 provided a valuable contrast to the sensor-only model. This model provided guidance but was not an effective framework for predicting labels. Our first iteration included all sensor summary statistics and self-report data. Unsurprisingly, this led to a gross overfitting of the model with a 100% training accuracy. Through thorough evaluation, we found that the questionnaires themselves joined upon the sensors allowed the model to predict with certainty which state each participant was in. 
To reform the model for overfitting we trimmed down feature selection to only include PANAS variables that were highly correlated with the predicting label outputs, regarding NaN values and shuffling data to reduce predictability. Although this resolved the overfitting issue, the model performed poorly (likely due to loss of data granularity) and we navigated to a different approach that explored 


**Model 2: Sensor Data Only**

Model 2 demonstrated that wearable sensor data alone can successfully distinguish between emotional states, supporting the broader goal of passive stress detection through wearable technology. The higher max depth and greater number of estimators improved the model’s ability to capture nonlinear relationships between sensor statistics and affective labels. However, model performance may still be hindered by overlapping features between Stress and Amusement, both of which can involve heightened arousal, and by inter-individual variability.
The confusion matrix for Model 2 highlighted this challenge, as some emotional states, especially Stress vs. Amusement, were occasionally misclassified. This points to the nuanced nature of physiological responses: similar arousal levels can emerge from both positive and negative stimuli, complicating label boundaries. In future iterations we think investigating other models beyond what we have learned in our coursework could be used to process time-series specific data.

<<ADD IN FEATURE IMPORTANCE FIGURE FOR RANDOM FOREST >>

**Model 3: Sensor Data + Survey Data**
Model 3 tries a different multiclass classification by incorporating sensor data and the survey data (NOT PANAS). These questions include variables such as age, gender, health behaviors and lifestyle factors. We wanted to see if these context-specific variables could improve the performance of our models 
Model 3 outperformed Model 1 and 2 in labeling the four emotional states. Feature importance showed that both sensor statistics and survey contributed to these results. However, there is overfitting with training being over 91%. Dropping features with almost zero importance improved the model generalization by a little bit.

**Model 4: Sensor Data + Survey Data (Binary Classification)**
Model 4 reframed the problem as a binary classification problem. We wanted to distinguish stress (label 2) from all the other states. The result was really good - showing the highest performance across all the models.
Feature pruning, like removing low importance features from the survey, helped remove some of the noise and improve generalization. The class balancing and hyperparameter tuning helped.
The plots showed that the model’s confidence in predicting stress increased when features like min, max and median fell within certain ranges. Overall, by focusing solely on stress detection, Model 4 delivered highly reliable predictions.

**Model 5: Sensor Data + Survey + PANAS Questionnaire Data**
Model 5 is our final model - it combines all available data sources: sensor-derived features, demographic survey responses, and psychological self-report questionnaires (PANAS, STAI, SAM, SSSQ)—to predict the four affective states. This comprehensive approach produced the highest multiclass test accuracy across all models at 78.2%, with a balanced performance across classes.
The model achieved strong results for Baseline (89.4%) and Stress (74.5%), but struggled more with Amusement (53.7%), consistent with earlier models. Including self-reported affect measures likely helped disambiguate subtle states like Meditation and Stress, while demographic features added context that improved generalization.
Despite using many features, overfitting remained minimal (training accuracy ~91%), likely due to the model’s depth and regularization. The integrated data provided richer information and led to more nuanced predictions.
Model 5 demonstrates the value of multimodal data fusion—pairing physiological signals with behavioral and psychological context can significantly enhance emotion classification from wearable devices.
There are several limitations to note in our study:
Generalizability: Our models were trained on data collected in a lab setting. Real-world noise and contextual variability could significantly affect model accuracy in practical deployment.


Chunk Size Rigidity: Fixed-size chunking (1,000 samples) may not align well with the natural boundaries of emotional episodes, potentially splitting or truncating relevant events.


Class Imbalance: Although moderate, class imbalance likely impacted classifier sensitivity, especially for underrepresented states like Amusement.


Model Constraints: We focused on Random Forests due to their interpretability and robustness, but more specialized time-series models might better capture complex temporal dependencies.


**Conclusion**

One of the most rewarding aspects of this project was the interdisciplinary lens through which we approached physiological state detection. Integrating knowledge from psychology, physiology, and machine learning gave us a more nuanced understanding of how emotion manifests in sensor data. By interpreting sensor data through a biopsychological framework, we translated data grounded in human experience, an area we are all passionate about exploring.

For this model, some future directions we would have liked to continue to explore the model:


Normalize/standardize summary statistics: Applying normalization or standardization to each entry’s summary statistics could help reduce variance caused by differing sensor scales.


Optimize time-series chunk size: We currently segment data into 1,000-reading chunks, but the ideal chunk size remains an open question. Future work will explore different segment lengths and dynamic segmentation strategies to enhance model performance.


Improve hyperparameter tuning: We performed a limited grid search due to time constraints. Future iterations could use more efficient methods like randomized search or Bayesian optimization to better tune model parameters.


Explore additional algorithms: Beyond the models discussed in class, we would like to test more advanced or domain-specific algorithms, including deep learning approaches tailored to time-series data.


Reconsider data preprocessing strategy: Instead of chunking, using individual data points could increase the number of training examples and potentially improve generalization.
While our final model integrated many of the features proposed in Milestone 3, the feature engineering process was inherently iterative, and there are still many promising directions to pursue.
Statement of Collaboration 
Our team collaborated very equally and worked very well together by leaning on one another. The entire team worked well on managing deadlines and we shared planning for meetings/scheduling. We met over 10+ times over the course of this quarter as a team to meet our deadlines and make significant intellectual contributions. With that said we each contributed as follows:

Ingrid Altamirano: Developer, Contribution: Git Integration and general environment setup, Model 2 build and exploration 

Tatianna Sanchez: Developer, Model 3 and 4 build, preprocessing questionnaires and survey data, 

Marianne Sawires: Developer, Contribution: partial data exploration, preprocessing pkl files, model exploration, readme writing 

Vanessa Scott: Developer, Contribution: Domain specific contribution for processing approach (previous research with sensor modeling in infants), Preprocessing of self-report data with visualizations. Creating PANAS + Sensor Data Model, Writing Results section for Model 1, Writing Discussion Section.

Arely Vasquez: Developer, Contribution: Preprocessing for chunking dataset for summary statistics, baseline model exploration.



Citations:
Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven, "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection", ICMI 2018, Boulder, USA, 2018.

WESAD: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html




