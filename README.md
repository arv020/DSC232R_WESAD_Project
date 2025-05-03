# DSC232R_WESAD_Project
WESAD Wearable Stress and Affect Detection Analysis Project for DSC 232R
https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html 

## Abstract
Wearable Stress and Affect Detection Dataset (17GB)
The dataset is collected from a controlled lab study that focuses on classifying participants’ emotional states, specifically neutral, stress, and amusement, based on physiological and motion data ultimately collecting the WESAD (Wearable Stress and Affect Detection) dataset. Data is collected from both wrist and chest-worn devices with 15 subjects reported from both medical professionals as well as self reports from the subjects. The following sensor modalities included are: blood volume pulse, electrocardiogram, electrodermal activity, electromyogram, respiration, body temperature, and acceleration. With these sensors we are able to explore the important features and trends between the stress classifications of: baseline vs. stress vs. amusement, and compare both devices.

## Data Exploration
The majority of our data is formatted as timeseries data with synchronization points captured by wearable chest and wrist sensors. The sensors were worn while participants were asked to do tasks to induce stress, amusement, meditation and a baseline was collected. For preprocessing we cosolidated the files into one dataframe per file type (pickle, csv etc.). Additionally, we removed some of the labels we did not need identified by the data structure to be undefined or unneccescary for processing. We visualized participant S2s data to gain and understanding of the distribution for the chest and wrist sensor data associated with the study. We evaluated the data to understand the structure of the files, the sensor descriptions and the number of observations and null values. Because of the nature of collection using body worn sensors, there is no missingness in sensor data. As part of our model building we intend to explore the connection between the self report surveys and the sensor data. 

