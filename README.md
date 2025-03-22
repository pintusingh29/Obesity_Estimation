# Obesity_Estimation
Estimation of Obesity Levels Based On Lifestyle

Project Overview
This project applies Machine Learning (ML) techniques in R to predict obesity levels based on individuals' eating habits and physical conditions. The dataset, sourced from the UCI Machine Learning Repository, includes lifestyle, dietary, and physical activity factors influencing obesity.

The study compares the performance of Random Forest (RF) and Artificial Neural Networks (ANNs) in classifying obesity levels.

Dataset
Name: Estimation of Obesity Levels Based on Eating Habits and Physical Condition

Source: UCI Machine Learning Repository

Features: 17 (including age, gender, food consumption, physical activity, BMI, etc.)

Target Variable: Obesity level classification (Normal, Overweight, Obese, etc.)

Machine Learning Methods Used
Random Forest (RF) – Chosen for its ability to handle mixed data types and provide feature importance.

Artificial Neural Networks (ANNs) – Used for capturing complex non-linear relationships in the data.

Installation & Setup
Clone this repository:

git clone https://github.com/yourusername/Obesity-Level-Prediction.git
cd Obesity-Level-Prediction
Install required R packages:
Open R and run:


install.packages(c("randomForest", "nnet", "ggplot2", "caret", "dplyr"))
Alternatively, install from requirements.R:

source("requirements.R")
Run the main script:

source("obesity_prediction.R")
Results & Insights
Feature Importance Analysis: Identifies key factors influencing obesity levels.

Model Performance Evaluation: Comparison of accuracy, precision, recall, and F1-score for RF and ANN models in R.

References
Dataset: UCI Machine Learning Repository

Random Forest in R: Liaw & Wiener, 2002

Neural Networks in R: Ripley, 1996

Contributors
Pintu Kumar Singh – MSc Data Science, University of East London

License
This project is open-source under the MIT License.
