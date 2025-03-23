Estimation of Obesity Levels Based on Lifestyle
📌 Introduction
This project aims to predict obesity levels based on demographic, lifestyle, and health-related factors using machine learning techniques. The dataset, sourced from the UCI Machine Learning Repository, contains a mix of numerical and categorical features related to diet, physical activity, and personal attributes. The goal is to compare the performance of Random Forest and Artificial Neural Networks (ANNs) to determine the most effective approach for obesity prediction.

📂 Dataset Overview
Source: UCI Machine Learning Repository

Size: 1345 instances with 17 features and 1 target variable

Attributes:

Demographic: Age, Gender, Height, Weight

Lifestyle: Physical Activity, Daily Water Consumption, Smoking, Alcohol Consumption, Mode of Transportation

Dietary: Frequency of High-Caloric Food Consumption, Vegetable Consumption, Meal Frequency

Target Variable: Obesity Level (Seven categories from Insufficient Weight to Obese Type III)

🛠️ Methodology
Data Preprocessing

Handled missing values using mean/mode imputation

Encoded categorical variables

Normalized numerical features using min-max scaling

Model Building

Random Forest:

Trained using randomForest with 500 trees

Hyperparameter tuning with cross-validation

Outputs feature importance for interpretability

Artificial Neural Network (ANN):

Built using nnet package

Single hidden layer with 10 neurons

300 epochs with weight decay to prevent overfitting

Model Evaluation and Comparison

Performance evaluation based on accuracy, precision, recall, and F1-score

Random Forest: 94.74% accuracy

ANN: 90.98% accuracy

Feature importance analysis identified Weight, Height, and Age as the top predictors

💻 Technologies Used
Language: R

Libraries: caret, randomForest, nnet, ggplot2, dplyr, corrplot, doParallel

🚀 Results and Insights
✅ Random Forest achieved higher accuracy and interpretability
✅ ANN demonstrated strong predictive power but required more tuning
✅ BMI and Weight showed the strongest correlation with obesity levels
✅ Class imbalance was addressed through oversampling
