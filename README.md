# Crop Yield Prediction Using Machine Learning

## Overview

This project develops a machine learning pipeline to predict agricultural crop production using Malaysian agricultural and climate datasets.
The objective of this project is to explore how machine learning techniques can be applied to estimate crop yield based on historical production data and environmental factors such as temperature, rainfall, humidity, and cultivation area.

This project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model development, evaluation, and optimisation.

---

## Objectives

- Analyse Malaysian crop production trends using historical datasets.
- Integrate agricultural and climate-related datasets.
- Develop regression models to predict crop yield.
- Compare different machine learning algorithms.
- Identify the most suitable model based on evaluation metrics.

---

## Dataset

The datasets used in this project are obtained from OpenDOSM.

### Agricultural Dataset
- Crop district area data
- Crop district production data

### Weather Dataset
Includes environmental features:
- Mean temperature
- Rainfall volume
- Relative humidity

---

## Machine Learning Workflow

### 1. Data Preprocessing

Performed data cleaning and preparation:
- Handling missing values
- Standardising dataset formats
- Filtering relevant years
- Merging agricultural and weather datasets
- Feature selection

### 2. Feature Engineering

Generated meaningful features including:
- Cultivated area
- Weather conditions
- Historical production patterns

### 3. Model Development

The following regression models were trained and evaluated:

- Linear Regression
- K-Nearest Neighbours Regression
- Support Vector Regression (SVR)
- Random Forest Regression
- Gradient Boosting Regression

### 4. Model Optimisation

Hyperparameter tuning was performed using:

- GridSearchCV
to improve model performance and identify optimal parameters.

---

## Model Evaluation

Models were evaluated using:

- Mean Squared Error (MSE)
- R² Score
- Root Mean Squared Error (RMSE)

Example evaluation:

| Model | MSE | R² Score |
|------|------|------|
| Random Forest | 0.00112 | 0.266 |
| Gradient Boosting | Best performance | - |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Google Colab

---

## Future Improvements

- Incorporate additional climate variables.
- Apply deep learning approaches such as neural networks.
- Develop a web dashboard for crop prediction.
- Include more recent agricultural datasets.

---

## Author

Nur Sahira Batrisyia  
As part of group project,
WIA1006: Machine Learning

Bachelor of Computer Science (Artificial Intelligence)  
Universiti Malaya

(September 2025)
