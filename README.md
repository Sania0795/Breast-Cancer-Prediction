# Breast Cancer Classification

This project applies **Logistic Regression** and **Random Forest** to a **Breast Cancer dataset** to classify tumors as **Malignant (M)** or **Benign (B)**.  

The workflow includes **data preprocessing, visualization, model training, evaluation**, and **predictions**.

## Dataset

- **Source**: Breast Cancer Wisconsin Dataset  
- **Samples**: 569 rows  
- **Features**: 30 numeric tumor features (e.g., radius, texture, smoothness)  
- **Target column**: `diagnosis`  
  - `M` → Malignant (cancer present)  
  - `B` → Benign (no cancer)  

## Project Workflow

### 1. Data Preprocessing
- Handled missing values  
- Encoded categorical column (`diagnosis`)  
- One-hot encoded other object columns  
- Scaled numeric features using **StandardScaler**

### 2. Exploratory Data Analysis (EDA)
- Histograms of numeric features  
- Correlation heatmap of features  
- Feature importance (Random Forest)

### 3. Model Training
- **Logistic Regression**
- **Random Forest Classifier** (100 trees)

Both models were trained with an 80/20 train-test split.

### 4. Model Evaluation
- **Accuracy score**
- **Classification report** (Precision, Recall, F1)  
- **Confusion matrix** (visualized with Seaborn)  
- **Top 10 important features** (Random Forest)

### 5. Model Deployment
- Trained models were saved with `joblib`:
  - `breast_cancer_rf_model.pkl`
  - `breast_cancer_scaler.pkl`
- A **user template CSV** was created for predictions (`breast_cancer_user_template.csv`).  
- New input data is preprocessed, scaled, and predictions are added as a new column:  
  - `Breast_Cancer_Prediction` → `1 = Malignant`, `0 = Benign`

## Example Results

- **Logistic Regression Accuracy**: ~95%  
- **Random Forest Accuracy**: ~97%  

Confusion Matrix (Logistic Regression)

```
[70,  2],
[ 3, 39]
```

Random Forest Feature Importance (Top 10):  
- worst concave points 
- mean perimeter 
- mean concave points 
- ...

## Files in Repository
breast-cancer.csv → dataset

breast_cancer_rf_model.pkl → saved Random Forest model

breast_cancer_scaler.pkl → saved StandardScaler

breast_cancer_classification.ipynb → Jupyter notebook containing the full code

## Conclusion
Random Forest performed slightly better than Logistic Regression.

Feature importance highlights which tumor features are most predictive.

The pipeline supports reproducible predictions on new patient data.
