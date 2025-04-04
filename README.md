# 🌾 Agriculture Yield Prediction Web App

## 📌 Description

The **Agriculture Yield Prediction Web App** is a machine learning-powered tool that predicts the expected crop yield (tons per hectare) based on key inputs such as crop type, temperature, rainfall, and pesticide usage. Built using Streamlit, it helps farmers, researchers, and policymakers make informed decisions about crop planning and resource allocation.

---

## 🚀 Live Demo

👉 **[Click Here to Use the App](https://btop6kdb68bjhqupheccjm.streamlit.app/)**  
*(Replace the link with your deployed Streamlit app URL.)*

---

## ✨ Features

- **Instant Yield Prediction:** Enter inputs to get real-time crop yield predictions.
- **User-friendly Interface:** Designed with an intuitive Streamlit UI.
- **Model Retraining:** Option to retrain the model with new data.
- **Batch Predictions:** Supports CSV input for multiple predictions.
- **Robust ML Models:** Built with Random Forest Regression, Gradient Boosting, and Linear Regression.

---

## 📊 Dataset Overview

| Feature Name         | Description                                          |
|----------------------|------------------------------------------------------|
| **Farm_ID**          | Unique identifier (dropped during preprocessing)     |
| **Crop_Type**        | Type of crop grown (categorical)                      |
| **Soil_Type**        | Soil classification (categorical)                     |
| **Irrigation_Type**  | Irrigation method used (categorical)                  |
| **Season**           | Season during cultivation (categorical)               |
| **Farm_Area**        | Farm size (in acres)                                  |
| **Fertilizer_Used**  | Amount of fertilizer used (in tons)                   |
| **Pesticide_Used**   | Pesticide quantity (in kg)                            |
| **Water_Usage**      | Water consumption (in cubic meters)                   |
| **Yield**            | Crop yield (target variable, in tons)                 |

---

## 🔍 Exploratory Data Analysis (EDA)

- **Missing Value Checks:** Identified and handled missing data.
- **Categorical Analysis:** Used `value_counts` for class distribution.
- **Numerical Analysis:** Visualized data with histograms and boxplots.
- **Outlier Detection:** Employed the IQR method to detect outliers.
- **Correlation Analysis:** Created heatmaps to identify multicollinearity.
- **Visual Insights:** Pairplots and distribution plots provided additional insights.

---

## 🔧 Data Preprocessing

- **Dropped Columns:** Removed `Farm_ID` as it is non-informative.
- **Encoding:** Applied `LabelEncoder` on categorical variables:
  - `Crop_Type`
  - `Soil_Type`
  - `Irrigation_Type`
  - `Season`
- **Scaling:** Standardized numerical features using `StandardScaler`.
- **Train-Test Split:** Data split into an 80/20 ratio for training and testing.

---

## 🤖 Machine Learning Models

### ✅ Models Evaluated

- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

### 📈 Evaluation Metrics

- **R² Score**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

---

## 🛠️ Hyperparameter Tuning

Used **GridSearchCV** to optimize model performance:

- **Random Forest Regressor:**
  ```python
  {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 10
  }

## 🔁 Cross Validation
Performed 5-Fold Cross-Validation to ensure model robustness and prevent overfitting.

---

## 🏆 Final Model Performance (After Tuning)
| Model                         | R² Score | RMSE  | MAE   |
|-------------------------------|----------|-------|-------|
| Random Forest Regressor       | -0.00    | 12.64 | 10.52 |
| Gradient Boosting Regressor   | -0.10    | 13.23 | 11.42 |

**Note:** The low R² scores suggest there is room for improvement. This could be due to data limitations, noise, or the need for additional features.

---

## 📦 Output Artifacts
- **random_forest_agri_model.pkl:** Final trained Random Forest model.
- **scaler_agri.pkl:** Fitted StandardScaler for numerical features.
- **Label Encoders:**
  - crop_encoder.pkl
  - soil_encoder.pkl
  - irrigation_encoder.pkl
  - season_encoder.pkl

---

## 📚 Tools & Libraries
- **Python**
- **Pandas & NumPy** – Data manipulation
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning, preprocessing, and evaluation
- **Statsmodels** – For VIF analysis to check multicollinearity
- **Pickle** – Model serialization
- **Streamlit** – Web app framework

---

## 📬 Contact
- **Author:** Yash
- **[Email](yd811822@gmail.com)** 
- **[LinkedIn](https://www.linkedin.com/in/yashcoding/)** 

---

## ✅ Lessons Learned
- **Feature Engineering:** Effective encoding and scaling significantly impact model performance.
- **Modeling Insights:** Even slight performance differences can highlight potential areas for improvement.
- **Data Visualization:** Critical for identifying outliers, skewness, and feature correlations.
- **Domain Expertise:** In-depth agricultural knowledge can drive better feature selection.
- **Model Evaluation:** Cross-validation and hyperparameter tuning are key to enhancing generalizability.

---

### Streamlit App UI
![Image](https://github.com/user-attachments/assets/6c3754b9-2ff1-4586-9e59-d3a5ffd73383)
