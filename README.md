# Medical Insurance Price Predictor

A machine learning project that estimates medical insurance charges from personal and lifestyle information. The project includes data cleaning, exploratory data analysis (EDA), feature engineering, model comparison, and a Streamlit web app for interactive predictions.

> **Note:** This project is for educational and demonstration purposes only. Predictions are estimates, not medical, financial, or insurance advice.

## Project Overview

The goal is to predict an individual's medical insurance charges using these input features:

- Age
- Sex
- Body Mass Index (BMI)
- Number of children
- Smoking status
- Region
- BMI category (derived from BMI)

The notebook compares Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor. In the reported experiment, **Random Forest performed best** among the three models.

## Features

- Loads and inspects the medical insurance dataset
- Checks missing values and removes duplicate records
- Performs exploratory data analysis with Pandas, Matplotlib, and Seaborn
- Creates BMI categories: underweight, normal, overweight, and obese
- Applies one-hot encoding to categorical variables
- Uses `log1p` transformation on insurance charges to reduce target skewness
- Trains and evaluates multiple regression models
- Saves the selected model and feature columns with Joblib
- Provides an interactive Streamlit interface for entering details and estimating charges
- Displays the predicted annual charge and a comparison chart

## Model Results

The notebook reports the following evaluation results on its train/test split:

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Random Forest Regressor | 0.8855 | 1,829.77 | 4,010.76 |
| Gradient Boosting Regressor | 0.8761 | 1,885.96 | 4,173.13 |
| Linear Regression | 0.5023 | 4,220.19 | 8,362.57 |

These are the results recorded in the notebook and may vary when the model is retrained, particularly if the train/test split is randomized.

## Tech Stack

- **Python**
- **Pandas** — data manipulation and cleaning
- **NumPy** — numerical operations and inverse log transformation
- **Matplotlib & Seaborn** — visualizations and EDA
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Joblib** — saving and loading trained model artifacts
- **Streamlit** — interactive web application
- **Jupyter Notebook** — experimentation and model development

## Repository Structure

```text
medical-insurance-price-predictor/
├── app.py
├── Insurance Cost Prediction Project.ipynb
├── medical_insurance.csv
├── insurance_model.pkl
├── model_features.pkl
├── model_name.pkl                 # optional, if included in your project
├── requirements.txt
└── README.md
```

Make sure the filenames in your repository match the filenames used in `app.py`. The notebook shown in this project exports `model_feature.pkl` (singular), while the Streamlit code may expect `model_features.pkl` (plural). Use one consistent filename in both files before running the app.

## Getting Started

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR-USERNAME/medical-insurance-price-predictor.git
cd medical-insurance-price-predictor
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Data and Preprocessing

The notebook loads `medical_insurance.csv` and examines the dataset's shape, data types, descriptive statistics, missing values, duplicates, category distributions, and numerical distributions.

The notebook reports 2,772 rows and 7 columns before duplicate removal, then 1,337 rows after removing duplicate records. It derives BMI categories and applies one-hot encoding to `sex`, `smoker`, `region`, and `bmi_category`. The target is transformed with:

```python
df["log_charges"] = np.log1p(df["charges"])
```

Predictions are converted back to the original charge scale with `np.expm1()` in the app.

## Model Evaluation

The notebook compares models using:

- **R² score** — indicates how much variation in the target is explained by the model.
- **MAE (Mean Absolute Error)** — average absolute prediction error.
- **RMSE (Root Mean Squared Error)** — penalizes larger errors more strongly than MAE.

The selected model is the model with the highest R² in the notebook's comparison.

## Important Compatibility Note

The notebook records these versions in its training environment:

- Python via Anaconda
- Scikit-learn `1.2.2`
- NumPy `1.26.4`

The Streamlit app should use compatible dependencies with the environment that created the `.pkl` model. Scikit-learn does not guarantee that pickle/joblib model files can be loaded across different scikit-learn versions. If you retrain the model, save the model artifacts again and keep the app's feature names in sync.

## Future Improvements

- Add cross-validation and a reproducible `random_state`
- Tune model hyperparameters
- Add feature-importance visualizations
- Improve input validation and error handling
- Deploy the Streamlit app
- Add automated tests for preprocessing and prediction
- Track model versions and dependency versions

## Disclaimer

This project is intended for learning and portfolio demonstration. It estimates charges from patterns in the supplied dataset and should not be used to make real-world insurance or healthcare decisions.

## Author

**Siddharth Kanodia**

If you found this project useful, feel free to star the repository.
