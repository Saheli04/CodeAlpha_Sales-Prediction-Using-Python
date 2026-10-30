# 📈 Sales Prediction Using Machine Learning

## 📌 Project Overview
This project predicts product sales based on advertising expenditure across TV, Radio, and Newspaper channels using machine learning regression models. It includes data analysis, model evaluation, advertising impact analysis, and an interactive Streamlit dashboard.

## 🎯 Objectives
- Clean and prepare advertising data for machine learning.
- Explore relationships between advertising budgets and sales.
- Train and compare Linear Regression and Random Forest models.
- Evaluate models using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- Estimate sales for different advertising budget combinations.
- Present insights through an interactive dashboard.

## 🛠️ Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib

## 📂 Project Structure
```text
Sales_Prediction/
├── Advertising.csv
├── train_model.py
├── dashboard.py
├── model_results.csv
├── requirements.txt
├── README.md
└── models/
    ├── best_model.pkl
    └── features.pkl
```

## 🤖 Machine Learning Models
1. Linear Regression
2. Random Forest Regressor

The models are compared using R², MAE, RMSE, and cross-validation scores. The selected model is saved for use in the dashboard.

## 📊 Initial Model Results
| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.8994 | 1.4608 | 1.7816 |
| Random Forest | 0.9802 | 0.6625 | 0.7911 |

These results come from an initial 80/20 train-test split. Results can vary with code changes and package versions. See `model_results.csv` for the actual run's metrics.

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_REPOSITORY_URL
cd Sales_Prediction
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the models
```bash
python train_model.py
```

### 5. Launch the dashboard
```bash
streamlit run dashboard.py
```

## 💡 Business Applications
- Compare estimated sales under different advertising budgets.
- Explore which advertising channels have stronger relationships with sales.
- Support preliminary advertising budget planning.

## ⚠️ Limitations
The dataset does not contain dates, audience segments, or platform identifiers. Therefore, this project estimates sales from advertising budgets rather than forecasting a time series or comparing customer segments. Observed relationships do not establish causation.

## 👩‍💻 Author
Saheli Debnath

**Project type:** Machine Learning | Regression | Data Analytics | Business Intelligence
