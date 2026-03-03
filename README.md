# Medical Insurance Cost Prediction 🏥

Machine Learning model to predict medical insurance costs based on patient demographics and health factors.

## Dataset

- **Size:** 2,772 records
- **Features:** 6 (Age, Sex, BMI, Children, Smoker, Region)
- **Target:** Insurance Charges
- **Cost Range:** $1,121.87 - $63,770.43
- **Average Cost:** $13,261.37

## Model Results

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.7399 | $6,318.42 | $4,167.30 |
| **Random Forest** | **0.9507** | **$2,750.00** | **$1,306.31** |
| Support Vector Regression | -0.0657 | $12,789.45 | $8,297.13 |

## 🏆 Best Model: Random Forest

- **R² Score:** 0.9507 (95.07% Accuracy!)
- **RMSE:** $2,750.00
- **MAE:** $1,306.31

## Sample Predictions

| Sample | Actual Cost | Predicted Cost | Error |
|--------|------------|-----------------|-------|
| 1 | $8,988.16 | $9,350.00 | $361.84 |
| 2 | $28,101.33 | $28,181.66 | $80.33 |
| 3 | $12,032.33 | $12,696.98 | $664.66 |
| 4 | $1,682.60 | $1,660.20 | $22.40 |
| 5 | $3,393.36 | $5,213.90 | $1,820.54 |
| 6 | $24,106.91 | $24,173.64 | $66.73 |
| 7 | $4,746.34 | $4,810.77 | $64.43 |
| 8 | $47,269.85 | $47,354.66 | $84.80 |
| 9 | $8,556.91 | $8,660.23 | $103.33 |
| 10 | $2,639.04 | $3,403.47 | $764.42 |

## Features

- **Age:** Patient age
- **Sex:** Male/Female
- **BMI:** Body Mass Index
- **Children:** Number of dependents
- **Smoker:** Smoking status (Yes/No)
- **Region:** Geographic region (Northeast, Northwest, Southeast, Southwest)

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib