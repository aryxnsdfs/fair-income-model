# Fairness-Aware Income Prediction

This project uses the Adult Census Income dataset to train a logistic regression model and evaluate group fairness by gender using `fairlearn`.

## Key Features
- Logistic Regression on Adult dataset
- Fairness analysis with `fairlearn`
- Visualized group metrics and model coefficients
  
## Output
Bar plot comparing accuracy and selection rate for each gender

Feature importance for top predictors

Demographic parity difference printed in console

## Setup

```bash
pip install -r requirements.txt
python fair_income_model.py

