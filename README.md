# FinTrust Data Science Track

This project finalizes the FinTrust predictive risk model for the Week 4 assignment. It combines the customer and transaction datasets, engineers a final feature set, compares candidate models, selects the final model, and exports reproducible outputs.

## Project structure

- `customer.csv` — customer master data
- `transactions.csv` — transaction records with the synthetic `Risk_Review_Flag` label
- `src/fintrust_risk_model.py` — end-to-end modelling pipeline
- `outputs/final_modelling_dataset.csv` — final modelling dataset used for training and evaluation
- `outputs/model_metrics.json` — model comparison metrics and final selection data
- `outputs/confusion_matrix.png` — confusion matrix for the final model
- `docs/final_data_science_report.md` — final technical report for the Data Science track

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the final model pipeline

```bash
python src/fintrust_risk_model.py
```

## Important note

`Risk_Review_Flag` is a synthetic educational target, not evidence of actual fraud or financial crime. The project is intended for learning and analytical experimentation only.
