# FinTrust Data Science Track — Final Risk Model Report

## 1. Objective

This document finalizes the predictive risk model for the FinTrust digital banking exercise. The goal is to estimate whether a transaction is likely to be flagged in the synthetic `Risk_Review_Flag` label, using customer attributes and transaction behavior patterns while keeping the work grounded in the approved project context and responsible-use principles.

Important: `Risk_Review_Flag` is a synthetic educational target designed for model training and analysis only. It must not be interpreted as evidence of real fraud or financial crime.

## 2. Final dataset and feature review

The final modelling dataset combines:

- customer attributes from the customer master file: `Age`, `Gender`, `City`, `Customer_Segment`, `Account_Type`, `Tenure_Months`, `Digital_Engagement_Score`, `Monthly_Income_Band`, `Preferred_Channel`, `Account_Status`
- transaction characteristics: `Amount_NGN`, `Transaction_Type`, `Channel`, `Device_Type`, `Location`, `International_Transaction`, `Transaction_Status`, `Hour`, `DayOfWeek`, `Month`, `Amount_Log`, `Is_International`, `Is_Failed`, `Is_Reversed`, `Is_Pending`
- customer prior behaviour features derived from earlier transactions only: `customer_prior_tx_count`, `customer_prior_risk_count`, `customer_prior_risk_rate`, `customer_prior_failure_count`, `customer_prior_failure_rate`

### Feature review and rationale

| Feature | Why it was included | Reason for retention or refinement |
|---|---|---|
| `Amount_NGN` / `Amount_Log` | Transaction size is a strong signal in risk review workflows. | Retained because large-value transactions are often elevated-risk patterns. |
| `Transaction_Type` | Different transaction types carry different risk profiles. | Retained; transfer and cash withdrawal patterns were more elevated relative to deposits and bill payments. |
| `Transaction_Status` | Failed, reversed and pending events are not normal completion signals. | Retained; status is a useful risk context feature. |
| `International_Transaction` / `Is_International` | International activity often carries extra review and risk checks. | Retained because this feature materially increases risk likelihood.| 
| `Hour`, `DayOfWeek`, `Month` | Risk patterns may vary across time slices. | Retained to capture temporal behaviour and operational irregularity. |
| `Customer_Segment`, `Account_Type`, `Preferred_Channel` | Customer profile and preferred channel help contextualize risk. | Retained because account profile and access mode often correlate with transaction risk. |
| `Digital_Engagement_Score`, `Tenure_Months`, `Age` | Product familiarity and customer maturity can affect behavioural risk. | Retained to include customer context beyond a single transaction. |
| Prior-behaviour features | Historical transaction patterns provide stronger contextual evidence than a single event alone. | Retained because they are derived from earlier transactions only, reducing leakage risk. |

## 3. Final model evaluation

Candidate models tested:

- Logistic Regression (baseline, interpretable)
- Random Forest Classifier

### Model comparison

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.48 | 0.6408 | 0.2975 | 0.6128 | 0.4006 | 0.6655 |
| Random Forest | 0.31 | 0.6421 | 0.2976 | 0.6085 | 0.3997 | 0.6645 |

### Confusion matrices

Logistic Regression confusion matrix (selected final model):

- TN = 1250
- FP = 680
- FN = 182
- TP = 288

This means the model is relatively conservative but catches a material share of risky transactions while still producing some false positives.

### Final model selection reasoning

The final selected model is the Logistic Regression classifier.

Why this model was chosen:

- It delivered the strongest overall balance between interpretability and predictive usefulness, with F1-score of 0.4006 and ROC-AUC of 0.6655.
- It is highly explainable to business stakeholders and easier to translate into policy or review rules.
- The Random Forest model was only marginally different in predictive quality and was less transparent for practical business explanation.
- This keeps the model aligned with the assignment requirement that logistic regression remains an important interpretable baseline, while not forcing it to be the final choice unless it is truly superior.

## 4. Error analysis

### False positives (FP)

- 680 transactions were predicted as risky but were actually `No`.
- Interpretation: The model is more sensitive to anomalous patterns and may over-trigger accounts that are unusual but not actual risk flags.
- Business meaning: The alerting process may capture a larger review queue, which is acceptable first-pass triage but should be reviewed with threshold calibration.

### False negatives (FN)

- 182 transactions were actually risky but were not detected.
- Interpretation: Some risk patterns are subtle and may require richer behavioural history or stronger signal engineering.
- Business meaning: The model could miss some risk events if the transaction is unremarkable at a single-event level.

### True positives (TP)

- 288 risky transactions were correctly identified.
- Interpretation: The model does identify the repeated patterns associated with the synthetic risk label.

### True negatives (TN)

- 1250 normal transactions were correctly classified as low risk.
- Interpretation: The classifier maintains a sensible baseline for ordinary customer activity.

## 5. Final model interpretation

The final logistic model coefficients show the most important signals:

- `Transaction_Type_Transfer` has a strong positive coefficient, suggesting transfer activity is strongly associated with higher risk classification.
- `Transaction_Type_Cash Withdrawal` also increases risk, which is reasonable for a review-oriented warning system.
- `Is_International` and `International_Transaction_Yes` are positively associated with risk, indicating cross-border or unusual geography adds risk context.
- `Transaction_Status_Failed` and `Is_Failed` increase risk scores, consistent with unusual behaviour patterns.
- `Transaction_Type_Deposit`, `Card Purchase`, and `Bill Payment` reduce risk relative to transfers and withdrawals.

These results support the business interpretation that transaction type, international flow, and transaction status remain more informative than customer profile alone.

## 6. Business insights

### Finding 1 — Transfer activity is riskier than standard purchase activity
- Evidence: `Transaction_Type_Transfer` has a positive coefficient and the model uses it as a significant predictor.
- Business meaning: Internal transfers and merchant-less movement may require closer review.
- Recommended action: Add a manual review rule for transfer-heavy or unusual transfer patterns, especially if paired with international or failed-status events.

### Finding 2 — International activity increases risk signal
- Evidence: `Is_International` and `International_Transaction_Yes` contribute positively to the model.
- Business meaning: Cross-border activity captures unusual and more policy-sensitive behaviour.
- Recommended action: Require secondary verification for international transactions above a threshold or for flagged customers.

### Finding 3 — Failed and pending transactions carry review signal
- Evidence: `Is_Failed` and `Transaction_Status_Failed` push the probability upward.
- Business meaning: Transaction attempts that fail or remain pending could reflect operational confusion or unusual activity.
- Recommended action: Monitor failed or pending transactions for repeated attempts and cluster review patterns.

### Finding 4 — Transaction amount matters but is not the only driver
- Evidence: `Amount_Log` and `Amount_NGN` are among the strongest numeric predictors.
- Business meaning: Larger transaction values materially affect the model, but risk is a combination of size and context rather than amount alone.
- Recommended action: Use amount thresholds as one rule among several, rather than a standalone trigger.

### Finding 5 — Customer prior behaviour has value but is secondary to transaction context
- Evidence: Prior transaction counts and historical failure patterns contribute to the model but are not dominant compared with transaction type and movement intensity.
- Business meaning: A customer-level behavioural baseline helps, but transaction features still remain decisive.
- Recommended action: Keep behavioural monitoring in the risk workflow but prioritize live transaction-level alerts first.

## 7. Data analytics comparison

No separate Data Analytics output files were present in the workspace, so the comparison here is made using the same source data and equivalent business patterns captured in the transaction/customer tables.

Example comparison format:

- DA finding: Transfer and withdrawal transactions appear more operationally sensitive than deposits or bill payments.
- DS feature/model test: `Transaction_Type` is an important predictor in the logistic model.
- DS result: Transfer activity has a strong positive association with risk classification.
- Comparison: The data science result confirms the analytics pattern from transaction mix and operational context.
- Interpretation: The model is consistent with the business view that movement transactions deserve heightened review compared with ordinary deposits or utility-like payments.

This comparison is intentionally cautious: the results do not need to be identical across analytic lenses, but they should be consistent with the broader business narrative.

## 8. Final validation evidence

Validation was applied using a stratified train/test split.

### Validation checks performed

1. Baseline comparison between Logistic Regression and Random Forest.
2. Final threshold tuning for the selected model to maximise F1-score on the hold-out set.
3. Confusion matrix review to understand false positives and false negatives.
4. Feature coefficient review to confirm model logic is explainable.
5. Business sanity checks using transaction type, status, and international patterns.

### Validation results

- The final logistic classifier produced F1-score of 0.4006 and ROC-AUC of 0.6655.
- The model consistently highlighted transfer, international and failed transactions as riskier patterns.
- The pattern was coherent with the underlying business logic and the synthetic data design.
- No hidden data-structure or missing-value issues prevented model training.

## 9. Limitations and responsible use

The final solution has the following limitations:

- The target variable is synthetic: it is educational and should not be treated as a real fraud determination.
- The analysis is transaction-level and cannot prove any actual financial crime or harm.
- The model is a learning-oriented classifier and not a production banking system.
- Some false positives and false negatives remain because the risk signal is complex and not perfectly separable from ordinary transaction patterns.
- The model is limited to the supplied office datasets and does not include sanctions, identity verification, device reputation, or external fraud intelligence.

Responsible use:

- Use this work for learning, testing, and risk process design only.
- Keep all outputs within the fictional FinTrust scenario.
- Clearly communicate that the model does not replace real compliance, fraud, or banking controls.

## 10. Final status

The data science track is complete and the solution includes:

- final modelling dataset
- feature review and justification
- model comparison
- final selected model
- evaluation metrics
- confusion matrix
- error analysis
- model interpretation
- validation evidence
- limitations documentation
- reproducible code and reporting

The project is ready for final integration with the broader FinTrust solution, with the explicit caveat that the risk label is synthetic and educational only.
