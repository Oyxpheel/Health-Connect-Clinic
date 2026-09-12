# HealthConnect Clinic — No-Show Prediction

AnalystLab Africa Experience Lab | Data Science Track
**Author:** Fatimah Odumuyiwa

## Overview

HealthConnect Clinic is a fictional healthcare provider facing challenges with missed appointments, inefficient use of appointment slots, and limited insight into the factors driving no-shows. This project explores how data and machine learning can be used to predict patient no-shows, allowing the clinic to intervene proactively (e.g. targeted reminders, phone confirmations) rather than discovering missed slots after the fact.

**Central Project Question:** How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

## Track Responsibility

As the Data Science track, the responsibility for this project is to define the machine learning problem and assess whether the available appointment data can support a no-show prediction solution.

## Week 4 Progress — Problem Understanding

**Status:** Complete

- Defined the ML problem as binary classification: predicting `appointment_outcome` (Attended vs No-Show)
- Conducted an initial data quality assessment on the HealthConnect appointment dataset (5,000 records, 1,696 unique patients)
- Proposed and encoded the target variable, excluding Cancelled appointments (263 records) as a separate, patient-initiated outcome
- Identified 11 candidate input features with clinical/behavioural rationale
- Proposed an initial modelling approach: Logistic Regression baseline, 80/20 stratified split, recall/F1 as priority metrics
- Documented key modelling considerations, assumptions, limitations, risks, and dependencies

## Week 5 Progress — Data Preparation, Feature Engineering & Baseline Model

**Status:** Complete

- Prepared the dataset for modelling (4,737 records after excluding Cancelled appointments); handled missing values via median imputation (`distance_to_clinic_km`, `waiting_time_minutes`) and an explicit "No Reminder" category for `reminder_channel`
- Ran EDA across numeric and categorical variables, then confirmed relationships statistically: t-tests for numeric features, chi-square tests for categorical features against `appointment_outcome`
- Finalised the feature list based on statistical significance, dropping weak/non-significant predictors (`gender`, `age_group`, `appointment_day`, `appointment_time`, `waiting_time_minutes`)
- Engineered two new features: `prior_no_show_rate` and `is_first_time_patient`, both statistically validated
- **Change from Week 4:** the train/test approach was updated from a stratified 80/20 split to a **patient-level (grouped) 80/20 split** (`GroupShuffleSplit`), since the dataset contains multiple appointments per patient and a standard split risked patient overlap between train and test sets
- Trained a Logistic Regression baseline model (`class_weight='balanced'`)
- Evaluated the baseline: Accuracy 0.63, Precision 0.63, Recall 0.62, F1 0.62, ROC-AUC 0.68 — recall prioritised as the key metric, since missing a real no-show is costlier than a false alarm
- Shared feature list, preprocessing steps, and baseline model details with the Machine Learning Engineering track to inform their pipeline work

## Week 6 Progress — Model Improvement, Error Analysis & Validation

**Status:** Complete

- Reloaded the Week 5 baseline model and train/test split (saved separately as `.pkl`/`.csv` files) rather than re-running the Week 5 analysis, to continue from the existing validated work
- Conducted error analysis on the baseline's misclassifications: isolated false negatives (183) and false positives (178), then compared their feature values against correctly-predicted no-shows — found the model systematically misses no-shows with "safe-looking" profiles (recent booking, clean history, reminder received)
- Verified the Week 6 test set was identical to Week 5's saved test set (966/966 rows) before comparing models, confirming an apples-to-apples comparison
- Collaborated with two Data Analytics interns:
  - **Claudia:** validated her finding on `reminder_channel` and no-show rates — confirmed the direction but found a smaller effect size than reported (54.6% vs 49.9%, not near-universal); flagged a separate, unresolved disagreement on `distance_to_clinic_km`'s importance
  - **Angela:** validated her finding that booking lead time and prior no-show history compound rather than act independently; engineered and tested an interaction feature (`lead_and_prior_noshow`) based on it
- Engineered and statistically tested two interaction features: `recent_and_sms` (T = 10.500, p < 0.0001) and `lead_and_prior_noshow` (T = -14.282, p < 0.0001) — both highly significant individually
- Trained and compared a Random Forest and a refined Logistic Regression (both including the two new features) against the Week 5 baseline: **neither improved recall** (baseline 0.62 vs. 0.61 for both alternatives), despite the features' strong individual significance
- **Key finding:** statistically significant features did not translate into model improvement, since both engineered features overlapped with variables already in the model — pointing to a model-level limitation (linear structure / untuned complexity) rather than a missing-feature problem
- **Candidate model recommendation:** the original Week 5 baseline Logistic Regression, unchanged — the two tested alternatives did not outperform it
- Documented limitations, risks, and dependencies (resolved/unresolved issues, new issues, technical/data/modelling limitations, integration challenges, and recommended mitigations), and defined Week 7 testing requirements

## Files

- `week4/Health_connect_clinic.ipynb` — Week 4 notebook (problem definition through modelling approach)
- `week4/Health Connect Clinic.docx` — Week 4 Project Summary
- `HealthConnect_Clinic_Week5_DataScience.ipynb` — Week 5 notebook (data preparation, EDA, feature selection, feature engineering, baseline model, evaluation)
- `Week 5 Project Summary.docx` — Week 5 Project Summary
- `week6/HealthConnect_Clinic_Week6_DataScience.ipynb` — Week 6 notebook (error analysis, cross-track validation, feature refinement, model comparison, evaluation)
- `week6/Project Summary(Week 6).docx` — Week 6 Project Summary
- `HealthConnect_Appointment_Data.csv` — original appointment dataset, unmodified (provided by AnalystLab Africa)
- `HealthConnect_Data_Dictionary.xlsx` — variable reference for the appointment dataset
- `HealthConnect_ML_Processed.csv` — derived, modelling-ready dataset (Cancelled appointments excluded, target encoded)
- `week6/week5_baseline_model.pkl`, `week6/week5_X_train.csv`, `week6/week5_X_test.csv`, `week6/week5_y_train.csv`, `week6/week5_y_test.csv` — Week 5 model and split artefacts, saved for reuse in Week 6

**Note on structure:** Week 4 and Week 6 deliverables are organised into their own folders. The original dataset, data dictionary, and processed dataset are shared inputs used across weeks and are kept at the repo root rather than duplicated per folder. Week 5's notebook and summary remain at the root for now and will move into a `week5/` folder once marked.

## Next Steps (Week 7)

Tune the Random Forest's hyperparameters and re-compare against the baseline, test alternative decision thresholds to prioritise recall, investigate the age feature's unexpectedly high importance, test Angela's reminder-channel × lead-time interaction suggestion, and resolve the open distance_to_clinic_km disagreement with Claudia.
