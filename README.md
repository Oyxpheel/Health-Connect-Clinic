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

## Files

- `Health_connect_clinic.ipynb` — Week 4 notebook (problem definition through modelling approach)
- `Health Connect Clinic.docx` — Week 4 Project Summary
- `HealthConnect_Clinic_Week5_DataScience.ipynb` — Week 5 notebook (data preparation, EDA, feature selection, feature engineering, baseline model, evaluation)
- `HealthConnect_Appointment_Data.csv` — original appointment dataset, unmodified (provided by AnalystLab Africa)
- `HealthConnect_Data_Dictionary.xlsx` — variable reference for the appointment dataset
- `HealthConnect_ML_Processed.csv` — derived, modelling-ready dataset (Cancelled appointments excluded, target encoded)

## Next Steps (Week 6)

Test a tree-based model (e.g. Random Forest) to check for non-linear patterns the Logistic Regression baseline may miss, review feature importance to prune weak predictors, explore feature interactions (e.g. `booking_lead_days` × `previous_no_shows`), and tune specifically for recall given its priority for this business problem.
