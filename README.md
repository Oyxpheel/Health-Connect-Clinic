# HealthConnect Clinic — No-Show Prediction

AnalystLab Africa Experience Lab | Data Science Track
**Author:** Fatimah Odumuyiwa

## Overview

HealthConnect Clinic is a fictional healthcare provider facing challenges with missed appointments, inefficient use of appointment slots, and limited insight into the factors driving no-shows. This project explores how data and machine learning can be used to predict patient no-shows, allowing the clinic to intervene proactively (e.g. targeted reminders, phone confirmations) rather than discovering missed slots after the fact.

**Central Project Question:** How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

## Track Responsibility

As the Data Science track, the responsibility for this project is to define the machine learning problem and assess whether the available appointment data can support a no-show prediction solution.

## Week 4 Progress — Problem Understanding

**Status:** In progress

- Defined the ML problem as binary classification: predicting `appointment_outcome` (Attended vs No-Show)
- Conducted an initial data quality assessment on the HealthConnect appointment dataset (5,000 records, 1,696 unique patients)
- Proposed and encoded the target variable, excluding Cancelled appointments (263 records) as a separate, patient-initiated outcome
- Identified 11 candidate input features with clinical/behavioural rationale
- Proposed an initial modelling approach: Logistic Regression baseline, 80/20 stratified split, recall/F1 as priority metrics
- Documented key modelling considerations, assumptions, limitations, risks, and dependencies

## Files

- `HealthConnect_Week4_DataScience_Fatimah.ipynb` — Week 4 notebook (problem definition through modelling approach)
- `Week4_Project_Summary_Fatimah.docx` — concise Week 4 summary and Week 5 focus
- `HealthConnect_Appointment_Data.csv` — appointment dataset (provided by AnalystLab Africa)

## Next Steps (Week 5)

Statistically test candidate features against `appointment_outcome`, finalise the feature list based on this analysis, and begin initial model training using the Logistic Regression baseline.
