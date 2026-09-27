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

## Week 7 Progress — Testing, Refinement & End-to-End Validation

**Status:** Complete

- Continued from the Week 6 candidate work: tuned Random Forest (`max_depth=10`, `min_samples_leaf=5`, `n_estimators=200`) and re-evaluated against the baseline — tuning reduced overfitting but did not close the recall gap (recall unchanged at 0.61), confirming Week 6's model-level limitation conclusion with direct evidence
- Trained and tested Gradient Boosting as a new candidate algorithm; initially appeared to beat the baseline on the test split (recall 0.64 vs 0.62), validated against predicted-vs-actual no-show rate to rule out over-flagging
- Ran a threshold sweep (0.3–0.6) on Gradient Boosting and selected 0.4 as the recommended cutoff (recall 0.81, precision 0.57 at that threshold)
- Checked Gradient Boosting's performance across Data Analytics' key segments (appointment type, reminder channel) — found weaker recall on Specialist Consultation appointments (0.72) and weaker precision on the WhatsApp reminder group (0.51)
- Conducted error analysis comparing false negatives against true positives: found first-time patients are ~4x more common among false negatives, and false negatives generally carry less prior no-show history and less extreme booking lead time — the model's two strongest signals have no data to work with for patients with no track record
- Compared Gradient Boosting's false negatives against the baseline's: 86 of 92 (93%) were shared misses, confirming this is a model-independent, data-level blind spot rather than an algorithm-choice problem; Gradient Boosting still recovered 97 patients the baseline missed while losing only 6
- **Key finding:** ran 5-fold cross-validation to test whether the recall improvement was statistically real — it was not. Baseline and Gradient Boosting are statistically tied (0.625 ± 0.013 vs 0.627 ± 0.009). **The earlier "Gradient Boosting beats baseline" claim is withdrawn.** Gradient Boosting is retained as the Week 7 candidate on the basis of more explainable errors and slightly better generalisation, not superior recall
- Re-selected the 0.4 decision threshold using out-of-fold predictions (rather than the test set) after identifying the original sweep risked being fitted to that specific split — the leakage-free result closely matched the original (recall 0.79, precision 0.58), confirming the threshold holds up
- Ran a train-vs-test overfitting check: found tuned Random Forest still carries a real train-test gap (0.74 vs 0.61 recall) despite the earlier "fixed most of the overfitting" claim; Gradient Boosting shows a smaller gap (0.71 vs 0.64) and generalises somewhat more reliably
- Corrected the age feature importance comparison: the original figure (0.202) was mistakenly taken from the untuned Random Forest; the correct tuned-model figure is 0.138, close to Gradient Boosting's 0.132 — age is confirmed as a real but more modest signal than first reported
- Collaborated with **Hudu Yusuf Ibrahim** (Data Analytics) to resolve a no-show rate discrepancy: my figure (70.5%) for a patient segment didn't match his reported figure (64.1%) for what he described as the same segment. Traced the gap to Cancelled-appointment handling and a mismatched lead-time cutoff definition — retesting with his exact "3+ weeks" (21+ days) cutoff produced 63.4%, within 0.7 points of his number, fully resolving the discrepancy
- Documented a model suitability assessment, an ML Engineering handoff spec (feature list, output format, decision threshold, known limitations), and Week 8 recommendations
- Updated limitations, risks, and dependencies to reflect all Week 7 findings, including the withdrawn recall claim and the persistent first-time-patient blind spot

## Files

- `week4/Health_connect_clinic.ipynb` — Week 4 notebook (problem definition through modelling approach)
- `week4/Health Connect Clinic.docx` — Week 4 Project Summary
- `HealthConnect_Clinic_Week5_DataScience.ipynb` — Week 5 notebook (data preparation, EDA, feature selection, feature engineering, baseline model, evaluation)
- `Week 5 Project Summary.docx` — Week 5 Project Summary
- `week6/HealthConnect_Clinic_Week6_DataScience.ipynb` — Week 6 notebook (error analysis, cross-track validation, feature refinement, model comparison, evaluation)
- `week6/Project Summary(Week 6).docx` — Week 6 Project Summary
- `week7/HealthConnect_Clinic_Week7_DataScience.ipynb` — Week 7 notebook (testing, error analysis, statistical validation, threshold correction, cross-track testing)
- `week7/PROJECT_SUMMARY_WEEK_7.docx` — Week 7 Project Summary
- `week7/week7_gradient_boosting_model.pkl` — Week 7 candidate model, saved for reuse in Week 8
- `HealthConnect_Appointment_Data.csv` — original appointment dataset, unmodified (provided by AnalystLab Africa)
- `HealthConnect_Data_Dictionary.xlsx` — variable reference for the appointment dataset
- `HealthConnect_ML_Processed.csv` — derived, modelling-ready dataset (Cancelled appointments excluded, target encoded; missing `reminder_channel` values filled as "No Reminder" for the Week 8 cross-track check with Data Analytics)
- `week6/week5_baseline_model.pkl`, `week6/week5_X_train.csv`, `week6/week5_X_test.csv`, `week6/week5_y_train.csv`, `week6/week5_y_test.csv` — Week 5 model and split artefacts, saved for reuse in Week 6
- `week6_X_train.csv`, `week6_X_test.csv`, `week6_y_train.csv`, `week6_y_test.csv` — 18-feature train/test split used across Week 7 and Week 8

**Note on structure:** Week 4, Week 6, and Week 7 deliverables are organised into their own folders. The original dataset, data dictionary, and processed dataset are shared inputs used across weeks and are kept at the repo root rather than duplicated per folder. Week 5's notebook and summary remain at the root for now and will move into a `week5/` folder once marked.

## Week 8 Progress — Final Model Selection, Documentation & Presentation

**Status:** Complete

- Set out to resolve Week 7's open question — model choice between the baseline and Gradient Boosting was still undecided, since recall alone couldn't distinguish them
- Discovered a feature mismatch that had gone unnoticed through Weeks 6–7: the Week 5 baseline was trained on 14 features, while Gradient Boosting (and the dataset from Week 6 onward) used 18. Any recall/precision gap between the two was therefore partly a feature-count confound, not a clean algorithm comparison
- Retrained the baseline Logistic Regression on the same 18-feature set as Gradient Boosting to remove that confound
- **Key finding:** once compared fairly at the matched 0.4 threshold, the earlier apparent Gradient Boosting advantage disappeared — Logistic Regression (recall 0.803, precision 0.583) and Gradient Boosting (recall 0.81, precision 0.57) were effectively tied, with differences well within normal variation
- Ran a train-vs-test overfitting check on the retrained baseline: train-test recall gap of just 0.004, roughly 17x smaller than Gradient Boosting's 0.07 — meaning the baseline's test performance is a substantially more reliable estimate of real-world behaviour
- **Decision:** selected the retrained Logistic Regression (18 features, threshold 0.4) as the final model, replacing Gradient Boosting — driven by reliability and interpretability rather than a raw performance edge, since the two were statistically tied
- Re-ran the Week 6 error analysis pattern on the final model directly rather than assuming it still held: confirmed with statistical tests that missed no-shows (n=95) book more recently (t = -17.53, p < 0.0001) and have cleaner prior-no-show histories (t = -4.04, p < 0.0001) than caught no-shows
- Formally tested the two segment weaknesses flagged in Week 7 for statistical significance rather than treating them as confirmed:
  - **Specialist Consultation recall gap (0.745 vs 0.817 overall):** not statistically significant (p = 0.111) — ruled out as sample-size noise, not a genuine model weakness
  - **WhatsApp-reminder precision gap (0.512 vs 0.605 overall):** statistically significant (p = 0.037) — confirmed as a real, unresolved weakness
- Investigated the confirmed WhatsApp gap against the model's three strongest features (`is_first_time_patient`, `prior_no_show_rate`, `booking_lead_days`) — none statistically explained the gap; logged as an open limitation with cause not yet identified
- Collaborated with four cross-track contacts on final integration:
  - **zakheni (Data Analytics):** joint investigation into the WhatsApp precision gap, ruling out three plausible causes together
  - **Claudia (Data Analytics):** resolved the Week 6 `distance_to_clinic_km` disagreement — a point-biserial correlation check (r = 0.039, p = 0.230) confirmed her no-correlation finding without contradicting this track's feature-importance result, since the two tests answer different questions
  - **Angela (Data Analytics):** tested her second, previously untested interaction suggestion (reminder channel × lead time) — direction matched her finding (21.1% vs 29.5%) but did not reach statistical significance (p = 0.147)
  - **ML Engineering:** sent the final model type, threshold, feature list, and known limitations on September 25th; pipeline-side confirmation outstanding as of this writeup
- Documented final model strengths, weaknesses, and business suitability: suitable for low-cost interventions (reminders) given precision of 0.583, not currently suitable for high-cost actions (overbooking, slot reassignment)
- Wrote a plain-language model summary for non-technical stakeholders
- Saved the final model artefact and produced a final baseline-vs-Gradient-Boosting comparison chart
- Prepared final presentation materials (slide deck) and this README update as the closing documentation deliverable

## Final Model Summary

| Metric | Value |
|---|---|
| Final model | Logistic Regression (18 features) |
| Decision threshold | 0.4 |
| Recall | 0.803 |
| Precision | 0.583 |
| Train-test recall gap | 0.004 |

**Why this model:** Gradient Boosting initially looked stronger on recall (0.64 vs 0.62 at the default threshold), but 5-fold cross-validation showed the two models were statistically tied, and the apparent edge disappeared entirely once the baseline was retrained on the same feature set. Logistic Regression was selected for its far smaller train-test gap and easier interpretability for ML Engineering and non-technical stakeholders.

## Files (Week 8)

- `HealthConnect_Clinic_Week8_DataScience.ipynb` — Week 8 notebook (final integration readiness, final model selection, error analysis, segment significance testing, business interpretation, cross-track collaboration records, limitations)
- `week8_final_model.pkl` — final selected model (Logistic Regression), saved via `joblib.dump`; apply at threshold 0.4 using `predict_proba()[:, 1]`, not `.predict()`
- `final_model_comparison.png` — Week 8 recall/precision comparison chart, Logistic Regression vs Gradient Boosting at matched threshold
- `HealthConnect_Week8_DataScience_Presentation.pptx` — final presentation deck (project journey, model comparison, key metrics, business recommendation, cross-track collaboration)

## Known Limitations

- **Performance ceiling:** roughly 4 in 10 patients flagged as likely no-shows will actually attend. Suitable for low-cost interventions (reminders); not currently suitable for high-cost actions (overbooking, slot reassignment)
- **Model choice was not a decisive win:** Logistic Regression and Gradient Boosting were statistically tied on recall and precision at matched threshold; the final call rested on reliability and interpretability, not a clear performance advantage
- **Residual train-test gap:** even the chosen model retains a small train-test gap (0.004) — some overfitting is still present
- **WhatsApp-reminder segment:** statistically significant precision drop (p = 0.037), cause not yet identified after testing the model's three strongest features
- **First-time patients:** under-flagged, since the model's strongest signals (prior history, lead time) have little to work with for these patients
- **ML Engineering pipeline integration:** final model spec sent September 25th; pipeline-side confirmation outstanding as of this writeup
- **Reminder-channel/lead-time interaction:** Angela's suggested Email-vs-SMS effect for short-notice appointments matched direction but did not reach statistical significance (p = 0.147); not yet confirmed evidence for reminder-channel strategy decisions

## Project Links

- GitHub: https://github.com/Oyxpheel/Health-Connect-Clinic
- LinkedIn: https://www.linkedin.com/in/fatimahodumuyiwa/
