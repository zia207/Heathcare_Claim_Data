# Healthcare Claims Data Science — Python Tutorial

A practical, end-to-end Python tutorial covering **healthcare claims data**, **patient classification systems**, and **financial risk modeling** for Medicare Advantage and value-based care programs.

---

## Table of Contents

1. [Overview](#overview)
2. [Tutorial Structure](#tutorial-structure)
3. [What You Will Learn](#what-you-will-learn)
4. [Getting Started](#getting-started)
5. [Notebooks](#notebooks)
6. [Key Concepts Covered](#key-concepts-covered)
7. [Dataset](#dataset)
8. [Dependencies](#dependencies)

---

## Overview

This repository provides a structured, hands-on Python tutorial for data scientists, analysts, and clinical informaticists working with healthcare claims data. The material spans foundational concepts through production-grade analytical workflows used by Medicare Advantage health plans.

**Domain**: Healthcare Analytics · Medicare Advantage · Value-Based Care  
**Audience**: Data Scientists, Healthcare Analysts, Clinical Informaticists  
**Level**: Intermediate Python / Beginner-to-Intermediate Healthcare Domain  
**Tools**: Python, pandas, scikit-learn, matplotlib, seaborn

---

## Tutorial Structure

Work through the three notebooks in this order:

```
Part 1 ── patient_classification_tools.ipynb
          Learn what healthcare claims contain and how patients are classified
              ↓
Part 2 ── ds_responsibilities_claims_classification.ipynb
          Understand the full scope of a data scientist's role in healthcare
              ↓
Part 3 ── healthcare_claims_workflow.ipynb
          Build an end-to-end pipeline with real Python code
```

---

## What You Will Learn

| Topic | Notebook |
|---|---|
| What claim data is and why it matters | Part 1 |
| ICD-10, CPT, HCPCS, DRG, HCC coding systems | Part 1 |
| APR-DRG severity & risk of mortality levels | Part 1 |
| Charlson Comorbidity Index (CCI) — code it yourself | Part 1 |
| LACE Index for 30-day readmission risk | Part 1 |
| RBRVS relative value units and Medicare payments | Part 1 |
| DSM-5 and mental health classification | Part 1 |
| Data scientist's end-to-end responsibilities | Part 2 |
| Feature engineering on claims (PMPM, HCC, comorbidities) | Part 2 & 3 |
| Risk stratification tiers and disease registries | Part 2 & 3 |
| Financial risk model design (prospective cost prediction) | Part 2 & 3 |
| Value-based care / ACO shared savings analytics | Part 2 & 3 |
| HEDIS-like quality measure calculation | Part 3 |
| Executive dashboard with matplotlib | Part 3 |
| Actionable KPI summary for leadership | Part 3 |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Heathcare_Claim_Data.git
cd Heathcare_Claim_Data
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open the notebooks in order (Part 1 → 2 → 3).

---

## Notebooks

### Part 1 — `patient_classification_tools.ipynb`
**Patient Classification Tools in Healthcare**

Conceptual and code-based introduction to every major classification system used in the US healthcare system.

**Sections:**
- What is claim data and how does it flow through the system?
- Diagnosis-based tools: ICD-10, DRG/MS-DRG, HCC
- Procedure-based tools: CPT, HCPCS, ICD-10-PCS
- Severity & acuity: APR-DRG, APACHE, SOFA, ASA
- Ambulatory tools: APC, RBRVS, ACG
- Post-acute tools: RUG, PDPM, OASIS
- Mental health: DSM-5, GAF
- Risk stratification: Charlson Comorbidity Index, LACE Index
- Hands-on: implement CCI and LACE calculators in Python
- Visualizations: APR-DRG severity charts, LACE distribution

**Key Code Skills:**
```python
# Example: Calculate Charlson Comorbidity Index from ICD-10 codes
def calculate_cci(diagnoses: list[str]) -> int:
    ...

# Example: LACE index for 30-day readmission risk
def calculate_lace(los, acuity, cci, er_visits) -> int:
    ...
```

---

### Part 2 — `ds_responsibilities_claims_classification.ipynb`
**Data Scientist Responsibilities in Claims Data & Patient Classification**

A comprehensive reference guide mapping the full scope of a healthcare data scientist's role — from raw data ingestion through executive reporting.

**Sections:**
1. Core Data Responsibilities — acquisition, cleaning, validation, feature engineering
2. Patient Classification — HCC mapping, DRG grouping, disease registries
3. Financial Risk Model Design — prospective models, retrospective cost analysis, stop-loss
4. Value-Based Program Analytics — ACO attribution, benchmarking, shared savings, HEDIS
5. Advanced Analytics — readmission prediction, NLP on clinical notes, dashboards
6. Collaboration & Stakeholder Responsibilities

**Key Frameworks:**
```
Raw Claims Data
      ↓
Data Cleaning & Validation
      ↓
Feature Engineering (HCC, DRG, PMPM, comorbidities)
      ↓
Patient Classification & Risk Stratification
      ↓
Financial Risk Model Development
      ↓
Value-Based Program Design & Benchmarking
      ↓
Quality Measurement & Care Gap Analysis
      ↓
Dashboards, Reports & Actionable Insights
```

---

### Part 3 — `healthcare_claims_workflow.ipynb`
**End-to-End Healthcare Claims Data Science Workflow**

A fully executable Python implementation of the complete analytics pipeline for a Medicare Advantage health plan with 2,000 simulated members and 12,000 simulated claims.

**Steps:**

| Step | Description |
|---|---|
| **Step 0** | Simulate realistic claims and eligibility data (with injected data quality issues) |
| **Step 1** | Data cleaning — deduplication, status filtering, imputation, financial validation |
| **Step 2** | Feature engineering — PMPM, HCC risk score, Charlson CCI, utilization metrics, age bands |
| **Step 3** | Patient classification — 4-tier risk stratification, disease registries |
| **Step 4** | Financial risk model — Linear Regression, Random Forest, Gradient Boosting comparison |
| **Step 5** | ACO / Value-Based Care analytics — attribution, risk-adjusted benchmarks, shared savings |
| **Step 6** | Quality measures — HbA1c testing, mammography, HTN adherence, 30-day readmissions |
| **Step 7** | 12-panel executive dashboard |
| **Step 8** | Actionable insights summary report |

**Models trained:**
```python
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=6),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05),
}
```

**Sample output — Plan KPI Summary:**
```
Total Members      :   1,847
Avg PMPM Cost      :  $  498
Avg HCC Score      :   1.082  (national avg = 1.000)
High/Critical Risk :     312 members (16.9%)
ER Rate / 1,000    :   423.1
Model R² (GB)      :   0.741
Total Shared Savings: $1,240,000
Care Gap Members   :      78 diabetics
```

---

## Key Concepts Covered

### Clinical Coding Systems

| Code Type | Full Name | Used For |
|---|---|---|
| ICD-10 | International Classification of Diseases, 10th Revision | Diagnoses |
| CPT | Current Procedural Terminology | Outpatient procedures |
| HCPCS | Healthcare Common Procedure Coding System | Medicare supplies & drugs |
| DRG / MS-DRG | Diagnosis Related Groups | Inpatient hospital payment |
| HCC | Hierarchical Condition Categories | Medicare Advantage risk adjustment |
| APR-DRG | All Patient Refined DRG | Severity of illness measurement |
| APC | Ambulatory Payment Classifications | Outpatient hospital payment |

### Actuarial & Risk Metrics

- **PMPM** — Per Member Per Month cost; primary actuarial metric
- **HCC Risk Score** — normalized to 1.0 = national average; drives capitation
- **Charlson Comorbidity Index (CCI)** — weighted 10-year mortality predictor
- **LACE Index** — 30-day readmission / death risk after hospitalization
- **MLR** — Medical Loss Ratio (claims paid / premiums collected)

### Value-Based Care Programs

| Program | Data Science Role |
|---|---|
| ACO (Accountable Care Organization) | Attribution models, benchmark spend, savings calculation |
| PCMH (Patient-Centered Medical Home) | Care coordination measurement, ER avoidance |
| Bundled Payments (BPCI) | Episode window definition, cost variance analysis |
| Pay-for-Performance (P4P) | HEDIS measure calculation, incentive payout modeling |
| Capitation / Global Risk | Prospective cost models, utilization management |

---

## Dataset

All data in this tutorial is **fully simulated** — no real patient information is used.

| Table | Rows | Description |
|---|---|---|
| `members` | 2,000 | Member eligibility with demographics and chronic conditions |
| `claims` | ~12,360 | Service-level claims with ICD-10, CPT, financials (includes ~3% injected duplicates) |

**Injected data quality issues** (to mimic real-world messiness):
- ~5% of members have missing age or unknown gender
- ~3% duplicate claims (same member / date / CPT / amount)
- Financial anomalies where `paid_amount > allowed_amount`

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
Heathcare_Claim_Data/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── patient_classification_tools.ipynb           # Part 1 — Foundations
├── ds_responsibilities_claims_classification.ipynb  # Part 2 — DS Role
└── healthcare_claims_workflow.ipynb             # Part 3 — Full Pipeline
```

---

## License

This project is for educational purposes. All data is synthetically generated.

---

*Built with Python · pandas · scikit-learn · matplotlib*
