# COREX (COntribution via Removal and EXplained variability) FOR MCDM WEIGHTING METHOD

This Streamlit application demonstrates the **COREX method** for determining criteria weights in Multi-Criteria Decision-Making (MCDM).  
The app follows the **8 labeled steps** of the COREX procedure, as outlined in the documentation.

## Features
- Upload your dataset (CSV or Excel) or use a provided sample dataset.
- Assign criterion types: **Benefit, Cost, Target** (with target values).
- Compute and display results in **8 labeled steps**:
  1. Normalization of the Decision Matrix  
  2. The Overall Performance Score  
  3. The Performance Score under Criterion Removal  
  4. Removal Impact Score  
  5. The Standard Deviation of Each Criterion  
  6. The Sum of Absolute Correlations for Each Criterion  
  7. Explained Variability Score  
  8. COREX Weight Scores  
- Download intermediate results at each step as CSV.
- Interactive chart of final COREX weights.

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/corex-app.git
cd corex-app
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app_corex_pdf_steps.py
```

This will launch a local server at `http://localhost:8501`.

## Deployment on Streamlit Cloud

1. Push this repository to your GitHub account.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app, select this repository, branch, and `app_corex_pdf_steps.py` as the entry file.
4. Click Deploy.

## File structure

```
.
├── app_corex_pdf_steps.py   # Streamlit application (8 labeled steps)
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---
Developed for research on **COREX weighting in MCDM**.
