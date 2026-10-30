# HealthCover — Online Pitching Demo

Streamlit web demo for a university project on personalized healthcare coverage needs recommendations.

## Start locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud

- Publish this repository on GitHub.
- Visit https://share.streamlit.io, create an app from GitHub, and set **Main file path = app.py**.
- Streamlit will install dependencies from the root `requirements.txt`.
- Upload both real, matching exported model files to `models/` for CatBoost Top-3 and SHAP:
  - `models/catboost_health_coverage_model.cbm`
  - `models/health_coverage_system_config.pkl`
- Without the model, the app supports health questionnaire and rule-based Need Scores only, not model Top-3.

**Use synthetic examples only.** This is a research demo, not a medical diagnosis or insurance product offer. Do not upload raw health data, private data, credentials, or the entire research notebook into a public repository.
