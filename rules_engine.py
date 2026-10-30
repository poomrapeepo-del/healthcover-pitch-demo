"""Need Score + rule inference for the independent 5-minute pitch demo.

Derived from the formulas and labeling thresholds in notebook
006_FINAL_reviewed_Alcohol_Low0_COMPLETE(2).ipynb.

Critical difference from the Low=0 experiment: the *UI* has an explicit
"ไม่ทราบ" choice. This is unknown (not zero), unlike the source CSV's None
category meaning 'does not drink'. Unknown alcohol is excluded from the
available-mean behavior component and encoded as 'Unknown' for CatBoost.
Unknown primary weighted components leave that Need Score unavailable,
preventing the app from silently pretending incomplete data are low risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math
import numpy as np
import pandas as pd

UNKNOWN = "ไม่ทราบ"
FEATURES = [
    "BMI", "Heart_Disease_Risk", "Diabetes_Risk", "Family_History_CVD",
    "Family_History_T2D", "Smoking_Status", "Alcohol_Consumption",
    "Physical_Activity_Level", "Stress_Level", "Depression_Score",
    "Anxiety_Score", "Social_Isolation_Index", "Sleep_Quality",
]
CATEGORICAL_FEATURES = [
    "Heart_Disease_Risk", "Diabetes_Risk", "Smoking_Status",
    "Alcohol_Consumption", "Physical_Activity_Level", "Sleep_Quality",
]
NUMERIC_FEATURES = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]
THAI_FEATURES = {
    "BMI": "ดัชนีมวลกาย (BMI)",
    "Heart_Disease_Risk": "ระดับความเสี่ยงโรคหัวใจ",
    "Diabetes_Risk": "ระดับความเสี่ยงโรคเบาหวาน",
    "Family_History_CVD": "ประวัติครอบครัวโรคหัวใจ",
    "Family_History_T2D": "ประวัติครอบครัวโรคเบาหวาน",
    "Smoking_Status": "การสูบบุหรี่",
    "Alcohol_Consumption": "การดื่มแอลกอฮอล์",
    "Physical_Activity_Level": "ระดับกิจกรรมทางกาย",
    "Stress_Level": "ระดับความเครียด",
    "Depression_Score": "คะแนนภาวะซึมเศร้า",
    "Anxiety_Score": "คะแนนความวิตกกังวล",
    "Social_Isolation_Index": "คะแนนความโดดเดี่ยวทางสังคม",
    "Sleep_Quality": "คุณภาพการนอนหลับ",
}
LABELS = {
    0: ("ความคุ้มครองสุขภาพแบบครอบคลุม", "Comprehensive Health Coverage"),
    1: ("ความคุ้มครองด้านโรคหัวใจ", "Heart Protection Coverage"),
    2: ("ความคุ้มครองด้านโรคเบาหวาน", "Diabetes Protection Coverage"),
    3: ("ความคุ้มครองด้านสุขภาวะทางจิต", "Mental Wellness Coverage"),
    4: ("ความคุ้มครองส่งเสริมวิถีชีวิตสุขภาพ", "Healthy Lifestyle Coverage"),
    5: ("ความคุ้มครองเพื่อป้องกันและเฝ้าระวัง", "Preventive Health Coverage"),
    6: ("ความคุ้มครองสุขภาพทั่วไป", "General Health Coverage"),
}
RISK = {"Low": 0., "Moderate": .5, "High": 1.}
SMOKING = {"Non-smoker": 0., "Former smoker": .5, "Current smoker": 1.}
ALCOHOL = {"Low": 0., "Moderate": .5, "High": 1.}
ACTIVITY = {"Highly Active": 0., "Moderately Active": .25,
            "Lightly Active": .5, "Sedentary": 1.}
SLEEP = {"Excellent": 0., "Good": .25, "Fair": .5, "Poor": 1.}
NUMERIC_RANGES = {
    "BMI": (0.01, 100.),
    "Stress_Level": (1., 10.),
    "Depression_Score": (0., 19.),
    "Anxiety_Score": (0., 16.),
    "Social_Isolation_Index": (0., 10.),
}
CATEGORICAL_OPTIONS = {
    "Heart_Disease_Risk": [UNKNOWN, "Low", "Moderate", "High"],
    "Diabetes_Risk": [UNKNOWN, "Low", "Moderate", "High"],
    "Family_History_CVD": [UNKNOWN, "No", "Yes"],
    "Family_History_T2D": [UNKNOWN, "No", "Yes"],
    "Smoking_Status": [UNKNOWN, "Non-smoker", "Former smoker", "Current smoker"],
    "Alcohol_Consumption": [UNKNOWN, "None (ไม่ดื่ม)", "Low", "Moderate", "High"],
    "Physical_Activity_Level": [UNKNOWN, "Highly Active", "Moderately Active", "Lightly Active", "Sedentary"],
    "Sleep_Quality": [UNKNOWN, "Excellent", "Good", "Fair", "Poor"],
}
DEFAULT_CONFIG = {
    "clinical_weight": .8, "family_weight": .1, "lifestyle_weight": .1,
    "high_threshold": .65, "mid_threshold": .45,
    "dominant_threshold": .50, "dominance_margin": .10,
}


def is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "ไม่ทราบ", "unknown", "missing", "n/a", "nan", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_user_inputs(user: Mapping[str, Any]) -> dict[str, Any]:
    """Return notebook-aligned model input; real missing is *not* Low=0."""
    if any(f not in user for f in FEATURES):
        raise ValueError("มีตัวแปรที่ยังไม่ได้ระบุในฟอร์ม")
    values: dict[str, Any] = {}
    for feature in FEATURES:
        raw = user[feature]
        if feature in NUMERIC_RANGES:
            if is_unknown(raw):
                values[feature] = np.nan
                continue
            try:
                number = float(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{THAI_FEATURES[feature]} ต้องเป็นตัวเลขหรือ 'ไม่ทราบ'") from exc
            low, high = NUMERIC_RANGES[feature]
            if not math.isfinite(number) or not low <= number <= high:
                raise ValueError(f"{THAI_FEATURES[feature]} ต้องอยู่ระหว่าง {low:g} ถึง {high:g}")
            values[feature] = number
        elif feature in ("Family_History_CVD", "Family_History_T2D"):
            if is_unknown(raw):
                values[feature] = np.nan
            elif raw in ("No", 0, "0"):
                values[feature] = 0
            elif raw in ("Yes", 1, "1"):
                values[feature] = 1
            else:
                raise ValueError(f"{THAI_FEATURES[feature]}: ค่าที่เลือกไม่ถูกต้อง")
        else:
            if is_unknown(raw):
                values[feature] = "Unknown"
            elif feature == "Alcohol_Consumption" and raw in ("None", "None (ไม่ดื่ม)"):
                values[feature] = "Low"  # None=non-drinker in dataset, score 0
            else:
                allowed = CATEGORICAL_OPTIONS[feature]
                if raw not in allowed:
                    raise ValueError(f"{THAI_FEATURES[feature]}: ค่าที่เลือกไม่ถูกต้อง")
                values[feature] = raw
    return values


def numeric_or_nan(val: Any) -> float:
    return float(val) if not is_unknown(val) else float("nan")


def mapped(mapping: dict, val: Any) -> float:
    return mapping.get(val, float("nan")) if isinstance(val, str) else float("nan")


def mean_available(*values: float) -> float:
    arr = [float(v) for v in values if not pd.isna(v)]
    return float(np.mean(arr)) if arr else float("nan")


def bmi_score(bmi: float) -> float:
    if pd.isna(bmi):
        return float("nan")
    if bmi < 18.5:
        return .5
    if bmi < 25:
        return 0.
    if bmi < 30:
        return .5
    return 1.


def weighted_three(a: float, b: float, c: float, config: Mapping[str, Any]) -> float:
    """Preserve original formula; do not replace missing main components with 0."""
    if any(pd.isna(v) for v in (a, b, c)):
        return float("nan")
    return (float(config["clinical_weight"]) * a
            + float(config["family_weight"]) * b
            + float(config["lifestyle_weight"]) * c)


def compute_need_scores(user: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, float]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    u = parse_user_inputs(user)
    heart_risk = mapped(RISK, u["Heart_Disease_Risk"])
    diabetes_risk = mapped(RISK, u["Diabetes_Risk"])
    family_heart = numeric_or_nan(u["Family_History_CVD"])
    family_diabetes = numeric_or_nan(u["Family_History_T2D"])
    smoking = mapped(SMOKING, u["Smoking_Status"])
    alcohol = mapped(ALCOHOL, u["Alcohol_Consumption"])
    inactive = mapped(ACTIVITY, u["Physical_Activity_Level"])
    bmi = bmi_score(numeric_or_nan(u["BMI"]))
    stress_in = numeric_or_nan(u["Stress_Level"])
    depression_in = numeric_or_nan(u["Depression_Score"])
    anxiety_in = numeric_or_nan(u["Anxiety_Score"])
    isolation_in = numeric_or_nan(u["Social_Isolation_Index"])
    stress = (stress_in - 1) / 9 if not pd.isna(stress_in) else float("nan")
    depression = depression_in / 19
    anxiety = anxiety_in / 16
    isolation = isolation_in / 10
    sleep = mapped(SLEEP, u["Sleep_Quality"])
    return {
        "Heart_Need_Score": weighted_three(heart_risk, family_heart,
            mean_available(smoking, inactive, bmi), cfg),
        "Diabetes_Need_Score": weighted_three(diabetes_risk, family_diabetes,
            mean_available(bmi, inactive), cfg),
        "Mental_Need_Score": weighted_three(mean_available(stress, depression, anxiety),
            isolation, sleep, cfg),
        "Lifestyle_Need_Score": weighted_three(mean_available(smoking, alcohol, inactive),
            bmi, mean_available(sleep, stress), cfg),
    }


def assign_rule_label(scores: Mapping[str, float], config: Mapping[str, Any] | None = None) -> int | None:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    ordered_keys = ("Heart_Need_Score", "Diabetes_Need_Score", "Mental_Need_Score", "Lifestyle_Need_Score")
    if any(pd.isna(scores[key]) for key in ordered_keys):
        return None
    v = [float(scores[key]) for key in ordered_keys]
    highs = [i for i, value in enumerate(v) if value >= float(cfg["high_threshold"])]
    if len(highs) >= 2:
        return 0
    if len(highs) == 1:
        return highs[0] + 1
    order = sorted(enumerate(v), key=lambda t: t[1], reverse=True)
    if (order[0][1] >= float(cfg["dominant_threshold"])
        and order[0][1] - order[1][1] >= float(cfg["dominance_margin"])):
        return order[0][0] + 1
    return 5 if order[0][1] >= float(cfg["mid_threshold"]) else 6


def model_frame(user: Mapping[str, Any]) -> pd.DataFrame:
    model_input = parse_user_inputs(user)
    frame = pd.DataFrame([model_input], columns=FEATURES)
    for feature in CATEGORICAL_FEATURES:
        frame[feature] = frame[feature].astype(str)
    for feature in NUMERIC_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame


def unknown_fields(user: Mapping[str, Any]) -> list[str]:
    return [feature for feature in FEATURES if is_unknown(user[feature])]


SAMPLE_PROFILES = {
    "หัวใจเด่น": {
        "BMI": "30.5", "Heart_Disease_Risk": "High", "Diabetes_Risk": "Low",
        "Family_History_CVD": "Yes", "Family_History_T2D": "No", "Smoking_Status": "Current smoker",
        "Alcohol_Consumption": "None (ไม่ดื่ม)", "Physical_Activity_Level": "Sedentary",
        "Stress_Level": "7", "Depression_Score": "5", "Anxiety_Score": "5",
        "Social_Isolation_Index": "3", "Sleep_Quality": "Fair",
    },
    "สุขภาพค่อนข้างดี": {
        "BMI": "22.1", "Heart_Disease_Risk": "Low", "Diabetes_Risk": "Low",
        "Family_History_CVD": "No", "Family_History_T2D": "No", "Smoking_Status": "Non-smoker",
        "Alcohol_Consumption": "Low", "Physical_Activity_Level": "Highly Active",
        "Stress_Level": "2", "Depression_Score": "1", "Anxiety_Score": "1",
        "Social_Isolation_Index": "1", "Sleep_Quality": "Excellent",
    },
    "มีข้อมูลไม่ทราบ": {
        "BMI": "25.5", "Heart_Disease_Risk": "High", "Diabetes_Risk": "Low",
        "Family_History_CVD": "Yes", "Family_History_T2D": "No", "Smoking_Status": "Former smoker",
        "Alcohol_Consumption": UNKNOWN, "Physical_Activity_Level": "Lightly Active",
        "Stress_Level": "6", "Depression_Score": "4", "Anxiety_Score": "3",
        "Social_Isolation_Index": "2", "Sleep_Quality": "Good",
    },
}
