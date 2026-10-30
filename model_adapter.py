"""Load the CatBoost trained by the supplied experiment notebook; no fake predictions."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from rules_engine import FEATURES, CATEGORICAL_FEATURES, DEFAULT_CONFIG, model_frame

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILE = MODELS_DIR / "catboost_health_coverage_model.cbm"
CONFIG_FILE = MODELS_DIR / "health_coverage_system_config.pkl"


@dataclass
class ModelBundle:
    model: object
    settings: dict
    config: dict


def has_model_files() -> bool:
    return MODEL_FILE.is_file() and CONFIG_FILE.is_file()


def load_bundle() -> ModelBundle:
    if not has_model_files():
        raise FileNotFoundError("ยังไม่พบไฟล์โมเดลที่ส่งออกจาก Notebook ในโฟลเดอร์ models/")
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(MODEL_FILE))
    settings = joblib.load(CONFIG_FILE)
    if list(settings.get("feature_columns", [])) != FEATURES:
        raise ValueError("ลำดับ Feature ในไฟล์ config ไม่ตรงกับ 13 Features ของเว็บแอป")
    if set(settings.get("categorical_features", [])) != set(CATEGORICAL_FEATURES):
        raise ValueError("รายการ Categorical Features ไม่ตรงกับโมเดล")
    cfg = dict(DEFAULT_CONFIG)
    for key in cfg:
        if key in settings:
            cfg[key] = settings[key]
    if not np.isclose(sum(float(cfg[k]) for k in ("clinical_weight", "family_weight", "lifestyle_weight")), 1.0):
        raise ValueError("น้ำหนักในไฟล์ config รวมกันไม่เท่ากับ 1")
    class_ids = {int(x) for x in np.asarray(model.classes_).ravel()}
    if class_ids != set(range(7)):
        raise ValueError("โมเดลไม่ครบ 7 ประเภทความคุ้มครอง")
    return ModelBundle(model=model, settings=settings, config=cfg)


def predict_top3(bundle: ModelBundle, user: dict) -> tuple[list[dict], pd.DataFrame]:
    frame = model_frame(user)
    probabilities = np.asarray(bundle.model.predict_proba(frame))[0]
    class_ids = np.asarray(bundle.model.classes_).astype(int)
    if len(probabilities) != len(class_ids):
        raise ValueError("ขนาดผลทำนายไม่ตรงกับคลาส")
    top3 = []
    for pos, index in enumerate(np.argsort(probabilities)[::-1][:3], start=1):
        top3.append({"rank": pos, "label": int(class_ids[index]), "probability": float(probabilities[index])})
    return top3, frame


def explain_top3(bundle: ModelBundle, frame: pd.DataFrame, top3: list[dict]) -> dict[int, list[dict]]:
    """Positive SHAP features on raw CatBoost logits, not on probabilities."""
    from catboost import Pool
    pool = Pool(data=frame, cat_features=CATEGORICAL_FEATURES, feature_names=FEATURES)
    shp = np.asarray(bundle.model.get_feature_importance(
        data=pool, type="ShapValues", shap_calc_type="Regular", thread_count=2, verbose=False
    ))
    expected = (1, len(bundle.model.classes_), len(FEATURES) + 1)
    if shp.shape != expected:
        raise ValueError(f"รูปแบบผล SHAP ไม่ตรง: ได้ {shp.shape}, ต้องเป็น {expected}")
    class_indices = {int(class_id): idx for idx, class_id in enumerate(bundle.model.classes_)}
    output = {}
    for result in top3:
        cls = class_indices[result["label"]]
        values = shp[0, cls, :-1]
        supporting = sorted(
            [{"feature": feature, "value": float(value), "user_value": frame.iloc[0][feature]}
             for feature, value in zip(FEATURES, values) if value > 0],
            key=lambda row: row["value"], reverse=True
        )
        output[result["label"]] = supporting[:3]
    return output
