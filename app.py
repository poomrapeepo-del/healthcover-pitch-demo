"""Healthcare coverage needs recommender — independent Streamlit pitch demo.

Run: python -m streamlit run app.py
The standalone app NEVER edits or trains the original thesis notebook.
For real CatBoost Top-3 and SHAP copy both exported model assets into models/.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import streamlit as st
from rules_engine import (
    UNKNOWN, FEATURES, THAI_FEATURES, LABELS, DEFAULT_CONFIG, NUMERIC_RANGES,
    CATEGORICAL_OPTIONS, SAMPLE_PROFILES, is_unknown, parse_user_inputs,
    compute_need_scores, assign_rule_label, unknown_fields,
)
from model_adapter import has_model_files, load_bundle, predict_top3, explain_top3

st.set_page_config(
    page_title="HealthCover | Personalized Recommendation",
    page_icon="💙", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], [data-testid="stApp"] {font-family:'Noto Sans Thai','Tahoma',sans-serif;}
    [data-testid="stAppViewContainer"] {background:linear-gradient(135deg,#f5faff 0%,#ffffff 48%,#f8fcff 100%);}
    [data-testid="stSidebar"] {background:#f0f6ff;}
    .hero {background:linear-gradient(115deg,#12386c,#1769ba 65%,#2f95c9);padding:28px 34px;
           border-radius:18px;color:white;margin-bottom:22px;box-shadow:0 12px 26px rgba(22,80,160,.12)}
    .hero h1 {font-size:30px; margin:5px 0 3px 0; color:white; letter-spacing:-.3px;}
    .hero p {font-size:14px; color:#e9f5ff; margin:5px 0;}
    .eyebrow {font-size:12px;letter-spacing:1.7px;font-weight:700;color:#c9edff}
    .step-name {font-weight:700; font-size:16px; color:#153a70; margin:20px 0 4px 0;}
    .score-card {padding:14px 18px;border:1px solid #e0ebf5;border-radius:13px;background:white;margin:8px 0;}
    .score-title {font-size:13px;color:#486787;margin-bottom:3px}
    .score-value {font-size:26px;font-weight:700;color:#164575}
    .result-card {border:1px solid #dbe8f6;border-left:5px solid #1769ba;background:white;border-radius:14px;
                  padding:15px 18px;margin:11px 0;box-shadow:0 2px 10px rgba(22,80,160,.05);}
    .rank {color:#1972b9;font-size:12px;font-weight:700;letter-spacing:1px}
    .result-title {font-weight:700;color:#15385e;font-size:17px;margin:3px 0}
    .result-score {font-size:25px;font-weight:700;color:#164575;margin-top:3px}
    .little {font-size:12px;color:#667b8e;line-height:1.6}
    #MainMenu, footer {visibility: hidden;}
    div[data-testid="stFormSubmitButton"] button {background:#166db7; color:white;border-radius:10px;border:0;font-weight:700;}

    /* -----------------------------------------------------------------
       ACCESSIBILITY FIX: readable text on light AND dark device settings.
       Leave hero/result custom colors unchanged; scope labels to the form.
       ----------------------------------------------------------------- */
    [data-testid="stAppViewContainer"] {color:#19324f !important;}
    [data-testid="stSidebar"] {color:#19324f !important;}
    [data-testid="stForm"] {
        background:rgba(255,255,255,0.95) !important;
        border:1px solid #d5e0ed !important;
        border-radius:15px !important;
    }
    [data-testid="stForm"] [data-testid="stWidgetLabel"],
    [data-testid="stForm"] [data-testid="stWidgetLabel"] *,
    [data-testid="stForm"] label,
    [data-testid="stForm"] label * {
        color:#172f4a !important;
        opacity:1 !important;
        -webkit-text-fill-color:#172f4a !important;
        font-weight:600 !important;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] input {
        background-color:#edf3f9 !important;
        color:#172f4a !important;
        -webkit-text-fill-color:#172f4a !important;
        caret-color:#172f4a !important;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {
        color:#586f87 !important;
        opacity:1 !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color:#edf3f9 !important;
        border-color:#c9d8e8 !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stForm"] [data-testid="stSelectbox"] [data-baseweb="select"] input {
        color:#172f4a !important;
        -webkit-text-fill-color:#172f4a !important;
        opacity:1 !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] svg {
        fill:#172f4a !important;
    }
    /* Dropdown menus are portalled outside the form in recent Streamlit. */
    div[data-baseweb="popover"] {background:#ffffff !important;}
    div[data-baseweb="popover"] [role="listbox"] {background:#ffffff !important;}
    div[data-baseweb="popover"] [role="option"] {
        color:#172f4a !important;
        background-color:#ffffff !important;
        -webkit-text-fill-color:#172f4a !important;
    }
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background-color:#e5f2ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color:#1b3655 !important;
    }
    [data-testid="stForm"] button[kind="primaryFormSubmit"],
    [data-testid="stForm"] div[data-testid="stFormSubmitButton"] button {
        background-color:#166db7 !important;
        color:#ffffff !important;
        -webkit-text-fill-color:#ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

DISPLAY_OPTIONS = {
    "Low": "ต่ำ (Low)", "Moderate": "ปานกลาง (Moderate)", "High": "สูง (High)",
    "No": "ไม่มี (No)", "Yes": "มี (Yes)", "Non-smoker": "ไม่สูบบุหรี่",
    "Former smoker": "เคยสูบ แต่เลิกแล้ว", "Current smoker": "สูบบุหรี่ปัจจุบัน",
    "None (ไม่ดื่ม)": "ไม่ดื่มเลย (None)",
    "Highly Active": "ออกกำลังกายมาก", "Moderately Active": "ออกกำลังกายปานกลาง",
    "Lightly Active": "ออกกำลังกายเล็กน้อย", "Sedentary": "ไม่ค่อยเคลื่อนไหว",
    "Excellent": "ดีเยี่ยม", "Good": "ดี", "Fair": "พอใช้", "Poor": "ไม่ดี",
}

def calculate_bmi(height_text: str, weight_text: str):
    """Convert user-entered centimetres and kilograms into the BMI feature.

    Keep BMI in the original CatBoost feature schema. Empty/unknown values
    propagate as UNKNOWN to the existing missing-value handling.
    """
    height_raw = str(height_text).strip()
    weight_raw = str(weight_text).strip()

    if is_unknown(height_raw) or is_unknown(weight_raw) or not height_raw or not weight_raw:
        return UNKNOWN

    try:
        height_cm = float(height_raw.replace(",", "."))
        weight_kg = float(weight_raw.replace(",", "."))
    except ValueError as exc:
        raise ValueError("กรุณากรอกส่วนสูงและน้ำหนักเป็นตัวเลข เช่น 170 และ 65") from exc

    if not (math.isfinite(height_cm) and math.isfinite(weight_kg)):
        raise ValueError("ส่วนสูงและน้ำหนักต้องเป็นตัวเลขที่ใช้คำนวณได้")
    if not (80 <= height_cm <= 250):
        raise ValueError("ส่วนสูงต้องอยู่ระหว่าง 80–250 เซนติเมตร")
    if not (20 <= weight_kg <= 350):
        raise ValueError("น้ำหนักต้องอยู่ระหว่าง 20–350 กิโลกรัม")

    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi_min, bmi_max = NUMERIC_RANGES["BMI"]
    if not (bmi_min <= bmi <= bmi_max):
        raise ValueError(
            f"ค่า BMI ที่คำนวณได้ ({bmi:.2f}) อยู่นอกช่วงที่ระบบรองรับ "
            f"({bmi_min:g}–{bmi_max:g}) กรุณาตรวจสอบส่วนสูงและน้ำหนัก"
        )
    return round(bmi, 2)


def reset_profile(profile: str) -> None:
    preset = SAMPLE_PROFILES[profile]
    for feature, val in preset.items():
        if feature != "BMI":
            st.session_state[f"input_{feature}"] = val

    # ตัวอย่างเดิมมี BMI อย่างเดียว: ใช้ส่วนสูงสมมติ 170 ซม.
    # และแปลงกลับเป็นน้ำหนักให้โปรไฟล์ตัวอย่าง โดยไม่แตะโมเดล
    preset_bmi = preset.get("BMI", UNKNOWN)
    if is_unknown(preset_bmi):
        st.session_state["input_Height_cm"] = ""
        st.session_state["input_Weight_kg"] = ""
    else:
        example_height_cm = 170.0
        example_weight_kg = float(preset_bmi) * (example_height_cm / 100) ** 2
        st.session_state["input_Height_cm"] = f"{example_height_cm:g}"
        st.session_state["input_Weight_kg"] = f"{example_weight_kg:.2f}"
    st.session_state.pop("prediction_result", None)


if "initial_preset_done" not in st.session_state:
    reset_profile("หัวใจเด่น")
    st.session_state["initial_preset_done"] = True

st.sidebar.markdown("### 💙 HealthCover Demo")
st.sidebar.caption("ต้นแบบสาธิตจากโครงงานการแนะนำความต้องการความคุ้มครองสุขภาพ")
st.sidebar.markdown("**กรอกข้อมูลตัวอย่างได้ทันที**")
for profile, subtitle in [("หัวใจเด่น", "กรณีความต้องการด้านหัวใจ"),
                          ("สุขภาพค่อนข้างดี", "กรณีคะแนนความต้องการต่ำ"),
                          ("มีข้อมูลไม่ทราบ", "แสดงการจัดการ Unknown")]:
    st.sidebar.button(f"โหลดตัวอย่าง: {profile}", help=subtitle, use_container_width=True,
                      key=f"preset_{profile}", on_click=reset_profile, args=(profile,))
st.sidebar.divider()
if has_model_files():
    st.sidebar.success("ตรวจพบไฟล์โมเดล CatBoost และ config")
else:
    st.sidebar.warning("ยังไม่มีไฟล์โมเดล: ยังไม่สามารถแสดงคำแนะนำ Top 3 ได้")
st.sidebar.caption("สำหรับ Top 3 ที่เป็นผลทำนายจริง ให้คัดลอกไฟล์ .cbm และ .pkl ที่ Notebook สร้างไว้ลงในโฟลเดอร์ models/")

st.markdown("""<div class="hero"><div class="eyebrow">PERSONALIZED HEALTH COVERAGE • RESEARCH DEMO</div>
<h1>ระบบแนะนำความต้องการความคุ้มครองสุขภาพ</h1>
</div>""", unsafe_allow_html=True)

st.info("**การสาธิต ไม่ใช่การตรวจวินิจฉัยหรือขายประกัน**", icon="ℹ️")

with st.form("health_form", clear_on_submit=False):
    st.markdown('<div class="step-name"> ข้อมูลสุขภาพ</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    def select_feature(feature: str, col):
        with col:
            st.selectbox(THAI_FEATURES[feature], options=CATEGORICAL_OPTIONS[feature],
                         key=f"input_{feature}", format_func=lambda v: DISPLAY_OPTIONS.get(v, v))

    def numeric_feature(feature: str, col):
        low, high = NUMERIC_RANGES[feature]
        with col:
            # Native help=? is a hover tooltip. Replace it with a tap-friendly
            # Streamlit popover so visitors on phones can read instructions.
            st.text_input(
                THAI_FEATURES[feature],
                key=f"input_{feature}",
            )
            with st.popover("ⓘ วิธีกรอก", type="tertiary"):
                st.markdown(f"**{THAI_FEATURES[feature]}**")
                st.write(f"กรอกตัวเลขในช่วง {low:g}–{high:g}")
                st.caption("หากไม่ทราบข้อมูล สามารถพิมพ์ 'ไม่ทราบ' หรือเว้นว่างได้")

    # รับส่วนสูง/น้ำหนักแทนการให้ผู้ใช้คำนวณ BMI เอง
    with left:
        st.text_input("ส่วนสูง (เซนติเมตร)", key="input_Height_cm", placeholder="เช่น 170")
    with right:
        st.text_input("น้ำหนัก (กิโลกรัม)", key="input_Weight_kg", placeholder="เช่น 65")
    st.caption("")

    select_feature("Heart_Disease_Risk", left)
    select_feature("Diabetes_Risk", right)
    select_feature("Family_History_CVD", left)
    select_feature("Family_History_T2D", right)

    select_feature("Smoking_Status", left)
    select_feature("Alcohol_Consumption", right)
    select_feature("Physical_Activity_Level", left)
    select_feature("Sleep_Quality", right)

    numeric_feature("Stress_Level", left)
    numeric_feature("Depression_Score", right)
    numeric_feature("Anxiety_Score", left)
    numeric_feature("Social_Isolation_Index", right)
    st.markdown(" ")
    submitted = st.form_submit_button("วิเคราะห์และแสดงผลคำแนะนำ", use_container_width=True)

if submitted:
    try:
        # ส่วนสูง/น้ำหนักใช้คำนวณในเว็บเท่านั้น โมเดลยังรับฟีเจอร์ BMI เดิม
        raw_user = {f: st.session_state[f"input_{f}"] for f in FEATURES if f != "BMI"}
        raw_user["BMI"] = calculate_bmi(
            st.session_state["input_Height_cm"],
            st.session_state["input_Weight_kg"],
        )
        parsed = parse_user_inputs(raw_user)
        unknowns = unknown_fields(raw_user)
        model_available = has_model_files()
        bundle = load_bundle() if model_available else None
        cfg = bundle.config if bundle is not None else dict(DEFAULT_CONFIG)
        scores = compute_need_scores(raw_user, cfg)
        rule_label = assign_rule_label(scores, cfg)
        top3, frame, factors = None, None, None
        error_shap = None
        if bundle is not None:
            top3, frame = predict_top3(bundle, raw_user)
            try:
                factors = explain_top3(bundle, frame, top3)
            except Exception as exc:
                error_shap = str(exc)
        st.session_state["prediction_result"] = {
            "raw_user": raw_user, "scores": scores, "rule_label": rule_label,
            "top3": top3, "factors": factors, "missing": unknowns,
            "weights": cfg, "shap_error": error_shap,
        }
    except Exception as exc:
        st.session_state.pop("prediction_result", None)
        st.error(f"ไม่สามารถวิเคราะห์ข้อมูลได้: {exc}")

result = st.session_state.get("prediction_result")
if result is not None:
    st.divider()
    st.subheader("ผลวิเคราะห์สำหรับข้อมูลตัวอย่าง")
    result_bmi = result["raw_user"].get("BMI", UNKNOWN)
    if not is_unknown(result_bmi):
        st.caption(f"ดัชนีมวลกาย (BMI) ที่คำนวณจากส่วนสูงและน้ำหนัก: {float(result_bmi):.2f} กก./ม²")
    missing = result["missing"]
    if missing:
        st.warning(
            f"กรอกไม่ทราบ {len(missing)} ตัวแปร: "
            + ", ".join(THAI_FEATURES[f] for f in missing)
            + ". คำแนะนำเป็นผลประมาณจากข้อมูลที่ยังไม่ครบ และยังไม่ได้ยืนยันความแม่นยำสำหรับกรณีนี้"
        )
    st.markdown("#### 3 อันดับคำแนะนำจาก CatBoost")
    if result["top3"] is None:
        st.info("ยังไม่ได้เชื่อมไฟล์โมเดลที่ฝึกจริง จึงไม่สร้างเปอร์เซ็นต์ Top 3 ขึ้นเอง โปรดคัดลอกโมเดล .cbm และ config .pkl จาก Notebook ลงใน models/ ก่อนสาธิตส่วนนี้")
    else:
        cols = st.columns(3, gap="medium")
        for col, item in zip(cols, result["top3"]):
            label = item["label"]
            with col:
                st.markdown(
                    f'<div class="result-card"><div class="rank">อันดับที่ {item["rank"]}</div>'
                    f'<div class="result-title">{LABELS[label][0]}</div>'
                    f'<div class="little">{LABELS[label][1]}</div>'
                    f'<div class="result-score">{item["probability"]*100:.2f}%</div>'
                    f'<div class="little">ความน่าจะเป็นของประเภทคำแนะนำจากโมเดล</div></div>',
                    unsafe_allow_html=True,
                )
                supporting = (result["factors"] or {}).get(label, [])
                if supporting:
                    st.markdown("**ปัจจัย SHAP ที่สนับสนุนอันดับนี้**")
                    for feature in supporting[:3]:
                        st.caption(f'• {THAI_FEATURES[feature["feature"]]}')
                elif result["factors"] is not None:
                    st.caption("ไม่มีค่า SHAP เชิงบวกที่แสดงในตัวอย่างนี้")
        if result["shap_error"]:
            st.warning(f"ส่วน SHAP ไม่พร้อมใช้งาน: {result['shap_error']}")
        

    st.markdown("---")
    st.caption("**หมายเหตุ:** ผลนี้เป็นการประมาณประเภทความต้องการความคุ้มครอง เพื่อการสาธิต ไม่ใช่โอกาสเกิดโรค การวินิจฉัย หรือคำแนะนำซื้อกรมธรรม์จริง")

st.markdown("---")
