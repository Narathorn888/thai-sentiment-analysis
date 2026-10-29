import os
import sys

# Ensure project root is in Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Internal modules
from src.preprocessor import clean_text, tokenize_thai, preprocess_for_display
from src.model import load_trained_model, load_metrics, predict_single
from src.issue_classifier import classify_crm_issue, determine_crm_priority
from src.data_loader import load_train_data, load_test_data, load_sample_data, get_dataset_overview

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Thai Review Sentiment & CRM Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS (Dark Mode + Aurora Gradient + Liquid Glass UI)
# ---------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Prompt:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
    /* Global Font Override */
    html, body, [class*="css"], .stMarkdown, .stText, p, div, span, button, input, textarea {
        font-family: 'Prompt', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Full App Canvas: Deep Aurora Gradient Background */
    .stApp {
        background: 
            radial-gradient(circle at 12% 15%, rgba(99, 102, 241, 0.18) 0%, transparent 40%),
            radial-gradient(circle at 88% 18%, rgba(236, 72, 153, 0.14) 0%, transparent 38%),
            radial-gradient(circle at 50% 82%, rgba(14, 165, 233, 0.14) 0%, transparent 42%),
            radial-gradient(circle at 92% 88%, rgba(168, 85, 247, 0.18) 0%, transparent 35%),
            linear-gradient(140deg, #060911 0%, #0B1120 40%, #0F172A 75%, #070B16 100%) !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
    }

    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1360px;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 29, 0.75) !important;
        backdrop-filter: blur(20px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(190%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Hero Liquid Glass Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.55) 0%, rgba(15, 23, 42, 0.75) 100%) !important;
        backdrop-filter: blur(20px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(190%) !important;
        border-radius: 22px;
        padding: 30px 36px;
        color: #FFFFFF;
        box-shadow: inset 0 1px 1.5px rgba(255, 255, 255, 0.2), 
                    0 15px 35px -5px rgba(0, 0, 0, 0.45);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -80px;
        right: -80px;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.3) 0%, rgba(236, 72, 153, 0.15) 50%, transparent 70%);
        border-radius: 50%;
        filter: blur(30px);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.25);
        color: #C7D2FE;
        border: 1px solid rgba(165, 180, 252, 0.35);
        padding: 5px 16px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 27px;
        font-weight: 700;
        line-height: 1.35;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #FFFFFF 30%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 14.5px;
        color: #94A3B8;
        font-weight: 300;
        max-width: 900px;
        line-height: 1.6;
    }

    /* Liquid Glass KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.55) 100%) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 18px;
        padding: 22px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.15), 
                    0 10px 25px -4px rgba(0, 0, 0, 0.35);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 255, 255, 0.22) !important;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.25), 
                    0 15px 35px -5px rgba(0, 0, 0, 0.5),
                    0 0 20px rgba(99, 102, 241, 0.25);
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 34px;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 13px;
        color: #34D399;
        font-weight: 500;
        margin-top: 6px;
    }
    .kpi-sub.neutral {
        color: #94A3B8;
    }

    /* ---------------------------------------------------- */
    /* Liquid Glass UI Buttons (Buttons & Form Actions)     */
    /* ---------------------------------------------------- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.03) 100%) !important;
        backdrop-filter: blur(16px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 14px !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: inset 0 1px 1.5px 0 rgba(255, 255, 255, 0.35),
                    inset 0 -1px 1px 0 rgba(0, 0, 0, 0.2),
                    0 8px 22px -3px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px) scale(1.015) !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.22) 0%, rgba(255, 255, 255, 0.07) 100%) !important;
        border-color: rgba(255, 255, 255, 0.45) !important;
        box-shadow: inset 0 1px 2px 0 rgba(255, 255, 255, 0.6),
                    0 12px 28px -4px rgba(0, 0, 0, 0.5),
                    0 0 25px rgba(99, 102, 241, 0.4) !important;
        color: #FFFFFF !important;
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(1px) scale(0.99) !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
    }

    /* Primary Liquid Glass CTA Button (e.g. Run Analysis) */
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.8) 0%, rgba(139, 92, 246, 0.6) 50%, rgba(59, 130, 246, 0.45) 100%) !important;
        backdrop-filter: blur(18px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(200%) !important;
        border: 1px solid rgba(199, 210, 254, 0.5) !important;
        box-shadow: inset 0 1px 2px 0 rgba(255, 255, 255, 0.7),
                    0 10px 28px -2px rgba(99, 102, 241, 0.55),
                    0 0 25px rgba(139, 92, 246, 0.35) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.95) 0%, rgba(139, 92, 246, 0.75) 50%, rgba(59, 130, 246, 0.6) 100%) !important;
        border-color: rgba(255, 255, 255, 0.7) !important;
        box-shadow: inset 0 1px 2.5px 0 rgba(255, 255, 255, 0.9),
                    0 14px 35px -2px rgba(99, 102, 241, 0.7),
                    0 0 35px rgba(139, 92, 246, 0.55) !important;
    }

    /* Tabs Styling with Liquid Glass Navigation Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        padding: 6px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 12px;
        padding: 0 22px;
        font-size: 14px;
        font-weight: 500;
        color: #94A3B8;
        background-color: transparent;
        border: none;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0.04) 100%) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.4), 
                    0 4px 18px rgba(0, 0, 0, 0.35) !important;
    }

    /* Liquid Glass Sentiment Result Cards */
    .sentiment-card {
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
        backdrop-filter: blur(18px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(190%) !important;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .sentiment-card.pos {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(6, 95, 70, 0.25) 100%);
        border: 1.5px solid rgba(52, 211, 153, 0.45);
        box-shadow: inset 0 1px 1.5px rgba(52, 211, 153, 0.5), 0 10px 25px rgba(16, 185, 129, 0.2);
    }
    .sentiment-card.neu {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.45) 0%, rgba(146, 64, 14, 0.2) 100%);
        border: 1.5px solid rgba(251, 191, 36, 0.45);
        box-shadow: inset 0 1px 1.5px rgba(251, 191, 36, 0.5), 0 10px 25px rgba(245, 158, 11, 0.2);
    }
    .sentiment-card.neg {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.5) 0%, rgba(153, 27, 27, 0.25) 100%);
        border: 1.5px solid rgba(248, 113, 113, 0.45);
        box-shadow: inset 0 1px 1.5px rgba(248, 113, 113, 0.5), 0 10px 25px rgba(239, 68, 68, 0.2);
    }
    
    .sentiment-title {
        font-size: 24px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .sentiment-title.pos { color: #6EE7B7; }
    .sentiment-title.neu { color: #FCD34D; }
    .sentiment-title.neg { color: #FCA5A5; }

    /* CRM Alert Box (Glass) */
    .crm-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.65) 100%) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border-radius: 16px;
        padding: 18px 22px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.15), 0 8px 20px rgba(0,0,0,0.3);
        margin-top: 14px;
    }
    .crm-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
        backdrop-filter: blur(8px);
    }

    /* Form Controls Glass */
    .stTextArea textarea, .stTextInput input {
        background: rgba(15, 23, 42, 0.55) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        color: #F8FAFC !important;
        border-radius: 14px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(129, 140, 248, 0.7) !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
    }

    [data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.55) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
    }

    /* Expander Glass */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(14px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #E2E8F0 !important;
    }
    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.4) !important;
        border-radius: 0 0 12px 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: none !important;
    }

    .sample-pill-title {
        font-size: 13px;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Cached Resources
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    return load_trained_model()

@st.cache_data
def get_overview():
    return get_dataset_overview()

model = get_model()
dataset_info = get_overview()

# ---------------------------------------------------------
# Sidebar UI (Dark Liquid Glass Theme)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 14px;'>
        <div style='background: linear-gradient(135deg, rgba(99, 102, 241, 0.8) 0%, rgba(168, 85, 247, 0.7) 100%); 
                    width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; 
                    justify-content: center; font-size: 22px; color: white;
                    box-shadow: inset 0 1px 1px rgba(255,255,255,0.6), 0 8px 18px rgba(99,102,241,0.4);'>
            ⚡
        </div>
        <div>
            <div style='font-size: 17px; font-weight: 700; color: #F8FAFC;'>ThaiNLP CRM</div>
            <div style='font-size: 12px; color: #94A3B8;'>Enterprise Liquid Glass AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: rgba(30, 41, 59, 0.5); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 14px; padding: 14px; margin-bottom: 18px; box-shadow: inset 0 1px 1px rgba(255,255,255,0.15);'>
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span style='height: 10px; width: 10px; background-color: #10B981; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px #10B981;'></span>
            <span style='font-size: 13px; font-weight: 600; color: #F8FAFC;'>Dark Mode: Active</span>
        </div>
        <div style='font-size: 12px; color: #94A3B8; margin-top: 5px;'>TF-IDF + Balanced Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚙️ การตั้งค่าระบบ")
    model_choice = st.selectbox(
        "โมเดลประมวลผล",
        (
            "TF-IDF + Logistic Regression (Active)",
            "Bi-LSTM (Roadmap Week 5-7)",
            "WangchanBERTa (Roadmap Week 5-7)"
        )
    )
    
    st.divider()
    st.markdown("#### 👥 คณะผู้จัดทำ")
    st.markdown("""
    <div style='background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 14px; font-size: 12.5px; color: #CBD5E1;'>
        <strong style='color: #818CF8;'>CS461 Final Project:</strong>
        <div style='margin-top: 8px; line-height: 1.7;'>
            • นราทร ศุภสุวรณพงษ์ (1660704568)<br>
            • วิษณุ เดวี (1660704048)<br>
            • ณัฏฐกิตติ์ สุทธิเกษมคุณ (1660702562)<br>
            • นฤเบศ จีนล้ำเลิศ (1660703750)
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero Header (Dark Liquid Glass)
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span>●</span> AI Powered CRM Platform • PyThaiNLP Engine • Liquid Glass UI
    </div>
    <div class="hero-title">
        ระบบวิเคราะห์ความรู้สึกและจำแนกประเด็นปัญหาบริการลูกค้าจากข้อความรีวิวภาษาไทย
    </div>
    <div class="hero-subtitle">
        Thai Customer Review Sentiment and Issue Classification for E-Commerce & CRM — ตรวจจับข้อร้องเรียนเร่งด่วน คัดแยกความรู้สึก และยกระดับประสบการณ์ดูแลลูกค้าแบบเรียลไทม์
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Product Tabs
# ---------------------------------------------------------
tab_live, tab_batch, tab_explore = st.tabs([
    "✍️ วิเคราะห์รีวิวเดี่ยว (Live Review & Issue Analyzer)",
    "📁 วิเคราะห์ไฟล์ชุดข้อมูล (Batch CSV & CRM Dashboard)",
    "🔍 สำรวจคลังข้อมูลรีวิว (Customer Reviews Explorer)"
])

# =========================================================
# TAB 1: Live Inference (Single Review)
# =========================================================
with tab_live:
    col_input, col_result = st.columns([1.2, 1], gap="large")
    
    with col_input:
        st.markdown("""
        <div style='font-size: 18px; font-weight: 600; color: #F8FAFC; margin-bottom: 4px;'>
            กรอกข้อความรีวิวหรือข้อร้องเรียนลูกค้า
        </div>
        <div style='font-size: 13.5px; color: #94A3B8; margin-bottom: 12px;'>
            พิมพ์ข้อความรีวิวภาษาไทยเพื่อส่งให้โมเดล AI วิเคราะห์ความรู้สึกและระบุประเด็นปัญหาทันที
        </div>
        """, unsafe_allow_html=True)
        
        user_text = st.text_area(
            "Review Input",
            value="ส่งของไวมากครับ สินค้าคุณภาพดีตรงปก แพ็คมาแน่นหนา แอดมินตอบแชทสุภาพมาก ประทับใจร้านนี้",
            height=140,
            label_visibility="collapsed"
        )
        
        st.markdown("<div class='sample-pill-title'>⚡ ตัวอย่างรีวิวกดทดสอบด่วน:</div>", unsafe_allow_html=True)
        pill_col1, pill_col2, pill_col3 = st.columns(3)
        
        if pill_col1.button("🟢 รีวิวชมเชย (Positive)", use_container_width=True):
            user_text = "ของส่งไวมาก คุณภาพดีเยี่ยม คุ้มค่าเกินราคา มีของแถมให้ด้วย ชอบมากๆ"
        if pill_col2.button("🟡 สอบถามทั่วไป (Neutral)", use_container_width=True):
            user_text = "ได้รับพัสดุเรียบร้อยแล้ว สินค้าครบถ้วนตามสั่ง ยังไม่ได้ลองเปิดใช้งาน"
        if pill_col3.button("🔴 เคสร้องเรียน (Negative)", use_container_width=True):
            user_text = "ส่งช้ามาก กล่องพังยับเยิน สินค้าด้านในแตกหัก แอดมินตอบช้าปัดความรับผิดชอบ แย่สุดๆ"

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🚀 วิเคราะห์ความรู้สึก & ประเด็นปัญหา", type="primary", use_container_width=True)

    with col_result:
        st.markdown("""
        <div style='font-size: 18px; font-weight: 600; color: #F8FAFC; margin-bottom: 4px;'>
            ผลลัพธ์การวิเคราะห์ (AI Analysis Output)
        </div>
        <div style='font-size: 13.5px; color: #94A3B8; margin-bottom: 12px;'>
            ระดับความมั่นใจ ประเด็นปัญหา และข้อเสนอแนะสำหรับทีม CRM
        </div>
        """, unsafe_allow_html=True)
        
        if user_text.strip() and model is not None:
            pred_data = predict_single(model, user_text)
            sentiment = pred_data["prediction"]
            confidence = pred_data["confidence"]
            probabilities = pred_data["probabilities"]
            
            issue_category, matched_kws = classify_crm_issue(user_text)
            crm_meta = determine_crm_priority(sentiment, issue_category)
            
            # Sentiment Card
            if sentiment == "เชิงบวก":
                card_cls = "pos"
                title_icon = "🟢 เชิงบวก (Positive Sentiment)"
            elif sentiment == "เชิงลบ":
                card_cls = "neg"
                title_icon = "🔴 เชิงลบ (Negative Sentiment)"
            else:
                card_cls = "neu"
                title_icon = "🟡 เชิงกลาง (Neutral Sentiment)"
                
            st.markdown(f"""
            <div class="sentiment-card {card_cls}">
                <div class="sentiment-title {card_cls}">{title_icon}</div>
                <div style='font-size: 14.5px; color: #E2E8F0; margin-top: 4px;'>
                    ระดับความมั่นใจของโมเดล: <strong style='font-size: 22px; color: #FFFFFF;'>{confidence:.1%}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution Chart (Dark Liquid Glass Theme)
            df_prob = pd.DataFrame(list(probabilities.items()), columns=["Sentiment", "Probability"])
            fig_prob = px.bar(
                df_prob,
                x="Probability",
                y="Sentiment",
                orientation="h",
                text=df_prob["Probability"].apply(lambda x: f"{x:.1%}"),
                color="Sentiment",
                color_discrete_map={"เชิงบวก": "#10B981", "เชิงกลาง": "#F59E0B", "เชิงลบ": "#EF4444"}
            )
            fig_prob.update_layout(
                xaxis=dict(range=[0, 1], showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickformat=".0%", tickfont=dict(color="#94A3B8")),
                yaxis=dict(title="", tickfont=dict(color="#F8FAFC")),
                height=150,
                margin=dict(t=5, b=5, l=10, r=10),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # CRM Action Box
            st.markdown(f"""
            <div class="crm-box">
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-size: 13px; font-weight: 600; color: #94A3B8;'>CRM ISSUE TAG</span>
                    <span class='crm-badge' style='background: {crm_meta["color"]}33; color: {crm_meta["color"]}; border: 1px solid {crm_meta["color"]}66;'>
                        {crm_meta["level"]}
                    </span>
                </div>
                <div style='font-size: 16.5px; font-weight: 600; color: #F8FAFC; margin: 6px 0;'>
                    {issue_category}
                </div>
                <div style='font-size: 13.5px; color: #CBD5E1;'>
                    <strong>การดำเนินการ:</strong> {crm_meta["action"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🛠️ ตรวจสอบการตัดคำและ Text Preprocessing"):
                prep = preprocess_for_display(user_text)
                st.markdown(f"**คำที่ตรวจพบในพจนานุกรมประเด็นปัญหา:** `{', '.join(matched_kws) if matched_kws else 'ไม่มี'}`")
                st.markdown(f"**ข้อความหลังทำความสะอาด:**")
                st.code(prep["cleaned_text"], language="text")
                st.markdown(f"**ผลการตัดคำ PyThaiNLP (`newmm`) — {prep['token_count']} คำ:**")
                st.code(prep["tokenized_display"], language="text")
        else:
            st.info("กรุณากรอกข้อความเพื่อเริ่มการวิเคราะห์")

# =========================================================
# TAB 2: Batch Analysis & CRM Dashboard
# =========================================================
with tab_batch:
    st.markdown("""
    <div style='font-size: 20px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;'>
        📁 ระบบวิเคราะห์ชุดข้อมูลจำนวนมากและแดชบอร์ดฝ่าย CRM
    </div>
    <div style='font-size: 14px; color: #94A3B8; margin-bottom: 20px;'>
        อัปโหลดไฟล์ CSV รีวิวลูกค้า หรือกดทดสอบด้วยชุดข้อมูลตัวอย่างจริง 100 ข้อความจาก Wisesight
    </div>
    """, unsafe_allow_html=True)
    
    col_u1, col_u2 = st.columns([1.5, 1], gap="medium")
    with col_u1:
        uploaded_csv = st.file_uploader("เลือกไฟล์ CSV ที่ต้องการวิเคราะห์", type=["csv"])
    with col_u2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("📥 โหลดตัวอย่างรีวิวจริง 100 ข้อความจาก Wisesight", use_container_width=True)
        
    df_batch = None
    if uploaded_csv is not None:
        try:
            df_batch = pd.read_csv(uploaded_csv)
        except Exception:
            df_batch = pd.read_csv(uploaded_csv, encoding='utf-8-sig')
    elif load_sample:
        df_batch = load_sample_data()
        st.toast("โหลดข้อมูลตัวอย่าง 100 แถวสำเร็จ!", icon="✅")

    if df_batch is not None and not df_batch.empty:
        st.markdown("##### 📌 ตัวอย่างข้อมูลนำเข้า (Preview)")
        st.dataframe(df_batch.head(3), use_container_width=True)
        
        c_sel1, c_sel2 = st.columns([2, 1])
        with c_sel1:
            target_col = st.selectbox("เลือกคอลัมน์ข้อความรีวิว:", df_batch.columns)
        with c_sel2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_btn = st.button("🚀 ประมวลผลทั้งชุดข้อมูล (Run Batch)", type="primary", use_container_width=True)
            
        if run_btn:
            with st.spinner("โมเดลกำลังตัดคำและวิเคราะห์ความรู้สึก..."):
                sents, confs, issues, prios = [], [], [], []
                total_len = len(df_batch)
                prog_bar = st.progress(0)
                
                for i, row_text in enumerate(df_batch[target_col]):
                    res = predict_single(model, str(row_text))
                    sent = res["prediction"]
                    conf = res["confidence"]
                    issue, _ = classify_crm_issue(str(row_text))
                    prio = determine_crm_priority(sent, issue)["level"]
                    
                    sents.append(sent)
                    confs.append(conf)
                    issues.append(issue)
                    prios.append(prio)
                    prog_bar.progress((i + 1) / total_len)
                    
                prog_bar.empty()
                
                df_batch["Predicted_Sentiment"] = sents
                df_batch["Confidence"] = confs
                df_batch["Issue_Category"] = issues
                df_batch["CRM_Priority"] = prios
                
            st.success("✅ วิเคราะห์ชุดข้อมูลเสร็จสมบูรณ์!")
            
            # Dashboard KPI Cards (Liquid Glass)
            st.divider()
            st.markdown("### 📊 แดชบอร์ดสรุปภาพรวมสำหรับผู้บริหาร (Executive CRM KPIs)")
            
            n_total = len(df_batch)
            n_pos = (df_batch["Predicted_Sentiment"] == "เชิงบวก").sum()
            n_neu = (df_batch["Predicted_Sentiment"] == "เชิงกลาง").sum()
            n_neg = (df_batch["Predicted_Sentiment"] == "เชิงลบ").sum()
            n_crit = df_batch["CRM_Priority"].str.contains("วิกฤต|เร่งด่วน", na=False).sum()
            
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">รีวิวทั้งหมด</div>
                    <div class="kpi-val">{n_total:,}</div>
                    <div class="kpi-sub neutral">ข้อความทั้งหมด</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">🟢 เชิงบวก (Positive)</div>
                    <div class="kpi-val" style="color: #34D399;">{n_pos}</div>
                    <div class="kpi-sub">{n_pos/n_total:.1%} ของทั้งหมด</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">🟡 เชิงกลาง (Neutral)</div>
                    <div class="kpi-val" style="color: #FBBF24;">{n_neu}</div>
                    <div class="kpi-sub neutral">{n_neu/n_total:.1%} ของทั้งหมด</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col4:
                st.markdown(f"""
                <div class="kpi-card" style="border-color: rgba(248, 113, 113, 0.45) !important;">
                    <div class="kpi-label" style="color: #F87171;">🔴 เคสเชิงลบ & เร่งด่วน</div>
                    <div class="kpi-val" style="color: #F87171;">{n_neg}</div>
                    <div class="kpi-sub" style="color: #FCA5A5;">🚨 ต้องดูแลเร่งด่วน {n_crit} เคส</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                fig_pie_b = px.pie(
                    df_batch,
                    names="Predicted_Sentiment",
                    title="สัดส่วนความรู้สึกของลูกค้า (Sentiment Share)",
                    color="Predicted_Sentiment",
                    hole=0.48,
                    color_discrete_map={"เชิงบวก": "#10B981", "เชิงกลาง": "#F59E0B", "เชิงลบ": "#EF4444"}
                )
                fig_pie_b.update_layout(
                    height=320, 
                    margin=dict(t=40, b=20, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC")
                )
                st.plotly_chart(fig_pie_b, use_container_width=True)
                
            with g_col2:
                df_issue_c = df_batch["Issue_Category"].value_counts().reset_index()
                df_issue_c.columns = ["Issue", "Count"]
                fig_bar_i = px.bar(
                    df_issue_c,
                    x="Count",
                    y="Issue",
                    orientation="h",
                    title="ประเด็นปัญหาที่พบบ่อยที่สุด (Issue Breakdown)",
                    color="Count",
                    color_continuous_scale="Reds"
                )
                fig_bar_i.update_layout(
                    height=320, 
                    margin=dict(t=40, b=20, l=10, r=10), 
                    yaxis=dict(autorange="reversed", tickfont=dict(color="#F8FAFC")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC")
                )
                st.plotly_chart(fig_bar_i, use_container_width=True)
                
            st.markdown("##### 📋 ตารางผลการวิเคราะห์ฉบับสมบูรณ์")
            st.dataframe(df_batch, use_container_width=True)
            
            csv_data = df_batch.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 ดาวน์โหลดผลการวิเคราะห์เป็น CSV (UTF-8 with BOM)",
                data=csv_data,
                file_name="crm_sentiment_issue_results.csv",
                mime="text/csv",
                type="primary"
            )

# =========================================================
# TAB 3: Dataset Explorer (Clean Tool)
# =========================================================
with tab_explore:
    st.markdown("""
    <div style='font-size: 20px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;'>
        🔍 สำรวจคลังข้อมูลความคิดเห็นรีวิวภาษาไทย (Customer Reviews Explorer)
    </div>
    <div style='font-size: 14px; color: #94A3B8; margin-bottom: 20px;'>
        ค้นหาและตรวจสอบข้อความรีวิวผู้บริโภคภาษาไทยจริงในระบบ (Wisesight Corpus)
    </div>
    """, unsafe_allow_html=True)
    
    df_train_explore = load_train_data()
    if not df_train_explore.empty:
        c_filter1, c_filter2 = st.columns([1, 2])
        with c_filter1:
            f_sent = st.selectbox("กรองตามความรู้สึก:", ["ทั้งหมด", "เชิงบวก", "เชิงกลาง", "เชิงลบ"])
        with c_filter2:
            search_txt = st.text_input("ค้นหาคีย์เวิร์ดในรีวิว:", placeholder="เช่น ขนส่ง, ส่งช้า, พัง, คุณภาพ, แอดมิน...")
            
        df_filtered = df_train_explore
        if f_sent != "ทั้งหมด":
            df_filtered = df_filtered[df_filtered["sentiment_th"] == f_sent]
        if search_txt.strip():
            df_filtered = df_filtered[df_filtered["review_text"].str.contains(search_txt, na=False)]
            
        st.write(f"พบรีวิวที่ตรงกับเงื่อนไข: **{len(df_filtered):,}** ข้อความ")
        st.dataframe(
            df_filtered[["review_text", "sentiment_th"]].rename(columns={
                "review_text": "ข้อความรีวิว (Review Text)",
                "sentiment_th": "คลาสความรู้สึก"
            }),
            use_container_width=True,
            height=380
        )
