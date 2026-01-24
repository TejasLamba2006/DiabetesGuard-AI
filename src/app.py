from rag_engine import create_rag_engine
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import shap
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="DiabetesGuard AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")


@st.cache_resource
def load_models():
    with open(f"{MODEL_DIR}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/shap_explainer.pkl", "rb") as f:
        shap_explainer = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, label_encoders, scaler, shap_explainer, feature_cols


@st.cache_resource
def load_rag_engine():
    api_key = st.secrets.get(
        "OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
    if not api_key:
        return None
    return create_rag_engine(api_key)


def encode_value(encoder, value):
    value = str(value).strip().lower()
    if value not in encoder.classes_:
        value = encoder.classes_[0]
    return encoder.transform([value])[0]


def get_risk_category(prob):
    if prob < 0.30:
        return "Low Risk", "#10B981", "low"
    elif prob < 0.60:
        return "Moderate Risk", "#F59E0B", "moderate"
    return "High Risk", "#EF4444", "high"


def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 20, 'color': '#1E293B'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#1E293B'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#4F46E5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E5E7EB",
            'steps': [
                {'range': [0, 30], 'color': '#D1FAE5'},
                {'range': [30, 60], 'color': '#FEF3C7'},
                {'range': [60, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': "#1E293B", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    return fig


def create_shap_waterfall(shap_values, feature_names, base_value):
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]

    fig = go.Figure()

    cumulative = base_value
    for i, idx in enumerate(sorted_idx[:6]):
        val = shap_values[idx]
        fig.add_trace(go.Bar(
            x=[val],
            y=[feature_names[idx]],
            orientation='h',
            marker_color='#EF4444' if val > 0 else '#10B981',
            text=f"{val:+.3f}",
            textposition='outside',
            showlegend=False
        ))

    fig.update_layout(
        title="Risk Factor Contributions (SHAP Analysis)",
        xaxis_title="Impact on Risk",
        yaxis_title="",
        height=350,
        margin=dict(l=20, r=80, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#E5E7EB'),
        yaxis=dict(autorange="reversed")
    )
    return fig


def create_comparison_chart(patient_values, dataset_means, feature_names):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Your Values',
        x=feature_names,
        y=patient_values,
        marker_color='#4F46E5'
    ))

    fig.add_trace(go.Bar(
        name='Dataset Average',
        x=feature_names,
        y=dataset_means,
        marker_color='#94A3B8'
    ))

    fig.update_layout(
        title="Your Metrics vs Dataset Average",
        barmode='group',
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#E5E7EB'),
        yaxis=dict(gridcolor='#E5E7EB')
    )
    return fig


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'patient_context' not in st.session_state:
    st.session_state.patient_context = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "patient"

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #0F172A;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    
    .main-header h1 {
        color: #F8FAFC !important;
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .main-header p {
        color: #94A3B8 !important;
        margin: 0.75rem 0 0 0;
        font-size: 1rem;
    }
    
    .header-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
    }
    
    .metric-card {
        background: #1E293B;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin: 1rem 0;
    }
    
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .risk-badge.low {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .risk-badge.moderate {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .risk-badge.high {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .chat-container {
        background: #1E293B;
        border-radius: 12px;
        border: 1px solid #334155;
        padding: 1.5rem;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        line-height: 1.7;
    }
    
    .user-message {
        background: #334155;
        margin-left: 2rem;
        border-left: 3px solid #4F46E5;
        color: #F1F5F9;
    }
    
    .assistant-message {
        background: #1E293B;
        border: 1px solid #334155;
        margin-right: 2rem;
        border-left: 3px solid #10B981;
        color: #E2E8F0;
    }
    
    .message-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .user-message .message-label {
        color: #94A3B8;
    }
    
    .assistant-message .message-label {
        color: #10B981;
    }
    
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 2rem 0;
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        background: #334155;
        color: #F1F5F9;
        border: 1px solid #475569;
    }
    
    .stButton > button:hover {
        background: #475569;
        border-color: #64748B;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #1E293B;
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        color: #94A3B8;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: #334155 !important;
        color: #F1F5F9 !important;
    }
    
    div[data-testid="stMetric"] {
        background: #1E293B;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-weight: 600;
    }
    
    .stForm {
        background: #1E293B;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    
    .stForm [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        color: #F1F5F9 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
    }
    
    .stSlider > div > div > div {
        background: #334155;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #4F46E5, #7C3AED) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #F1F5F9 !important;
    }
    
    p, span, label {
        color: #CBD5E1;
    }
    
    .stMarkdown {
        color: #CBD5E1;
    }
    
    .stAlert {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    
    .stInfo {
        background: rgba(79, 70, 229, 0.1) !important;
        border: 1px solid rgba(79, 70, 229, 0.3) !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
    }
    
    .stExpander {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    
    .stDataFrame {
        background: #1E293B;
    }
    
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #64748B;
    }
    
    .empty-state i {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #475569;
    }
    
    footer {
        text-align: center;
        padding: 2rem;
        color: #64748B;
        font-size: 0.875rem;
        margin-top: 3rem;
        border-top: 1px solid #334155;
    }
    
    .sidebar .stRadio > label {
        color: #94A3B8;
    }
    
    section[data-testid="stSidebar"] {
        background: #1E293B;
        border-right: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

try:
    model, label_encoders, scaler, shap_explainer, feature_cols = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Please run train_model.py first to generate models. Error: {e}")

rag_engine = load_rag_engine()
if rag_engine is None:
    st.sidebar.warning(
        "Set OPENROUTER_API_KEY in secrets or environment for AI chat features")

with st.sidebar:
    st.markdown("### Settings")

    view_mode = st.radio(
        "View Mode",
        ["Patient View", "Doctor View"],
        index=0 if st.session_state.view_mode == "patient" else 1,
        help="Switch between patient-friendly and clinical views"
    )
    st.session_state.view_mode = "patient" if view_mode == "Patient View" else "doctor"

    st.markdown("---")
    st.markdown("### Technology Stack")
    st.info("""
    **DiabetesGuard AI** uses:
    - XGBoost for risk prediction
    - SHAP for explainability
    - FAISS + RAG for grounded AI
    - DeepSeek V3 via OpenRouter
    """)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("""
<div class="main-header">
    <h1><span class="header-icon"><i class="fa-solid fa-heart-pulse"></i></span> DiabetesGuard AI</h1>
    <p>Clinical Decision Support System with Explainable Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

if models_loaded:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Risk Assessment", "AI Chat", "What-If Analysis", "Insights"])

    with tab1:
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.markdown("### Patient Information")

            with st.form("patient_form"):
                c1, c2 = st.columns(2)
                with c1:
                    gender = st.selectbox("Gender", ["Female", "Male"])
                    age = st.number_input("Age", 1, 120, 45)
                    bmi = st.number_input("BMI", 10.0, 60.0, 28.0, 0.1)
                    hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 6.0, 0.1)

                with c2:
                    glucose = st.number_input(
                        "Blood Glucose (mg/dL)", 50, 400, 120)
                    hypertension = st.selectbox(
                        "Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
                    heart_disease = st.selectbox(
                        "Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
                    smoking = st.selectbox("Smoking History", [
                                           "never", "former", "current", "ever", "not current", "unknown"])

                submitted = st.form_submit_button(
                    "Analyze Risk", use_container_width=True, type="primary")

        with col2:
            if submitted:
                features = np.array([[
                    encode_value(label_encoders["gender"], gender.lower()),
                    age,
                    hypertension,
                    heart_disease,
                    encode_value(
                        label_encoders["smoking_history"], smoking.lower()),
                    bmi,
                    hba1c,
                    glucose
                ]])

                features_scaled = scaler.transform(features)
                prob = model.predict_proba(features_scaled)[0][1]
                category, color, emoji = get_risk_category(prob)

                shap_values = shap_explainer.shap_values(features_scaled)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]
                else:
                    shap_vals = shap_values[0]

                st.session_state.patient_context = {
                    'age': age,
                    'gender': gender,
                    'bmi': bmi,
                    'hba1c': hba1c,
                    'glucose': glucose,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'smoking': smoking,
                    'risk_score': prob * 100,
                    'risk_category': category.lower().replace(" ", "_"),
                    'shap_values': shap_vals
                }

                st.markdown("### Risk Assessment Results")

                m1, m2 = st.columns(2)
                with m1:
                    st.plotly_chart(create_risk_gauge(prob),
                                    use_container_width=True)

                with m2:
                    icon_map = {'low': 'fa-check-circle', 'moderate': 'fa-exclamation-circle', 'high': 'fa-times-circle'}
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; margin-top: 2rem;">
                        <div class="risk-badge {emoji}">
                            <i class="fa-solid {icon_map[emoji]}"></i> {category}
                        </div>
                        <p style="margin-top: 1rem; color: #94A3B8;">
                            Based on analysis of 8 clinical parameters
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.session_state.view_mode == "doctor":
                        st.markdown(f"""
                        **Clinical Summary:**
                        - Predicted probability: {prob*100:.2f}%
                        - Confidence interval: ±{(prob*(1-prob)*100/len(features))**.5*1.96:.1f}%
                        - Model: XGBoost Classifier
                        """)

                st.markdown('<div class="section-divider"></div>',
                            unsafe_allow_html=True)

                feature_names = ['Gender', 'Age', 'Hypertension', 'Heart Disease',
                                 'Smoking', 'BMI', 'HbA1c', 'Glucose']

                st.markdown("### Risk Factor Analysis")
                st.plotly_chart(create_shap_waterfall(
                    shap_vals, feature_names, 0), use_container_width=True)

                if rag_engine:
                    st.markdown('<div class="section-divider"></div>',
                                unsafe_allow_html=True)
                    st.markdown("### AI Explanation")

                    with st.spinner("Generating personalized explanation..."):
                        explanation = rag_engine.explain_risk_factors(
                            st.session_state.patient_context,
                            shap_vals,
                            mode=st.session_state.view_mode
                        )
                    st.markdown(explanation)
            else:
                st.markdown("""
                <div class="empty-state">
                    <i class="fa-solid fa-clipboard-list"></i>
                    <h3>Enter patient data and click "Analyze Risk"</h3>
                    <p>Get risk assessment with explainable insights</p>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Chat with DiabetesGuard AI")

        if not rag_engine:
            st.warning(
                "Please configure OPENROUTER_API_KEY to enable chat features")
        else:
            st.markdown("""
            <p style="color: #94A3B8; margin-bottom: 1.5rem;">
            Ask questions about diabetes risk, lifestyle modifications, or your assessment. 
            Responses are grounded in our medical knowledge base and dataset.
            </p>
            """, unsafe_allow_html=True)

            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-label"><i class="fa-solid fa-user"></i> You</div>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="message-label"><i class="fa-solid fa-robot"></i> DiabetesGuard</div>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    if 'sources' in msg and msg['sources']:
                        with st.expander("Sources"):
                            st.write(", ".join(msg['sources']))

            st.markdown('</div>', unsafe_allow_html=True)

            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Your question:",
                    placeholder="e.g., Why is my risk score high? What can I do to lower my HbA1c?"
                )
                send_btn = st.form_submit_button(
                    "Send", use_container_width=True)

            if send_btn and user_input:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })

                with st.spinner("Thinking..."):
                    response = rag_engine.generate_response(
                        user_input,
                        patient_context=st.session_state.patient_context,
                        mode=st.session_state.view_mode
                    )

                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['response'],
                    'sources': response.get('sources', []),
                    'is_grounded': response.get('is_grounded', False)
                })

                st.rerun()

    with tab3:
        st.markdown("### What-If Analysis")
        st.markdown("""
        <p style="color: #94A3B8;">
        Explore how changes in your health metrics could impact your diabetes risk.
        </p>
        """, unsafe_allow_html=True)

        if st.session_state.patient_context:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Modify Parameters")

                new_bmi = st.slider(
                    "Target BMI",
                    min_value=18.5,
                    max_value=45.0,
                    value=float(st.session_state.patient_context['bmi']),
                    step=0.5
                )

                new_hba1c = st.slider(
                    "Target HbA1c (%)",
                    min_value=4.0,
                    max_value=12.0,
                    value=float(st.session_state.patient_context['hba1c']),
                    step=0.1
                )

                new_glucose = st.slider(
                    "Target Blood Glucose (mg/dL)",
                    min_value=70,
                    max_value=300,
                    value=int(st.session_state.patient_context['glucose']),
                    step=5
                )

                new_smoking = st.selectbox(
                    "Smoking Status",
                    ["never", "former", "current"],
                    index=["never", "former", "current"].index(
                        st.session_state.patient_context['smoking']
                    ) if st.session_state.patient_context['smoking'] in ["never", "former", "current"] else 0
                )

            with col2:
                st.markdown("#### Predicted Impact")

                new_features = np.array([[
                    encode_value(
                        label_encoders["gender"], st.session_state.patient_context['gender'].lower()),
                    st.session_state.patient_context['age'],
                    st.session_state.patient_context['hypertension'],
                    st.session_state.patient_context['heart_disease'],
                    encode_value(
                        label_encoders["smoking_history"], new_smoking.lower()),
                    new_bmi,
                    new_hba1c,
                    new_glucose
                ]])

                new_features_scaled = scaler.transform(new_features)
                new_prob = model.predict_proba(new_features_scaled)[0][1]
                new_category, new_color, new_emoji = get_risk_category(
                    new_prob)

                original_risk = st.session_state.patient_context['risk_score']
                new_risk = new_prob * 100
                risk_change = new_risk - original_risk

                st.metric(
                    "New Risk Score",
                    f"{new_risk:.1f}%",
                    f"{risk_change:+.1f}%",
                    delta_color="inverse"
                )

                icon_map2 = {'low': 'fa-check-circle', 'moderate': 'fa-exclamation-circle', 'high': 'fa-times-circle'}
                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>Original:</strong> {original_risk:.1f}% → <strong>New:</strong> {new_risk:.1f}%</p>
                    <div class="risk-badge {new_emoji}">
                        <i class="fa-solid {icon_map2[new_emoji]}"></i> {new_category}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>',
                        unsafe_allow_html=True)

            if rag_engine:
                st.markdown("#### AI-Generated Recommendations")

                if st.button("Generate Personalized Plan"):
                    with st.spinner("Creating personalized recommendations..."):
                        modified_context = st.session_state.patient_context.copy()
                        modified_context['target_bmi'] = new_bmi
                        modified_context['target_hba1c'] = new_hba1c
                        modified_context['target_glucose'] = new_glucose

                        counterfactual = rag_engine.generate_counterfactual(
                            modified_context)
                    st.markdown(counterfactual)
        else:
            st.info(
                "Please complete a risk assessment first to use What-If analysis")

    with tab4:
        st.markdown("### Dataset Insights")

        try:
            with open(f"{MODEL_DIR}/dataset_stats.pkl", "rb") as f:
                stats = pickle.load(f)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Samples", f"{stats['total_samples']:,}")
            m2.metric("Diabetic Cases", f"{stats['diabetic_samples']:,}")
            m3.metric("Non-Diabetic", f"{stats['non_diabetic_samples']:,}")
            m4.metric(
                "Diabetes Rate", f"{stats['diabetic_samples']/stats['total_samples']*100:.1f}%")

            st.markdown('<div class="section-divider"></div>',
                        unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Feature Distributions")

                feature_data = []
                for feat, vals in stats['feature_ranges'].items():
                    feature_data.append({
                        'Feature': feat,
                        'Min': vals['min'],
                        'Max': vals['max'],
                        'Mean': vals['mean']
                    })

                df = pd.DataFrame(feature_data)
                st.dataframe(df, use_container_width=True)

            with col2:
                st.markdown("#### Categories")

                st.write("**Gender:**", ", ".join(stats['gender_categories']))
                st.write("**Smoking History:**",
                         ", ".join(stats['smoking_categories']))

                if st.session_state.patient_context:
                    st.markdown("#### Your Values vs Average")

                    patient_vals = [
                        st.session_state.patient_context['age'],
                        st.session_state.patient_context['bmi'],
                        st.session_state.patient_context['hba1c'],
                        st.session_state.patient_context['glucose']
                    ]

                    dataset_means = [
                        stats['feature_ranges']['age']['mean'],
                        stats['feature_ranges']['bmi']['mean'],
                        stats['feature_ranges']['HbA1c_level']['mean'],
                        stats['feature_ranges']['blood_glucose_level']['mean']
                    ]

                    fig = create_comparison_chart(
                        patient_vals,
                        dataset_means,
                        ['Age', 'BMI', 'HbA1c', 'Glucose']
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except FileNotFoundError:
            st.warning(
                "Dataset statistics not found. Please run train_model.py first.")

st.markdown("""
<div style="text-align: center; padding: 2rem; color: #64748B; font-size: 0.875rem; margin-top: 3rem; border-top: 1px solid #334155;">
    <p><strong>DiabetesGuard AI</strong> | Built for GDG Praxis 2.0</p>
    <p><i class="fa-solid fa-triangle-exclamation"></i> For educational purposes only. Not a diagnostic tool. Consult healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)
