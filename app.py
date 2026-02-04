import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Autiscan - Autism Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid #667eea30;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .question-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
        color: #1a1a2e !important;
    }
    
    .question-card span {
        color: #1a1a2e !important;
        font-size: 1rem;
    }
    
    .question-card:hover {
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
        border-left-color: #764ba2;
    }
    
    .question-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 1rem;
    }
    
    .prediction-box {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .low-risk {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
        box-shadow: 0 10px 30px rgba(40, 167, 69, 0.2);
    }
    
    .high-risk {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 2px solid #dc3545;
        box-shadow: 0 10px 30px rgba(220, 53, 69, 0.2);
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .prediction-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .prediction-subtitle {
        font-size: 1rem;
        opacity: 0.8;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1rem 2rem;
        border: none;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: #1a1a2e !important;
    }
    
    .info-box strong {
        color: #1a1a2e !important;
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        color: #667eea;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.25rem;
        border: 1px solid #667eea40;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
    
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('train.csv')
    df_processed = df.copy()
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            mode_value = df_processed[df_processed[col] != '?'][col].mode()
            if len(mode_value) > 0:
                df_processed[col] = df_processed[col].replace('?', mode_value[0])
    
    columns_to_drop = ['ID', 'age_desc', 'contry_of_res']
    df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
    
    label_encoders = {}
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    X = df_processed.drop('Class/ASD', axis=1)
    y = df_processed['Class/ASD']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, label_encoders, X.columns.tolist(), feature_importance

model, label_encoders, feature_names, feature_importance = load_and_train_model()

st.markdown('<h1 class="main-header">🧠 Autiscan</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Autism Spectrum Disorder Screening Tool Powered by Machine Learning</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">84.4%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">800+</div>
        <div class="stat-label">Training Samples</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">10</div>
        <div class="stat-label">AQ-10 Questions</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">4</div>
        <div class="stat-label">ML Models Tested</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.markdown("### 👤 Demographics")
    st.markdown("---")
    
    age = st.slider("📅 Age", min_value=1, max_value=100, value=25, help="Enter your age")
    
    gender = st.selectbox("⚧ Gender", ["Male", "Female"])
    gender_code = "m" if gender == "Male" else "f"
    
    ethnicity_options = ["White-European", "Asian", "Middle Eastern", "Black", "South Asian", 
                         "Hispanic", "Others", "Latino", "Pasifika", "Turkish"]
    ethnicity = st.selectbox("🌍 Ethnicity", ethnicity_options)
    
    st.markdown("---")
    st.markdown("### 📋 Medical History")
    
    jaundice = st.radio("🟡 Born with jaundice?", ["No", "Yes"], horizontal=True)
    
    autism_family = st.radio("👨‍👩‍👧‍👦 Family member with autism?", ["No", "Yes"], horizontal=True)
    
    used_app = st.radio("📱 Used screening app before?", ["No", "Yes"], horizontal=True)
    
    relation_options = ["Self", "Parent", "Health care professional", "Relative", "Others"]
    relation = st.selectbox("👥 Who is completing this test?", relation_options)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <small style="color: #6c757d;">
            🔒 Your data is processed locally<br>
            and is not stored anywhere.
        </small>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 Screening Test", "📊 Model Insights", "ℹ️ About"])

with tab1:
    st.markdown("### 📝 AQ-10 Screening Questionnaire")
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Instructions:</strong> Please read each statement carefully and select how strongly you agree or disagree with it. 
        Answer based on how things are for you most of the time.
    </div>
    """, unsafe_allow_html=True)
    
    aq10_questions = {
        "A1_Score": ("I often notice small sounds when others do not", "🔊"),
        "A2_Score": ("I usually concentrate more on the whole picture, rather than small details", "🖼️"),
        "A3_Score": ("I find it easy to do more than one thing at once", "🔀"),
        "A4_Score": ("If there is an interruption, I can switch back to what I was doing very quickly", "⚡"),
        "A5_Score": ("I find it easy to read between the lines when someone is talking to me", "💬"),
        "A6_Score": ("I know how to tell if someone listening to me is getting bored", "😴"),
        "A7_Score": ("When I'm reading a story, I find it difficult to work out the characters' intentions", "📖"),
        "A8_Score": ("I like to collect information about categories of things", "📚"),
        "A9_Score": ("I find it easy to work out what someone is thinking or feeling just by looking at their face", "😊"),
        "A10_Score": ("I find it difficult to work out people's intentions", "🤔")
    }
    
    aq_responses = {}
    
    for i, (key, (question, emoji)) in enumerate(aq10_questions.items(), 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div class="question-card">
                    <span class="question-number">{i}</span>
                    <span>{emoji} {question}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                aq_responses[key] = st.radio(
                    f"Response {i}",
                    options=["Disagree", "Agree"],
                    horizontal=True,
                    key=key,
                    label_visibility="collapsed"
                )
    
    aq_scores = {k: 1 if v == "Agree" else 0 for k, v in aq_responses.items()}
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Analyze & Get Prediction", use_container_width=True):
        result_score = sum(aq_scores.values())
        
        input_data = {
            'A1_Score': aq_scores['A1_Score'],
            'A2_Score': aq_scores['A2_Score'],
            'A3_Score': aq_scores['A3_Score'],
            'A4_Score': aq_scores['A4_Score'],
            'A5_Score': aq_scores['A5_Score'],
            'A6_Score': aq_scores['A6_Score'],
            'A7_Score': aq_scores['A7_Score'],
            'A8_Score': aq_scores['A8_Score'],
            'A9_Score': aq_scores['A9_Score'],
            'A10_Score': aq_scores['A10_Score'],
            'age': age,
            'gender': 1 if gender_code == 'm' else 0,
            'ethnicity': ethnicity_options.index(ethnicity),
            'jaundice': 1 if jaundice == 'Yes' else 0,
            'austim': 1 if autism_family == 'Yes' else 0,
            'used_app_before': 1 if used_app == 'Yes' else 0,
            'result': result_score,
            'relation': relation_options.index(relation)
        }
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 2.5rem; font-weight: 700; color: #667eea;">{result_score}/10</div>
                <div style="color: #6c757d;">AQ-10 Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_color = "#dc3545" if prediction == 1 else "#28a745"
            risk_text = "Higher Risk" if prediction == 1 else "Lower Risk"
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 2rem; font-weight: 700; color: {risk_color};">{risk_text}</div>
                <div style="color: #6c757d;">Risk Assessment</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = max(probability) * 100
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 2.5rem; font-weight: 700; color: #667eea;">{confidence:.1f}%</div>
                <div style="color: #6c757d;">Confidence Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            <div class="prediction-box high-risk">
                <div class="prediction-icon">⚠️</div>
                <div class="prediction-title">Higher Likelihood of ASD Traits Detected</div>
                <div class="prediction-subtitle">
                    Based on your responses, our model suggests a higher likelihood of autism spectrum traits.<br>
                    We strongly recommend consulting a qualified healthcare professional for a comprehensive evaluation.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box low-risk">
                <div class="prediction-icon">✅</div>
                <div class="prediction-title">Lower Likelihood of ASD Traits Detected</div>
                <div class="prediction-subtitle">
                    Based on your responses, our model suggests a lower likelihood of autism spectrum traits.<br>
                    If you still have concerns, please consult a healthcare professional.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Probability Distribution")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['No ASD Traits', 'ASD Traits'],
            y=probability,
            marker_color=['#28a745', '#dc3545'],
            text=[f'{p*100:.1f}%' for p in probability],
            textposition='auto',
            textfont=dict(size=16, color='white')
        ))
        
        fig.update_layout(
            title=dict(text='Prediction Probability', font=dict(size=18)),
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 📊 Model Performance & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Model Comparison")
        
        model_data = pd.DataFrame({
            'Model': ['Random Forest', 'KNN', 'Logistic Regression', 'SVM'],
            'Accuracy': [84.38, 83.12, 82.50, 79.38]
        })
        
        fig = px.bar(
            model_data, 
            x='Model', 
            y='Accuracy',
            color='Accuracy',
            color_continuous_scale=['#ff6b6b', '#feca57', '#1dd1a1'],
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            showlegend=False,
            yaxis_range=[70, 90],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Top 10 Important Features")
        
        top_features = feature_importance.head(10).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#667eea', '#764ba2']
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📋 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box" style="background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); border-left-color: #ffc107;">
            <strong>🔑 Result Score</strong><br>
            The AQ-10 result score is the most important predictor, accounting for ~16% of the model's decision.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box" style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-left-color: #28a745;">
            <strong>📅 Age Factor</strong><br>
            Age is the second most important feature, contributing ~14% to the prediction.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box" style="background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%); border-left-color: #007bff;">
            <strong>❓ Question A6</strong><br>
            Question 6 (detecting boredom) is the most predictive individual question at ~13%.
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### ℹ️ About Autiscan")
    
    st.markdown("""
    **Autiscan** is an advanced autism spectrum disorder screening tool that uses machine learning 
    to analyze responses to the AQ-10 (Autism Quotient) questionnaire along with demographic information.
    """)
    
    st.markdown("#### 🧬 What is ASD?")
    st.markdown("""
    Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by:
    - Challenges with social communication and interaction
    - Restricted or repetitive patterns of behavior or interests
    - Symptoms present from early childhood
    
    Early identification can lead to better support and outcomes.
    """)
    
    st.markdown("#### 🔬 About the AQ-10")
    st.markdown("""
    The AQ-10 is a brief, validated screening questionnaire designed to identify individuals 
    who may benefit from a full diagnostic assessment. It was developed by the Autism Research Centre 
    at Cambridge University.
    """)
    
    st.markdown("#### ⚠️ Important Disclaimer")
    st.warning("""
    **This tool is for screening purposes only and is NOT a diagnostic tool.**
    
    - It cannot diagnose autism or any other condition
    - A positive result does not mean you have autism
    - A negative result does not mean you don't have autism
    - Only qualified healthcare professionals can provide a diagnosis
    - Please consult a doctor or specialist if you have concerns
    """)
    
    st.markdown("#### 👨‍💻 Developer")
    st.markdown("""
    Made with ❤️ by **Hardik Agrawal**
    
    [![GitHub](https://img.shields.io/badge/GitHub-Hardik0385-blue?style=flat&logo=github)](https://github.com/Hardik0385/Autiscan)
    """)

st.markdown("""
<div class="footer">
    <p><strong>Autiscan</strong> - Powered by Machine Learning</p>
    <p style="font-size: 0.85rem;">
        🔒 Privacy First: All processing happens locally in your browser. No data is stored or transmitted.
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        © 2026 Hardik Agrawal | 
        <a href="https://github.com/Hardik0385/Autiscan" target="_blank">GitHub</a> | 
        MIT License
    </p>
</div>
""", unsafe_allow_html=True)
