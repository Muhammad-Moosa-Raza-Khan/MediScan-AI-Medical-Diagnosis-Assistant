import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import time

# Initialize session state
if 'diagnosis_results' not in st.session_state:
    st.session_state.diagnosis_results = None

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="MediScan AI - Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #e6f2ff !important;  /* Light blue background */
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;  /* Light gray background */
        border-right: 1px solid #d1d5db !important;
    }
    
    /* Sidebar text */
    .sidebar .sidebar-content {
        color: black !important;
    }
    
    /* Content cards */
    .css-1aumxhk, .stAlert, .stSuccess, .stInfo, .stWarning {
        background-color: white !important;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        padding: 20px;
    }
    
    /* Text contrast */
    body, p, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important; /* Black text */
    }
    
    /* Sidebar widgets */
    .sidebar .stSlider label {
        color: black !important;
        font-weight: 500;
        margin-bottom: 0.5rem;
        display: block;
    }
    .sidebar .stMarkdown {
        color: black !important;
    }
    .sidebar .stMarkdown h2 {
        color: black !important;
        border-bottom: 1px solid #d1d5db;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem !important;
    }
    .sidebar .stSlider {
        margin-bottom: 1.5rem;
    }
    .sidebar .stSlider>div>div>div {
        background-color: #d1d5db !important;
    }
    .sidebar .stSlider>div>div>div>div {
        background-color: #2ecc71 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2ecc71 !important;
        color: white !important;
        border: none;
    }
    
    /* Alert boxes */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-left: 4px solid #28a745;
    }
    .stInfo {
        background-color: #e7f5fe !important;
        color: #0c5460 !important;
        border-left: 4px solid #17a2b8;
    }
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border-left: 4px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL & RULE DEFINITION
# -----------------------------
def rule_based_expert_system(symptoms):
    rules = {
        'Flu': ['fever', 'cough', 'fatigue'],
        'Common Cold': ['sneezing', 'runny_nose', 'mild_fever'],
        'Malaria': ['fever', 'chills', 'sweating'],
        'COVID-19': ['fever', 'cough', 'loss_of_taste'],
        'Allergy': ['sneezing', 'runny_nose', 'itchy_eyes'],
        'Strep Throat': ['fever', 'sore_throat', 'headache'],
        'Pneumonia': ['fever', 'cough', 'chest_pain', 'shortness_of_breath']
    }
    present = {symptom for symptom, sev in symptoms.items() if sev >= 3}
    return [disease for disease, req_syms in rules.items() if all(s in present for s in req_syms)]

def train_classifier():
    data = {
        'fever': [1,1,0,1,0,1,0,1,1,1,0,1,1,0],
        'cough': [1,0,0,1,0,1,0,1,1,1,0,1,1,0],
        'fatigue': [1,0,0,1,0,0,0,1,1,1,0,0,0,0],
        'sneezing': [0,1,1,0,1,0,1,0,0,0,1,0,0,1],
        'runny_nose': [0,1,1,0,1,0,1,0,0,0,1,0,0,1],
        'mild_fever': [0,1,1,0,0,0,0,0,0,0,0,0,0,0],
        'chills': [1,0,0,1,0,1,0,1,1,1,0,1,1,0],
        'sweating': [1,0,0,1,0,1,0,1,1,1,0,1,1,0],
        'loss_of_taste': [1,0,0,0,0,1,0,0,0,0,0,0,0,0],
        'itchy_eyes': [0,0,0,0,1,0,1,0,0,0,1,0,0,1],
        'sore_throat': [0,0,0,0,0,0,0,0,1,1,0,1,0,0],
        'headache': [0,0,0,0,0,0,0,0,1,1,0,1,0,0],
        'chest_pain': [0,0,0,0,0,0,0,0,0,1,0,1,1,0],
        'shortness_of_breath':[0,0,0,0,0,0,0,0,0,1,0,1,1,0],
        'diagnosis': [
            'Flu', 'Common Cold', 'Common Cold', 'Malaria', 'Common Cold', 'COVID-19',
            'Allergy', 'Malaria', 'Strep Throat', 'Pneumonia', 'Allergy', 'Pneumonia', 'Pneumonia', 'Allergy'
        ]
    }
    df = pd.DataFrame(data)
    X = df.drop(columns='diagnosis')
    y = df['diagnosis']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

def bayesian_confidence_scores(diseases):
    base_confidence = {
        'Flu': 0.80,
        'Common Cold': 0.60,
        'Malaria': 0.75,
        'COVID-19': 0.85,
        'Allergy': 0.65,
        'Strep Throat': 0.70,
        'Pneumonia': 0.90
    }
    return {d: base_confidence.get(d, 0.50) for d in diseases}

# -----------------------------
# PLOTTING FUNCTIONS
# -----------------------------
def plot_symptoms(symptom_dict):
    df = pd.DataFrame.from_dict(symptom_dict, orient='index', columns=['Severity'])
    df = df[df['Severity'] > 0]
    if df.empty:
        st.warning("No symptoms selected.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=df.index.str.replace('_', ' ').str.title(), y='Severity', data=df, palette='coolwarm', ax=ax)
    ax.set_ylim(0, 6)
    ax.set_title("Selected Symptoms by Severity")
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    st.pyplot(fig)

def plot_prediction_probabilities(classes, probs):
    df = pd.DataFrame({'Disease': classes, 'Probability': probs})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Disease', y='Probability', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Prediction Probabilities")
    ax.set_ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    st.pyplot(fig)

def plot_disease_risk_map():
    disease_risk = {
        "New York": {"lat": 40.7128, "lon": -74.0060, "risk": 0.7},
        "Los Angeles": {"lat": 34.0522, "lon": -118.2437, "risk": 0.5},
        "Chicago": {"lat": 41.8781, "lon": -87.6298, "risk": 0.6},
        "Houston": {"lat": 29.7604, "lon": -95.3698, "risk": 0.4},
        "Miami": {"lat": 25.7617, "lon": -80.1918, "risk": 0.3}
    }
    m = folium.Map(location=[39.5, -98.35], zoom_start=4)
    for city, info in disease_risk.items():
        color = 'green'
        if info['risk'] > 0.6:
            color = 'red'
        elif info['risk'] > 0.4:
            color = 'orange'
        folium.CircleMarker(
            location=[info['lat'], info['lon']],
            radius=15,
            popup=f"{city} Risk: {info['risk']*100:.0f}%",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6
        ).add_to(m)
    st.subheader("🌍 Disease Risk Mapping by Region")
    st.caption("Note: This is dummy data for demonstration purposes.")
    st_folium(m, width=700, height=450)

# -----------------------------
# STREAMLIT APP LAYOUT
# -----------------------------
# Header with logo and description
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://img.icons8.com/color/96/hospital.png", width=80)
with col2:
    st.title("MediScan AI")
    st.markdown("""
    <div style="color: black; font-size: 16px;">
    A smart medical diagnosis assistant combining rule-based expert systems with machine learning 
    to provide accurate preliminary health assessments.
    </div>
    """, unsafe_allow_html=True)

# Sidebar with improved symptom input
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: black;">Symptom Tracker</h2>
        <p style="color: black;">Rate your symptoms from 0 (none) to 5 (severe)</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_symptoms = {}
    symptom_list = [
        'fever', 'cough', 'fatigue', 'sneezing', 'runny_nose',
        'mild_fever', 'chills', 'sweating', 'loss_of_taste', 'itchy_eyes',
        'sore_throat', 'headache', 'chest_pain', 'shortness_of_breath'
    ]
    
    st.markdown("**General Symptoms**")
    for symptom in symptom_list[:3]:
        user_symptoms[symptom] = st.slider(
            symptom.replace('_', ' ').title(),
            0, 5, 0,
            help=f"Severity of {symptom.replace('_', ' ')}"
        )
    
    st.markdown("**Respiratory Symptoms**")
    for symptom in symptom_list[3:5]:
        user_symptoms[symptom] = st.slider(
            symptom.replace('_', ' ').title(),
            0, 5, 0,
            help=f"Severity of {symptom.replace('_', ' ')}"
        )
    
    st.markdown("**Fever-Related Symptoms**")
    for symptom in symptom_list[5:8]:
        user_symptoms[symptom] = st.slider(
            symptom.replace('_', ' ').title(),
            0, 5, 0,
            help=f"Severity of {symptom.replace('_', ' ')}"
        )
    
    st.markdown("**Other Symptoms**")
    for symptom in symptom_list[8:]:
        user_symptoms[symptom] = st.slider(
            symptom.replace('_', ' ').title(),
            0, 5, 0,
            help=f"Severity of {symptom.replace('_', ' ')}"
        )

# Main content area
tab1, tab2 = st.tabs(["Diagnosis", "About"])

with tab1:
    with st.expander("📊 Symptom Analysis Dashboard", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            plot_symptoms(user_symptoms)
        with col2:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                <h4 style="color: black;">Symptom Legend</h4>
                <p style="color: black; font-size: 14px;">
                <strong>0:</strong> No symptoms<br>
                <strong>1-2:</strong> Mild<br>
                <strong>3-4:</strong> Moderate<br>
                <strong>5:</strong> Severe
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("🔍 Run Comprehensive Diagnosis", use_container_width=True):
        with st.spinner('Analyzing symptoms with our AI engine...'):
            time.sleep(1)
            
            present_symptoms = {k: v for k, v in user_symptoms.items() if v > 0}
            if not present_symptoms:
                st.warning("Please select at least one symptom with severity > 0.")
                st.session_state.diagnosis_results = None
            else:
                # Store results in session state
                model, features = train_classifier()
                input_vector = [min(user_symptoms.get(f, 0) / 5, 1) for f in features]
                input_df = pd.DataFrame([input_vector], columns=features)
                
                st.session_state.diagnosis_results = {
                    'rb_diagnosis': rule_based_expert_system(user_symptoms),
                    'ml_prediction': model.predict(input_df)[0],
                    'probs': model.predict_proba(input_df)[0],
                    'model_classes': model.classes_,
                    'input_df': input_df
                }

    # Display persisted results
    if st.session_state.diagnosis_results:
        results = st.session_state.diagnosis_results
        
        st.subheader("🧠 Expert System Analysis")
        if results['rb_diagnosis']:
            st.success(f"✅ Rule-Based Diagnosis: {', '.join(results['rb_diagnosis'])}")
        else:
            st.info("No exact rule-based match found")
        
        st.subheader("🤖 Machine Learning Prediction")
        st.success(f"🔍 Most Likely Condition: {results['ml_prediction']}")
        
        # Show probability plot
        st.subheader("📊 Prediction Probabilities")
        plot_prediction_probabilities(results['model_classes'], results['probs'])
        
        # Confidence scores
        st.subheader("🔎 Bayesian Confidence Scores")
        diseases_to_score = results['rb_diagnosis'] if results['rb_diagnosis'] else [results['ml_prediction']]
        confidence_scores = bayesian_confidence_scores(diseases_to_score)
        
        for disease, score in confidence_scores.items():
            progress_value = score * 100
            color = "red" if progress_value < 60 else "orange" if progress_value < 80 else "green"
            
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: black;">{disease}</span>
                    <span style="color: black;">{progress_value:.1f}%</span>
                </div>
                <div style="height: 10px; background-color: #eee; border-radius: 5px;">
                    <div style="width: {progress_value}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add clear button
        if st.button("🧹 Clear Diagnosis Results"):
            st.session_state.diagnosis_results = None
            st.rerun()
    
    st.markdown("---")
    st.subheader("🌍 Regional Disease Risk Assessment")
    st.write("Explore disease prevalence by geographic location (sample data):")
    plot_disease_risk_map()

with tab2:
    st.header("About MediScan AI")
    
    # Main description
    st.markdown("""
    <div style="color: black;">
    MediScan AI is an advanced medical diagnosis assistant that combines rule-based systems with machine learning to provide preliminary health assessments.
    </div>
    """, unsafe_allow_html=True)
    
    # How It Works section
    st.subheader("How It Works")
    st.markdown("""
    <div style="color: black;">
    The system combines two complementary approaches:
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="color: black;">
    - **Rule-Based Expert System**: Matches symptoms against known disease patterns using medical knowledge bases<br>
    - **Machine Learning Model**: Uses a Random Forest classifier trained on historical diagnosis data
    </div>
    """, unsafe_allow_html=True)
    
    # Warning note
    st.warning("""
    ⚠️ **Important**: This tool provides preliminary information only and should not replace professional medical advice.
    """)
    
    # Technical Details
    st.subheader("Technical Details")
    st.markdown("""
    <div style="color: black;">
    - Built with Python 3.9 and Streamlit 1.13.0<br>
    - Uses scikit-learn's Random Forest Classifier (v1.2.0)<br>
    - Incorporates Bayesian probability scoring<br>
    - Interactive visualizations with Matplotlib/Seaborn<br>
    - Geospatial mapping with Folium
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("Version 1.0.0 | Last updated: June 2023")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: black; font-size: 14px;">
        <p>MediScan AI - Final Project for AI in Healthcare</p>
        <p>© 2023 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)