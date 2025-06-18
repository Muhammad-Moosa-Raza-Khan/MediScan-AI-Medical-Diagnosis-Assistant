# MediScan AI - Medical Diagnosis Assistant

![MediScan AI Logo](https://img.icons8.com/color/96/hospital.png)  
*A hybrid rule-based and machine learning system for preliminary medical diagnosis.*

---

## 🚀 Overview
**MediScan AI** is a Streamlit-powered web application that combines **rule-based expert systems** with **machine learning** (Random Forest) to provide preliminary health assessments based on user-reported symptoms. Designed for educational purposes, it demonstrates how AI can assist in healthcare diagnostics while emphasizing the importance of professional medical advice.

---

## ✨ Key Features
- **Dual-Diagnosis Engine**:  
  - **Rule-Based System**: Matches symptoms against predefined medical rules (e.g., "Fever + Cough + Fatigue = Flu").  
  - **ML Model**: Random Forest classifier trained on synthetic diagnosis data.  
- **Interactive Dashboard**:  
  - Symptom severity visualization (`Matplotlib/Seaborn`).  
  - Prediction probability charts.  
  - Geospatial disease risk map (`Folium`).  
- **Confidence Scoring**: Bayesian confidence metrics for each diagnosis.  
- **User-Friendly UI**:  
  - Clean, accessible design with a responsive layout.  
  - Symptom severity sliders (0-5) categorized by type.  

---

## 🛠️ Technical Stack
| Component               | Technology/Package       |
|-------------------------|--------------------------|
| Frontend                | Streamlit                |
| Backend                 | Python 3.9+              |
| Machine Learning        | scikit-learn (Random Forest) |
| Data Handling           | pandas, NumPy            |
| Visualizations          | Matplotlib, Seaborn, Folium |
| Deployment              | Streamlit Cloud, Docker  |

---

