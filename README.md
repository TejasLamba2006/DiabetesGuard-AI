# 🩺 DiabetesGuard AI

AI-Powered Clinical Decision Support System with Explainable Risk Assessment

Built for **GDG on Campus Praxis 2.0** - GenAI + Machine Learning Innovation Showcase

---

## 🎯 Overview

DiabetesGuard AI is a comprehensive diabetes risk assessment platform that combines:
- **Machine Learning** for risk prediction (XGBoost)
- **Explainable AI** using SHAP values
- **Retrieval-Augmented Generation (RAG)** with FAISS for grounded AI responses
- **Generative AI** (Google Gemini) for natural language explanations

### Key Differentiators
- **Dual View Mode**: Doctor (clinical) vs Patient (simplified) interfaces
- **Grounded Responses**: AI only answers from curated medical knowledge base
- **What-If Analysis**: Counterfactual reasoning for behavior change motivation
- **SHAP Explanations**: Transparent risk factor contributions

---

## 🖥️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost Classifier |
| Explainability | SHAP TreeExplainer |
| Vector Store | FAISS |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| GenAI | Google Gemini 1.5 Flash |
| Frontend | Streamlit |
| Visualization | Plotly |

---

## 📂 Project Structure

```
DiabetesGuard-AI/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── train_model.py         # ML model training script
│   ├── knowledge_builder.py   # FAISS knowledge base builder
│   └── rag_engine.py          # RAG query engine
├── models/                    # Trained models (generated)
├── knowledge_base/            # FAISS index (generated)
├── data/
│   └── diabetes_dataset.csv   # Training dataset
├── requirements.txt
├── .streamlit/
│   └── secrets.toml           # API keys (create this)
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/DiabetesGuard-AI.git
cd DiabetesGuard-AI
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Or set environment variable:
```bash
# Windows
set GEMINI_API_KEY=your-key-here

# Linux/Mac
export GEMINI_API_KEY=your-key-here
```

Get your free API key from: https://makersuite.google.com/app/apikey

### 5. Train the model and build knowledge base
```bash
cd src
python train_model.py
python knowledge_builder.py
```

### 6. Run the application
```bash
streamlit run app.py
```

---

## 🚀 Features

### 📋 Risk Assessment
- Input patient clinical data
- Get ML-powered risk prediction
- View SHAP-based explanations
- AI-generated personalized insights

### 💬 AI Chat
- Ask questions about diabetes
- Responses grounded in knowledge base
- Dual mode: Clinical vs Patient-friendly
- Source citations provided

### 🔮 What-If Analysis
- Modify health parameters
- See projected risk changes
- Get actionable recommendations
- Evidence-based interventions

### 📈 Dataset Insights
- View training data statistics
- Compare your metrics to averages
- Understand feature distributions

---

## 🔒 Safety & Grounding

The RAG engine ensures AI responses are:
1. **Relevant**: Only answers diabetes-related queries
2. **Grounded**: Uses only curated medical knowledge
3. **Transparent**: Shows source documents
4. **Safe**: Includes clinical disclaimers

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**.

- Not a diagnostic tool
- Not a substitute for professional medical advice
- All outputs require interpretation by qualified healthcare professionals
- Always consult your doctor for medical decisions

---

## 🏆 Built For

**GDG on Campus Praxis 2.0**
- Theme: Healthcare | Preventive Medicine | Human-AI Collaboration
- Problem: Preventive Risk & Clinical Decision Support

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

Built with ❤️ for Praxis 2.0
