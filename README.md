# DiabetesGuard AI

**Clinical Decision Support System with Explainable Risk Assessment**

Built for GDG on Campus Praxis 2.0 | GenAI + Machine Learning Track

---

## Abstract

DiabetesGuard AI implements a multi-layered approach to diabetes risk prediction combining gradient boosting classification with retrieval-augmented generation (RAG) for grounded natural language explanations. The system achieves 97.78% ROC-AUC on the test set while maintaining full explainability through SHAP (SHapley Additive exPlanations) values.

---

## Architecture Overview

```
+---------------------------------------------------------------------+
|                        DiabetesGuard AI                             |
+---------------------------------------------------------------------+
|  +--------------+    +--------------+    +----------------------+   |
|  |   Streamlit  |--->|  XGBoost     |--->|  SHAP Explainer      |   |
|  |   Frontend   |    |  Classifier  |    |  (TreeExplainer)     |   |
|  +--------------+    +--------------+    +----------------------+   |
|         |                                          |                |
|         v                                          v                |
|  +--------------+    +--------------+    +----------------------+   |
|  |  RAG Engine  |--->|    FAISS     |--->|  DeepSeek V3         |   |
|  |  + BERT      |    |  IndexFlatIP |    |  (OpenRouter)        |   |
|  +--------------+    +--------------+    +----------------------+   |
+---------------------------------------------------------------------+
```

---

## Technical Stack

| Layer | Component | Specification |
|-------|-----------|---------------|
| ML Model | XGBoost Classifier | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` |
| Explainability | SHAP TreeExplainer | Shapley value computation for feature attribution |
| Vector Store | FAISS IndexFlatIP | Inner product similarity with L2 normalization |
| Embeddings | Sentence-BERT | `all-MiniLM-L6-v2` (384-dim vectors) |
| Tokenization | BERT Tokenizer | `bert-base-uncased` for query validation |
| LLM | DeepSeek V3 | Via OpenRouter API (free tier) |
| Frontend | Streamlit | v1.28+ with custom CSS |
| Visualization | Plotly | Interactive gauge and waterfall charts |

---

## Project Structure

```
DiabetesGuard-AI/
├── src/
│   ├── app.py                 # Streamlit application (755 LOC)
│   ├── train_model.py         # Model training pipeline
│   ├── knowledge_builder.py   # FAISS index construction
│   └── rag_engine.py          # RAG query engine with grounding
├── models/                    # Serialized models (pkl)
│   ├── model.pkl              # XGBoost classifier
│   ├── shap_explainer.pkl     # SHAP TreeExplainer
│   ├── scaler.pkl             # StandardScaler
│   └── label_encoders.pkl     # Categorical encoders
├── knowledge_base/            # Vector store
│   ├── knowledge_index.faiss  # FAISS index (27 documents)
│   └── knowledge_documents.pkl
├── data/
│   └── diabetes_dataset.csv   # 100k samples
├── requirements.txt
├── .env                       # API credentials
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip package manager
- OpenRouter API key (free)

### Setup

```bash
# Clone repository
git clone https://github.com/TejasLamba2006/DiabetesGuard-AI.git
cd DiabetesGuard-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix

# Install dependencies
pip install -r requirements.txt

# Configure API
echo "OPENROUTER_API_KEY=your-key-here" > .env

# Train model and build knowledge base
python src/train_model.py
python src/knowledge_builder.py

# Launch application
streamlit run src/app.py
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9778 |
| Accuracy | 96.2% |
| Precision | 0.94 |
| Recall | 0.91 |
| F1 Score | 0.92 |

### Feature Importance (SHAP)

1. HbA1c Level (0.342)
2. Blood Glucose (0.287)
3. BMI (0.156)
4. Age (0.098)
5. Smoking History (0.067)
6. Hypertension (0.031)
7. Heart Disease (0.012)
8. Gender (0.007)

---

## RAG Implementation

### Query Processing Pipeline

1. **Tokenization**: BERT tokenizer extracts query tokens
2. **Relevance Check**: Keyword matching + semantic similarity scoring
3. **Retrieval**: FAISS k-NN search (k=5, threshold=0.3)
4. **Context Assembly**: Top documents concatenated with patient context
5. **Generation**: DeepSeek V3 with grounding constraints

### Knowledge Base Composition

- 18 curated medical guidelines (ADA, WHO standards)
- 9 dataset-derived statistical insights
- Total: 27 indexed documents

---

## API Reference

### RAGEngine Class

```python
class RAGEngine:
    def __init__(self, api_key: str)
    def generate_response(query: str, patient_context: dict, mode: str) -> dict
    def explain_risk_factors(patient_context: dict, shap_values: np.ndarray) -> str
    def generate_counterfactual(patient_context: dict) -> str
    def get_recommendations(patient_context: dict, mode: str) -> str
```

### Response Schema

```json
{
  "response": "string",
  "relevant_documents": ["array"],
  "is_grounded": "boolean",
  "relevance_score": "float",
  "sources": ["array"]
}
```

---

## Security Considerations

- API keys stored in environment variables (`.env` gitignored)
- No patient data persistence
- Query validation prevents prompt injection
- Grounded responses prevent hallucination

---

## Limitations

- Model trained on single dataset (Provided by GDG)
- Free tier LLM rate limits may affect response time
- Not validated for clinical deployment
- English language only

---

## Disclaimer

**For educational and research purposes only.**

This system is not a diagnostic tool and should not be used for medical decisions. All outputs require interpretation by qualified healthcare professionals. Always consult a licensed physician for medical advice.

---

## License

MIT License

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

---

Built for GDG on Campus Praxis 2.0 Hackathon
