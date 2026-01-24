import os
import pickle
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from openai import OpenAI
from typing import List, Dict, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
MODEL_DIR = os.path.join(BASE_DIR, "models")


class RAGEngine:
    def __init__(self, api_key: str):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.index = faiss.read_index(
            os.path.join(KNOWLEDGE_DIR, "knowledge_index.faiss"))

        with open(os.path.join(KNOWLEDGE_DIR, "knowledge_documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = "deepseek/deepseek-chat-v3.1"
        self.max_retries = 3
        self.base_delay = 2

        self.relevance_threshold = 0.3
        self.max_context_docs = 5

    def _generate_text(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)
                        time.sleep(delay)
                        continue
                    return self._get_fallback_response(prompt)
                return f"Error generating response: {error_str}"
        return self._get_fallback_response(prompt)

    def _get_fallback_response(self, prompt: str) -> str:
        if "risk factors" in prompt.lower() or "contributing" in prompt.lower():
            return "Based on your health data, key factors affecting diabetes risk include BMI, HbA1c levels, blood glucose, age, and lifestyle factors like smoking. Higher BMI (>25) and HbA1c (>5.7%) indicate increased risk. Please consult your healthcare provider for personalized advice."
        elif "recommendation" in prompt.lower() or "action plan" in prompt.lower():
            return "General recommendations for diabetes prevention: 1) Maintain healthy weight (BMI 18.5-24.9), 2) Exercise 150 minutes weekly, 3) Eat balanced diet rich in fiber, 4) Monitor blood sugar regularly, 5) Quit smoking if applicable. Consult your doctor for personalized guidance."
        elif "what-if" in prompt.lower() or "counterfactual" in prompt.lower():
            return "Lifestyle changes can significantly impact diabetes risk. Reducing BMI by 5-7% can lower risk by up to 58%. Improving HbA1c through diet and exercise also helps. Small, consistent changes lead to better outcomes over time."
        return "I'm currently experiencing high demand. Based on your health data, I recommend consulting with your healthcare provider for personalized diabetes risk assessment and management strategies."

    def tokenize_query(self, query: str) -> List[str]:
        tokens = self.tokenizer.tokenize(query.lower())
        return tokens

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        relevant_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.relevance_threshold:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                relevant_docs.append(doc)

        return relevant_docs

    def is_query_relevant(self, query: str) -> Tuple[bool, float]:
        diabetes_keywords = [
            'diabetes', 'diabetic', 'blood sugar', 'glucose', 'hba1c', 'a1c',
            'insulin', 'bmi', 'weight', 'obesity', 'hypertension', 'blood pressure',
            'heart disease', 'cardiovascular', 'smoking', 'lifestyle', 'diet',
            'exercise', 'risk', 'prediction', 'health', 'medical', 'patient',
            'age', 'family history', 'complication', 'medication', 'treatment',
            'prediabetes', 'prevention', 'symptom', 'diagnosis', 'screening',
            'cholesterol', 'metabolic', 'pancreas', 'hormone'
        ]

        tokens = self.tokenize_query(query)
        query_lower = query.lower()

        keyword_matches = sum(
            1 for kw in diabetes_keywords if kw in query_lower)
        keyword_score = min(keyword_matches / 3, 1.0)

        relevant_docs = self.retrieve_context(query, top_k=3)
        if relevant_docs:
            semantic_score = max(doc['relevance_score']
                                 for doc in relevant_docs)
        else:
            semantic_score = 0.0

        combined_score = 0.4 * keyword_score + 0.6 * semantic_score
        is_relevant = combined_score >= 0.25

        return is_relevant, combined_score

    def generate_response(self, query: str, patient_context: Optional[Dict] = None,
                          mode: str = "patient") -> Dict:
        is_relevant, relevance_score = self.is_query_relevant(query)

        if not is_relevant:
            return {
                "response": "I can only answer questions related to diabetes, health risk factors, and the clinical data in our system. Please ask about diabetes risk, blood sugar, HbA1c, BMI, lifestyle factors, or related health topics.",
                "relevant_documents": [],
                "is_grounded": False,
                "relevance_score": relevance_score
            }

        relevant_docs = self.retrieve_context(
            query, top_k=self.max_context_docs)

        if not relevant_docs:
            return {
                "response": "I don't have enough information in my knowledge base to answer this question accurately. Please ask about topics covered in our diabetes dataset and medical guidelines.",
                "relevant_documents": [],
                "is_grounded": False,
                "relevance_score": relevance_score
            }

        context_text = "\n\n".join([
            f"[Source: {doc['topic']}]\n{doc['content']}"
            for doc in relevant_docs
        ])

        patient_info = ""
        if patient_context:
            patient_info = f"""
Current Patient Information:
- Age: {patient_context.get('age', 'N/A')} years
- Gender: {patient_context.get('gender', 'N/A')}
- BMI: {patient_context.get('bmi', 'N/A')}
- HbA1c: {patient_context.get('hba1c', 'N/A')}%
- Blood Glucose: {patient_context.get('glucose', 'N/A')} mg/dL
- Hypertension: {'Yes' if patient_context.get('hypertension') else 'No'}
- Heart Disease: {'Yes' if patient_context.get('heart_disease') else 'No'}
- Smoking History: {patient_context.get('smoking', 'N/A')}
- Diabetes Risk Score: {patient_context.get('risk_score', 'N/A')}%
- Risk Category: {patient_context.get('risk_category', 'N/A')}
"""

        if mode == "doctor":
            system_prompt = """You are a clinical decision support assistant for healthcare professionals.
Provide responses using medical terminology and clinical guidelines.
Focus on evidence-based recommendations and diagnostic considerations.
IMPORTANT: Only use information from the provided context. Do not make up facts.
If the context doesn't contain relevant information, say so clearly."""
        else:
            system_prompt = """You are a friendly health assistant helping patients understand their diabetes risk.
Use simple, clear language avoiding complex medical terms.
Be empathetic and encouraging while being accurate.
IMPORTANT: Only use information from the provided context. Do not make up facts.
Always recommend consulting healthcare providers for medical decisions."""

        prompt = f"""{system_prompt}

KNOWLEDGE BASE CONTEXT:
{context_text}

{patient_info}

USER QUESTION: {query}

Provide a helpful, accurate response based ONLY on the context provided above. If you cannot answer from the context, clearly state that."""

        response_text = self._generate_text(prompt)

        return {
            "response": response_text,
            "relevant_documents": relevant_docs,
            "is_grounded": True,
            "relevance_score": relevance_score,
            "sources": [doc['topic'] for doc in relevant_docs]
        }

    def explain_risk_factors(self, patient_context: Dict, shap_values: Optional[np.ndarray] = None,
                             mode: str = "patient") -> str:
        relevant_docs = self.retrieve_context(
            "diabetes risk factors explanation", top_k=4)
        context_text = "\n\n".join([doc['content'] for doc in relevant_docs])

        feature_names = ['Gender', 'Age', 'Hypertension', 'Heart Disease',
                         'Smoking History', 'BMI', 'HbA1c', 'Blood Glucose']

        shap_info = ""
        if shap_values is not None:
            sorted_indices = np.argsort(np.abs(shap_values))[::-1]
            top_factors = [(feature_names[i], shap_values[i])
                           for i in sorted_indices[:4]]
            shap_info = "Key contributing factors (from model analysis):\n"
            for feat, val in top_factors:
                direction = "increases" if val > 0 else "decreases"
                shap_info += f"- {feat}: {direction} risk (impact: {abs(val):.3f})\n"

        if mode == "doctor":
            prompt = f"""As a clinical decision support system, explain this patient's diabetes risk factors.

MEDICAL CONTEXT:
{context_text}

PATIENT DATA:
- Age: {patient_context.get('age')} years
- Gender: {patient_context.get('gender')}
- BMI: {patient_context.get('bmi'):.1f}
- HbA1c: {patient_context.get('hba1c'):.1f}%
- Fasting Glucose: {patient_context.get('glucose')} mg/dL
- Hypertension: {'Present' if patient_context.get('hypertension') else 'Absent'}
- Heart Disease: {'Present' if patient_context.get('heart_disease') else 'Absent'}
- Smoking: {patient_context.get('smoking')}
- Calculated Risk: {patient_context.get('risk_score'):.1f}%

{shap_info}

Provide a clinical interpretation of these risk factors with evidence-based recommendations. Use appropriate medical terminology."""
        else:
            prompt = f"""Help this patient understand their diabetes risk in simple terms.

HEALTH INFORMATION:
{context_text}

YOUR HEALTH DATA:
- Age: {patient_context.get('age')} years
- BMI: {patient_context.get('bmi'):.1f}
- HbA1c Level: {patient_context.get('hba1c'):.1f}%
- Blood Sugar: {patient_context.get('glucose')} mg/dL
- High Blood Pressure: {'Yes' if patient_context.get('hypertension') else 'No'}
- Heart Problems: {'Yes' if patient_context.get('heart_disease') else 'No'}
- Smoking: {patient_context.get('smoking')}
- Your Risk Level: {patient_context.get('risk_score'):.1f}%

{shap_info}

Explain in friendly, simple language what factors contribute to their risk and what they can do about it. Be encouraging and positive."""

        return self._generate_text(prompt)

    def generate_counterfactual(self, patient_context: Dict, target_reduction: float = 0.2) -> str:
        relevant_docs = self.retrieve_context(
            "diabetes lifestyle intervention weight loss exercise", top_k=3)
        context_text = "\n\n".join([doc['content'] for doc in relevant_docs])

        current_risk = patient_context.get('risk_score', 50)
        target_risk = max(10, current_risk * (1 - target_reduction))

        prompt = f"""Based on the medical context below, suggest specific, achievable changes this patient could make to reduce their diabetes risk.

MEDICAL CONTEXT:
{context_text}

CURRENT PATIENT STATUS:
- Age: {patient_context.get('age')} years
- Current BMI: {patient_context.get('bmi'):.1f}
- Current HbA1c: {patient_context.get('hba1c'):.1f}%
- Blood Glucose: {patient_context.get('glucose')} mg/dL
- Hypertension: {'Yes' if patient_context.get('hypertension') else 'No'}
- Smoking: {patient_context.get('smoking')}
- Current Risk Score: {current_risk:.1f}%
- Target Risk Score: {target_risk:.1f}%

Provide specific, actionable "what-if" scenarios. For example:
- "If you reduce your BMI from X to Y, your risk could decrease by..."
- "If you improve HbA1c from X% to Y%, expected risk reduction is..."

Be specific with numbers and timeframes. Base recommendations ONLY on the medical context provided."""

        return self._generate_text(prompt)

    def get_recommendations(self, patient_context: Dict, mode: str = "patient") -> str:
        risk_level = patient_context.get('risk_category', 'moderate')
        query = f"diabetes {risk_level} risk recommendations treatment lifestyle"
        relevant_docs = self.retrieve_context(query, top_k=4)
        context_text = "\n\n".join([doc['content'] for doc in relevant_docs])

        if mode == "doctor":
            prompt = f"""Generate clinical recommendations for this patient.

CLINICAL GUIDELINES:
{context_text}

PATIENT PROFILE:
- Age: {patient_context.get('age')} years
- Risk Category: {patient_context.get('risk_category')}
- Risk Score: {patient_context.get('risk_score'):.1f}%
- BMI: {patient_context.get('bmi'):.1f}
- HbA1c: {patient_context.get('hba1c'):.1f}%
- Glucose: {patient_context.get('glucose')} mg/dL
- Comorbidities: {'Hypertension' if patient_context.get('hypertension') else 'None'}{', Heart Disease' if patient_context.get('heart_disease') else ''}

Provide prioritized clinical recommendations with:
1. Immediate actions (within 1 week)
2. Short-term goals (1-3 months)
3. Long-term management (3-12 months)
Include specific targets and follow-up schedule."""
        else:
            prompt = f"""Create a friendly action plan for this patient.

HEALTH GUIDANCE:
{context_text}

YOUR PROFILE:
- Risk Level: {patient_context.get('risk_category')}
- Risk Score: {patient_context.get('risk_score'):.1f}%
- BMI: {patient_context.get('bmi'):.1f}
- Blood Sugar: {patient_context.get('glucose')} mg/dL

Provide a simple, encouraging action plan with:
1. First steps you can take today
2. Goals for the next month
3. Long-term healthy habits

Keep it positive and achievable. Use simple language."""

        return self._generate_text(prompt)


def create_rag_engine(api_key: str) -> RAGEngine:
    return RAGEngine(api_key)
