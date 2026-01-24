import os
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = "../knowledge_base"
DATA_PATH = "../data/diabetes_dataset.csv"
MODEL_DIR = "../models"

MEDICAL_KNOWLEDGE = [
    {
        "topic": "diabetes_definition",
        "content": "Diabetes is a chronic metabolic disorder characterized by elevated blood glucose levels. Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't produce enough insulin. It is often associated with obesity, sedentary lifestyle, and genetic factors.",
        "category": "general"
    },
    {
        "topic": "hba1c_interpretation",
        "content": "HbA1c (glycated hemoglobin) measures average blood sugar over 2-3 months. Normal: below 5.7%. Prediabetes: 5.7% to 6.4%. Diabetes: 6.5% or higher. HbA1c is a key indicator for diabetes diagnosis and management. Higher HbA1c levels indicate poorer blood sugar control and increased risk of complications.",
        "category": "biomarkers"
    },
    {
        "topic": "blood_glucose_interpretation",
        "content": "Fasting blood glucose levels interpretation: Normal: below 100 mg/dL. Prediabetes: 100-125 mg/dL. Diabetes: 126 mg/dL or higher. Random blood glucose above 200 mg/dL with symptoms indicates diabetes. Blood glucose monitoring is essential for diabetes management.",
        "category": "biomarkers"
    },
    {
        "topic": "bmi_and_diabetes",
        "content": "Body Mass Index (BMI) strongly correlates with diabetes risk. Normal BMI: 18.5-24.9. Overweight: 25-29.9. Obese: 30 or higher. Obesity increases diabetes risk by 7x compared to normal weight. Weight loss of 5-10% can significantly reduce diabetes risk and improve blood sugar control.",
        "category": "risk_factors"
    },
    {
        "topic": "age_and_diabetes",
        "content": "Age is a significant risk factor for Type 2 diabetes. Risk increases substantially after age 45. People over 65 have the highest rates of diabetes. However, Type 2 diabetes is increasingly diagnosed in younger adults and children due to rising obesity rates.",
        "category": "risk_factors"
    },
    {
        "topic": "hypertension_diabetes_link",
        "content": "Hypertension and diabetes frequently coexist. About 75% of adults with diabetes have hypertension. Both conditions share common risk factors including obesity and insulin resistance. Having both conditions significantly increases risk of heart disease, stroke, and kidney disease.",
        "category": "comorbidities"
    },
    {
        "topic": "heart_disease_diabetes",
        "content": "People with diabetes have 2-4 times higher risk of heart disease. Diabetes damages blood vessels and nerves controlling the heart. Cardiovascular disease is the leading cause of death among diabetics. Managing blood sugar, blood pressure, and cholesterol is crucial for heart health.",
        "category": "comorbidities"
    },
    {
        "topic": "smoking_diabetes_risk",
        "content": "Smoking increases diabetes risk by 30-40%. Smokers with diabetes have higher risk of complications including heart disease, kidney disease, and poor circulation. Quitting smoking improves insulin sensitivity and reduces complication risk. Former smokers still have elevated risk for several years.",
        "category": "risk_factors"
    },
    {
        "topic": "gender_diabetes",
        "content": "Diabetes affects both genders but with some differences. Men develop diabetes at lower BMI levels. Women with gestational diabetes have 50% higher risk of Type 2 diabetes later. Hormonal changes during menopause can affect blood sugar control in women.",
        "category": "demographics"
    },
    {
        "topic": "lifestyle_modifications",
        "content": "Lifestyle modifications can prevent or delay Type 2 diabetes. Recommendations include: 150 minutes of moderate exercise weekly, healthy diet rich in fiber and low in processed foods, maintaining healthy weight, limiting alcohol consumption, stress management, and adequate sleep of 7-9 hours.",
        "category": "interventions"
    },
    {
        "topic": "diet_recommendations",
        "content": "Diet recommendations for diabetes prevention and management: Focus on whole grains, legumes, vegetables, and lean proteins. Limit refined carbohydrates, sugary beverages, and processed foods. Mediterranean diet and DASH diet have shown benefits. Portion control and regular meal timing help maintain stable blood sugar.",
        "category": "interventions"
    },
    {
        "topic": "exercise_benefits",
        "content": "Regular physical activity improves insulin sensitivity and helps control blood sugar. Benefits include: reduced HbA1c by 0.5-0.7%, improved cardiovascular health, weight management, and better mental health. Both aerobic exercise and resistance training are beneficial. Start gradually and aim for consistency.",
        "category": "interventions"
    },
    {
        "topic": "monitoring_frequency",
        "content": "Blood glucose monitoring recommendations vary by individual needs. For prediabetes: HbA1c test every 6-12 months. For diabetes: HbA1c every 3 months initially, then every 6 months if stable. Self-monitoring frequency depends on treatment type and blood sugar control.",
        "category": "clinical_guidelines"
    },
    {
        "topic": "medication_options",
        "content": "Diabetes medications include: Metformin (first-line for Type 2), Sulfonylureas, DPP-4 inhibitors, GLP-1 receptor agonists, SGLT2 inhibitors, and Insulin. Treatment choice depends on individual factors. Many patients require combination therapy. Regular medication review is important.",
        "category": "treatment"
    },
    {
        "topic": "complications_prevention",
        "content": "Diabetes complications can be prevented through good control. Key targets: HbA1c below 7%, blood pressure below 130/80 mmHg, LDL cholesterol below 100 mg/dL. Regular screening for eye, kidney, and nerve complications is essential. Foot care and dental care are also important.",
        "category": "clinical_guidelines"
    },
    {
        "topic": "risk_stratification",
        "content": "Diabetes risk stratification helps prioritize interventions. Low risk: probability below 30%, moderate risk monitoring needed. Moderate risk: 30-60%, lifestyle intervention recommended. High risk: above 60%, immediate medical consultation and possibly pharmacological intervention required.",
        "category": "clinical_guidelines"
    },
    {
        "topic": "prediabetes_management",
        "content": "Prediabetes is reversible with proper intervention. Without intervention, 15-30% progress to diabetes within 5 years. Intensive lifestyle modification can reduce progression by 58%. Weight loss of 7% body weight and 150 minutes weekly exercise are key targets.",
        "category": "interventions"
    },
    {
        "topic": "family_history",
        "content": "Family history significantly impacts diabetes risk. Having one parent with Type 2 diabetes increases risk by 40%. Both parents: risk increases to 70%. First-degree relatives should undergo regular screening starting at age 35 or earlier if overweight.",
        "category": "risk_factors"
    }
]


def generate_dataset_insights(df):
    insights = []

    diabetic = df[df['diabetes'] == 1]
    non_diabetic = df[df['diabetes'] == 0]

    insights.append({
        "topic": "dataset_overview",
        "content": f"Our diabetes dataset contains {len(df)} patient records with {len(diabetic)} diabetic cases ({len(diabetic)/len(df)*100:.1f}%) and {len(non_diabetic)} non-diabetic cases ({len(non_diabetic)/len(df)*100:.1f}%). This data is used for risk prediction and analysis.",
        "category": "dataset"
    })

    insights.append({
        "topic": "age_distribution",
        "content": f"In our dataset, patient ages range from {df['age'].min()} to {df['age'].max()} years with mean age of {df['age'].mean():.1f} years. Diabetic patients have mean age of {diabetic['age'].mean():.1f} years compared to {non_diabetic['age'].mean():.1f} for non-diabetic patients.",
        "category": "dataset"
    })

    insights.append({
        "topic": "bmi_statistics",
        "content": f"BMI in our dataset ranges from {df['bmi'].min():.1f} to {df['bmi'].max():.1f} with mean of {df['bmi'].mean():.1f}. Diabetic patients have mean BMI of {diabetic['bmi'].mean():.1f} compared to {non_diabetic['bmi'].mean():.1f} for non-diabetic, showing higher BMI correlates with diabetes.",
        "category": "dataset"
    })

    insights.append({
        "topic": "hba1c_statistics",
        "content": f"HbA1c levels in our dataset range from {df['HbA1c_level'].min():.1f}% to {df['HbA1c_level'].max():.1f}% with mean of {df['HbA1c_level'].mean():.2f}%. Diabetic patients show mean HbA1c of {diabetic['HbA1c_level'].mean():.2f}% versus {non_diabetic['HbA1c_level'].mean():.2f}% for non-diabetic.",
        "category": "dataset"
    })

    insights.append({
        "topic": "glucose_statistics",
        "content": f"Blood glucose levels range from {df['blood_glucose_level'].min()} to {df['blood_glucose_level'].max()} mg/dL with mean of {df['blood_glucose_level'].mean():.1f} mg/dL. Diabetic patients average {diabetic['blood_glucose_level'].mean():.1f} mg/dL compared to {non_diabetic['blood_glucose_level'].mean():.1f} mg/dL for non-diabetic.",
        "category": "dataset"
    })

    hyp_diabetic = diabetic['hypertension'].mean() * 100
    hyp_non_diabetic = non_diabetic['hypertension'].mean() * 100
    insights.append({
        "topic": "hypertension_correlation",
        "content": f"In our dataset, {hyp_diabetic:.1f}% of diabetic patients have hypertension compared to {hyp_non_diabetic:.1f}% of non-diabetic patients. This confirms the strong correlation between hypertension and diabetes risk.",
        "category": "dataset"
    })

    hd_diabetic = diabetic['heart_disease'].mean() * 100
    hd_non_diabetic = non_diabetic['heart_disease'].mean() * 100
    insights.append({
        "topic": "heart_disease_correlation",
        "content": f"Heart disease is present in {hd_diabetic:.1f}% of diabetic patients versus {hd_non_diabetic:.1f}% of non-diabetic patients in our dataset, demonstrating the cardiovascular-diabetes connection.",
        "category": "dataset"
    })

    smoking_stats = df.groupby('smoking_history')['diabetes'].mean() * 100
    smoking_text = ", ".join(
        [f"{cat}: {val:.1f}% diabetes rate" for cat, val in smoking_stats.items()])
    insights.append({
        "topic": "smoking_correlation",
        "content": f"Smoking history and diabetes rates in our dataset: {smoking_text}. Current and former smokers show elevated diabetes risk.",
        "category": "dataset"
    })

    gender_stats = df.groupby('gender')['diabetes'].mean() * 100
    gender_text = ", ".join(
        [f"{g}: {v:.1f}% diabetes rate" for g, v in gender_stats.items()])
    insights.append({
        "topic": "gender_distribution",
        "content": f"Gender distribution in our dataset shows: {gender_text}. Our model accounts for gender differences in diabetes risk assessment.",
        "category": "dataset"
    })

    return insights


def build_knowledge_base():
    print("Loading dataset for insights...")
    df = pd.read_csv(DATA_PATH)

    print("Generating dataset insights...")
    dataset_insights = generate_dataset_insights(df)

    all_knowledge = MEDICAL_KNOWLEDGE + dataset_insights

    print("Loading sentence transformer model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding knowledge documents...")
    documents = [item['content'] for item in all_knowledge]
    embeddings = encoder.encode(documents, show_progress_bar=True)
    embeddings = embeddings.astype('float32')

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    faiss.write_index(index, f"{KNOWLEDGE_DIR}/knowledge_index.faiss")

    with open(f"{KNOWLEDGE_DIR}/knowledge_documents.pkl", "wb") as f:
        pickle.dump(all_knowledge, f)

    with open(f"{KNOWLEDGE_DIR}/knowledge_metadata.json", "w") as f:
        metadata = {
            "total_documents": len(all_knowledge),
            "medical_guidelines": len(MEDICAL_KNOWLEDGE),
            "dataset_insights": len(dataset_insights),
            "embedding_dimension": dimension,
            "encoder_model": "all-MiniLM-L6-v2",
            "categories": list(set([item['category'] for item in all_knowledge]))
        }
        json.dump(metadata, f, indent=2)

    print(f"\nKnowledge base built successfully!")
    print(f"Total documents: {len(all_knowledge)}")
    print(f"Medical guidelines: {len(MEDICAL_KNOWLEDGE)}")
    print(f"Dataset insights: {len(dataset_insights)}")
    print(f"Files saved in: {KNOWLEDGE_DIR}")

    return index, all_knowledge, encoder


if __name__ == "__main__":
    build_knowledge_base()
