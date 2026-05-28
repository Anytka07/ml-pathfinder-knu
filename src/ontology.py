import json
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class SkillOntology:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = get_model()
        self.skills = self._define_skills()
        self.skill_embeddings = None
        
        self.skill_to_idx = {skill: i for i, skill in enumerate(self.skills)}
    
    def _define_skills(self):
        return [
            "Linear Algebra", "Calculus", "Probability and Statistics", "Optimization",
            "Python Programming", "SQL", "Data Cleaning", "Feature Engineering",
            "Supervised Learning", "Unsupervised Learning", "Deep Learning",
            "Computer Vision", "Natural Language Processing", "Reinforcement Learning",
            "Model Evaluation", "Cross Validation", "Hyperparameter Tuning",
            "MLOps", "Docker", "Cloud Computing", "Big Data", "Git",
            "Research Methodology", "Scientific Writing", "Time Series"
        ]
    
    def get_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=False)
    
    def save(self, path="ontology/"):
        Path(path).mkdir(exist_ok=True)
        with open(f"{path}skills.json", "w", encoding="utf-8") as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)
        with open(f"{path}embeddings.pkl", "wb") as f:
            pickle.dump(self.skill_embeddings, f)