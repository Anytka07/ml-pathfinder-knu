import pandas as pd
import numpy as np
import random

class Recommender:
    def __init__(self, ontology):
        self.ontology = ontology
        self.udemy = self._load_udemy()
        self.kaggle_items = self._load_kaggle_placeholder()
    
    def _load_udemy(self):
        try:
            df = pd.read_csv("data/udemy/udemy_courses.csv")
            if 'url' not in df.columns:
                df['url'] = df.get('course_id', pd.Series(range(len(df)))).apply(
                    lambda x: f"https://www.udemy.com/course/{x}/"
                )
            print(f"✅ Завантажено {len(df)} Udemy курсів")
            return df
        except:
            print("⚠️ Udemy не знайдено → використовуємо заглушку")
            return pd.DataFrame({
                'course_title': [
                    'Python for Data Science and Machine Learning Bootcamp',
                    'Deep Learning A-Z 2025: Neural Networks and AI',
                    'Machine Learning A-Z: Hands-On Python & R',
                    'Statistics for Data Science and Business Analysis',
                    'Complete A.I. & Machine Learning Bootcamp'
                ],
                'url': [
                    'https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/',
                    'https://www.udemy.com/course/deeplearning/',
                    'https://www.udemy.com/course/machinelearning/',
                    'https://www.udemy.com/course/statistics-for-data-science-and-business-analysis/',
                    'https://www.udemy.com/course/ai-and-machine-learning-bootcamp/'
                ],
                'level': ['Intermediate', 'Advanced', 'Intermediate', 'Beginner', 'All Levels']
            })
    
    def _load_kaggle_placeholder(self):
        return [
            {"title": "Titanic - Machine Learning from Disaster", "type": "Competition", "tags": "classification, beginner"},
            {"title": "House Prices - Advanced Regression Techniques", "type": "Competition", "tags": "regression"},
            {"title": "Digit Recognizer", "type": "Competition", "tags": "computer vision"},
            {"title": "Natural Language Processing with Disaster Tweets", "type": "Competition", "tags": "nlp"},
            {"title": "Store Sales - Time Series Forecasting", "type": "Competition", "tags": "time series"},
        ]
    
    def recommend_courses(
      self,
      observed_profile,
      target_gaps,
      top_n=6
     ):

      recommendations = []

      sample_df = self.udemy.sample(
      min(150, len(self.udemy)),
      random_state=42
     )

      for _, row in sample_df.iterrows():
        
        title = str(
            row.get(
                'course_title',
                ''
            )
        )

        text = title.lower()

        desc = row.get('description', '')
        if pd.notna(desc):
            text += " " + str(desc).lower()


        score = 0
        explanations = []

        # =====================================
        # GAP MATCHING
        # =====================================

        for skill, gap in target_gaps.items():

            if gap < 0.15:
                continue

            keywords = skill.lower().split()

            if any(
                kw in text
                for kw in keywords
            ):

                score += gap

                explanations.append(
                    f"закриває gap по {skill}"
                )

        # =====================================
        # BONUS
        # =====================================

        difficulty = str(
            row.get(
                'level',
                'Intermediate'
            )
        )

        if difficulty == "Beginner":
            score += 0.05

        if score >= 0.15:

            recommendations.append({

                "title": title,

                "explanation": (
                    " • ".join(explanations[:2])
                    if explanations
                    else "розвиває ML skills"
                ),

                "relevance": round(score, 2),

                "difficulty": difficulty,

                "url": row.get(
                    'url',
                    '#'
                )
            })

      recommendations.sort(
        key=lambda x: x['relevance'],
        reverse=True
      )

      return recommendations[:top_n]
    


    def recommend_kaggle(self, target_gaps=None, top_n=5):
        """Рекомендації Kaggle"""
        return self.kaggle_items[:top_n]
    
    def generate_learning_path(self, gaps):
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
        path = []
        for skill, gap_value in sorted_gaps[:7]:
            path.append(f"**{skill}** (gap: {gap_value:.2f}) → теорія + практика на Kaggle")
        return path

