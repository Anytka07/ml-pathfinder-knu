import numpy as np
import pandas as pd
from src.ontology import SkillOntology


class ProfileManager:

    def __init__(self, ontology):

        self.ontology = ontology

        self.target_profiles = self._define_target_profiles()

    # =====================================================
    # TARGET ROLES
    # =====================================================

    def _define_target_profiles(self):

        return {

            "ML Engineer": {

                "Python Programming": 0.95,
                "Supervised Learning": 0.90,
                "Deep Learning": 0.85,
                "MLOps": 0.90,
                "Docker": 0.80,
                "Cloud Computing": 0.75,
                "Model Evaluation": 0.85
            },

            "Data Analyst": {

                "SQL": 0.90,
                "Probability and Statistics": 0.90,
                "Data Cleaning": 0.95,
                "Supervised Learning": 0.80,
                "Visualization": 0.85
            },

            "Researcher": {

                "Research Methodology": 0.95,
                "Scientific Writing": 0.90,
                "Probability and Statistics": 0.90,
                "Deep Learning": 0.80
            }
        }

    # =====================================================
    # CREATE OBSERVED PROFILE
    # =====================================================

    def create_observed_profile(
        self,
        syllabi_texts,
        self_assessment=None,
        udemy_texts=None
    ):

        """
        Формує observed profile
        як explicit skill vector
        """

        observed = {

            skill: 0.0
            for skill in self.ontology.skills
        }

        # =================================================
        # 1. SELF-ASSESSMENT
        # =================================================

        if self_assessment:

            for skill, score in self_assessment.items():

                if skill in observed:

                    observed[skill] = max(
                        observed[skill],
                        float(score)
                    )

        # =================================================
        # 2. SYLLABUS ANALYSIS
        # =================================================

        if syllabi_texts:

            if isinstance(syllabi_texts, list):

                text = " ".join(
                    syllabi_texts
                ).lower()

            else:

                text = str(
                    syllabi_texts
                ).lower()

            keyword_map = {

                "Python Programming": [
                    "python",
                    "numpy",
                    "pandas"
                ],

                "SQL": [
                    "sql",
                    "database",
                    "mysql",
                    "postgresql"
                ],

                "Probability and Statistics": [
                    "statistics",
                    "probability",
                    "statistical",
                    "distribution"
                ],

                "Data Cleaning": [
                    "cleaning",
                    "preprocessing",
                    "missing values",
                    "data preparation"
                ],

                "Visualization": [
                    "visualization",
                    "matplotlib",
                    "seaborn",
                    "plotly",
                    "dashboard"
                ],

                "Supervised Learning": [
                    "classification",
                    "regression",
                    "supervised learning"
                ],

                "Deep Learning": [
                    "deep learning",
                    "neural network",
                    "cnn",
                    "rnn",
                    "transformer"
                ],

                "MLOps": [
                    "mlops",
                    "deployment",
                    "pipeline"
                ],

                "Docker": [
                    "docker",
                    "containerization"
                ],

                "Cloud Computing": [
                    "aws",
                    "azure",
                    "gcp",
                    "cloud"
                ],

                "Model Evaluation": [
                    "cross validation",
                    "evaluation",
                    "metrics",
                    "f1-score"
                ],

                "Research Methodology": [
                    "research",
                    "methodology",
                    "scientific research"
                ],

                "Scientific Writing": [
                    "paper",
                    "scientific writing",
                    "publication"
                ]
            }

            for skill, keywords in keyword_map.items():

                matches = sum(
                    kw in text
                    for kw in keywords
                )

                if matches > 0:

                    boost = min(
                        0.15 * matches,
                        0.35
                    )

                    observed[skill] = min(
                        1.0,
                        observed[skill] + boost
                    )

        # =================================================
        # 3. EXTERNAL COURSES / UDEMY
        # =================================================

        if udemy_texts:

            if isinstance(udemy_texts, list):

                joined = " ".join(
                    udemy_texts
                ).lower()

            else:

                joined = str(
                    udemy_texts
                ).lower()

            for skill in observed:

                if skill.lower() in joined:

                    observed[skill] = min(
                        1.0,
                        observed[skill] + 0.20
                    )

        return observed

    # =====================================================
    # COMPUTE SKILL GAPS
    # =====================================================

    def compute_skill_gaps(
        self,
        observed_profile,
        target_role
    ):

        """
        gap = target - observed
        """

        target_dict = self.target_profiles.get(
            target_role,
            {}
        )

        gaps = {}

        for skill, target_score in target_dict.items():

            observed_score = observed_profile.get(
                skill,
                0.0
            )

            gap = max(
                0,
                target_score - observed_score
            )

            gaps[skill] = round(
                gap,
                3
            )

        return gaps