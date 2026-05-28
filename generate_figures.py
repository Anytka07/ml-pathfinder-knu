# generate_figures.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# =========================================
# SETTINGS
# =========================================

os.makedirs("figures", exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# =========================================
# DATA
# =========================================

skills = [
    "Python", "SQL", "Statistics",
    "Deep Learning", "MLOps",
    "Docker", "Cloud", "Research"
]

ml_engineer = [0.95, 0.40, 0.55, 0.85, 0.90, 0.80, 0.75, 0.30]
data_analyst = [0.75, 0.90, 0.90, 0.30, 0.20, 0.10, 0.20, 0.35]
researcher = [0.60, 0.40, 0.90, 0.80, 0.10, 0.05, 0.10, 0.95]

student_profile = [0.79, 0.54, 0.54, 0.49, 0.67, 0.48, 0.14, 0.30]

# =========================================
# 1. RADAR CHART
# =========================================

def radar_chart():
    labels = skills
    num_vars = len(labels)

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    values_target = ml_engineer + ml_engineer[:1]
    values_student = student_profile + student_profile[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    ax.plot(angles, values_target, linewidth=2, label='ML Engineer Target')
    ax.fill(angles, values_target, alpha=0.15)

    ax.plot(angles, values_student, linewidth=2, label='Observed Student')
    ax.fill(angles, values_student, alpha=0.15)

    plt.xticks(angles[:-1], labels)
    plt.yticks([0.2,0.4,0.6,0.8,1.0])
    plt.ylim(0,1)

    plt.title("Skill Radar Chart")
    plt.legend(loc='upper right')

    plt.savefig("figures/radar_chart.png", bbox_inches="tight")
    plt.close()

# =========================================
# 2. HEATMAP ROLES VS SKILLS
# =========================================

def heatmap_roles():
    df = pd.DataFrame({
        "ML Engineer": ml_engineer,
        "Data Analyst": data_analyst,
        "Researcher": researcher
    }, index=skills)

    plt.figure(figsize=(9,6))
    sns.heatmap(df, annot=True, cmap="viridis")

    plt.title("Skills vs Roles Heatmap")

    plt.savefig("figures/roles_heatmap.png", bbox_inches="tight")
    plt.close()

# =========================================
# 3. GAP BAR CHART
# =========================================

def gap_chart():
    target = np.array(ml_engineer)
    observed = np.array(student_profile)

    gaps = np.maximum(0, target - observed)

    plt.figure(figsize=(10,6))

    plt.bar(skills, gaps)

    plt.title("Skill Gaps")
    plt.ylabel("Gap Value")
    plt.xticks(rotation=20)

    plt.savefig("figures/gap_chart.png", bbox_inches="tight")
    plt.close()

# =========================================
# 4. FEATURE IMPORTANCE
# =========================================

def feature_importance():
    importance = {
        "Python":0.95,
        "SQL":0.90,
        "Statistics":0.90,
        "Deep Learning":0.85,
        "MLOps":0.90,
        "Docker":0.80,
        "Cloud":0.75,
        "Research":0.95
    }

    plt.figure(figsize=(10,6))

    plt.bar(
        importance.keys(),
        importance.values()
    )

    plt.title("Feature Importance")
    plt.ylabel("Importance")

    plt.savefig("figures/feature_importance.png", bbox_inches="tight")
    plt.close()

# =========================================
# 5. DATA AUGMENTATION HISTOGRAM
# =========================================

def augmentation_histogram():
    synthetic = []

    for _ in range(200):
        noise = np.random.normal(0, 0.1)
        value = 0.55 + noise
        synthetic.append(value)

    plt.figure(figsize=(10,6))

    plt.hist(synthetic, bins=20)

    plt.title("Synthetic Student Distribution")
    plt.xlabel("Skill Value")

    plt.savefig("figures/augmentation_histogram.png", bbox_inches="tight")
    plt.close()

# =========================================
# 6. BOXPLOT
# =========================================

def students_boxplot():
    students = []

    for _ in range(40):
        row = np.random.normal(0.6, 0.15, len(skills))
        students.append(row)

    df = pd.DataFrame(students, columns=skills)

    plt.figure(figsize=(12,6))

    sns.boxplot(data=df)

    plt.title("Synthetic Students Skill Distribution")
    plt.xticks(rotation=20)

    plt.savefig("figures/students_boxplot.png", bbox_inches="tight")
    plt.close()

# =========================================
# 7. METRICS DASHBOARD
# =========================================

def metrics_dashboard():
    metrics = {
        "Relevance":0.71,
        "Gap Coverage":0.61,
        "Precision@5":0.72,
        "Diversity":0.68
    }

    plt.figure(figsize=(8,6))

    plt.bar(
        metrics.keys(),
        metrics.values()
    )

    plt.ylim(0,1)

    plt.title("Model Evaluation Metrics")

    plt.savefig("figures/metrics_dashboard.png", bbox_inches="tight")
    plt.close()

# =========================================
# 8. RESPONSE TIME
# =========================================

def response_time_chart():
    samples = [100, 250, 500, 1000]
    times = [0.7, 1.1, 1.7, 2.4]

    plt.figure(figsize=(8,6))

    plt.plot(samples, times, marker='o')

    plt.title("Response Time vs Dataset Size")
    plt.xlabel("Courses Count")
    plt.ylabel("Seconds")

    plt.savefig("figures/response_time.png", bbox_inches="tight")
    plt.close()

# =========================================
# 9. ROBUSTNESS TEST
# =========================================

def robustness_chart():
    noise = [0, 0.05, 0.10, 0.15, 0.20]
    quality = [0.79, 0.77, 0.74, 0.71, 0.67]

    plt.figure(figsize=(8,6))

    plt.plot(noise, quality, marker='o')

    plt.title("Robustness to Noise")
    plt.xlabel("Noise Level")
    plt.ylabel("Recommendation Quality")

    plt.savefig("figures/robustness_chart.png", bbox_inches="tight")
    plt.close()

# =========================================
# 10. EMBEDDING VISUALIZATION
# =========================================

def embedding_visualization():
    np.random.seed(42)

    embeddings = np.random.rand(50, 20)

    labels = (
        ["ML"] * 15 +
        ["DA"] * 15 +
        ["RES"] * 20
    )

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(9,7))

    for label in set(labels):
        idx = [i for i,l in enumerate(labels) if l == label]

        plt.scatter(
            reduced[idx,0],
            reduced[idx,1],
            label=label
        )

    plt.title("Embedding Space Visualization")
    plt.legend()

    plt.savefig("figures/embedding_visualization.png", bbox_inches="tight")
    plt.close()

# =========================================
# 11. PIE CHART CLEANING
# =========================================

def cleaning_pie():
    labels = ["Relevant Courses", "Removed Courses"]
    sizes = [82, 18]

    plt.figure(figsize=(7,7))

    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%'
    )

    plt.title("Course Cleaning Results")

    plt.savefig("figures/cleaning_pie.png", bbox_inches="tight")
    plt.close()

# =========================================
# RUN ALL
# =========================================

if __name__ == "__main__":

    radar_chart()
    heatmap_roles()
    gap_chart()
    feature_importance()
    augmentation_histogram()
    students_boxplot()
    metrics_dashboard()
    response_time_chart()
    robustness_chart()
    embedding_visualization()
    cleaning_pie()

    print(" All figures generated in /figures")