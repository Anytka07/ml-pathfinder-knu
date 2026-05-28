import streamlit as st
import numpy as np
import pandas as pd

from src.ontology import SkillOntology
from src.profiles import ProfileManager
from src.recommender import Recommender


st.set_page_config(page_title="ML PathFinder KNU", layout="wide")
st.title("🧭 ML PathFinder KNU")
st.caption("Рекомендаційна система-навігатор для студентів КН ЧНУ")

# ==================================================
# CACHE LAYER 
# ==================================================
@st.cache_resource
def get_ontology():
    return SkillOntology()


@st.cache_resource
def get_profile_manager():
    ontology = SkillOntology()
    return ProfileManager(ontology)

profile_manager = get_profile_manager()

@st.cache_resource
def get_recommender():
    ontology = SkillOntology()
    return Recommender(ontology)

recommender = get_recommender()

ontology = get_ontology()
profile_manager = get_profile_manager()
recommender = get_recommender()

# ==================================================
# ONLY USER STATE
# ==================================================

if "students" not in st.session_state:
    st.session_state.students = []


tab1, tab2, tab3 = st.tabs(["👨🎓 Student", "👨🏫 Teacher", "🛠 Admin"])


# ===================== STUDENT =====================
with tab1:

    student_name = st.text_input("👤 Введіть ваше ім’я")

    role = st.selectbox(
        "Оберіть цільову роль",
        ["ML Engineer", "Data Analyst", "Researcher"]
    )

    st.subheader("1. Силабуси пройдених дисциплін")

    syllabi_input = st.text_area(
        "Встав текст силабусів:",
        "Машинне навчання: регресія, класифікація\n"
        "Статистика та ймовірності\n"
        "Python для Data Science\n"
        "Лінійна алгебра",
        height=130
    )

    # ==================================================
    # SELF-ASSESSMENT
    # ==================================================

    st.subheader("2. Self-Assessment")

    skills_list = list(profile_manager.target_profiles[role].keys())

    cols = st.columns(3)
    assessment = {}

    for i, skill in enumerate(skills_list):

        default_value = profile_manager.target_profiles[role][skill] * 0.6

        assessment[skill] = cols[i % 3].slider(
            skill,
            0.0,
            1.0,
            float(default_value)
        )

    # ==================================================
    # BUTTON
    # ==================================================

    if st.button("🚀 Отримати персональні рекомендації", type="primary"):

        with st.spinner("Обробка рекомендацій..."):

            observed = profile_manager.create_observed_profile(
                syllabi_input,
                assessment
            )

            gaps = profile_manager.compute_skill_gaps(
                observed,
                role
            )

            total_gap = sum(gaps.values())
            avg_gap = total_gap / len(gaps)

            st.session_state.students.append({
                "name": student_name,
                "role": role,
                "assessment": assessment,
                "gaps": gaps,
                "total_gap": total_gap
            })

        st.success(
            f"Середній skill gap до ролі **{role}**: **{avg_gap:.1%}**"
        )

        col1, col2 = st.columns(2)

        # ================= COURSES =================
        with col1:

            st.subheader("📚 Рекомендовані курси")

            courses = recommender.recommend_courses(observed, gaps)

            for c in courses:
                st.markdown(f"### {c['title']}")
                st.caption(c['explanation'])
                st.caption(
                    f"Релевантність: {c['relevance']:.2f} | "
                    f"Складність: {c['difficulty']}"
                )

                if c.get("url") and c["url"] != "#":
                    st.markdown(f"[🔗 Перейти]({c['url']})")

                st.divider()

        # ================= KAGGLE =================
        with col2:

            st.subheader("🏆 Kaggle рекомендації")

            for ds in recommender.recommend_kaggle():
                st.markdown(f"• **{ds['title']}**")
                st.caption(ds.get("tags", ""))

            st.subheader("📍 Оптимальна навчальна траєкторія")

            for step in recommender.generate_learning_path(gaps):
                st.write(step)


# ===================== TEACHER =====================
with tab2:

    st.subheader("👨🏫 Аналітика студентів")

    if not st.session_state.students:
        st.warning("Поки немає студентів.")

    else:
        students = st.session_state.students

        st.success(f"Студентів: {len(students)}")

        avg_gaps = {}
        first = students[0]["gaps"]

        for skill in first.keys():
            avg_gaps[skill] = np.mean([s["gaps"].get(skill, 0) for s in students])

        st.subheader("📊 Середній Gap")
        st.bar_chart(pd.Series(avg_gaps))

        st.subheader("🔥 Найслабші навички")

        sorted_gaps = sorted(avg_gaps.items(), key=lambda x: x[1], reverse=True)

        for skill, val in sorted_gaps[:5]:
            st.write(f"• {skill}: {val:.2f}")


# ===================== ADMIN =====================
with tab3:

    st.subheader("🛠 Target Profiles")

    selected_role = st.selectbox(
        "Оберіть роль",
        ["ML Engineer", "Data Analyst", "Researcher"]
    )

    current = profile_manager.target_profiles[selected_role]

    new_profile = {}

    for skill, val in current.items():
        new_profile[skill] = st.slider(
            skill,
            0.0,
            1.0,
            float(val),
            key=f"admin_{selected_role}_{skill}"
        )

    if st.button("💾 Зберегти"):
        profile_manager.target_profiles[selected_role] = new_profile
        st.success("Оновлено!")