import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Student Placement Analytics",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load Data and Model
# -----------------------------

data = pd.read_csv("data/student_data.csv")
model = joblib.load("model/placement_model.pkl")

# -----------------------------
# Title
# -----------------------------

st.title("🎓 Student Performance & Placement Prediction System")

st.write(
    "A Data Analytics and Machine Learning dashboard "
    "for analyzing student performance and predicting placement outcomes."
)

st.divider()

# -----------------------------
# Key Metrics
# -----------------------------

total_students = len(data)
placed_students = len(data[data["Placement"] == "Yes"])
placement_rate = (placed_students / total_students) * 100
average_cgpa = data["CGPA"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Students", total_students)

with col2:
    st.metric("Placed Students", placed_students)

with col3:
    st.metric("Placement Rate", f"{placement_rate:.1f}%")

with col4:
    st.metric("Average CGPA", f"{average_cgpa:.2f}")

st.divider()

# -----------------------------
# Data Preview
# -----------------------------

st.header("📋 Student Dataset")

st.dataframe(
    data,
    use_container_width=True,
    height=300
)

# -----------------------------
# Analytics
# -----------------------------

st.header("📊 Performance Analytics")

col1, col2 = st.columns(2)

# CGPA vs Placement
with col1:

    st.subheader("CGPA Distribution by Placement")

    fig, ax = plt.subplots()

    sns.boxplot(
        data=data,
        x="Placement",
        y="CGPA",
        ax=ax
    )

    ax.set_xlabel("Placement")
    ax.set_ylabel("CGPA")

    st.pyplot(fig)

# Attendance vs Placement
with col2:

    st.subheader("Attendance vs Placement")

    fig, ax = plt.subplots()

    sns.boxplot(
        data=data,
        x="Placement",
        y="Attendance",
        ax=ax
    )

    ax.set_xlabel("Placement")
    ax.set_ylabel("Attendance (%)")

    st.pyplot(fig)

# -----------------------------
# Skills Analysis
# -----------------------------

st.subheader("💻 Coding Skill vs Placement")

fig, ax = plt.subplots()

sns.countplot(
    data=data,
    x="Coding_Skill",
    hue="Placement",
    ax=ax
)

ax.set_xlabel("Coding Skill (1-10)")
ax.set_ylabel("Number of Students")

st.pyplot(fig)

# -----------------------------
# Internship Analysis
# -----------------------------

st.subheader("💼 Internship Experience vs Placement")

internship_data = (
    data.groupby(["Internships", "Placement"])
    .size()
    .unstack(fill_value=0)
)

st.bar_chart(internship_data)

# -----------------------------
# Project Analysis
# -----------------------------

st.subheader("📁 Projects vs Placement")

project_data = (
    data.groupby(["Projects", "Placement"])
    .size()
    .unstack(fill_value=0)
)

st.bar_chart(project_data)

st.divider()

# -----------------------------
# Placement Prediction
# -----------------------------

st.header("🤖 AI Placement Prediction")

st.write(
    "Enter a student's details below to predict their placement outcome."
)

col1, col2 = st.columns(2)

with col1:

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    coding = st.slider(
        "Coding Skill",
        min_value=1,
        max_value=10,
        value=7
    )

with col2:

    communication = st.slider(
        "Communication Skill",
        min_value=1,
        max_value=10,
        value=7
    )

    projects = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )

    internships = st.number_input(
        "Number of Internships",
        min_value=0,
        max_value=5,
        value=1,
        step=1
    )

    aptitude = st.slider(
        "Aptitude Score",
        min_value=40,
        max_value=100,
        value=75
    )

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("🔮 Predict Placement", type="primary"):

    input_data = pd.DataFrame({
        "CGPA": [cgpa],
        "Attendance": [attendance],
        "Backlogs": [backlogs],
        "Coding_Skill": [coding],
        "Communication_Skill": [communication],
        "Projects": [projects],
        "Internships": [internships],
        "Aptitude_Score": [aptitude]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    if prediction == "Yes":

        st.success("🎉 Prediction: Likely to be Placed")

        st.write(
            f"Placement Probability: "
            f"{probability[list(model.classes_).index('Yes')] * 100:.2f}%"
        )

    else:

        st.warning("⚠️ Prediction: May Not Be Placed")

        st.write(
            f"Placement Probability: "
            f"{probability[list(model.classes_).index('Yes')] * 100:.2f}%"
        )

st.divider()

st.caption(
    "Student Performance & Placement Prediction System | "
    "Data Analytics & AI Internship Project"
)