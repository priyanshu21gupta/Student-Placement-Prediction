# Student Performance & Placement Prediction System

## Project Overview

The Student Performance & Placement Prediction System is a Data Analytics and Machine Learning project designed to analyze student academic and skill-related factors and predict their placement outcome.

The system analyzes factors such as CGPA, attendance, backlogs, coding skills, communication skills, projects, internships, and aptitude scores.

A Random Forest Machine Learning model is used to predict whether a student is likely to be placed.

## Objectives

- Analyze student academic performance.
- Identify factors related to placement outcomes.
- Visualize student performance using charts.
- Predict student placement outcomes using Machine Learning.
- Provide an interactive dashboard for analysis and prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset

The project uses a synthetically generated dataset containing 500 student records.

The dataset contains the following features:

- Student ID
- CGPA
- Attendance
- Backlogs
- Coding Skill
- Communication Skill
- Number of Projects
- Number of Internships
- Aptitude Score
- Placement Outcome

The dataset is generated for educational and demonstration purposes.

## Data Analytics

The dashboard provides analysis of:

- CGPA and placement
- Attendance and placement
- Coding skills and placement
- Internship experience and placement
- Project experience and placement
- Overall placement distribution

## Machine Learning

A Random Forest Classifier is used for placement prediction.

### Input Features

- CGPA
- Attendance
- Backlogs
- Coding Skill
- Communication Skill
- Projects
- Internships
- Aptitude Score

### Output

The model predicts:

- Likely to be Placed
- May Not Be Placed

The dashboard also displays the estimated placement probability.

## Dashboard

The Streamlit dashboard provides:

- Total number of students
- Number of placed students
- Placement rate
- Average CGPA
- Student dataset
- Performance analytics
- Placement prediction form

## Project Structure

Student-Placement-Prediction/

    data/
        student_data.csv

    model/
        placement_model.pkl

    app.py
    train_model.py
    requirements.txt
    README.md

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt