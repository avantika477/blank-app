import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Title
st.write("# Personal Fitness Tracker 🏋️‍♂️")
st.write(
    "This app predicts the **calories burned** during exercise based on input parameters like `Age`, `BMI`, `Duration`, `Heart Rate`, etc."
)

# Sidebar - User Inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 22)
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 150, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36, 42, 37.5)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])

    gender_encoded = 1 if gender == "Male" else 0

    data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_encoded,
    }

    return pd.DataFrame(data, index=[0])

df = user_input_features()

st.write("---")
st.subheader("Your Input Parameters")
st.write(df)

# Load data
@st.cache_data
def load_data():
    try:
        calories = pd.read_csv("calories.csv")
        exercise = pd.read_csv("exercise.csv")
    except FileNotFoundError:
        st.error("Error: Data files not found. Please check the file paths.")
        return None, None

    # Merge datasets
    exercise_df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])

    # Add BMI column
    exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

    # Select relevant features
    features = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
    exercise_df = exercise_df[features]

    # One-hot encoding for gender
    exercise_df = pd.get_dummies(exercise_df, drop_first=True)

    return exercise_df

exercise_df = load_data()

if exercise_df is not None:
    # Split data
    train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    # Prepare training and testing sets
    X_train = train_data.drop("Calories", axis=1)
    y_train = train_data["Calories"]

    X_test = test_data.drop("Calories", axis=1)
    y_test = test_data["Calories"]

    # Train model
    model = RandomForestRegressor(n_estimators=1000, max_depth=6, max_features=3, random_state=1)
    model.fit(X_train, y_train)

    # Align user input with trained model features
    df = df.reindex(columns=X_train.columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)[0]

    # Display prediction
    st.write("---")
    st.subheader("Predicted Calories Burned 🔥")
    st.metric(label="Estimated Calories Burned", value=f"{round(prediction, 2)} kcal")

    # Similar results
    st.write("---")
    st.subheader("Similar Cases from Data 📊")

    calorie_range = [prediction - 10, prediction + 10]
    similar_data = exercise_df[
        (exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])
    ]

    if not similar_data.empty:
        st.write(similar_data.sample(min(5, len(similar_data))))
    else:
        st.write("No similar cases found.")

    # General insights
    st.write("---")
    st.subheader("Comparison with Other Users 📈")

    def percentage_below(value, column):
        return round((exercise_df[column] < value).mean() * 100, 2)

    st.write(f"You are older than **{percentage_below(df['Age'][0], 'Age')}%** of people.")
    st.write(f"Your exercise duration is longer than **{percentage_below(df['Duration'][0], 'Duration')}%** of people.")
    st.write(f"Your heart rate is higher than **{percentage_below(df['Heart_Rate'][0], 'Heart_Rate')}%** of people.")
    st.write(f"Your body temperature is higher than **{percentage_below(df['Body_Temp'][0], 'Body_Temp')}%** of people.")

    # Visualization - Distribution of Calories Burned
    st.write("---")
    st.subheader("Calories Burned Distribution 📊")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(exercise_df["Calories"], bins=30, kde=True, color="blue", alpha=0.6)
    ax.axvline(prediction, color="red", linestyle="dashed", linewidth=2)
    plt.xlabel("Calories Burned")
    plt.ylabel("Frequency")
    plt.title("Distribution of Calories Burned")
    st.pyplot(fig)

else:
    st.error("Data not loaded properly. Please check your dataset.")
