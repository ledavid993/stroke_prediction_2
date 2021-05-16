import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from predict2 import predict

from sklearn.model_selection import train_test_split, RandomizedSearchCV

st.set_option('deprecation.showPyplotGlobalUse', False)

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/onehotencoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

features = ['gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level',
            'bmi', 'smoking_status']

target = 'stroke'

num_features = ['age', 'avg_glucose_level', 'bmi']

cat_features = ['gender', 'hypertension', 'heart_disease',
                'ever_married', 'work_type', 'Residence_type',
                'smoking_status', 'age_group', 'bmi_group']

pie_features = ['gender', 'hypertension', 'heart_disease',
                'ever_married', 'work_type', 'Residence_type',
                'smoking_status']


def categorize_bmi_group(x):
    if x < 18.5:
        return "UnderWeight"
    elif 18.5 < x < 25:
        return "Healthy"
    elif 25 < x < 30:
        return "OverWeight"
    else:
        return "Obese"


def categorize_age_group(x):
    if x < 13:
        return "Child"
    elif 13 < x < 20:
        return "Teenager"
    elif 20 < x <= 60:
        return "Adult"
    else:
        return "Elder"


def transform_features(x):
    X = pd.DataFrame(x, columns=features)
    # converting numerical features as float dtype
    X.loc[:, num_features] = X.loc[:, num_features].astype('float64')
    # add new features
    X["age_group"] = X.age.apply(categorize_age_group)
    X["bmi_group"] = X.age.apply(categorize_bmi_group)

    # converting categorical features as category dtype
    X.loc[:, cat_features] = X.loc[:, cat_features].astype('category')
    # Categorical encoding
    cols = encoder.get_feature_names(cat_features)

    X.loc[:, cols] = encoder.transform(X[cat_features])

    # Drop categorical features
    X.drop(cat_features, axis=1, inplace=True)

    # Feature scaling
    X.loc[:, num_features] = scaler.transform(X[num_features])
    return X


st.title("Stroke Disease Analysis")

dataset_name = st.sidebar.selectbox(
    "Select Dataset", ("Stroke Disease", "Stroke Disease"))


def get_dataset(dataset_name):
    if dataset_name == "Stroke Disease":
        df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    return df, X, y


if dataset_name != "--":
    get_prediction = st.sidebar.selectbox(
        "Would you like a prediction?", ("No", "Yes"))

    classifier_name = st.sidebar.selectbox(
        "Select visualizations", ("--", "Histogram", "Bar", "Boxplot", "Pie chart", "Correlation"))

    df, X, y = get_dataset(dataset_name)
    df_train, df_test = train_test_split(
        df, random_state=3, test_size=0.25, stratify=df.stroke)
    X.gender.replace({'Other': "Female"}, inplace=True)
    X.drop('id', axis=1, inplace=True)
    df_train_pie = df_train
    transform_features(X)

    st.write("shape of dataset", X.shape)
    st.markdown("""---""")
    st.write("Data Samples", df.sample(5))

    def get_hist():
        fig, axes = plt.subplots(nrows=3, figsize=(10, 10))
        for idx, feature in enumerate(num_features):
            hist = df_train[feature].plot(
                kind="hist", ax=axes[idx], title=feature, bins=30)

    def get_bar():
        fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

        df_train[target].value_counts(normalize=True).plot.bar(
            width=0.2, ax=axes[0], title="Train Set")

        df_test[target].value_counts(normalize=True).plot.bar(
            width=0.2, ax=axes[1], title="Test Set")

    if classifier_name == "Histogram":
        get_hist()

    if classifier_name == "Bar":
        get_bar()

    if classifier_name == "Boxplot":
        fig, axes = plt.subplots(nrows=3, figsize=(8, 7))
        for idx, feature in enumerate(num_features):
            box = df_train[feature].plot(kind='box', ax=axes[idx],
                                         vert=False)
        plt.tight_layout()

    if classifier_name == "Pie chart":
        fig, axes = plt.subplots(4, 2, figsize=(20, 20))
        axes = [ax for axes_row in axes for ax in axes_row]

        for idx, feature in enumerate(pie_features):
            df_train[feature].value_counts().plot(
                kind='pie', ax=axes[idx], title=feature, autopct="%.2f", fontsize=18)
            axes[idx].set_ylabel('')

        plt.tight_layout()

    if classifier_name == "Correlation":
        corr_matrix = df_train.corr()[target].sort_values(
            ascending=False).to_frame()
        fig, ax = plt.subplots(figsize=(5, 10))
        ax = sns.heatmap(corr_matrix, annot=True,
                         linewidths=.5, fmt=".2f", cmap="YlGnBu")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.2, top - 0.2)

    if get_prediction == "Yes":
        st.write("Please fill out the patient's information.")

        form = st.form(key='my-form')
        gender = form.selectbox("Gender", ("Male", "Female"))
        age = form.slider("Age", 1, 120, value=20)
        hyper_tension = form.checkbox("History of Hypertension", value=False)
        heart_disease = form.checkbox("History of Heart Disease", value=False)
        ever_married = form.selectbox("Ever Married?", ("Yes", "No"))
        work_type = form.selectbox(
            "Type of Work?", ("Private", "children", "Self-employed", "Never_worked", "Govt_job"))
        residence_type = form.selectbox("Residence Type", ("Urban", "Rural"))
        avg_glucose_level = form.slider(
            "Average Glucose Level", 1, 500, value=20)
        bmi = form.slider("BMI", 1, 200, value=20)
        smoking_status = form.selectbox(
            "Smoking Status", ("Unknown", "never smoked", "smokes", "formerly smoked"))
        submit = form.form_submit_button('Submit')

        x = [['Male', 67.0, 0, 1, 'Yes', 'Private',
              'Urban', 228.69, 36.6, 'formerly smoked']]

        if submit:
            result = predict([[gender, age, hyper_tension, heart_disease, ever_married, work_type, residence_type,
                               avg_glucose_level, bmi, smoking_status]])
            if result == 1:
                st.title("The patient is will to get a stroke")
            else:
                st.title("The patient is will not to get a stroke")


st.pyplot()
