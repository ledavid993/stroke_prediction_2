import os
import pickle
import pandas as pd

FILE_DIR = os.path.dirname(__file__)

with open(os.path.join(FILE_DIR, 'model', 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(FILE_DIR, 'model', 'onehotencoder.pkl'), 'rb') as f:
    encoder = pickle.load(f)

with open(os.path.join(FILE_DIR, 'model', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

features = ['gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level',
            'bmi', 'smoking_status']

target = 'stroke'

num_features = ['age', 'avg_glucose_level', 'bmi']

cat_features = ['gender', 'hypertension', 'heart_disease',
                'ever_married', 'work_type', 'Residence_type',
                'smoking_status', 'age_group', 'bmi_group']


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


def predict(x):
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
    return model.predict(X)[0]


# x = [['Male', 67.0, 0, 1, 'Yes', 'Private',
#       'Urban', 228.69, 36.6, 'formerly smoked']]

# y_pred = predict(x)

# if y_pred:
#     print("The patient is likely to have a stroke")
# else:
#     print("The patient is not likely to have a stroke")
