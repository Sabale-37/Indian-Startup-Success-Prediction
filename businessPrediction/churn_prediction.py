import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def train_and_predict_hr_model(satisfaction_level, last_evaluation, number_project, average_montly_hours, 
                               time_spend_company, work_accident, promotion_last_5years, salary):
    # Importing the dataset
    dataset = pd.read_csv(r'C:\Users\Admin\Desktop\Indian-Startup-Success-Prediction\businessPrediction\HR.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Encoding independent variables
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7])], remainder='passthrough')
    X = ct.fit_transform(X)
    X = np.array(X)

    # Encoding dependent variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Importing and training the classifier
    classifier = XGBClassifier(random_state=49)
    classifier.fit(X_train, y_train)

    # Making the Confusion Matrix
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Prepare user input
    user_input = [[satisfaction_level, last_evaluation, number_project, average_montly_hours,
                   time_spend_company, work_accident, promotion_last_5years, salary]]

    # Process the user input
    user_input = ct.transform(user_input)
    user_input = sc.transform(user_input)

    # Make a prediction
    prediction = classifier.predict(user_input)

    # Interpret the prediction
    result = "Employee is likely to leave." if prediction[0] == 1 else "Employee is likely to stay."

    return result

# Example usage
result = train_and_predict_hr_model(0.5, 0.8, 3, 150, 3, 0, 0, 'low')
print(result)
