import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def process_and_train_model(
                            sector, city, startup_name, founded_year, funding_amount, 
                            funding_rounds, investor_count, team_size, revenue, 
                            profit_margin, customer_count, growth_rate, 
                            founder_experience, market_size):
    
    # Importing dataset
    dataset = pd.read_csv(r'C:\Users\Dell\Desktop\hackthon\Indian-Startup-Success-Prediction\businessPrediction\tanya.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Encoding categorical data
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 2])], remainder='passthrough')
    X = ct.fit_transform(X).toarray()

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the model
    classifier = GradientBoostingClassifier(random_state=49)
    classifier.fit(X_train, y_train)

    # Making predictions on the test set for evaluation
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Preprocessing the input data
    input_data = pd.DataFrame([[sector, city, startup_name, founded_year, funding_amount, 
                                funding_rounds, investor_count, team_size, revenue, 
                                profit_margin, customer_count, growth_rate, 
                                founder_experience, market_size]], 
                              columns=['Sector', 'City', 'Startup_Name', 'Founded_Year', 'Funding_Amount', 
                                       'Funding_Rounds', 'Investor_Count', 'Team_Size', 'Revenue', 
                                       'Profit_Margin', 'Customer_Count', 'Growth_Rate', 
                                       'Founder_Experience', 'Market_Size'])
    
    input_data = ct.transform(input_data).toarray()
    input_data = sc.transform(input_data)

    # Making prediction on the input data
    prediction = classifier.predict(input_data)

    # Mapping prediction to performance description
    performance_mapping = {0: "underperforming", 1: "moderate success", 2: "outstanding"}
    performance = performance_mapping.get(prediction[0], "Unknown")

    return performance


performance= process_and_train_model( 
                                    sector="FinTech", 
                                                city="Bangalore", 
                                                startup_name="HealthInc", 
                                                founded_year=2020, 
                                                funding_amount=5000000, 
                                                funding_rounds=3, 
                                                investor_count=10, 
                                                team_size=50, 
                                                revenue=2000000, 
                                                profit_margin=0.2, 
                                                customer_count=10000, 
                                                growth_rate=0.15, 
                                                founder_experience=10, 
                                                market_size=100000000)

print("Performance:", performance)

