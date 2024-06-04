import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def process_and_train_model(
                            sector, city, startup_name, founded_year, funding_amount, 
                            funding_rounds, investor_count, team_size, revenue, 
                            profit_margin, customer_count, growth_rate, 
                            founder_experience, market_size):
    
    # Importing dataset
    dataset = pd.read_csv(r'C:\Users\Asus\Desktop\git hackathon\Indian-Startup-Success-Prediction\businessPrediction\tanya.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Encoding categorical data
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [0, 1, 2])], remainder='passthrough')
    X = ct.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Feature scaling
    sc = StandardScaler(with_mean=False)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the model
    classifier = GradientBoostingClassifier(random_state=49)
    classifier.fit(X_train, y_train)

    # Making predictions on the test set for evaluation
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    # Preprocessing the input data
    input_data = pd.DataFrame([[sector, city, startup_name, founded_year, funding_amount, 
                                funding_rounds, investor_count, team_size, revenue, 
                                profit_margin, customer_count, growth_rate, 
                                founder_experience, market_size]], 
                              columns=['Sector', 'City', 'Startup_Name', 'Founded_Year', 'Funding_Amount', 
                                       'Funding_Rounds', 'Investor_Count', 'Team_Size', 'Revenue', 
                                       'Profit_Margin', 'Customer_Count', 'Growth_Rate', 
                                       'Founder_Experience', 'Market_Size'])
    
    input_data = ct.transform(input_data)
    input_data = sc.transform(input_data)

    # Making prediction on the input data
    prediction = classifier.predict(input_data)

    # Mapping prediction to performance description
    performance_mapping = {0: "underperforming", 1: "moderate success", 2: "outstanding"}
    performance = performance_mapping.get(prediction[0], "Unknown")

    return performance


performance = process_and_train_model(
                                    sector="Gomatesh", 
                                    city="NewCity", 
                                    startup_name="NewStartup", 
                                    founded_year=2021, 
                                    funding_amount=1000000, 
                                    funding_rounds=2, 
                                    investor_count=5, 
                                    team_size=20, 
                                    revenue=500000, 
                                    profit_margin=0.1, 
                                    customer_count=5000, 
                                    growth_rate=0.1, 
                                    founder_experience=5, 
                                    market_size=50000000)

print("Performance:", performance)
