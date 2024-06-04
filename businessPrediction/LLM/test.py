import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Define the SimpleNN class as provided in your code
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def predict_startup_success(model, scaler, encoder, input_data):
    # Preprocess input data
    encoded_features = encoder.transform(input_data[['Sector', 'City', 'Startup_Name']]).toarray()
    scaled_features = scaler.transform(input_data[['Founded_Year', 'Funding_Amount', 'Funding_Rounds', 'Investor_Count', 
                                                    'Team_Size', 'Revenue', 'Profit_Margin', 'Customer_Count', 
                                                    'Growth_Rate', 'Founder_Experience', 'Market_Size']])

    # Combine features
    X = np.hstack((encoded_features, scaled_features))

    # Make prediction
    with torch.no_grad():
        output = model(torch.tensor(X, dtype=torch.float32))
        _, predicted = torch.max(output, 1)

    return predicted.item()

if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv(r'C:\Users\Asus\Desktop\git hackathon\Indian-Startup-Success-Prediction\businessPrediction\LLM\tanya.csv')

    # Preprocess categorical data
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features = ['Sector', 'City', 'Startup_Name']
    encoded_features = encoder.fit_transform(dataset[categorical_features]).toarray()

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = dataset[['Founded_Year', 'Funding_Amount', 'Funding_Rounds', 'Investor_Count', 
                                  'Team_Size', 'Revenue', 'Profit_Margin', 'Customer_Count', 
                                  'Growth_Rate', 'Founder_Experience', 'Market_Size']]
    scaled_features = scaler.fit_transform(numerical_features)

    # Combine features
    X = np.hstack((encoded_features, scaled_features))
    y = dataset['Predicti2n'].values

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Model, loss function, optimizer
    input_dim = X.shape[1]
    hidden_dim = 128
    output_dim = len(np.unique(y))

    model = SimpleNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Save trained model, scaler, and encoder
    joblib.dump(model, "trained_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")

    # Load trained model, scaler, and encoder
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")

    # Get input from user
    input_data = {}
    for feature in dataset.columns[:-1]:  # Exclude the target column
        input_data[feature] = input(f"Enter value for {feature}: ")

    input_df = pd.DataFrame(input_data, index=[0])
    
    prediction = predict_startup_success(model, scaler, encoder, input_df)

    # Map prediction to corresponding output
    if prediction == 0:
        print("Startup is likely to fail")
    elif prediction == 1:
        print("Startup is in the moderate stage")
    elif prediction == 2:
        print("Startup will be successful")
