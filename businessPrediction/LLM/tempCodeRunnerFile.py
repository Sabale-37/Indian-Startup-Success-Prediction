import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load dataset
dataset = pd.read_csv(r'C:\Users\Asus\Desktop\git hackathon\tanya.csv')

# Preprocess categorical data
encoder = OneHotEncoder()
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
y = dataset['Predicti2n'].values  # Assuming 'Success_Label' is your target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust num_labels as per your target

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()