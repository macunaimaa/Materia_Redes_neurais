import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Macky_glas import x
import time

def create_sliding_windows(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(sequences), np.array(targets)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_sequence = x
window_size = 50

# Create sliding windows and targets
sequences, targets = create_sliding_windows(data_sequence, window_size)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.1, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 2: Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Modelo LSTM 
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x) 
        return x
        
# Modelo GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
    
    
# Step 3: Train the model
def train_model(model, X_train, y_train, num_epochs=1000, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def EQM(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

def EQMN1(y_pred, y_true):
    return EQM(y_pred, y_true).item() / EQM(torch.mean(y_true), y_true).item()

def EQMN2(y_pred, y_true, last_value):
    return EQM(y_pred, y_true).item() / EQM(last_value, y_true).item()

def indicators(predicted, real, last):
    print(f"EQM:   {EQM(torch.tensor(predicted)  , torch.tensor(real))}")
    print(f"EQMN1: {EQMN1(torch.tensor(predicted), torch.tensor(real))}")
    print(f"EQMN2: {EQMN2(torch.tensor(predicted), torch.tensor(real), torch.tensor(last))}")
    print('')

# Step 4: Test the model
def test(model, data, window_size):
    sequences, _ = create_sliding_windows(data, window_size)

    sequences_tensor = torch.Tensor(sequences)

    predictions = []
    with torch.no_grad():
        for inputs in sequences_tensor:
            inputs = inputs.unsqueeze(0)  # Add batch dimension
            output = model(inputs)

            predictions.append(output.detach().squeeze().numpy())
            print(predictions)
    return np.array(predictions)

# Create and train the model
input_size = window_size
hidden_size = 64
output_size = 1

model1 = MLP(input_size, hidden_size, output_size)
model2 = LSTM(input_size, hidden_size, output_size)
model3 = GRU(input_size, hidden_size, output_size)
