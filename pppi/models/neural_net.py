"""
PyTorch neural network implementation for PPPI.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from .base import PPPIBaseModel

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class PPPINeuralNet(PPPIBaseModel):
    def __init__(self, batch_size=32, epochs=100, learning_rate=0.001):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X, y):
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        # Create model
        self.model = NeuralNetwork(X.shape[1]).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X, y.reshape(-1, 1))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict_proba(self, X):
        # Convert to PyTorch tensor
        X = torch.FloatTensor(X).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).cpu().numpy()
        
        # Return probabilities for both classes
        return np.column_stack([1-preds, preds])
    
    def get_params(self, deep=True):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "model": self.model
        } 