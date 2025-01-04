"""
Neural network model implementation for PPPI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from .base import PPPIBaseModel
import pandas as pd

class PressureNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Residual blocks with skip connections
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )
        
        # Output layers with strong regularization
        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.output(x)

class PPPINeuralNet(PPPIBaseModel):
    def __init__(self, input_dim=None, learning_rate=0.0002, batch_size=128, epochs=50):
        super().__init__()
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_ = None
        self.scaler = StandardScaler()
        self.classes_ = np.array([0, 1])
        self.early_stopping_patience = 15
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def _initialize_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim must be set before initializing the model")
        model = PressureNet(self.input_dim)
        model.to(self.device)
        return model
    
    def fit(self, X, y):
        """Fit the neural network model."""
        # Convert pandas Series to numpy array if necessary
        if hasattr(y, 'values'):
            y = y.values
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        # Initialize input_dim if not set
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Initialize model if not already initialized
        if self.model_ is None:
            self.model_ = self._initialize_model()
        
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Split data for validation
        val_size = int(0.1 * len(X))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        
        # Convert data to PyTorch tensors
        X_train = torch.FloatTensor(self.scaler.fit_transform(X[train_idx])).to(self.device)
        y_train = torch.LongTensor(y[train_idx].astype(int)).to(self.device)
        X_val = torch.FloatTensor(self.scaler.transform(X[val_idx])).to(self.device)
        y_val = torch.LongTensor(y[val_idx].astype(int)).to(self.device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Initialize optimizer with cosine annealing
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Initial restart interval
            T_mult=2  # Multiply interval by 2 after each restart
        )
        
        # Calculate class weights for imbalanced dataset
        class_counts = np.bincount(y_train.cpu().numpy())
        total_samples = len(y_train)
        class_weights = torch.FloatTensor([
            total_samples / (len(class_counts) * count) for count in class_counts
        ]).to(self.device)
        
        # Focal Loss with class weights
        criterion = FocalLoss(alpha=class_weights, gamma=2)
        
        # Training loop with early stopping and learning rate scheduling
        best_val_auc = 0.0
        for epoch in range(self.epochs):
            # Training phase
            self.model_.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update learning rate
                scheduler.step(epoch + batch_idx / len(train_loader))
                
                # Log training progress
                # if batch_idx % 10 == 0:
                #     current_lr = scheduler.get_last_lr()[0]
                #     print(f'Epoch {epoch+1}/{self.epochs} | Batch {batch_idx}/{len(train_loader)} | LR: {current_lr:.6f} | Loss: {loss.item():.4f}')
                
                train_loss += loss.item()
                train_preds.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())
            
            # Validation phase
            self.model_.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                val_outputs = self.model_(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_preds = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                val_targets = y_val.cpu().numpy()
            
            # Calculate metrics
            train_auc = roc_auc_score(train_targets, train_preds)
            val_auc = roc_auc_score(val_targets, val_preds)
            
            # Early stopping based on validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model_.state_dict()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"Best validation AUC: {best_val_auc:.4f}")
                self.model_.load_state_dict(best_model_state)
                break
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        # Check if model is fitted
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Prepare input data
        X = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
        
        # Make predictions
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update({
            'input_dim': self.input_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        })
        return params
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self 

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss 