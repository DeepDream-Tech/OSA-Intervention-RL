"""
OSA Pre-Event Risk Predictor
==============================

LSTM-based model that predicts imminent OSA events from multimodal feature
sequences. Provides the "trigger signal" for the intervention system.

Architecture:
  Input: Sequence of multimodal feature vectors (last N epochs)
  → Bidirectional LSTM (captures temporal patterns)
  → Attention mechanism (weights important time steps)
  → Risk classifier + time-to-event regressor

Design rationale (from literature):
  - OSA events have characteristic 30-120s prodromal signatures:
    • Progressive increase in respiratory effort (RIP amplitude ↑)
    • Thoracoabdominal asynchrony develops (phase angle ↑) 
    • Snoring crescendo pattern (audio RMS ↑, F0 instability ↑)
    • SpO2 begins to drift down (delayed ~15s from airway collapse)
  - LSTM can learn these temporal precursor patterns
  - Attention highlights which epochs are most predictive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class TemporalAttention(nn.Module):
    """
    Scaled dot-product attention over temporal dimension.
    Learns to weight which past epochs are most informative for OSA prediction.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim) - attention-weighted summary
            weights: (batch, seq_len) - attention weights for interpretability
        """
        scores = self.attention_weights(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class OSARiskPredictor(nn.Module):
    """
    Predicts OSA event risk from a sequence of multimodal feature vectors.
    
    Outputs:
      1. risk_score (float, 0-1): Probability of OSA event in next 1-3 epochs
      2. severity (float, 0-1): Predicted severity if event occurs
      3. time_to_event (float, epochs): Estimated epochs until event onset
    
    Input: Sequence of feature vectors from MultimodalFeatureExtractor
    """
    
    def __init__(
        self,
        input_dim: int = 33,       # MultimodalFeatureExtractor.FEATURE_DIM
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = 10,          # Look back 10 epochs (5 minutes at 30s/epoch)
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # Input projection with batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM for temporal pattern learning
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Temporal attention
        self.attention = TemporalAttention(hidden_dim * 2)  # *2 for bidirectional
        
        # Output heads
        # Head 1: OSA risk probability
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Head 2: Predicted severity
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Head 3: Time-to-event (regression)
        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensures positive output
        )
    
    def forward(
        self, 
        feature_sequence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feature_sequence: (batch, seq_len, input_dim) - sequence of feature vectors
        
        Returns:
            Dict with:
              'risk_score': (batch, 1) - OSA event probability
              'severity': (batch, 1) - predicted event severity
              'time_to_event': (batch, 1) - epochs until predicted event
              'attention_weights': (batch, seq_len) - temporal attention weights
        """
        batch_size, seq_len, feat_dim = feature_sequence.shape
        
        # Input projection
        x = feature_sequence.reshape(-1, feat_dim)  # (batch*seq, feat_dim)
        x = self.input_bn(x)
        x = self.input_proj(x)
        x = F.relu(x)
        x = x.reshape(batch_size, seq_len, -1)  # (batch, seq, hidden)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq, hidden*2)
        
        # Temporal attention
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden*2)
        
        # Multi-head predictions
        risk_score = self.risk_head(context)
        severity = self.severity_head(context)
        time_to_event = self.tte_head(context)
        
        return {
            'risk_score': risk_score,
            'severity': severity,
            'time_to_event': time_to_event,
            'attention_weights': attn_weights,
        }
    
    def predict(self, feature_sequence: np.ndarray) -> Dict[str, float]:
        """
        Convenience method for single-sample inference.
        
        Args:
            feature_sequence: (seq_len, input_dim) numpy array
        
        Returns:
            Dict with scalar predictions
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(feature_sequence).unsqueeze(0)  # (1, seq, feat)
            output = self.forward(x)
            
            return {
                'risk_score': float(output['risk_score'].item()),
                'severity': float(output['severity'].item()),
                'time_to_event': float(output['time_to_event'].item()),
                'attention_weights': output['attention_weights'].squeeze(0).numpy(),
            }


class OSARiskLoss(nn.Module):
    """
    Multi-task loss for OSA risk predictor training.
    
    Combines:
      - Binary cross-entropy for risk classification
      - MSE for severity prediction
      - Smooth L1 for time-to-event regression
      - Focal loss weighting for class imbalance (most epochs are non-OSA)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: Output from OSARiskPredictor
            targets: Dict with 'risk_label', 'severity_label', 'tte_label'
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components for logging
        """
        # 1. Focal loss for risk classification (handles class imbalance)
        risk_pred = predictions['risk_score'].squeeze(-1)
        risk_target = targets['risk_label'].float()
        
        bce = F.binary_cross_entropy(risk_pred, risk_target, reduction='none')
        pt = torch.where(risk_target == 1, risk_pred, 1 - risk_pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        risk_loss = (focal_weight * bce).mean()
        
        # 2. MSE for severity (only for positive samples)
        positive_mask = risk_target > 0.5
        if positive_mask.sum() > 0:
            severity_loss = F.mse_loss(
                predictions['severity'].squeeze(-1)[positive_mask],
                targets['severity_label'][positive_mask]
            )
        else:
            severity_loss = torch.tensor(0.0)
        
        # 3. Smooth L1 for time-to-event (only for positive samples)
        if positive_mask.sum() > 0:
            tte_loss = F.smooth_l1_loss(
                predictions['time_to_event'].squeeze(-1)[positive_mask],
                targets['tte_label'][positive_mask]
            )
        else:
            tte_loss = torch.tensor(0.0)
        
        # Weighted combination
        total_loss = 1.0 * risk_loss + 0.5 * severity_loss + 0.3 * tte_loss
        
        loss_dict = {
            'risk_loss': float(risk_loss.item()),
            'severity_loss': float(severity_loss.item()),
            'tte_loss': float(tte_loss.item()),
            'total_loss': float(total_loss.item()),
        }
        
        return total_loss, loss_dict
