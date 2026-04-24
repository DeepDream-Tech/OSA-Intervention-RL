"""
OSA Intervention System V2 — Classification-Based Architecture
=================================================================

New architecture based on real UCDDB data analysis:

  信号 → 分类模型 → 离散状态 {清醒, 正常睡眠, 打鼾, 呼吸中断}
    │                        ↓
    └→ 特征提取 → 严重度评分 (0~1 连续值)
                             ↓
    信号窗口(60-90s) → Bi-LSTM 趋势编码器 → trend_vector
                             ↓
                 ┌───────────┴───────────┐
                 │     规则决策引擎       │
                 │                       │
                 │  清醒       → 不干预   │
                 │  正常睡眠   → 不干预   │
                 │  打鼾 + 仰卧 + 恶化趋势│
                 │    → 方向性 cue        │
                 │  呼吸中断   → 短促 cue │
                 └───────────────────────┘
                             ↓
                    RL agent 调节 cue 参数
                    (响度/频率/ITD/ILD)

Key design decisions grounded in real data:
  - 4 states from UCDDB: 32.3% Awake, 52.6% Normal, 11.8% Snoring, 3.4% Apnea
  - Severe class imbalance → use focal loss + class weights
  - Severity score: continuous 0-1 calibrated against real SpO2 drops
  - Trend encoder: captures "snoring worsening" patterns (pre-apneic signature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


# =============================================================================
# State Definition (matching UCDDB-derived labels)
# =============================================================================

class OSAState(IntEnum):
    AWAKE = 0
    NORMAL_SLEEP = 1
    SNORING = 2      # Hypopnea / partial obstruction
    APNEA = 3        # Apnea / complete obstruction

STATE_NAMES = {0: '清醒', 1: '正常睡眠', 2: '打鼾', 3: '呼吸中断'}


# =============================================================================
# 1. State Classifier (分类模型)
# =============================================================================

class StateClassifier(nn.Module):
    """
    Classifies current epoch into 4 states + outputs severity score.
    
    Architecture: 1D-CNN for per-epoch feature extraction → classification head
    
    Input: Feature vector from signal processing (33-dim) for current epoch
    Output: 
      - state_logits: (4,) — probabilities for each state
      - severity: (1,) — continuous severity score [0, 1]
    
    Trained on UCDDB annotations (20,789 labeled epochs from 25 subjects).
    Uses focal loss to handle class imbalance (Apnea = only 3.4%).
    """
    
    def __init__(self, input_dim: int = 33, hidden_dim: int = 128, n_states: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_states = n_states
        
        # Shared feature backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Classification head (4 states)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_states),
        )
        
        # Severity regression head (continuous 0-1)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) — feature vector for current epoch
        Returns:
            Dict with 'state_logits', 'state_probs', 'severity'
        """
        features = self.backbone(x)
        state_logits = self.state_head(features)
        state_probs = F.softmax(state_logits, dim=-1)
        severity = self.severity_head(features)
        
        return {
            'state_logits': state_logits,
            'state_probs': state_probs,
            'severity': severity.squeeze(-1),
            'features': features,  # For downstream use
        }
    
    def predict(self, x: np.ndarray) -> Dict[str, any]:
        """Single-sample inference."""
        self.eval()
        with torch.no_grad():
            t = torch.FloatTensor(x).unsqueeze(0)
            out = self.forward(t)
            state = int(torch.argmax(out['state_probs'], dim=-1).item())
            return {
                'state': state,
                'state_name': STATE_NAMES[state],
                'state_probs': out['state_probs'].squeeze(0).numpy(),
                'severity': float(out['severity'].item()),
            }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in 4-state classification.
    
    From UCDDB data:
      Awake: 32.3%, Normal: 52.6%, Snoring: 11.8%, Apnea: 3.4%
    
    Focal loss down-weights easy examples (Awake/Normal) and focuses
    on hard, rare examples (Apnea).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        # Class weights inversely proportional to frequency
        # From UCDDB: [0.323, 0.526, 0.118, 0.034]
        if alpha is None:
            # Inverse frequency, normalized
            freq = torch.tensor([0.323, 0.526, 0.118, 0.034])
            alpha = 1.0 / freq
            alpha = alpha / alpha.sum() * 4  # Normalize to sum=n_classes
        self.register_buffer('alpha', alpha)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, n_classes)
            targets: (batch,) integer class labels
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        
        # Get per-sample alpha
        alpha_t = self.alpha[targets]
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# 2. Trend Encoder (趋势编码器)
# =============================================================================

class TrendEncoder(nn.Module):
    """
    Bi-LSTM trend encoder that compresses 60-90 seconds of multimodal
    signal history into a "current deterioration trend" vector.
    
    This is NOT a predictor (doesn't output a probability).
    It encodes temporal patterns that the decision engine uses:
      - "snoring is getting worse" (鼾声在加重)
      - "SpO2 is declining" (血氧在下降)
      - "respiratory effort is increasing" (呼吸努力在增加)
    
    Architecture: 1D-CNN (per-epoch features) → Bi-LSTM (temporal) → trend vector
    
    Validated by literature:
      - DeepArousal-Net (IEEE TBME 2025): CNN + Bi-LSTM achieves 81.3% at 30s forecasting
      - Wang et al. (IEEE JBHI 2023): 1D-CNN-LSTM at 83% with just RIP signals
    """
    
    def __init__(
        self,
        input_dim: int = 33,
        hidden_dim: int = 64,
        trend_dim: int = 32,     # Output trend vector dimension
        n_layers: int = 1,
        seq_len: int = 3,        # 3 epochs = 90 seconds lookback
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.trend_dim = trend_dim
        self.seq_len = seq_len
        
        # Per-epoch feature extraction
        self.epoch_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        # Compress to trend vector
        self.trend_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, trend_dim),  # *2 for bidirectional
            nn.Tanh(),  # Bounded output [-1, 1]
        )
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (batch, seq_len, input_dim) — sequence of feature vectors
        Returns:
            trend: (batch, trend_dim) — trend encoding vector
        """
        batch_size, seq_len, feat_dim = x_seq.shape
        
        # Encode each epoch
        x = x_seq.reshape(-1, feat_dim)
        x = self.epoch_encoder(x)
        x = x.reshape(batch_size, seq_len, -1)
        
        # LSTM over time
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output (captures full bidirectional context)
        last_output = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Project to trend vector
        trend = self.trend_proj(last_output)
        
        return trend
    
    def encode(self, feature_history: List[np.ndarray]) -> np.ndarray:
        """
        Convenience method for real-time use.
        
        Args:
            feature_history: List of feature vectors (most recent last)
        Returns:
            trend_vector: numpy array of shape (trend_dim,)
        """
        self.eval()
        with torch.no_grad():
            # Pad or truncate to seq_len
            if len(feature_history) < self.seq_len:
                pad = [np.zeros(self.input_dim)] * (self.seq_len - len(feature_history))
                seq = pad + list(feature_history)
            else:
                seq = list(feature_history[-self.seq_len:])
            
            x = torch.FloatTensor(np.array(seq)).unsqueeze(0)
            trend = self.forward(x)
            return trend.squeeze(0).numpy()


# =============================================================================
# 3. Decision Engine (规则决策引擎)
# =============================================================================

@dataclass
class InterventionDecision:
    """Output of the decision engine."""
    should_intervene: bool
    intervention_type: str       # 'none', 'directional_cue', 'burst_cue'
    urgency: float               # 0-1, how urgent
    reason: str                  # Human-readable explanation
    
    # Suggested parameters (can be refined by RL)
    suggested_loudness: float = 0.0
    suggested_frequency: float = 250.0
    suggested_duration: float = 0.0
    suggested_itd: float = 0.0
    suggested_ild: float = 0.0


class DecisionEngine:
    """
    Rule-based decision engine that maps 
    (state, severity, trend, position) → intervention decision.
    
    This is the core clinical logic — every decision is explainable.
    
    Decision Tree:
    
    State = AWAKE → No intervention (不干预)
    State = NORMAL_SLEEP → No intervention
    State = SNORING:
        Severity < 0.5 → Monitor (继续监测)
        Severity >= 0.5:
            Is supine? → Directional cue (方向性cue引导翻身)
            Not supine → Monitor, watch trend
            Trend worsening? → Lower threshold, pre-emptive directional cue
    State = APNEA:
        → Short burst cue (短促声音cue)
        If already tried burst, escalate loudness
    
    All thresholds are derived from UCDDB severity distributions:
        Snoring severity: mean=0.708, std=0.129
        Apnea severity: mean=0.817, std=0.089
    """
    
    def __init__(self):
        # Thresholds calibrated against UCDDB severity distributions
        self.snoring_intervention_threshold = 0.55  # ~1 std below snoring mean
        self.snoring_urgent_threshold = 0.80        # Above snoring mean
        self.trend_worsening_threshold = 0.1        # Trend vector L2 norm indicating deterioration
        
        # Intervention attempt tracking
        self.directional_attempts = 0
        self.burst_attempts = 0
        self.max_directional = 4
        self.max_burst = 3
        self.cooldown_remaining = 0
    
    def reset(self):
        """Reset for new session."""
        self.directional_attempts = 0
        self.burst_attempts = 0
        self.cooldown_remaining = 0
    
    def decide(
        self,
        state: int,
        severity: float,
        is_supine: bool,
        trend_vector: Optional[np.ndarray] = None,
        spo2: float = 95.0,
    ) -> InterventionDecision:
        """
        Main decision function. Every path is explainable.
        
        Args:
            state: 0=Awake, 1=Normal, 2=Snoring, 3=Apnea
            severity: 0.0 - 1.0 continuous
            is_supine: Whether patient is in supine position
            trend_vector: Output from TrendEncoder (optional)
            spo2: Current SpO2 value
        """
        # Cooldown check
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return InterventionDecision(
                should_intervene=False,
                intervention_type='none',
                urgency=0.0,
                reason=f'冷却期中，剩余{self.cooldown_remaining}个epoch',
            )
        
        # Compute trend deterioration magnitude
        trend_worsening = 0.0
        if trend_vector is not None:
            # Positive values in trend vector indicate worsening
            trend_worsening = float(np.mean(np.clip(trend_vector, 0, None)))
        
        # ===== STATE: AWAKE =====
        if state == OSAState.AWAKE:
            return InterventionDecision(
                should_intervene=False,
                intervention_type='none',
                urgency=0.0,
                reason='患者清醒，无需干预',
            )
        
        # ===== STATE: NORMAL SLEEP =====
        if state == OSAState.NORMAL_SLEEP:
            # Even during normal sleep, check if trend is worsening
            if trend_worsening > self.trend_worsening_threshold and is_supine:
                return InterventionDecision(
                    should_intervene=False,  # Don't intervene yet, just note it
                    intervention_type='none',
                    urgency=0.2,
                    reason=f'正常睡眠但趋势恶化(trend={trend_worsening:.2f})，仰卧位，密切监测',
                )
            return InterventionDecision(
                should_intervene=False,
                intervention_type='none',
                urgency=0.0,
                reason='正常睡眠，无需干预',
            )
        
        # ===== STATE: SNORING =====
        if state == OSAState.SNORING:
            # Adjust threshold based on trend
            effective_threshold = self.snoring_intervention_threshold
            if trend_worsening > self.trend_worsening_threshold:
                # Trend worsening → lower the threshold (intervene earlier)
                effective_threshold -= 0.1
            
            if severity < effective_threshold:
                return InterventionDecision(
                    should_intervene=False,
                    intervention_type='none',
                    urgency=severity,
                    reason=f'打鼾但严重度较低({severity:.2f} < {effective_threshold:.2f})，继续监测',
                )
            
            # Severity above threshold → intervene
            if is_supine:
                # Supine + snoring → directional cue (核心策略)
                self.directional_attempts += 1
                loudness = min(0.15 + 0.05 * (self.directional_attempts - 1), 0.45)
                
                return InterventionDecision(
                    should_intervene=True,
                    intervention_type='directional_cue',
                    urgency=severity,
                    reason=f'打鼾(严重度{severity:.2f}) + 仰卧位 → 方向性cue引导翻身 (第{self.directional_attempts}次)',
                    suggested_loudness=loudness,
                    suggested_frequency=250.0,
                    suggested_duration=2.0,
                    suggested_itd=1.2,
                    suggested_ild=15.0,
                )
            else:
                # Not supine but still snoring severely
                if severity >= self.snoring_urgent_threshold:
                    self.burst_attempts += 1
                    loudness = min(0.25 + 0.08 * (self.burst_attempts - 1), 0.5)
                    return InterventionDecision(
                        should_intervene=True,
                        intervention_type='burst_cue',
                        urgency=severity,
                        reason=f'打鼾严重({severity:.2f}) + 非仰卧但恶化 → 短促声音cue (第{self.burst_attempts}次)',
                        suggested_loudness=loudness,
                        suggested_frequency=800.0,
                        suggested_duration=0.5,
                    )
                return InterventionDecision(
                    should_intervene=False,
                    intervention_type='none',
                    urgency=severity,
                    reason=f'打鼾(严重度{severity:.2f})，非仰卧位，继续监测',
                )
        
        # ===== STATE: APNEA =====
        if state == OSAState.APNEA:
            self.burst_attempts += 1
            loudness = min(0.3 + 0.1 * (self.burst_attempts - 1), 0.7)
            
            # SpO2 critical → more aggressive
            if spo2 < 85:
                loudness = min(loudness + 0.15, 0.7)
                reason = f'呼吸中断(严重度{severity:.2f}) + SpO2危急({spo2:.0f}%) → 加强短促cue (第{self.burst_attempts}次)'
            else:
                reason = f'呼吸中断(严重度{severity:.2f}) → 短促声音cue (第{self.burst_attempts}次)'
            
            return InterventionDecision(
                should_intervene=True,
                intervention_type='burst_cue',
                urgency=min(severity + 0.2, 1.0),
                reason=reason,
                suggested_loudness=loudness,
                suggested_frequency=1000.0,
                suggested_duration=0.5,
            )
        
        # Fallback
        return InterventionDecision(
            should_intervene=False,
            intervention_type='none',
            urgency=0.0,
            reason='未知状态',
        )


# =============================================================================
# 4. Integrated System V2
# =============================================================================

class OSASystemV2:
    """
    Complete integrated system with the new architecture.
    
    Pipeline per epoch (30 seconds):
    
    1. Signal Processing → 33-dim feature vector
    2. State Classifier → {Awake, Normal, Snoring, Apnea} + severity
    3. Trend Encoder → 32-dim trend vector from last 90s
    4. Decision Engine → InterventionDecision (explainable)
    5. Audio Synthesis → Binaural audio for earphone
    """
    
    def __init__(self):
        self.classifier = StateClassifier(input_dim=33)
        self.trend_encoder = TrendEncoder(input_dim=33, trend_dim=32, seq_len=3)
        self.decision_engine = DecisionEngine()
        
        # Feature history for trend encoder
        self.feature_history: List[np.ndarray] = []
        self.max_history = 10
        
        # Session statistics
        self.epoch_count = 0
        self.intervention_count = 0
        self.state_history = []
    
    def reset(self):
        """Reset for new sleep session."""
        self.feature_history = []
        self.epoch_count = 0
        self.intervention_count = 0
        self.state_history = []
        self.decision_engine.reset()
    
    def process_epoch(
        self,
        feature_vector: np.ndarray,
        is_supine: bool,
        spo2: float = 95.0,
    ) -> Dict:
        """
        Process one 30-second epoch.
        
        Args:
            feature_vector: 33-dim feature vector from signal processing
            is_supine: Whether patient is in supine position
            spo2: Current SpO2 value
            
        Returns:
            Dict with state, severity, trend, decision, and explanation
        """
        self.epoch_count += 1
        
        # 1. Update feature history
        self.feature_history.append(feature_vector.copy())
        if len(self.feature_history) > self.max_history:
            self.feature_history = self.feature_history[-self.max_history:]
        
        # 2. Classify current state
        classification = self.classifier.predict(feature_vector)
        state = classification['state']
        severity = classification['severity']
        
        # 3. Encode trend (needs >= 1 epoch of history)
        trend_vector = None
        if len(self.feature_history) >= 2:
            trend_vector = self.trend_encoder.encode(self.feature_history)
        
        # 4. Make decision
        decision = self.decision_engine.decide(
            state=state,
            severity=severity,
            is_supine=is_supine,
            trend_vector=trend_vector,
            spo2=spo2,
        )
        
        # 5. Track
        self.state_history.append(state)
        if decision.should_intervene:
            self.intervention_count += 1
        
        return {
            'epoch': self.epoch_count,
            'state': state,
            'state_name': classification['state_name'],
            'state_probs': classification['state_probs'],
            'severity': severity,
            'trend_vector': trend_vector,
            'decision': decision,
            'is_supine': is_supine,
            'spo2': spo2,
        }
    
    def get_session_summary(self) -> Dict:
        """Get summary statistics for current session."""
        from collections import Counter
        state_counts = Counter(self.state_history)
        total = len(self.state_history)
        
        return {
            'total_epochs': total,
            'total_minutes': total * 0.5,
            'interventions': self.intervention_count,
            'intervention_rate': self.intervention_count / max(total, 1),
            'state_distribution': {
                STATE_NAMES[k]: state_counts.get(k, 0) / max(total, 1)
                for k in range(4)
            },
        }
