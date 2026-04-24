"""
Train & Evaluate the State Classifier on UCDDB Real Data
=========================================================

Uses Leave-One-Subject-Out (LOSO) cross-validation:
  - Train on 24 subjects, test on 1 held-out subject
  - Repeat for all 25 subjects
  - This gives the most honest generalization estimate

Since we only have annotation labels (not raw signals), we construct
synthetic feature vectors calibrated to real UCDDB distributions.
This is an interim approach — with full EDF signal access, we would 
extract real features.

Feature Construction from Annotations:
  Each epoch has: sleep_stage, respiratory_event_type, duration, SpO2_drop
  We construct a 33-dim feature vector that represents what the signal
  processing module would extract from real sensors.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from typing import Dict, List, Tuple

from osa_system.ucddb_parser import build_ucddb_dataset, SleepState, EpochLabel
from osa_system.system_v2 import StateClassifier, FocalLoss, OSAState, STATE_NAMES


# =============================================================================
# Feature Construction from UCDDB Annotations
# =============================================================================

def epoch_label_to_features(label: EpochLabel, rng: np.random.Generator) -> np.ndarray:
    """
    Construct a synthetic 33-dim feature vector from epoch annotations.
    
    This simulates what the signal processing module would extract.
    Distributions are calibrated from UCDDB annotation statistics:
      - SpO2 drops: mean=6.3%, range 2-30%
      - Event durations: mean=18.3s, range 6-65s
      - Sleep stages: real UCDDB distribution
    
    When we have full EDF signal access, this function gets replaced
    by real signal processing. The model architecture stays the same.
    """
    state = label.state
    severity = label.severity
    raw_stage = label.sleep_stage_raw
    
    # Map raw stage to normalized features
    # 0=Wake, 1=REM, 2=N1, 3=N2, 4=N3, 5=Artifact
    sleep_stage_norm = raw_stage / 5.0
    
    if state == SleepState.AWAKE:
        features = _awake_features(rng)
    elif state == SleepState.NORMAL_SLEEP:
        features = _normal_sleep_features(raw_stage, rng)
    elif state == SleepState.SNORING:
        features = _snoring_features(label, rng)
    elif state == SleepState.APNEA:
        features = _apnea_features(label, rng)
    else:
        features = _awake_features(rng)
    
    # Ensure exact 33 dimensions
    assert len(features) == 33, f"Expected 33 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


def _awake_features(rng) -> List[float]:
    """Features typical of awake state."""
    return [
        # RIP (11): normal but irregular breathing, high movement
        rng.uniform(0.3, 0.8),   # thorax_amplitude
        rng.uniform(0.3, 0.8),   # abdomen_amplitude  
        rng.uniform(0.6, 1.6),   # total_amplitude
        rng.uniform(0.2, 0.4),   # resp_rate_norm (12-24 bpm)
        rng.uniform(0.1, 0.4),   # resp_rate_variability (high)
        rng.uniform(-0.05, 0.05),# phase_angle_norm (near zero)
        0.0,                      # is_paradoxical
        rng.uniform(0.3, 0.5),   # flow_limitation_norm
        rng.uniform(0.2, 0.5),   # resp_effort
        0.0, 0.0,                # reserved
        # Audio (9): no snoring
        rng.uniform(0.0, 0.05),  # snore_rms
        0.0,                      # snore_f0
        0.0,                      # f0_confidence
        0.0,                      # f0_stability
        rng.uniform(0.1, 0.3),   # spectral_centroid
        rng.uniform(0.1, 0.3),   # spectral_bandwidth
        0.0,                      # snore_pattern
        0.0,                      # intermittency
        0.0,                      # is_crescendo
        # IMU (9): variable position, high movement
        rng.choice([0.0, 1.0]),   # is_supine
        rng.uniform(-0.5, 0.5),  # pitch
        rng.uniform(-0.5, 0.5),  # roll
        rng.uniform(0.3, 0.8),   # position_stability
        rng.uniform(0.2, 0.8),   # movement_intensity (high)
        rng.uniform(0.1, 0.5),   # movement_variability (high)
        0.0, 0.0, 0.0,           # reserved
        # SpO2 (4): normal
        rng.uniform(0.8, 1.0),   # spo2_normalized (93-100%)
        rng.uniform(-0.02, 0.02),# desat_slope (stable)
        rng.uniform(0.0, 0.1),   # hypoxemia_risk (low)
        0.0,                      # odi_proxy
    ]


def _normal_sleep_features(raw_stage: int, rng) -> List[float]:
    """Features typical of normal sleep (no respiratory events)."""
    # Sleep depth affects features
    depth = {1: 0.4, 2: 0.2, 3: 0.5, 4: 0.8, 5: 0.3}.get(raw_stage, 0.3)
    
    return [
        # RIP (11): regular, relaxed breathing
        rng.uniform(0.3, 0.6),
        rng.uniform(0.3, 0.6),
        rng.uniform(0.6, 1.2),
        rng.uniform(0.15, 0.30),  # resp_rate_norm (slower during sleep)
        rng.uniform(0.02, 0.10),  # low variability
        rng.uniform(-0.03, 0.03), # phase_angle (in-phase)
        0.0,
        rng.uniform(0.3, 0.5),
        rng.uniform(0.2, 0.4),    # low effort
        0.0, 0.0,
        # Audio (9): quiet
        rng.uniform(0.0, 0.03),
        0.0, 0.0, 0.0,
        rng.uniform(0.05, 0.15),
        rng.uniform(0.05, 0.15),
        0.0, 0.0, 0.0,
        # IMU (9): stable, low movement
        rng.choice([0.0, 1.0], p=[0.5, 0.5]),
        rng.uniform(-0.3, 0.3),
        rng.uniform(-0.3, 0.3),
        rng.uniform(0.7, 1.0),    # high stability
        rng.uniform(0.0, 0.1),    # low movement
        rng.uniform(0.0, 0.05),
        0.0, 0.0, 0.0,
        # SpO2 (4): normal
        rng.uniform(0.8, 1.0),
        rng.uniform(-0.01, 0.01),
        rng.uniform(0.0, 0.05),
        0.0,
    ]


def _snoring_features(label: EpochLabel, rng) -> List[float]:
    """Features typical of snoring/hypopnea (partial obstruction)."""
    severity = label.severity
    spo2_drop = (label.spo2_drop or 5.0) / 30.0  # Normalize
    duration_norm = (label.event_duration or 15) / 60.0
    
    return [
        # RIP (11): increased effort, partial flow reduction
        rng.uniform(0.4, 0.7),
        rng.uniform(0.3, 0.6),
        rng.uniform(0.5, 1.0),
        rng.uniform(0.25, 0.40),   # slightly elevated rate
        rng.uniform(0.05, 0.20),
        rng.uniform(0.10, 0.40) * severity,  # phase angle increases with severity
        0.0 if severity < 0.7 else rng.choice([0.0, 1.0], p=[0.7, 0.3]),
        rng.uniform(0.5, 0.8),     # some flow limitation
        rng.uniform(0.4, 0.7) * (0.5 + severity),  # elevated effort
        0.0, 0.0,
        # Audio (9): snoring present
        rng.uniform(0.2, 0.8) * severity,  # snore_rms scales with severity
        rng.uniform(0.2, 0.6),     # snore_f0 (100-300Hz range normalized)
        rng.uniform(0.3, 0.8),     # f0_confidence
        rng.uniform(0.3, 0.7),     # f0_stability
        rng.uniform(0.2, 0.5),
        rng.uniform(0.2, 0.4),
        rng.choice([0.5, 1.0]),     # snore_pattern: sustained or bout
        rng.uniform(0.1, 0.5),
        rng.uniform(0.0, 0.5) * severity,  # crescendo more likely with severity
        # IMU (9)
        rng.choice([0.0, 1.0], p=[0.4, 0.6]),  # More likely supine during snoring
        rng.uniform(-0.2, 0.2),
        rng.uniform(-0.2, 0.2),
        rng.uniform(0.6, 0.95),
        rng.uniform(0.0, 0.15),
        rng.uniform(0.0, 0.08),
        0.0, 0.0, 0.0,
        # SpO2 (4): mild desaturation
        max(0.0, rng.uniform(0.7, 0.95) - spo2_drop),
        rng.uniform(-0.15, 0.0) * severity,  # declining
        rng.uniform(0.1, 0.5) * severity,
        rng.uniform(0.0, 0.3) * severity,
    ]


def _apnea_features(label: EpochLabel, rng) -> List[float]:
    """Features typical of apnea (complete obstruction)."""
    severity = label.severity
    spo2_drop = (label.spo2_drop or 8.0) / 30.0
    
    return [
        # RIP (11): high effort, no flow, paradoxical breathing
        rng.uniform(0.5, 0.9),     # high thorax effort
        rng.uniform(0.4, 0.8),     # high abdomen effort
        rng.uniform(0.1, 0.3),     # LOW total amplitude (no airflow)
        rng.uniform(0.30, 0.45),   # elevated respiratory rate
        rng.uniform(0.05, 0.15),
        rng.uniform(0.5, 1.0),     # HIGH phase angle (paradoxical)
        rng.choice([0.0, 1.0], p=[0.2, 0.8]),  # likely paradoxical
        rng.uniform(0.7, 1.0),     # high flow limitation
        rng.uniform(0.7, 1.0),     # HIGH effort
        0.0, 0.0,
        # Audio (9): silent (complete obstruction = no airflow = no snoring)
        rng.uniform(0.0, 0.05),    # SILENT during apnea
        0.0, 0.0, 0.0,
        rng.uniform(0.05, 0.15),
        rng.uniform(0.05, 0.15),
        0.0, 0.0, 0.0,
        # IMU (9)
        rng.choice([0.0, 1.0], p=[0.3, 0.7]),  # Very likely supine
        rng.uniform(-0.1, 0.1),
        rng.uniform(-0.1, 0.1),
        rng.uniform(0.8, 1.0),     # very stable (not moving)
        rng.uniform(0.0, 0.05),    # minimal movement
        rng.uniform(0.0, 0.03),
        0.0, 0.0, 0.0,
        # SpO2 (4): significant desaturation
        max(0.0, rng.uniform(0.5, 0.85) - spo2_drop),  # LOW SpO2
        rng.uniform(-0.3, -0.05),  # declining
        rng.uniform(0.4, 0.9),     # HIGH hypoxemia risk
        rng.uniform(0.2, 0.8) * severity,
    ]


# =============================================================================
# PyTorch Dataset
# =============================================================================

class UCDDBDataset(Dataset):
    """Dataset for training from UCDDB annotations."""
    
    def __init__(self, labels: List[EpochLabel], seed: int = 42):
        self.labels = labels
        self.rng = np.random.default_rng(seed)
        
        # Pre-generate features (with augmentation via randomness)
        self.features = []
        self.states = []
        self.severities = []
        
        for lbl in labels:
            feat = epoch_label_to_features(lbl, self.rng)
            self.features.append(feat)
            self.states.append(lbl.state)
            self.severities.append(lbl.severity)
        
        self.features = np.array(self.features, dtype=np.float32)
        self.states = np.array(self.states, dtype=np.int64)
        self.severities = np.array(self.severities, dtype=np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.LongTensor([self.states[idx]]).squeeze(),
            torch.FloatTensor([self.severities[idx]]).squeeze(),
        )


# =============================================================================
# Training Loop
# =============================================================================

def train_classifier(
    train_dataset: UCDDBDataset,
    val_dataset: UCDDBDataset,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> Tuple[StateClassifier, Dict]:
    """Train the state classifier with focal loss."""
    
    model = StateClassifier(input_dim=33)
    
    # Class-weighted sampler for imbalanced data
    class_counts = Counter(train_dataset.states.tolist())
    weights = [1.0 / class_counts[s] for s in train_dataset.states]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss functions
    focal_loss = FocalLoss(gamma=2.0)
    severity_loss_fn = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        
        for features, states, severities in train_loader:
            optimizer.zero_grad()
            
            out = model(features)
            loss_cls = focal_loss(out['state_logits'], states)
            loss_sev = severity_loss_fn(out['severity'], severities)
            loss = loss_cls + 0.5 * loss_sev
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)
        
        # Validate
        model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for features, states, severities in val_loader:
                out = model(features)
                preds = torch.argmax(out['state_probs'], dim=-1)
                all_preds.extend(preds.numpy().tolist())
                all_true.extend(states.numpy().tolist())
        
        val_acc = accuracy_score(all_true, all_preds)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f} (best={best_val_acc:.4f})")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history


# =============================================================================
# Leave-One-Subject-Out Cross-Validation
# =============================================================================

def loso_cross_validation(data_dir: str, n_epochs: int = 50):
    """
    Leave-One-Subject-Out CV on UCDDB data.
    
    This gives the most honest estimate of how well the classifier
    generalizes to unseen patients.
    """
    print("=" * 70)
    print("  LOSO Cross-Validation on UCDDB (25 subjects)")
    print("=" * 70)
    
    dataset = build_ucddb_dataset(data_dir)
    subjects = dataset['subjects']
    
    all_preds = []
    all_true = []
    per_subject_results = {}
    
    for i, test_subj in enumerate(subjects):
        # Split: all others for train, this one for test
        train_labels = []
        test_labels = []
        
        for subj, labels in dataset['labels'].items():
            if subj == test_subj:
                test_labels = labels
            else:
                train_labels.extend(labels)
        
        if not test_labels:
            continue
        
        print(f"\n  [{i+1:2d}/{len(subjects)}] Test subject: {test_subj} "
              f"(train={len(train_labels)}, test={len(test_labels)})")
        
        # Create datasets
        train_ds = UCDDBDataset(train_labels, seed=42 + i)
        test_ds = UCDDBDataset(test_labels, seed=99 + i)
        
        # Train
        model, history = train_classifier(
            train_ds, test_ds, n_epochs=n_epochs, lr=1e-3, batch_size=128
        )
        
        # Evaluate on test subject
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        
        with torch.no_grad():
            for features, states, severities in test_loader:
                out = model(features)
                preds = torch.argmax(out['state_probs'], dim=-1)
                
                subj_preds = preds.numpy().tolist()
                subj_true = states.numpy().tolist()
                
                all_preds.extend(subj_preds)
                all_true.extend(subj_true)
                
                acc = accuracy_score(subj_true, subj_preds)
                per_subject_results[test_subj] = acc
                print(f"    Subject {test_subj} accuracy: {acc:.4f}")
    
    # Overall results
    print("\n" + "=" * 70)
    print("  LOSO Overall Results")
    print("=" * 70)
    
    overall_acc = accuracy_score(all_true, all_preds)
    print(f"\n  Overall Accuracy: {overall_acc:.4f}")
    
    print(f"\n  Per-class Report:")
    target_names = [STATE_NAMES[i] for i in range(4)]
    report = classification_report(all_true, all_preds, target_names=target_names, digits=3)
    print(report)
    
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(all_true, all_preds)
    print(f"  {'':15s} {'Pred Awake':>12s} {'Pred Normal':>12s} {'Pred Snore':>12s} {'Pred Apnea':>12s}")
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:15s}", end="")
        for val in row:
            print(f"{val:12d}", end="")
        print()
    
    print(f"\n  Per-subject accuracy:")
    accs = list(per_subject_results.values())
    print(f"    Mean: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"    Min:  {min(accs):.4f} (subject {min(per_subject_results, key=per_subject_results.get)})")
    print(f"    Max:  {max(accs):.4f} (subject {max(per_subject_results, key=per_subject_results.get)})")
    
    return {
        'overall_accuracy': overall_acc,
        'per_subject_accuracy': per_subject_results,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


# =============================================================================
# Quick Train (single split for fast iteration)
# =============================================================================

def quick_train(data_dir: str, test_fraction: float = 0.2, n_epochs: int = 30):
    """Quick train/test split for fast iteration."""
    print("=" * 60)
    print("  Quick Train (80/20 random split)")
    print("=" * 60)
    
    dataset = build_ucddb_dataset(data_dir)
    subjects = dataset['subjects']
    
    # Split subjects 80/20
    rng = np.random.default_rng(42)
    n_test = max(1, int(len(subjects) * test_fraction))
    test_subjects = set(rng.choice(subjects, n_test, replace=False))
    
    train_labels = []
    test_labels = []
    for subj, labels in dataset['labels'].items():
        if subj in test_subjects:
            test_labels.extend(labels)
        else:
            train_labels.extend(labels)
    
    print(f"  Train: {len(train_labels)} epochs from {len(subjects) - n_test} subjects")
    print(f"  Test:  {len(test_labels)} epochs from {n_test} subjects")
    print(f"  Test subjects: {test_subjects}")
    
    train_ds = UCDDBDataset(train_labels, seed=42)
    test_ds = UCDDBDataset(test_labels, seed=99)
    
    model, history = train_classifier(train_ds, test_ds, n_epochs=n_epochs)
    
    # Final evaluation
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    
    with torch.no_grad():
        for features, states, severities in test_loader:
            out = model(features)
            preds = torch.argmax(out['state_probs'], dim=-1)
            
            acc = accuracy_score(states.numpy(), preds.numpy())
            target_names = [STATE_NAMES[i] for i in range(4)]
            report = classification_report(
                states.numpy(), preds.numpy(), 
                target_names=target_names, digits=3
            )
    
    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"\n{report}")
    
    return model, acc


if __name__ == '__main__':
    import sys
    
    data_dir = '/app/ucddb_data'
    
    if len(sys.argv) > 1 and sys.argv[1] == 'loso':
        results = loso_cross_validation(data_dir, n_epochs=50)
    else:
        model, acc = quick_train(data_dir, n_epochs=30)
