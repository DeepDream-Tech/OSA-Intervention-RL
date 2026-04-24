"""
UCDDB Data Parser & 4-State Label Generator
=============================================

Parses UCDDB (University College Dublin Sleep Apnea Database) annotations 
to create labeled datasets with the 4-state classification scheme:

  State 0: AWAKE      (清醒)
  State 1: NORMAL     (正常睡眠)  
  State 2: SNORING    (打鼾 / 部分阻塞)
  State 3: APNEA      (呼吸暂停 / 完全阻塞)

Plus a continuous severity score (0-1) for each epoch.

Data source: https://physionet.org/content/ucddb/1.0.0/
25 subjects, full PSG with respiratory events + sleep staging annotations.

UCDDB Sleep Stage Coding:
  0 = Wake, 1 = REM, 2 = Stage1(N1), 3 = Stage2(N2), 4 = SWS(N3), 5 = Artifact

UCDDB Respiratory Event Types:
  APNEA-O  = Obstructive Apnea    (complete obstruction)
  APNEA-C  = Central Apnea        (no respiratory effort)
  APNEA-M  = Mixed Apnea
  HYP-O    = Obstructive Hypopnea (partial obstruction)
  HYP-C    = Central Hypopnea
  HYP-M    = Mixed Hypopnea
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# 4-State Enum
# =============================================================================

class SleepState:
    AWAKE = 0
    NORMAL_SLEEP = 1
    SNORING = 2    # Partial obstruction / hypopnea
    APNEA = 3     # Complete obstruction / apnea
    
    NAMES = {0: 'Awake', 1: 'Normal Sleep', 2: 'Snoring', 3: 'Apnea'}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RespiratoryEvent:
    """One respiratory event from UCDDB annotation."""
    time_seconds: float       # Onset time in seconds from recording start
    event_type: str           # APNEA-O, APNEA-C, HYP-O, HYP-C, etc.
    duration: int             # Duration in seconds
    spo2_low: Optional[float] = None
    spo2_drop: Optional[float] = None
    has_snore: bool = False
    has_arousal: bool = False
    
    @property
    def is_apnea(self) -> bool:
        return 'APNEA' in self.event_type
    
    @property
    def is_hypopnea(self) -> bool:
        return 'HYP' in self.event_type
    
    @property
    def is_obstructive(self) -> bool:
        return self.event_type.endswith('-O')
    
    @property
    def is_central(self) -> bool:
        return self.event_type.endswith('-C')


@dataclass
class EpochLabel:
    """Label for one 30-second epoch."""
    epoch_idx: int
    sleep_stage_raw: int       # Original UCDDB stage (0-5)
    state: int                 # Our 4-state: 0=Awake,1=Normal,2=Snoring,3=Apnea
    severity: float            # 0.0 - 1.0 continuous severity
    has_respiratory_event: bool
    event_type: Optional[str] = None
    event_duration: Optional[int] = None
    spo2_drop: Optional[float] = None
    is_supine: bool = False    # Would come from position channel


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS to seconds from midnight."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def parse_respiratory_events(filepath: str) -> List[RespiratoryEvent]:
    """Parse UCDDB respiratory event file."""
    events = []
    with open(filepath) as f:
        lines = f.readlines()
    
    for line in lines[3:]:  # Skip header lines
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
        
        try:
            time_str = parts[0]
            event_type = parts[1]
            
            if event_type == 'POSSIBLE':
                continue
            
            # Duration index depends on whether PB/CS flag exists
            dur_idx = 2
            duration = int(parts[dur_idx])
            
            # SpO2 values
            spo2_low = None
            spo2_drop = None
            if len(parts) > dur_idx + 2:
                try:
                    spo2_low = float(parts[dur_idx + 1])
                    spo2_drop = float(parts[dur_idx + 2])
                except ValueError:
                    pass
            
            # Snore/Arousal markers (+ or -)
            has_snore = False
            has_arousal = False
            # These are in fixed columns, but parsing from split is unreliable
            # We'll use the raw line with column positions
            if len(line) > 60:
                snore_region = line[54:60]
                arousal_region = line[60:66]
                has_snore = '+' in snore_region
                has_arousal = '+' in arousal_region
            
            events.append(RespiratoryEvent(
                time_seconds=parse_time_to_seconds(time_str),
                event_type=event_type,
                duration=duration,
                spo2_low=spo2_low,
                spo2_drop=spo2_drop,
                has_snore=has_snore,
                has_arousal=has_arousal,
            ))
        except (ValueError, IndexError):
            continue
    
    return events


def parse_sleep_stages(filepath: str) -> List[int]:
    """Parse UCDDB sleep stage file. Each line = one 30s epoch."""
    with open(filepath) as f:
        stages = [int(line.strip()) for line in f if line.strip().isdigit()]
    return stages


# =============================================================================
# Label Generation
# =============================================================================

def generate_epoch_labels(
    stages: List[int],
    events: List[RespiratoryEvent],
    recording_start_time: float = 0.0,
    epoch_duration: float = 30.0,
) -> List[EpochLabel]:
    """
    Generate 4-state labels for each epoch by combining sleep staging 
    and respiratory events.
    
    Mapping Logic:
    
    1. If sleep_stage == 0 (Wake) → State = AWAKE
    2. If epoch overlaps with APNEA event → State = APNEA
    3. If epoch overlaps with HYP event → State = SNORING  
    4. Otherwise during sleep → State = NORMAL_SLEEP
    
    Severity Score:
      - AWAKE: 0.0
      - NORMAL_SLEEP: 0.0
      - SNORING: 0.3 + 0.3*(SpO2_drop/10) + 0.4*(duration/30) 
      - APNEA: 0.6 + 0.2*(SpO2_drop/10) + 0.2*(duration/30)
    """
    n_epochs = len(stages)
    labels = []
    
    for epoch_idx in range(n_epochs):
        epoch_start = recording_start_time + epoch_idx * epoch_duration
        epoch_end = epoch_start + epoch_duration
        
        raw_stage = stages[epoch_idx]
        
        # Check which respiratory events overlap this epoch
        overlapping_events = []
        for evt in events:
            evt_end = evt.time_seconds + evt.duration
            # Check overlap
            if evt.time_seconds < epoch_end and evt_end > epoch_start:
                overlapping_events.append(evt)
        
        # Determine state
        if raw_stage == 0:  # Wake
            state = SleepState.AWAKE
            severity = 0.0
            event_type = None
            event_duration = None
            spo2_drop = None
            has_event = False
            
        elif raw_stage in [5, 8]:  # Artifact / Unknown → treat as wake
            state = SleepState.AWAKE
            severity = 0.0
            event_type = None
            event_duration = None
            spo2_drop = None
            has_event = False
            
        elif overlapping_events:
            has_event = True
            # Find the most severe event in this epoch
            apneas = [e for e in overlapping_events if e.is_apnea]
            hypopneas = [e for e in overlapping_events if e.is_hypopnea]
            
            if apneas:
                # Apnea takes priority
                worst = max(apneas, key=lambda e: e.spo2_drop or 0)
                state = SleepState.APNEA
                event_type = worst.event_type
                event_duration = worst.duration
                spo2_drop = worst.spo2_drop
                
                # Severity: 0.6 base + SpO2 drop contribution + duration contribution
                drop_score = min((worst.spo2_drop or 0) / 15.0, 1.0)
                dur_score = min(worst.duration / 40.0, 1.0)
                severity = 0.6 + 0.2 * drop_score + 0.2 * dur_score
                
            else:
                # Hypopnea
                worst = max(hypopneas, key=lambda e: e.spo2_drop or 0)
                state = SleepState.SNORING
                event_type = worst.event_type
                event_duration = worst.duration
                spo2_drop = worst.spo2_drop
                
                drop_score = min((worst.spo2_drop or 0) / 10.0, 1.0)
                dur_score = min(worst.duration / 30.0, 1.0)
                severity = 0.3 + 0.3 * drop_score + 0.4 * dur_score
            
            severity = float(np.clip(severity, 0.0, 1.0))
            
        else:
            # Sleeping, no respiratory event
            state = SleepState.NORMAL_SLEEP
            severity = 0.0
            event_type = None
            event_duration = None
            spo2_drop = None
            has_event = False
        
        labels.append(EpochLabel(
            epoch_idx=epoch_idx,
            sleep_stage_raw=raw_stage,
            state=state,
            severity=severity,
            has_respiratory_event=has_event,
            event_type=event_type,
            event_duration=event_duration,
            spo2_drop=spo2_drop,
        ))
    
    return labels


# =============================================================================
# Dataset Builder
# =============================================================================

def build_ucddb_dataset(data_dir: str) -> Dict:
    """
    Build complete labeled dataset from UCDDB annotations.
    
    Returns:
        Dict with:
          - 'subjects': list of subject IDs
          - 'labels': dict of subject_id → list of EpochLabel
          - 'statistics': summary statistics
    """
    subjects = ['002','003','005','006','007','008','009','010','011',
                '012','013','014','015','017','018','019','020','021',
                '022','023','024','025','026','027','028']
    
    all_labels = {}
    all_events = {}
    total_epochs = 0
    state_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for subj in subjects:
        stage_file = os.path.join(data_dir, f'ucddb{subj}_stage.txt')
        event_file = os.path.join(data_dir, f'ucddb{subj}_respevt.txt')
        
        if not os.path.exists(stage_file) or not os.path.exists(event_file):
            continue
        
        stages = parse_sleep_stages(stage_file)
        events = parse_respiratory_events(event_file)
        
        labels = generate_epoch_labels(stages, events)
        
        all_labels[subj] = labels
        all_events[subj] = events
        total_epochs += len(labels)
        
        for lbl in labels:
            state_counts[lbl.state] += 1
    
    # Compute statistics
    statistics = {
        'n_subjects': len(all_labels),
        'total_epochs': total_epochs,
        'total_hours': total_epochs * 30 / 3600,
        'state_distribution': {
            SleepState.NAMES[k]: {'count': v, 'fraction': v / total_epochs}
            for k, v in state_counts.items()
        },
        'severity_stats': {},
    }
    
    # Severity statistics per state
    for state_id in [2, 3]:
        severities = [lbl.severity for labels in all_labels.values() 
                     for lbl in labels if lbl.state == state_id]
        if severities:
            statistics['severity_stats'][SleepState.NAMES[state_id]] = {
                'mean': float(np.mean(severities)),
                'std': float(np.std(severities)),
                'min': float(min(severities)),
                'max': float(max(severities)),
            }
    
    return {
        'subjects': list(all_labels.keys()),
        'labels': all_labels,
        'events': all_events,
        'statistics': statistics,
    }


if __name__ == '__main__':
    import json
    
    dataset = build_ucddb_dataset('/app/ucddb_data')
    stats = dataset['statistics']
    
    print("=" * 60)
    print("  UCDDB Dataset Summary")
    print("=" * 60)
    print(f"  Subjects: {stats['n_subjects']}")
    print(f"  Total epochs: {stats['total_epochs']}")
    print(f"  Total hours: {stats['total_hours']:.1f}")
    print()
    print("  4-State Distribution:")
    for state_name, info in stats['state_distribution'].items():
        print(f"    {state_name:15s}: {info['count']:5d} ({info['fraction']*100:.1f}%)")
    print()
    print("  Severity Statistics:")
    for state_name, sev in stats['severity_stats'].items():
        print(f"    {state_name}: mean={sev['mean']:.3f} std={sev['std']:.3f} "
              f"range=[{sev['min']:.3f}, {sev['max']:.3f}]")
