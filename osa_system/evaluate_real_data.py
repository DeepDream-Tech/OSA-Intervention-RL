"""
End-to-End System Test on Real UCDDB Annotations
===================================================

This test uses REAL clinical annotations from 25 UCDDB patients to drive
the decision engine. It answers the key question:

  "Given the real sequence of sleep states and respiratory events 
   for each patient night, what would our system have done?"

We bypass the classifier (since we have ground-truth labels) and directly
test the decision engine + intervention protocol logic against real data.

This is the honest evaluation: we know the states perfectly, and we measure
whether the decision engine makes clinically sensible intervention choices.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List
import json

from osa_system.ucddb_parser import build_ucddb_dataset, SleepState, EpochLabel
from osa_system.system_v2 import (
    DecisionEngine, TrendEncoder, OSAState, STATE_NAMES, InterventionDecision
)


def simulate_night(
    labels: List[EpochLabel],
    decision_engine: DecisionEngine,
    verbose: bool = False,
) -> Dict:
    """
    Simulate one night using real UCDDB annotations.
    
    For each epoch, we feed the REAL state and severity to the decision engine
    and record what intervention it would have chosen.
    
    Args:
        labels: List of EpochLabel from UCDDB parser
        decision_engine: Our rule-based decision engine
        verbose: Print epoch-by-epoch details
    
    Returns:
        Night summary statistics
    """
    decision_engine.reset()
    
    n_epochs = len(labels)
    interventions = []
    decisions_by_state = defaultdict(list)
    
    # Simulate position (assume supine for epochs near apnea events)
    # In UCDDB, position data exists in the EDF but we use a simple heuristic:
    # Supine probability higher during respiratory events
    rng = np.random.default_rng(42)
    
    for i, label in enumerate(labels):
        state = label.state
        severity = label.severity
        
        # Estimate supine from context
        # UCDDB shows supine is more common during events
        if state in [SleepState.SNORING, SleepState.APNEA]:
            is_supine = rng.random() < 0.65  # 65% supine during events
        else:
            is_supine = rng.random() < 0.40  # 40% supine during normal sleep
        
        # Estimate SpO2 from annotation
        if label.spo2_drop is not None:
            spo2 = 96.0 - label.spo2_drop
        elif state == SleepState.APNEA:
            spo2 = 88.0  # Typical during apnea
        elif state == SleepState.SNORING:
            spo2 = 92.0  # Mild reduction during hypopnea
        else:
            spo2 = 96.0
        
        # Make decision
        decision = decision_engine.decide(
            state=state,
            severity=severity,
            is_supine=is_supine,
            trend_vector=None,  # No trend encoder for this test
            spo2=spo2,
        )
        
        decisions_by_state[state].append(decision)
        
        if decision.should_intervene:
            interventions.append({
                'epoch': i,
                'time_min': i * 0.5,
                'state': state,
                'state_name': STATE_NAMES.get(state, '?'),
                'severity': severity,
                'type': decision.intervention_type,
                'urgency': decision.urgency,
                'reason': decision.reason,
                'loudness': decision.suggested_loudness,
                'is_supine': is_supine,
                'spo2': spo2,
            })
        
        if verbose and decision.should_intervene:
            time_min = i * 0.5
            print(f"  [{time_min:6.1f}min] {decision.reason}")
    
    # Compute statistics
    state_counts = Counter(l.state for l in labels)
    intervention_by_type = Counter(iv['type'] for iv in interventions)
    intervention_during_state = Counter(iv['state'] for iv in interventions)
    
    # Key clinical metrics
    total_resp_events = sum(1 for l in labels if l.has_respiratory_event)
    apnea_epochs = state_counts.get(SleepState.APNEA, 0)
    snoring_epochs = state_counts.get(SleepState.SNORING, 0)
    
    # How many respiratory event epochs got intervention?
    events_with_intervention = sum(1 for iv in interventions 
                                   if iv['state'] in [SleepState.SNORING, SleepState.APNEA])
    
    # False interventions (during normal sleep or awake)
    false_interventions = sum(1 for iv in interventions
                             if iv['state'] in [SleepState.AWAKE, SleepState.NORMAL_SLEEP])
    
    # How many apnea events were caught?
    apnea_interventions = sum(1 for iv in interventions if iv['state'] == SleepState.APNEA)
    
    return {
        'n_epochs': n_epochs,
        'duration_hours': n_epochs * 30 / 3600,
        'state_distribution': {STATE_NAMES.get(k, '?'): v for k, v in state_counts.items()},
        'total_respiratory_events': total_resp_events,
        'apnea_epochs': apnea_epochs,
        'snoring_epochs': snoring_epochs,
        'total_interventions': len(interventions),
        'directional_cues': intervention_by_type.get('directional_cue', 0),
        'burst_cues': intervention_by_type.get('burst_cue', 0),
        'events_with_intervention': events_with_intervention,
        'event_intervention_rate': events_with_intervention / max(total_resp_events, 1),
        'apnea_catch_rate': apnea_interventions / max(apnea_epochs, 1),
        'false_interventions': false_interventions,
        'false_intervention_rate': false_interventions / max(len(interventions), 1),
        'interventions_per_hour': len(interventions) / max(n_epochs * 30 / 3600, 0.01),
        'intervention_details': interventions[:10],  # First 10 for inspection
    }


def run_full_evaluation(data_dir: str):
    """Run evaluation on all 25 UCDDB subjects."""
    
    print("=" * 70)
    print("  End-to-End Evaluation: Decision Engine on UCDDB Real Data")
    print("=" * 70)
    
    dataset = build_ucddb_dataset(data_dir)
    
    all_results = {}
    aggregate = defaultdict(list)
    
    for subj in dataset['subjects']:
        labels = dataset['labels'][subj]
        engine = DecisionEngine()
        
        result = simulate_night(labels, engine, verbose=False)
        all_results[subj] = result
        
        # Aggregate
        for key in ['event_intervention_rate', 'apnea_catch_rate', 
                    'false_intervention_rate', 'interventions_per_hour',
                    'total_interventions', 'total_respiratory_events']:
            aggregate[key].append(result[key])
        
        print(f"  Subject {subj}: {result['n_epochs']:4d} epochs, "
              f"{result['total_respiratory_events']:3d} events, "
              f"{result['total_interventions']:3d} interventions, "
              f"event_catch={result['event_intervention_rate']:.0%}, "
              f"apnea_catch={result['apnea_catch_rate']:.0%}, "
              f"false_rate={result['false_intervention_rate']:.0%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  Aggregate Results (25 subjects, 173 hours)")
    print("=" * 70)
    
    for key, values in aggregate.items():
        arr = np.array(values)
        print(f"  {key:35s}: {arr.mean():.3f} ± {arr.std():.3f}  "
              f"[{arr.min():.3f} — {arr.max():.3f}]")
    
    # Clinical interpretation
    print("\n" + "-" * 70)
    print("  Clinical Interpretation")
    print("-" * 70)
    
    mean_catch = np.mean(aggregate['event_intervention_rate'])
    mean_apnea_catch = np.mean(aggregate['apnea_catch_rate'])
    mean_false_rate = np.mean(aggregate['false_intervention_rate'])
    mean_per_hour = np.mean(aggregate['interventions_per_hour'])
    
    print(f"""
  呼吸事件干预率:    {mean_catch:.1%} 的打鼾/呼吸暂停epoch收到了干预
  呼吸暂停捕获率:    {mean_apnea_catch:.1%} 的呼吸暂停epoch收到了干预
  误干预率:          {mean_false_rate:.1%} 的干预发生在正常睡眠/清醒期间
  每小时干预次数:    {mean_per_hour:.1f} 次/小时

  解读:
  - 干预率反映了系统对呼吸事件的响应覆盖面
  - 误干预率越低越好（避免不必要的睡眠干扰）
  - 每小时干预次数应该适中（太高会严重影响睡眠质量）
""")
    
    # Show detailed example for one subject
    example_subj = max(all_results.keys(), 
                       key=lambda s: all_results[s]['total_respiratory_events'])
    example = all_results[example_subj]
    
    print(f"\n  详细示例: Subject {example_subj} (事件最多的患者)")
    print(f"  {'─'*50}")
    print(f"  记录时长: {example['duration_hours']:.1f} 小时")
    print(f"  状态分布: {example['state_distribution']}")
    print(f"  呼吸事件: {example['total_respiratory_events']} epochs")
    print(f"  干预次数: {example['total_interventions']}")
    print(f"    方向性cue: {example['directional_cues']}")
    print(f"    短促cue:   {example['burst_cues']}")
    print(f"\n  前10次干预:")
    for iv in example['intervention_details']:
        print(f"    [{iv['time_min']:6.1f}min] {iv['reason']}")
    
    return all_results


if __name__ == '__main__':
    results = run_full_evaluation('/app/ucddb_data')
