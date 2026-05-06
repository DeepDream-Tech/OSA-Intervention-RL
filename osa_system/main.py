"""
OSA Acoustic Intervention System V2 - Main Entry Point
=======================================================

V2 Classification-Based Architecture using real UCDDB data.

System pipeline:
  Sensors → Features (8-dim) → State Classifier → Decision Engine → Audio

Usage:
  # Demo mode (simulated data)
  python -m osa_system.main --mode demo --episodes 3

  # Evaluate on real UCDDB data
  python -m osa_system.main --mode evaluate

  # Train classifier on UCDDB
  python -m osa_system.main --mode train
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter

# V2 System components
from .system import (
    OSASystem, StateClassifier, DecisionEngine, TrendEncoder,
    OSAState, STATE_NAMES, InterventionDecision
)
from .signal_processing import SignalConfig, MultimodalFeatureExtractor
from .audio_synthesis import AudioSynthesizer, AudioConfig
from .ucddb_parser import build_ucddb_dataset, SleepState, EpochLabel


# ==============================================================================
# Demo Mode - Simulated Sleep Session
# ==============================================================================

def run_demo(args):
    """Run a demonstration of the V2 system with simulated data."""
    print("\n" + "="*70)
    print("  OSA Acoustic Intervention System V2 - Demo Mode")
    print("="*70 + "\n")

    # Initialize V2 system
    system = OSASystem()
    audio_synth = AudioSynthesizer()

    # Load trained classifier if available
    model_path = args.model_path or './osa_models/classifier_real.pt'
    if os.path.exists(model_path):
        system.classifier.load_state_dict(torch.load(model_path))
        print(f"[System] Loaded classifier from {model_path}")
    else:
        print(f"[System] No trained model found, using untrained classifier")

    for episode in range(args.episodes):
        print(f"\n{'━'*60}")
        print(f"  Episode {episode + 1}/{args.episodes} - Simulated Sleep Session")
        print(f"{'━'*60}")

        system.reset()

        # Simulate 30 minutes (60 epochs)
        n_epochs = 60
        rng = np.random.default_rng(42 + episode)

        for epoch in range(n_epochs):
            # Simulate feature vector (8-dim UCDDB-aligned)
            # Randomly generate sleep states with realistic distribution
            state_prob = rng.random()
            if state_prob < 0.05:  # 5% awake
                features = rng.uniform([0.3, 0.3, 0.2, -0.05, 0.0, 0.0, 0.95, -0.01],
                                      [0.8, 0.8, 0.4, 0.05, 0.05, 1.0, 0.98, 0.01], size=8)
                is_supine = rng.random() < 0.3
                spo2 = 96.0
            elif state_prob < 0.60:  # 55% normal sleep
                features = rng.uniform([0.4, 0.4, 0.15, -0.1, 0.0, 0.0, 0.94, -0.02],
                                      [0.7, 0.7, 0.35, 0.1, 0.1, 1.0, 0.97, 0.02], size=8)
                is_supine = rng.random() < 0.5
                spo2 = 95.0
            elif state_prob < 0.85:  # 25% snoring
                features = rng.uniform([0.3, 0.3, 0.1, 0.1, 0.3, 0.0, 0.90, -0.05],
                                      [0.6, 0.6, 0.3, 0.4, 0.7, 1.0, 0.94, 0.0], size=8)
                is_supine = rng.random() < 0.7
                spo2 = 92.0
            else:  # 15% apnea
                features = rng.uniform([0.1, 0.1, 0.05, 0.4, 0.5, 0.0, 0.80, -0.1],
                                      [0.3, 0.3, 0.15, 0.8, 0.9, 1.0, 0.88, -0.03], size=8)
                is_supine = rng.random() < 0.8
                spo2 = 88.0

            # Process epoch
            result = system.process_epoch(
                feature_vector=features.astype(np.float32),
                is_supine=is_supine,
                spo2=spo2,
            )

            # Display every 10 epochs or when intervening
            if epoch % 10 == 0 or result['decision'].should_intervene:
                time_min = epoch * 0.5
                print(f"\n  Epoch {epoch} ({time_min:.1f} min)")
                print(f"    State: {result['state_name']} (severity={result['severity']:.2f})")
                print(f"    Position: {'Supine' if is_supine else 'Non-supine'}, SpO2={spo2:.1f}%")

                if result['decision'].should_intervene:
                    dec = result['decision']
                    print(f"    → INTERVENTION: {dec.intervention_type}")
                    print(f"      Reason: {dec.reason}")
                    print(f"      Params: loudness={dec.suggested_loudness:.2f}, "
                          f"freq={dec.suggested_frequency:.0f}Hz, "
                          f"dur={dec.suggested_duration:.1f}s")

                    # Generate audio
                    action = np.array([
                        dec.suggested_loudness,
                        dec.suggested_frequency,
                        dec.suggested_duration,
                        0.5,  # timing
                        dec.suggested_itd,
                        dec.suggested_ild,
                    ])
                    left, right = audio_synth.action_to_audio(action)
                    if left is not None:
                        print(f"      Audio: {len(left)} samples, "
                              f"peak L={np.max(np.abs(left)):.3f} R={np.max(np.abs(right)):.3f}")

        # Episode summary
        summary = system.get_session_summary()
        print(f"\n  Episode Summary:")
        print(f"    Duration: {summary['total_minutes']:.1f} minutes")
        print(f"    Interventions: {summary['interventions']}")
        print(f"    Intervention rate: {summary['intervention_rate']:.1%}")
        print(f"    State distribution:")
        for state_name, pct in summary['state_distribution'].items():
            print(f"      {state_name}: {pct:.1%}")

    print(f"\n{'='*70}")
    print(f"  Demo complete!")
    print(f"{'='*70}\n")


# ==============================================================================
# Evaluation Mode - Real UCDDB Data
# ==============================================================================

def run_evaluation(args):
    """Evaluate V2 system on real UCDDB data."""
    print("\n" + "="*70)
    print("  OSA System V2 - Evaluation on Real UCDDB Data")
    print("="*70 + "\n")

    # Import evaluation logic
    from .evaluate_real_data import evaluate_all_subjects

    # Load UCDDB dataset
    print("Loading UCDDB dataset...")
    dataset = build_ucddb_dataset()
    print(f"Loaded {len(dataset)} subjects\n")

    # Run evaluation
    results = evaluate_all_subjects(dataset, verbose=args.verbose)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {results_path}")
    print(f"{'='*70}\n")


# ==============================================================================
# Training Mode - Train Classifier
# ==============================================================================

def run_training(args):
    """Train the state classifier on UCDDB data."""
    print("\n" + "="*70)
    print("  OSA System V2 - Classifier Training")
    print("="*70 + "\n")

    # Import training logic
    from .train_classifier import train_loso_cv

    # Load UCDDB dataset
    print("Loading UCDDB dataset...")
    dataset = build_ucddb_dataset()
    print(f"Loaded {len(dataset)} subjects\n")

    # Train with LOSO cross-validation
    print("Starting Leave-One-Subject-Out cross-validation...")
    results = train_loso_cv(
        dataset=dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
    )

    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"  Mean Accuracy: {results['mean_accuracy']:.2%}")
    print(f"  Std Accuracy: {results['std_accuracy']:.2%}")
    print(f"{'='*70}\n")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OSA Personalized Acoustic Intervention System V2'
    )
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'evaluate', 'train'],
                       help='Operation mode')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained classifier model')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of demo episodes')
    parser.add_argument('--save-dir', type=str, default='./osa_models',
                       help='Directory for model saving/loading')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    # Training-specific arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per fold')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    args = parser.parse_args()

    print(f"\n  OSA Acoustic Intervention System V2")
    print(f"  Mode: {args.mode}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'evaluate':
        run_evaluation(args)
    elif args.mode == 'train':
        run_training(args)


if __name__ == '__main__':
    main()
