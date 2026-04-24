"""
OSA Acoustic Intervention System - Main Entry Point
======================================================

Provides the complete integrated system and demonstration script.

System architecture:

  ┌──────────────────────────────────────────────────────────────┐
  │                    EARPHONE DEVICE                           │
  │                                                              │
  │  ┌─────────┐  ┌─────────┐  ┌────────┐  ┌──────────┐        │
  │  │ RIP Band│  │ Mic     │  │ IMU    │  │ SpO2     │        │
  │  │ (chest/ │  │ (snore  │  │(6-axis │  │(pulse ox)│        │
  │  │ abdomen)│  │ audio)  │  │accel/  │  │          │        │
  │  │         │  │         │  │gyro)   │  │          │        │
  │  └────┬────┘  └────┬────┘  └───┬────┘  └────┬─────┘        │
  │       │            │           │             │               │
  │  ┌────▼────────────▼───────────▼─────────────▼──────────┐   │
  │  │        Multimodal Feature Extractor (33-dim)         │   │
  │  │  • Bandpass filtering  • Online z-score normalization│   │
  │  │  • Phase angle calc    • Spectral analysis           │   │
  │  └─────────────────────┬────────────────────────────────┘   │
  │                        │                                     │
  │  ┌─────────────────────▼────────────────────────────────┐   │
  │  │        OSA Risk Predictor (Bi-LSTM + Attention)      │   │
  │  │  • Temporal pattern recognition (10-epoch lookback)  │   │
  │  │  • Risk score + severity + time-to-event output      │   │
  │  └─────────────────────┬────────────────────────────────┘   │
  │                        │                                     │
  │  ┌─────────────────────▼────────────────────────────────┐   │
  │  │     Hierarchical Intervention Protocol (FSM)         │   │
  │  │  MONITOR → DETECT → DIRECTIONAL CUE → EVALUATE      │   │
  │  │                              ↓ (if no response)      │   │
  │  │                       SHORT BURST CUE → COOLDOWN     │   │
  │  └─────────────────────┬────────────────────────────────┘   │
  │                        │                                     │
  │  ┌─────────────────────▼────────────────────────────────┐   │
  │  │     SAC RL Agent (6D Continuous Action Space)         │   │
  │  │  Action: [Loudness, Frequency, Duration,             │   │
  │  │          Timing, ITD, ILD]                           │   │
  │  │  • Personalized to patient physiology                │   │
  │  │  • Minimal effective dose optimization               │   │
  │  └─────────────────────┬────────────────────────────────┘   │
  │                        │                                     │
  │  ┌─────────────────────▼────────────────────────────────┐   │
  │  │     Audio Synthesizer (Binaural Rendering)           │   │
  │  │  • ITD/ILD spatial audio for directional cues        │   │
  │  │  • Fade-in/out envelope for comfort                  │   │
  │  │  • Safety-limited output amplitude                   │   │
  │  └─────────────────────┬────────────────────────────────┘   │
  │                        │                                     │
  │                   ┌────▼────┐                                │
  │                   │Speaker  │                                │
  │                   │(L/R ear)│                                │
  │                   └─────────┘                                │
  └──────────────────────────────────────────────────────────────┘

Usage:
  # Training
  python main.py --mode train --severity moderate --timesteps 500000
  
  # Evaluation  
  python main.py --mode evaluate --model-path ./osa_models/best/best_model
  
  # Simulation demo
  python main.py --mode demo --episodes 3
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional
from datetime import datetime

# System components
from osa_system.signal_processing import (
    SignalConfig, MultimodalFeatureExtractor
)
from osa_system.risk_predictor import OSARiskPredictor
from osa_system.environment import (
    OSAInterventionEnv, PatientProfile, AirwayState, SleepStage
)
from osa_system.intervention_protocol import (
    InterventionProtocol, InterventionConfig, ProtocolState
)
from osa_system.rl_agent import (
    OSAAgentTrainer, TrainingConfig, 
    RuleBasedAgent, NoInterventionAgent, RandomAgent
)
from osa_system.audio_synthesis import AudioSynthesizer


# ==============================================================================
# Integrated System
# ==============================================================================

class OSAInterventionSystem:
    """
    Complete integrated OSA acoustic intervention system.
    
    Orchestrates all components for real-time operation:
      Sensors → Features → Risk → Protocol → RL Agent → Audio → Earphone
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[SignalConfig] = None,
        intervention_config: Optional[InterventionConfig] = None,
    ):
        self.config = config or SignalConfig()
        
        # Initialize components
        self.feature_extractor = MultimodalFeatureExtractor(self.config)
        self.risk_predictor = OSARiskPredictor()
        self.protocol = InterventionProtocol(intervention_config)
        self.audio_synth = AudioSynthesizer()
        
        # Feature history for risk prediction
        self.feature_history: List[np.ndarray] = []
        self.max_history = 10  # Look-back window
        
        # Load RL model if provided
        self.rl_model = None
        if model_path and os.path.exists(model_path + '.zip'):
            from stable_baselines3 import SAC
            self.rl_model = SAC.load(model_path)
            print(f"[System] Loaded RL model from {model_path}")
        else:
            print("[System] No RL model loaded, using protocol-only mode")
    
    def process_epoch(
        self,
        thorax: np.ndarray,
        abdomen: np.ndarray,
        audio: np.ndarray,
        accel: np.ndarray,
        gyro: Optional[np.ndarray],
        spo2_values: np.ndarray,
    ) -> Dict:
        """
        Process one epoch of sensor data and generate intervention.
        
        This is the main real-time processing loop entry point.
        Called every 30 seconds with buffered sensor data.
        
        Returns:
            Dict with:
              - intervention_action: 6D action vector
              - left_audio / right_audio: stereo audio arrays (if intervention)
              - risk_score: current OSA risk
              - protocol_info: intervention protocol state info
              - features: extracted feature dict
        """
        # 1. Extract multimodal features
        feature_vector, raw_features = self.feature_extractor.extract_all(
            thorax, abdomen, audio, accel, gyro, spo2_values
        )
        
        # 2. Update feature history for temporal prediction
        self.feature_history.append(feature_vector)
        if len(self.feature_history) > self.max_history:
            self.feature_history = self.feature_history[-self.max_history:]
        
        # 3. Predict OSA risk
        if len(self.feature_history) >= 3:
            # Pad to fixed sequence length
            seq = np.array(self.feature_history)
            if len(seq) < self.max_history:
                pad = np.zeros((self.max_history - len(seq), seq.shape[1]))
                seq = np.vstack([pad, seq])
            
            risk_output = self.risk_predictor.predict(seq)
            risk_score = risk_output['risk_score']
        else:
            risk_score = 0.0
        
        # 4. Get RL agent action (if available)
        rl_action = None
        if self.rl_model is not None:
            # Convert features to environment-compatible observation
            obs = self._features_to_obs(feature_vector, raw_features)
            rl_action, _ = self.rl_model.predict(obs, deterministic=True)
        
        # 5. Protocol decides intervention
        is_supine = raw_features['imu']['is_supine'] > 0.5
        spo2 = raw_features['spo2']['spo2_current']
        airway_state = 0  # Would come from more detailed processing
        is_aroused = False  # Would come from EEG or movement detection
        
        # Estimate airway state from respiratory features
        if raw_features['rip']['is_paradoxical'] > 0.5:
            airway_state = 2  # Likely complete obstruction
        elif raw_features['rip']['phase_angle'] > 60:
            airway_state = 1  # Partial obstruction
        
        action, protocol_info = self.protocol.decide(
            risk_score=risk_score,
            is_supine=is_supine,
            spo2=spo2,
            airway_state=airway_state,
            is_aroused=is_aroused,
            rl_action=rl_action,
        )
        
        # 6. Generate audio if intervening
        left_audio = None
        right_audio = None
        if action[0] > 0.05:  # Loudness above threshold
            left_audio, right_audio = self.audio_synth.action_to_audio(action)
        
        return {
            'intervention_action': action,
            'left_audio': left_audio,
            'right_audio': right_audio,
            'risk_score': risk_score,
            'protocol_info': protocol_info,
            'features': raw_features,
            'feature_vector': feature_vector,
        }
    
    def _features_to_obs(
        self, 
        feature_vector: np.ndarray,
        raw_features: Dict,
    ) -> np.ndarray:
        """Convert extracted features to RL observation format."""
        # Map from feature extractor format to environment observation format
        rip = raw_features['rip']
        audio = raw_features['audio']
        imu = raw_features['imu']
        spo2 = raw_features['spo2']
        
        obs = np.array([
            spo2['spo2_normalized'],
            np.clip(spo2['desat_slope'] / 3.0, -1, 1),
            spo2['hypoxemia_risk'],
            0.0,  # PaCO2 (not available from sensors, estimated)
            
            rip['resp_rate'] / 30.0,
            rip['total_amplitude'],
            rip['resp_effort'],
            rip['phase_angle'] / 180.0,
            
            1.0 - rip['is_paradoxical'],  # Airway patency proxy
            rip['is_paradoxical'] / 2.0 + (rip['phase_angle'] > 60) * 0.5,
            0.0,  # Obstruction duration (estimated)
            
            audio['snore_rms'] * 10,
            audio['snore_f0'] / 500.0,
            audio['snore_pattern'] / 2.0,
            
            imu['is_supine'],
            imu['pitch'] / 90.0,
            
            0.0,  # Sleep stage (estimated)
            0.5,  # Sleep depth (estimated)
            
            0.0,  # Intervention level
            1.0,  # Epochs since intervention
            0.0,  # Arousal flag
        ], dtype=np.float32)
        
        return obs


# ==============================================================================
# Training & Evaluation Functions
# ==============================================================================

def train_agent(args):
    """Train the RL agent."""
    config = TrainingConfig(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        patient_severity=args.severity,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        use_curriculum=args.curriculum,
        buffer_size=min(args.timesteps, 100_000),
    )
    
    trainer = OSAAgentTrainer(config)
    trainer.setup()
    
    print("\n" + "="*70)
    print("  OSA Acoustic Intervention System - RL Agent Training")
    print("="*70)
    print(f"  Algorithm:      {config.algorithm}")
    print(f"  Timesteps:      {config.total_timesteps:,}")
    print(f"  Patient type:   {config.patient_severity}")
    print(f"  Curriculum:     {config.use_curriculum}")
    print(f"  Save dir:       {config.save_dir}")
    print("="*70 + "\n")
    
    # Train
    results = trainer.train()
    
    # Evaluate
    print("\n" + "="*70)
    print("  Post-Training Evaluation")
    print("="*70 + "\n")
    
    eval_metrics = trainer.evaluate(n_episodes=20)
    
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.save_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({**results, **eval_metrics}, f, indent=2)
    
    print(f"\n  Metrics saved to: {metrics_path}")
    
    return eval_metrics


def evaluate_agents(args):
    """Compare RL agent against baselines."""
    print("\n" + "="*70)
    print("  OSA Agent Comparative Evaluation")
    print("="*70 + "\n")
    
    env = OSAInterventionEnv(
        severity=args.severity,
        max_epochs=240,
        randomize_patient=True,
        seed=42,
    )
    
    agents = {
        'No Intervention': NoInterventionAgent(),
        'Random': RandomAgent(env.action_space),
        'Rule-Based': RuleBasedAgent(),
    }
    
    # Load RL agent if available
    if args.model_path and os.path.exists(args.model_path + '.zip'):
        from stable_baselines3 import SAC
        agents['SAC (Trained)'] = SAC.load(args.model_path)
    
    results = {}
    n_episodes = args.n_eval_episodes
    
    for name, agent in agents.items():
        print(f"\n  Evaluating: {name}")
        print(f"  {'─'*50}")
        
        all_rewards = []
        all_spo2_mins = []
        all_events = []
        all_arousals = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=42 + ep)
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            all_rewards.append(episode_reward)
            all_spo2_mins.append(env.episode_spo2_min)
            all_events.append(env.episode_events)
            all_arousals.append(env.episode_arousals)
        
        metrics = {
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'mean_spo2_min': float(np.mean(all_spo2_mins)),
            'mean_events': float(np.mean(all_events)),
            'std_events': float(np.std(all_events)),
            'mean_arousals': float(np.mean(all_arousals)),
        }
        results[name] = metrics
        
        print(f"    Reward:        {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"    SpO2 min:      {metrics['mean_spo2_min']:.1f}%")
        print(f"    OSA events:    {metrics['mean_events']:.1f} ± {metrics['std_events']:.1f}")
        print(f"    Arousals:      {metrics['mean_arousals']:.1f}")
    
    # Save comparison
    comparison_path = os.path.join(args.save_dir, 'agent_comparison.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Comparison saved to: {comparison_path}")
    return results


def run_demo(args):
    """Run a demonstration of the full system."""
    print("\n" + "="*70)
    print("  OSA Acoustic Intervention System - Live Demo")
    print("="*70 + "\n")
    
    # Create simulated environment
    env = OSAInterventionEnv(
        severity=args.severity,
        max_epochs=60,  # 30 minutes for demo
        randomize_patient=False,
        render_mode='ansi',
        seed=42,
    )
    
    # Create protocol
    protocol = InterventionProtocol()
    audio_synth = AudioSynthesizer()
    
    # Use rule-based agent for demo (no training needed)
    agent = RuleBasedAgent()
    
    for episode in range(args.episodes):
        print(f"\n{'━'*60}")
        print(f"  Episode {episode + 1}/{args.episodes}")
        print(f"{'━'*60}")
        
        obs, info = env.reset(seed=42 + episode)
        done = False
        step = 0
        
        while not done:
            # Agent decides action
            action, _ = agent.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # Display status every 10 epochs
            if step % 10 == 0 or action[0] > 0.05:
                render_str = env.render()
                if render_str:
                    print(f"\n{render_str}")
                
                if action[0] > 0.05:
                    print(f"  → INTERVENTION: loudness={action[0]:.2f}, "
                          f"freq={action[1]:.0f}Hz, dur={action[2]:.1f}s, "
                          f"ITD={action[4]:.1f}ms, ILD={action[5]:.0f}dB")
                    
                    # Generate audio (demo)
                    left, right = audio_synth.action_to_audio(action)
                    print(f"    Audio generated: {len(left)} samples, "
                          f"peak L={np.max(np.abs(left)):.3f} R={np.max(np.abs(right)):.3f}")
                
                print(f"  Reward: {reward:.3f}")
        
        # Episode summary
        print(f"\n  Episode Summary:")
        print(f"    Total reward:  {sum(env.episode_rewards):.2f}")
        print(f"    SpO2 minimum:  {env.episode_spo2_min:.1f}%")
        print(f"    OSA events:    {env.episode_events}")
        print(f"    Arousals:      {env.episode_arousals}")
    
    print(f"\n{'='*70}")
    print(f"  Demo complete!")
    print(f"{'='*70}\n")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OSA Personalized Acoustic Intervention System'
    )
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['train', 'evaluate', 'demo'],
                       help='Operation mode')
    parser.add_argument('--algorithm', type=str, default='SAC',
                       choices=['SAC', 'PPO'],
                       help='RL algorithm')
    parser.add_argument('--severity', type=str, default='moderate',
                       choices=['mild', 'moderate', 'severe'],
                       help='Patient severity profile')
    parser.add_argument('--timesteps', type=int, default=100_000,
                       help='Total training timesteps')
    parser.add_argument('--save-dir', type=str, default='./osa_models',
                       help='Directory for model saving')
    parser.add_argument('--log-dir', type=str, default='./osa_logs',
                       help='Directory for training logs')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (for evaluate mode)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of demo/eval episodes')
    parser.add_argument('--n-eval-episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--curriculum', action='store_true', default=True,
                       help='Use curriculum learning')
    parser.add_argument('--no-curriculum', action='store_false', dest='curriculum')
    
    args = parser.parse_args()
    
    print(f"\n  OSA Acoustic Intervention System v1.0")
    print(f"  Mode: {args.mode}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'evaluate':
        evaluate_agents(args)
    elif args.mode == 'demo':
        run_demo(args)


if __name__ == '__main__':
    main()
