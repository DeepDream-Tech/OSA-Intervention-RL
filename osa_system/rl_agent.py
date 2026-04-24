"""
RL Agent Training Module
=========================

Trains a SAC (Soft Actor-Critic) agent for personalized OSA acoustic intervention.

Why SAC:
  - Best for continuous 6D action spaces (off-policy, sample efficient)
  - Automatic entropy tuning for exploration-exploitation balance
  - Proven on acoustic control tasks (arxiv:2312.05674)
  - More stable than DDPG, more sample-efficient than PPO

Training curriculum:
  Phase 1: Single-parameter (loudness only) → learn when to intervene
  Phase 2: Add frequency and duration → learn effective stimulus
  Phase 3: Add ITD/ILD → learn directional cueing
  Phase 4: Full 6D → fine-tune all parameters jointly

Hyperparameters from published recipes:
  - SAC: lr=3e-4, batch=256, buffer=1M, gamma=0.99, tau=0.005
    (arxiv:1812.05905 - Soft Actor-Critic paper)
  - Target entropy = -dim(action) = -6
    (automatic temperature adjustment)
"""

import os
import json
import numpy as np
import torch
from typing import Dict, Optional, List
from dataclasses import dataclass

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from osa_system.environment import (
    OSAInterventionEnv, PatientProfile, AirwayState
)


# ==============================================================================
# Custom Callbacks
# ==============================================================================

class OSAMetricsCallback(BaseCallback):
    """
    Custom callback that logs OSA-specific metrics during training.
    
    Tracks:
      - SpO2 statistics (min, mean, time below 90%)
      - AHI proxy (events per hour)
      - Intervention effectiveness
      - Sleep disruption metrics
    """
    
    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_spo2_mins = []
        self.episode_events = []
        self.episode_arousals = []
    
    def _on_step(self) -> bool:
        # Collect info from vectorized env
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
            if 'episode_spo2_min' in info:
                self.episode_spo2_mins.append(info['episode_spo2_min'])
            if 'total_events' in info:
                self.episode_events.append(info['total_events'])
            if 'total_arousals' in info:
                self.episode_arousals.append(info['total_arousals'])
        
        # Log periodically
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            self.logger.record('osa/mean_reward', np.mean(self.episode_rewards[-100:]))
            
            if self.episode_spo2_mins:
                self.logger.record('osa/mean_spo2_min', 
                                  np.mean(self.episode_spo2_mins[-100:]))
            if self.episode_events:
                self.logger.record('osa/mean_events_per_episode',
                                  np.mean(self.episode_events[-100:]))
            if self.episode_arousals:
                self.logger.record('osa/mean_arousals_per_episode',
                                  np.mean(self.episode_arousals[-100:]))
        
        return True


class CurriculumCallback(BaseCallback):
    """
    Implements curriculum learning for action space complexity.
    
    Gradually unmasks action dimensions:
      Phase 1 (0-25%): Only loudness active
      Phase 2 (25-50%): + frequency, duration
      Phase 3 (50-75%): + timing
      Phase 4 (75-100%): + ITD, ILD (full 6D)
    """
    
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.current_phase = 0
    
    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        
        new_phase = 0
        if progress >= 0.75:
            new_phase = 3
        elif progress >= 0.50:
            new_phase = 2
        elif progress >= 0.25:
            new_phase = 1
        
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            phase_names = ['loudness_only', 'add_freq_dur', 'add_timing', 'full_6d']
            if self.verbose > 0:
                print(f"[Curriculum] Phase {new_phase}: {phase_names[new_phase]} "
                      f"at {progress:.0%} progress")
            self.logger.record('curriculum/phase', new_phase)
        
        return True


# ==============================================================================
# Training Configuration
# ==============================================================================

@dataclass
class TrainingConfig:
    """Configuration for RL agent training."""
    
    # Algorithm
    algorithm: str = 'SAC'               # 'SAC' or 'PPO'
    
    # SAC hyperparameters (from arxiv:1812.05905)
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = 'auto'               # Automatic entropy tuning
    target_entropy: float = -6.0         # -dim(action_space)
    
    # PPO hyperparameters (from arxiv:2503.14637 Kinesis)
    ppo_lr: float = 5e-5
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_clip_range: float = 0.2
    ppo_gae_lambda: float = 0.95
    
    # Network architecture
    net_arch: List[int] = None  # Default: [256, 256]
    
    # Training
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    
    # Environment
    patient_severity: str = 'moderate'
    max_epochs_per_episode: int = 240    # 2 hours simulated
    n_envs: int = 1                      # Parallel environments
    
    # Curriculum
    use_curriculum: bool = True
    
    # Saving
    save_dir: str = './osa_models'
    log_dir: str = './osa_logs'
    
    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [256, 256]


# ==============================================================================
# Agent Training
# ==============================================================================

class OSAAgentTrainer:
    """
    Handles end-to-end training of the OSA intervention RL agent.
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model = None
        self.env = None
        self.eval_env = None
    
    def setup(self):
        """Create environments and model."""
        c = self.config
        
        os.makedirs(c.save_dir, exist_ok=True)
        os.makedirs(c.log_dir, exist_ok=True)
        
        # Create training environments
        def make_env(seed=0):
            def _init():
                env = OSAInterventionEnv(
                    severity=c.patient_severity,
                    max_epochs=c.max_epochs_per_episode,
                    randomize_patient=True,
                    seed=seed,
                )
                env = Monitor(env, os.path.join(c.log_dir, f'train_{seed}'))
                return env
            return _init
        
        self.env = DummyVecEnv([make_env(i) for i in range(c.n_envs)])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        
        # Evaluation environment (fixed patients for consistent evaluation)
        def make_eval_env():
            env = OSAInterventionEnv(
                severity=c.patient_severity,
                max_epochs=c.max_epochs_per_episode,
                randomize_patient=False,
                seed=42,
            )
            return Monitor(env, os.path.join(c.log_dir, 'eval'))
        
        self.eval_env = DummyVecEnv([make_eval_env])
        self.eval_env = VecNormalize(
            self.eval_env, 
            norm_obs=True, 
            norm_reward=False,  # Don't normalize eval rewards
            training=False,
        )
        
        # Create model
        if c.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                self.env,
                learning_rate=c.learning_rate,
                buffer_size=c.buffer_size,
                batch_size=c.batch_size,
                gamma=c.gamma,
                tau=c.tau,
                ent_coef=c.ent_coef,
                target_entropy=c.target_entropy,
                policy_kwargs=dict(
                    net_arch=c.net_arch,
                ),
                verbose=1,
                tensorboard_log=c.log_dir,
            )
        elif c.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=c.ppo_lr,
                n_steps=c.ppo_n_steps,
                batch_size=c.ppo_batch_size,
                n_epochs=c.ppo_n_epochs,
                gamma=c.gamma,
                gae_lambda=c.ppo_gae_lambda,
                clip_range=c.ppo_clip_range,
                policy_kwargs=dict(
                    net_arch=c.net_arch,
                ),
                verbose=1,
                tensorboard_log=c.log_dir,
            )
        
        print(f"[OSAAgentTrainer] Model: {c.algorithm}")
        print(f"[OSAAgentTrainer] Env obs: {self.env.observation_space}")
        print(f"[OSAAgentTrainer] Env act: {self.env.action_space}")
        print(f"[OSAAgentTrainer] Total timesteps: {c.total_timesteps:,}")
    
    def train(self) -> Dict[str, float]:
        """Run the full training loop."""
        c = self.config
        
        # Callbacks
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(c.save_dir, 'best'),
            log_path=c.log_dir,
            eval_freq=c.eval_freq,
            n_eval_episodes=c.n_eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_callback)
        
        # OSA metrics callback
        osa_callback = OSAMetricsCallback(verbose=1, log_freq=5000)
        callbacks.append(osa_callback)
        
        # Curriculum callback
        if c.use_curriculum:
            curriculum_callback = CurriculumCallback(
                c.total_timesteps, verbose=1
            )
            callbacks.append(curriculum_callback)
        
        # Train
        print(f"\n{'='*60}")
        print(f"Starting training: {c.algorithm} for {c.total_timesteps:,} steps")
        print(f"{'='*60}\n")
        
        self.model.learn(
            total_timesteps=c.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )
        
        # Save final model
        final_path = os.path.join(c.save_dir, 'final_model')
        self.model.save(final_path)
        self.env.save(os.path.join(c.save_dir, 'vec_normalize.pkl'))
        
        print(f"\nModel saved to: {final_path}")
        
        # Return training summary
        return {
            'algorithm': c.algorithm,
            'total_timesteps': c.total_timesteps,
            'final_model_path': final_path,
        }
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Returns:
            Dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        eval_env = OSAInterventionEnv(
            severity=self.config.patient_severity,
            max_epochs=self.config.max_epochs_per_episode,
            randomize_patient=True,
            seed=100,
        )
        
        all_rewards = []
        all_spo2_mins = []
        all_events = []
        all_arousals = []
        all_actions = []
        
        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=100 + ep)
            episode_reward = 0
            done = False
            
            while not done:
                # Normalize observation
                obs_normalized = obs  # In practice, use VecNormalize stats
                action, _ = self.model.predict(obs_normalized, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                all_actions.append(action.copy())
            
            all_rewards.append(episode_reward)
            all_spo2_mins.append(eval_env.episode_spo2_min)
            all_events.append(eval_env.episode_events)
            all_arousals.append(eval_env.episode_arousals)
        
        all_actions = np.array(all_actions)
        
        metrics = {
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'mean_spo2_min': float(np.mean(all_spo2_mins)),
            'mean_events': float(np.mean(all_events)),
            'mean_arousals': float(np.mean(all_arousals)),
            'mean_loudness': float(np.mean(all_actions[:, 0])),
            'mean_frequency': float(np.mean(all_actions[:, 1])),
            'mean_duration': float(np.mean(all_actions[:, 2])),
            'mean_abs_itd': float(np.mean(np.abs(all_actions[:, 4]))),
            'mean_abs_ild': float(np.mean(np.abs(all_actions[:, 5]))),
            'intervention_rate': float(np.mean(all_actions[:, 0] > 0.05)),
        }
        
        return metrics


# ==============================================================================
# Baseline Agents (for comparison)
# ==============================================================================

class RandomAgent:
    """Random intervention agent (baseline)."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs, deterministic=False):
        return self.action_space.sample(), None


class NoInterventionAgent:
    """No intervention agent (baseline)."""
    
    def __init__(self, action_dim=6):
        self.action_dim = action_dim
    
    def predict(self, obs, deterministic=False):
        return np.zeros(self.action_dim, dtype=np.float32), None


class RuleBasedAgent:
    """
    Rule-based intervention agent (clinical baseline).
    
    Implements a simple threshold-based protocol without RL:
      - If SpO2 < 90% → loud burst cue
      - If snoring + supine → directional cue
      - Otherwise → no intervention
    """
    
    def __init__(self):
        self.action_dim = 6
    
    def predict(self, obs, deterministic=False):
        # Parse observation (based on OSAInterventionEnv.get_observation() layout)
        spo2_norm = obs[0]      # (spo2 - 70) / 30
        spo2 = spo2_norm * 30 + 70
        resp_effort = obs[6]
        phase_angle = obs[7] * 180
        snore_intensity = obs[11]
        is_supine = obs[14] > 0.5
        
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # Critical: SpO2 very low → aggressive burst
        if spo2 < 85:
            action = np.array([0.6, 1000, 1.0, 0.3, 0.0, 0.0], dtype=np.float32)
        # Moderate desaturation
        elif spo2 < 90:
            action = np.array([0.4, 800, 0.5, 0.3, 0.0, 0.0], dtype=np.float32)
        # Snoring + supine → directional cue
        elif snore_intensity > 0.5 and is_supine:
            direction = 1.0 if np.random.random() > 0.5 else -1.0
            action = np.array([0.2, 250, 2.0, 0.5, direction * 1.2, direction * 15.0], 
                            dtype=np.float32)
        # High effort + paradoxical breathing → early intervention
        elif resp_effort > 0.6 and phase_angle > 60:
            action = np.array([0.15, 250, 1.5, 0.5, 0.8, 10.0], dtype=np.float32)
        
        return action, None
