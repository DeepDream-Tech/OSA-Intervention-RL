"""
OSA Physiology Simulation Environment
========================================

A Gymnasium-compatible environment that simulates OSA physiology for training
the acoustic intervention RL agent.

Physiological model based on published OSA dynamics:
  - Airway compliance model (collapsibility as a function of position & sleep stage)
  - Respiratory drive oscillator (generates breathing patterns)
  - SpO2 dynamics (O2 stores, hemoglobin dissociation, circulation delay)
  - Arousal model (response to acoustic stimulation and hypoxemia)
  - Sleep stage cycling (N1→N2→N3→REM, with OSA-related fragmentation)

State transitions:
  Normal breathing ← →  Partial obstruction → Complete obstruction → Arousal
       ↑                                                               |
       └───────────────────────────────────────────────────────────────┘

References:
  - Acoustic bubble control MDP design (arxiv:2312.05674)
  - HealthGym medical RL environments (arxiv:2203.06369)
  - Portiloop closed-loop stimulation (arxiv:2107.13473)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum


# ==============================================================================
# Physiological Constants
# ==============================================================================

class SleepStage(IntEnum):
    WAKE = 0
    N1 = 1
    N2 = 2
    N3 = 3
    REM = 4

class AirwayState(IntEnum):
    OPEN = 0
    PARTIAL_OBSTRUCTION = 1
    COMPLETE_OBSTRUCTION = 2

class InterventionLevel(IntEnum):
    NONE = 0
    DIRECTIONAL_CUE = 1     # 方向性提示 (ITD/ILD spatial audio)
    SHORT_BURST_CUE = 2     # 短促声音刺激


@dataclass
class PatientProfile:
    """
    Individual patient characteristics affecting OSA dynamics.
    
    These parameters create inter-patient variability for robust RL training.
    Values sampled from distributions based on clinical literature.
    """
    # Airway collapsibility (Pcrit) - higher = more collapsible = more OSA-prone
    pcrit: float = -2.0             # cmH2O, normal=-13, severe OSA=+2
    
    # Position sensitivity (how much supine worsens airway)
    position_sensitivity: float = 1.5  # Multiplier: 1.0-3.0
    
    # Arousal threshold (how much stimulation needed to trigger arousal)
    arousal_threshold: float = 0.5  # Lower = easier to wake (light sleeper)
    
    # SpO2 reserve (baseline oxygen stores)
    spo2_baseline: float = 96.0    # % (normal 94-99)
    
    # Respiratory drive strength
    resp_drive: float = 1.0        # Relative drive (0.5-2.0)
    
    # Sleep stage distribution preferences
    rem_fraction: float = 0.20     # Fraction of sleep in REM
    deep_fraction: float = 0.15    # Fraction in N3
    
    # BMI category (affects airway collapsibility)
    bmi_category: int = 1          # 0=normal, 1=overweight, 2=obese
    
    # AHI severity target for simulation
    target_ahi: float = 15.0       # Events/hour (5=mild, 15=moderate, 30=severe)

    @classmethod
    def random_patient(cls, rng: np.random.Generator, severity: str = 'moderate'):
        """Generate a random patient profile."""
        severity_params = {
            'mild':     {'pcrit_mean': -5.0, 'pcrit_std': 2.0, 'ahi_mean': 10.0},
            'moderate': {'pcrit_mean': -1.0, 'pcrit_std': 2.0, 'ahi_mean': 20.0},
            'severe':   {'pcrit_mean': 2.0,  'pcrit_std': 1.5, 'ahi_mean': 40.0},
        }
        params = severity_params.get(severity, severity_params['moderate'])
        
        return cls(
            pcrit=float(rng.normal(params['pcrit_mean'], params['pcrit_std'])),
            position_sensitivity=float(rng.uniform(1.0, 3.0)),
            arousal_threshold=float(rng.uniform(0.2, 0.8)),
            spo2_baseline=float(rng.uniform(93.0, 98.0)),
            resp_drive=float(rng.uniform(0.5, 1.5)),
            rem_fraction=float(rng.uniform(0.15, 0.25)),
            deep_fraction=float(rng.uniform(0.10, 0.20)),
            bmi_category=int(rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3])),
            target_ahi=float(rng.normal(params['ahi_mean'], 5.0)),
        )


# ==============================================================================
# Physiology Simulator
# ==============================================================================

class PhysiologySimulator:
    """
    Simulates the core physiological dynamics of OSA.
    
    Models the closed-loop interaction between:
      Airway mechanics ↔ Respiratory effort ↔ Blood gases ↔ Neural drive
    
    Each step represents one 30-second epoch.
    """
    
    def __init__(self, patient: PatientProfile, rng: np.random.Generator):
        self.patient = patient
        self.rng = rng
        self.reset()
    
    def reset(self):
        """Reset physiological state to beginning of sleep."""
        p = self.patient
        
        self.state = {
            # Airway
            'airway_state': AirwayState.OPEN,
            'airway_patency': 1.0,  # 1.0 = fully open, 0.0 = fully closed
            'obstruction_duration': 0,  # Epochs in current obstruction
            
            # Respiratory
            'resp_rate': 15.0,       # Breaths per minute
            'resp_amplitude': 1.0,   # Normalized tidal volume
            'resp_effort': 0.3,      # Inspiratory effort (increases with obstruction)
            'phase_angle': 5.0,      # Thoracoabdominal phase angle (degrees)
            
            # Blood gases
            'spo2': p.spo2_baseline,
            'spo2_trend': 0.0,       # Rate of change per epoch
            'paco2': 40.0,           # Arterial CO2 (mmHg), rises during apnea
            
            # Snoring
            'snore_intensity': 0.0,  # 0-1 scale
            'snore_f0': 0.0,         # Hz
            'snore_pattern': 0,      # 0=none, 1=sustained, 2=bout
            
            # Sleep/Arousal
            'sleep_stage': SleepStage.N1,
            'sleep_depth': 0.3,      # 0=wake, 1=deepest N3
            'arousal_drive': 0.0,    # Cumulative arousal pressure
            'is_aroused': False,
            
            # Position
            'is_supine': True,       # Start in supine (worst case for training)
            'position_angle': 0.0,   # 0=supine, 90=lateral, 180=prone
            
            # Intervention tracking
            'intervention_level': InterventionLevel.NONE,
            'intervention_history': [],
            'epochs_since_intervention': 0,
            'total_arousals': 0,
            'total_events': 0,
        }
        
        self.epoch = 0
        self._init_sleep_cycle()
    
    def _init_sleep_cycle(self):
        """Initialize sleep cycle progression."""
        # Typical 90-minute NREM-REM cycle
        self.cycle_length = int(self.rng.normal(90, 10) * 2)  # In 30s epochs
        self.cycle_position = 0
    
    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Advance physiology by one epoch (30 seconds).
        
        Args:
            action: np.ndarray of shape (6,)
                [loudness, frequency, duration, timing, ITD, ILD]
                loudness ∈ [0, 1]: relative volume
                frequency ∈ [20, 4000]: Hz
                duration ∈ [0.1, 10]: seconds
                timing ∈ [0, 1]: within-epoch timing (0=start, 1=end)
                ITD ∈ [-1.5, 1.5]: interaural time difference (ms)
                ILD ∈ [-20, 20]: interaural level difference (dB)
        
        Returns:
            Updated physiological state dictionary
        """
        self.epoch += 1
        s = self.state
        p = self.patient
        
        # 1. Update sleep stage
        self._update_sleep_stage()
        
        # 2. Determine intervention type from action
        loudness = float(action[0])
        itd = float(action[4])
        ild = float(action[5])
        
        has_intervention = loudness > 0.05  # Minimal threshold
        is_directional = abs(itd) > 0.3 or abs(ild) > 5.0  # Spatial cue
        
        if has_intervention:
            if is_directional:
                s['intervention_level'] = InterventionLevel.DIRECTIONAL_CUE
            else:
                s['intervention_level'] = InterventionLevel.SHORT_BURST_CUE
            s['epochs_since_intervention'] = 0
        else:
            s['intervention_level'] = InterventionLevel.NONE
            s['epochs_since_intervention'] += 1
        
        # 3. Process acoustic intervention effect on position
        position_change = self._compute_position_response(action)
        self._update_position(position_change)
        
        # 4. Update airway mechanics
        self._update_airway()
        
        # 5. Update respiratory dynamics
        self._update_respiration()
        
        # 6. Update blood gases (SpO2, PaCO2)
        self._update_blood_gases()
        
        # 7. Update snoring characteristics
        self._update_snoring()
        
        # 8. Check for arousal (acoustic + hypoxemia driven)
        arousal_stimulus = self._compute_arousal_stimulus(action)
        self._update_arousal(arousal_stimulus)
        
        # 9. If aroused, resolve the obstruction
        if s['is_aroused']:
            self._handle_arousal()
        
        return self.state.copy()
    
    def _update_sleep_stage(self):
        """Progress through sleep cycle with realistic staging."""
        s = self.state
        self.cycle_position += 1
        
        # Simplified sleep cycle: N1→N2→N3→N2→REM→N2...
        cycle_frac = (self.cycle_position % self.cycle_length) / self.cycle_length
        
        noise = self.rng.normal(0, 0.05)
        
        if cycle_frac < 0.05:
            s['sleep_stage'] = SleepStage.N1
            s['sleep_depth'] = 0.2 + noise
        elif cycle_frac < 0.35:
            s['sleep_stage'] = SleepStage.N2
            s['sleep_depth'] = 0.5 + noise
        elif cycle_frac < 0.50:
            s['sleep_stage'] = SleepStage.N3
            s['sleep_depth'] = 0.8 + noise
        elif cycle_frac < 0.65:
            s['sleep_stage'] = SleepStage.N2
            s['sleep_depth'] = 0.5 + noise
        elif cycle_frac < 0.85:
            s['sleep_stage'] = SleepStage.REM
            s['sleep_depth'] = 0.4 + noise
            # REM: increased airway collapsibility
        else:
            s['sleep_stage'] = SleepStage.N1
            s['sleep_depth'] = 0.2 + noise
        
        s['sleep_depth'] = float(np.clip(s['sleep_depth'], 0.0, 1.0))
    
    def _compute_position_response(self, action: np.ndarray) -> float:
        """
        Compute position change from directional acoustic cue.
        
        Binaural cues (ITD/ILD) create spatial perception that can
        subconsciously encourage position change during light sleep.
        
        Returns:
            position_change: degrees of position shift
        """
        s = self.state
        loudness, freq, duration, timing, itd, ild = action
        
        if loudness < 0.05:
            return 0.0
        
        # Directional strength from binaural parameters
        directional_strength = np.sqrt(
            (itd / 1.5) ** 2 + (ild / 20.0) ** 2
        )
        directional_strength = np.clip(directional_strength, 0, 1)
        
        # Response probability depends on sleep depth
        # Lighter sleep → more responsive to subtle cues
        response_probability = (1.0 - s['sleep_depth'] * 0.8) * directional_strength * loudness
        
        # Stochastic response
        if self.rng.random() < response_probability:
            # Direction of movement based on ITD/ILD direction
            direction = np.sign(itd + ild / 20.0)
            magnitude = 10.0 + 20.0 * directional_strength  # 10-30 degrees
            return float(direction * magnitude)
        
        return 0.0
    
    def _update_position(self, position_change: float):
        """Update body position based on movement."""
        s = self.state
        
        # Natural random position shifts (small)
        natural_shift = self.rng.normal(0, 2.0)
        
        s['position_angle'] += position_change + natural_shift
        s['position_angle'] = float(np.clip(s['position_angle'], -180, 180))
        
        # Determine if supine (within ±30° of 0)
        s['is_supine'] = abs(s['position_angle']) < 30.0
    
    def _update_airway(self):
        """
        Update airway patency based on position, sleep stage, and patient anatomy.
        
        Pcrit model: airway collapses when tissue pressure exceeds Pcrit
          - Supine increases tissue pressure (gravity effect)
          - Deeper sleep reduces muscle tone
          - REM further reduces tone
        """
        s = self.state
        p = self.patient
        
        # Base collapsibility from Pcrit
        # More negative Pcrit = more stable airway
        collapse_tendency = (p.pcrit + 5.0) / 10.0  # Normalize to ~[0, 1]
        
        # Position effect: supine dramatically increases collapse risk
        position_factor = 1.0
        if s['is_supine']:
            position_factor = p.position_sensitivity
        
        # Sleep depth effect: deeper sleep = more muscle relaxation
        sleep_factor = 1.0 + s['sleep_depth'] * 0.5
        
        # REM effect: additional muscle atonia
        rem_factor = 1.3 if s['sleep_stage'] == SleepStage.REM else 1.0
        
        # BMI effect
        bmi_factor = 1.0 + p.bmi_category * 0.2
        
        # Total collapse probability this epoch
        collapse_prob = np.clip(
            collapse_tendency * position_factor * sleep_factor * rem_factor * bmi_factor * 0.15,
            0.0, 0.8
        )
        
        # Stochastic state transitions
        noise = self.rng.random()
        
        if s['airway_state'] == AirwayState.OPEN:
            if noise < collapse_prob:
                s['airway_state'] = AirwayState.PARTIAL_OBSTRUCTION
                s['airway_patency'] = self.rng.uniform(0.3, 0.7)
            else:
                s['airway_patency'] = min(1.0, s['airway_patency'] + 0.1)
        
        elif s['airway_state'] == AirwayState.PARTIAL_OBSTRUCTION:
            if noise < collapse_prob * 0.5:  # Can worsen
                s['airway_state'] = AirwayState.COMPLETE_OBSTRUCTION
                s['airway_patency'] = 0.0
                s['obstruction_duration'] = 0
                s['total_events'] += 1
            elif noise > (1 - collapse_prob * 0.3):  # Can resolve spontaneously
                s['airway_state'] = AirwayState.OPEN
                s['airway_patency'] = 0.8
            else:
                s['airway_patency'] = np.clip(
                    s['airway_patency'] + self.rng.normal(-0.05, 0.05), 0.1, 0.7
                )
        
        elif s['airway_state'] == AirwayState.COMPLETE_OBSTRUCTION:
            s['obstruction_duration'] += 1
            s['airway_patency'] = 0.0
            # Only resolves through arousal (handled in _handle_arousal)
    
    def _update_respiration(self):
        """Update respiratory parameters based on airway state."""
        s = self.state
        p = self.patient
        
        if s['airway_state'] == AirwayState.OPEN:
            # Normal breathing
            target_rate = 15.0 * p.resp_drive
            target_amplitude = 1.0
            target_effort = 0.3
            target_phase = 5.0 + self.rng.normal(0, 3)
        
        elif s['airway_state'] == AirwayState.PARTIAL_OBSTRUCTION:
            # Obstructive hypopnea: increased effort, reduced flow
            target_rate = 18.0 * p.resp_drive
            target_amplitude = s['airway_patency']  # Reduced by obstruction
            target_effort = 0.6 + (1.0 - s['airway_patency']) * 0.4
            # Phase angle increases with obstruction (paradoxical breathing)
            target_phase = 30.0 + (1.0 - s['airway_patency']) * 120.0
        
        elif s['airway_state'] == AirwayState.COMPLETE_OBSTRUCTION:
            # Obstructive apnea: high effort, no flow
            target_rate = 20.0 * p.resp_drive  # Respiratory drive increases
            target_amplitude = 0.05  # Near-zero flow
            target_effort = 1.0  # Maximum effort against closed airway
            target_phase = 170.0 + self.rng.normal(0, 5)  # Near-complete paradox
        
        # Smooth transitions
        alpha = 0.3
        s['resp_rate'] = float(alpha * target_rate + (1 - alpha) * s['resp_rate'])
        s['resp_amplitude'] = float(alpha * target_amplitude + (1 - alpha) * s['resp_amplitude'])
        s['resp_effort'] = float(alpha * target_effort + (1 - alpha) * s['resp_effort'])
        s['phase_angle'] = float(alpha * target_phase + (1 - alpha) * s['phase_angle'])
        
        # Add physiological noise
        s['resp_rate'] += float(self.rng.normal(0, 0.5))
        s['resp_amplitude'] = float(np.clip(s['resp_amplitude'] + self.rng.normal(0, 0.02), 0, 2))
    
    def _update_blood_gases(self):
        """
        Update SpO2 and PaCO2 based on ventilation status.
        
        SpO2 dynamics:
          - O2 stores provide ~15-30s buffer before desaturation begins
          - Desaturation rate depends on metabolic rate and O2 reserves
          - Recovery is faster than desaturation (hemoglobin loading curve)
        """
        s = self.state
        p = self.patient
        
        # Effective ventilation (0-1)
        ventilation = s['resp_amplitude'] * s['airway_patency']
        
        # SpO2 dynamics
        if ventilation > 0.7:
            # Good ventilation → recovery toward baseline
            recovery_rate = 0.5  # % per epoch
            target_spo2 = p.spo2_baseline
            spo2_change = min(recovery_rate, target_spo2 - s['spo2']) * 0.3
        elif ventilation > 0.3:
            # Partial obstruction → slow decline
            spo2_change = -0.5 * (1.0 - ventilation)
        else:
            # Severe obstruction / apnea → rapid desaturation
            # Rate depends on time in obstruction (O2 stores deplete)
            desat_rate = 0.8 + 0.3 * min(s['obstruction_duration'], 5)
            spo2_change = -desat_rate
        
        # Apply with noise
        spo2_change += self.rng.normal(0, 0.2)
        s['spo2'] = float(np.clip(s['spo2'] + spo2_change, 60.0, 100.0))
        s['spo2_trend'] = float(spo2_change)
        
        # PaCO2 dynamics (rises during hypoventilation)
        if ventilation > 0.7:
            paco2_change = -0.5  # Washout
        else:
            paco2_change = 0.5 * (1.0 - ventilation)
        
        s['paco2'] = float(np.clip(s['paco2'] + paco2_change, 30.0, 70.0))
    
    def _update_snoring(self):
        """Update snoring characteristics based on airway state."""
        s = self.state
        
        if s['airway_state'] == AirwayState.OPEN:
            s['snore_intensity'] = max(0, s['snore_intensity'] - 0.2)
            s['snore_f0'] = 0.0
            s['snore_pattern'] = 0
        
        elif s['airway_state'] == AirwayState.PARTIAL_OBSTRUCTION:
            # Snoring occurs with partial obstruction (vibrating tissue)
            obstruction_degree = 1.0 - s['airway_patency']
            s['snore_intensity'] = float(0.3 + 0.7 * obstruction_degree + 
                                        self.rng.normal(0, 0.05))
            s['snore_f0'] = float(100.0 + 200.0 * obstruction_degree +
                                 self.rng.normal(0, 20))
            s['snore_pattern'] = 1 if s['snore_intensity'] > 0.5 else 2
        
        elif s['airway_state'] == AirwayState.COMPLETE_OBSTRUCTION:
            # Silent during complete obstruction (no airflow)
            s['snore_intensity'] = 0.0
            s['snore_f0'] = 0.0
            s['snore_pattern'] = 0
        
        s['snore_intensity'] = float(np.clip(s['snore_intensity'], 0, 1))
    
    def _compute_arousal_stimulus(self, action: np.ndarray) -> float:
        """
        Compute total arousal stimulus from acoustic intervention and hypoxemia.
        
        Arousal threshold varies by sleep stage:
          N1: lowest threshold (easily aroused)
          N2: moderate
          N3: highest (hardest to arouse)
          REM: variable
        """
        s = self.state
        loudness, freq, duration, timing, itd, ild = action
        
        # Acoustic arousal stimulus
        # Higher loudness, certain frequencies, longer duration → more arousing
        freq_weight = 1.0  # Weight certain frequencies
        if 500 < freq < 2000:
            freq_weight = 1.3  # Speech frequencies most alerting
        
        acoustic_stimulus = loudness * duration / 10.0 * freq_weight
        
        # Hypoxemia arousal stimulus (body's defense mechanism)
        hypoxemia_stimulus = max(0, (90.0 - s['spo2']) / 20.0)
        
        # CO2-driven arousal
        co2_stimulus = max(0, (s['paco2'] - 45.0) / 15.0)
        
        return float(acoustic_stimulus + hypoxemia_stimulus * 0.5 + co2_stimulus * 0.3)
    
    def _update_arousal(self, stimulus: float):
        """Update arousal state based on cumulative stimulus."""
        s = self.state
        p = self.patient
        
        # Sleep stage modulates arousal threshold
        stage_threshold = {
            SleepStage.WAKE: 0.01,
            SleepStage.N1: 0.2,
            SleepStage.N2: 0.4,
            SleepStage.N3: 0.7,
            SleepStage.REM: 0.35,
        }
        
        threshold = stage_threshold[SleepStage(s['sleep_stage'])] * (1.0 + p.arousal_threshold)
        
        # Accumulate arousal drive
        s['arousal_drive'] += stimulus
        s['arousal_drive'] *= 0.8  # Decay over time
        
        # Check if arousal threshold exceeded
        s['is_aroused'] = s['arousal_drive'] > threshold
    
    def _handle_arousal(self):
        """Handle cortical arousal: resolve obstruction, reset state."""
        s = self.state
        
        s['airway_state'] = AirwayState.OPEN
        s['airway_patency'] = 1.0
        s['obstruction_duration'] = 0
        s['resp_effort'] = 0.4
        s['phase_angle'] = 10.0
        s['arousal_drive'] = 0.0
        s['is_aroused'] = False
        s['total_arousals'] += 1
        
        # Brief sleep stage regression after arousal
        s['sleep_stage'] = SleepStage.N1
        s['sleep_depth'] = 0.1
    
    def get_observation(self) -> np.ndarray:
        """
        Convert physiological state to observation vector for RL agent.
        
        Returns:
            obs: np.ndarray of shape (obs_dim,) — normalized observation
        """
        s = self.state
        
        obs = np.array([
            # SpO2 features (4)
            (s['spo2'] - 70.0) / 30.0,             # Normalized SpO2
            np.clip(s['spo2_trend'] / 3.0, -1, 1), # SpO2 trend
            max(0, (90.0 - s['spo2']) / 20.0),     # Hypoxemia risk
            (s['paco2'] - 35.0) / 25.0,             # Normalized PaCO2
            
            # Respiratory features (4)
            s['resp_rate'] / 30.0,                  # Normalized resp rate
            s['resp_amplitude'],                    # Tidal volume proxy
            s['resp_effort'],                       # Respiratory effort
            s['phase_angle'] / 180.0,               # Normalized phase angle
            
            # Airway features (3)
            s['airway_patency'],                    # Airway openness
            float(s['airway_state']) / 2.0,         # Airway state
            s['obstruction_duration'] / 10.0,       # Obstruction duration
            
            # Snoring features (3)
            s['snore_intensity'],                   # Snore energy
            s['snore_f0'] / 500.0,                  # Normalized F0
            s['snore_pattern'] / 2.0,               # Pattern type
            
            # Position features (2)
            float(s['is_supine']),                  # Supine indicator
            s['position_angle'] / 180.0,            # Normalized position
            
            # Sleep features (2)
            float(s['sleep_stage']) / 4.0,          # Sleep stage
            s['sleep_depth'],                       # Sleep depth
            
            # Intervention history (3)
            float(s['intervention_level']) / 2.0,   # Current level
            min(s['epochs_since_intervention'], 10) / 10.0,  # Time since last
            float(s['is_aroused']),                  # Arousal flag
        ], dtype=np.float32)
        
        return obs


# ==============================================================================
# Gymnasium Environment
# ==============================================================================

class OSAInterventionEnv(gym.Env):
    """
    Gymnasium environment for training OSA acoustic intervention RL agents.
    
    Observation space (21 dimensions):
        Physiological state vector from PhysiologySimulator
    
    Action space (6 dimensions, continuous):
        [loudness, frequency, duration, timing, ITD, ILD]
        
        loudness ∈ [0, 1]:     Relative sound level (0=silent, 1=max safe level)
        frequency ∈ [20, 4000]: Tone frequency in Hz
        duration ∈ [0.1, 10]:  Stimulus duration in seconds
        timing ∈ [0, 1]:       When in epoch to deliver (0=start, 1=end)
        ITD ∈ [-1.5, 1.5]:    Interaural time difference in ms (spatial cue)
        ILD ∈ [-20, 20]:      Interaural level difference in dB (spatial cue)
    
    Reward design (multi-objective):
        R = w_spo2 * R_spo2 + w_airway * R_airway + w_sleep * R_sleep + w_action * R_action
        
        R_spo2:   Reward for maintaining SpO2 > 90% (primary health objective)
        R_airway: Reward for preventing/resolving obstruction
        R_sleep:  Penalty for causing unnecessary arousals (sleep quality)
        R_action: Penalty for excessive intervention (minimal effective dose)
    
    Episode:
        - 2 hours of simulated sleep (240 epochs at 30s each)
        - Terminates early if SpO2 drops below critical threshold (safety)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    OBS_DIM = 21    # Observation vector dimension
    ACTION_DIM = 6  # Action vector dimension
    
    def __init__(
        self,
        patient: Optional[PatientProfile] = None,
        severity: str = 'moderate',
        max_epochs: int = 240,         # 2 hours
        render_mode: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        randomize_patient: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.severity = severity
        self.max_epochs = max_epochs
        self.render_mode = render_mode
        self.randomize_patient = randomize_patient
        
        # Reward weights (from acoustic bubble paper: shaped distance reward)
        self.reward_weights = reward_weights or {
            'spo2': 0.40,      # Primary: keep SpO2 healthy
            'airway': 0.25,    # Secondary: prevent obstruction
            'sleep': 0.20,     # Tertiary: minimize sleep disruption
            'action': 0.10,    # Quaternary: minimal intervention
            'event': 0.05,     # Bonus: prevent complete events
        }
        
        # Action space: 6D continuous
        self.action_space = spaces.Box(
            low=np.array([0.0, 20.0, 0.1, 0.0, -1.5, -20.0], dtype=np.float32),
            high=np.array([1.0, 4000.0, 10.0, 1.0, 1.5, 20.0], dtype=np.float32),
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        
        # Initialize
        self._rng = np.random.default_rng(seed)
        self.patient = patient or PatientProfile.random_patient(self._rng, severity)
        self.sim = PhysiologySimulator(self.patient, self._rng)
        self.epoch = 0
        
        # Tracking for metrics
        self.episode_rewards = []
        self.episode_spo2_min = 100.0
        self.episode_events = 0
        self.episode_arousals = 0
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to start of new sleep session."""
        super().reset(seed=seed)
        
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Optionally randomize patient for each episode (diversity)
        if self.randomize_patient:
            self.patient = PatientProfile.random_patient(self._rng, self.severity)
        
        self.sim = PhysiologySimulator(self.patient, self._rng)
        self.sim.reset()
        self.epoch = 0
        
        # Reset tracking
        self.episode_rewards = []
        self.episode_spo2_min = 100.0
        self.episode_events = 0
        self.episode_arousals = 0
        
        obs = self.sim.get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one epoch with the given acoustic intervention action.
        
        Implements the hierarchical intervention protocol:
        1. If supine → prefer directional cue (high ITD/ILD) to encourage position change
        2. If directional cue ineffective → escalate to short burst cue
        3. Always minimize intervention intensity (sleep quality preservation)
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Record pre-step state
        pre_spo2 = self.sim.state['spo2']
        pre_airway = self.sim.state['airway_state']
        pre_events = self.sim.state['total_events']
        
        # Advance physiology
        self.sim.step(action)
        self.epoch += 1
        
        # Get observation
        obs = self.sim.get_observation()
        
        # Compute reward
        reward = self._compute_reward(action, pre_spo2, pre_airway, pre_events)
        self.episode_rewards.append(reward)
        
        # Track metrics
        self.episode_spo2_min = min(self.episode_spo2_min, self.sim.state['spo2'])
        self.episode_events = self.sim.state['total_events']
        self.episode_arousals = self.sim.state['total_arousals']
        
        # Termination conditions
        terminated = self.sim.state['spo2'] < 70.0  # Critical safety threshold
        truncated = self.epoch >= self.max_epochs
        
        info = self._get_info()
        
        return obs, float(reward), bool(terminated), bool(truncated), info
    
    def _compute_reward(
        self, 
        action: np.ndarray,
        pre_spo2: float,
        pre_airway: int,
        pre_events: int,
    ) -> float:
        """
        Multi-component reward function.
        
        Design philosophy: r = 1 - (d/d_max)^k with k=0.2 (sub-linear)
        from acoustic bubble control paper (arxiv:2312.05674)
        """
        s = self.sim.state
        w = self.reward_weights
        
        # 1. SpO2 reward: shaped distance from target (95%)
        spo2_distance = max(0, 95.0 - s['spo2']) / 25.0  # Normalized [0, 1]
        r_spo2 = 1.0 - spo2_distance ** 0.2  # Sub-linear penalty
        
        # Bonus for SpO2 improvement
        spo2_improvement = s['spo2'] - pre_spo2
        r_spo2 += 0.1 * np.clip(spo2_improvement, -1, 1)
        
        # 2. Airway reward: maintaining open airway
        r_airway = s['airway_patency']  # 1.0 = open, 0.0 = closed
        
        # Bonus for resolving partial obstruction without full arousal
        if pre_airway > 0 and s['airway_state'] == AirwayState.OPEN and not s['is_aroused']:
            r_airway += 0.5  # Successfully resolved without waking patient
        
        # 3. Sleep quality reward: penalize arousals
        r_sleep = 0.0
        if s['is_aroused']:
            r_sleep = -1.0  # Strong penalty for causing arousal
        else:
            # Reward for maintaining sleep depth
            r_sleep = s['sleep_depth'] * 0.5
        
        # 4. Action economy: penalize unnecessary/excessive intervention
        loudness = action[0]
        r_action = 0.0
        
        # Normalize each action dimension to [0,1] before computing penalty
        action_ranges = self.action_space.high - self.action_space.low
        action_normalized = (action - self.action_space.low) / (action_ranges + 1e-8)
        
        if s['airway_state'] == AirwayState.OPEN and loudness > 0.1:
            # Penalize intervention when airway is open (unnecessary)
            r_action = -loudness * 0.5
        else:
            # Small penalty for intervention intensity (minimal effective dose)
            r_action = -0.05 * np.mean(action_normalized ** 2)
        
        # 5. Event prevention bonus
        r_event = 0.0
        new_events = s['total_events'] - pre_events
        if new_events > 0:
            r_event = -2.0  # Strong penalty for allowing complete obstruction
        elif s['airway_state'] == AirwayState.PARTIAL_OBSTRUCTION:
            # Partial obstruction - urgent but not yet an event
            r_event = -0.3
        
        # Weighted combination
        reward = (
            w['spo2'] * r_spo2 +
            w['airway'] * r_airway +
            w['sleep'] * r_sleep +
            w['action'] * r_action +
            w['event'] * r_event
        )
        
        return float(reward)
    
    def _get_info(self) -> Dict[str, Any]:
        """Return information dictionary for logging."""
        s = self.sim.state
        return {
            'spo2': s['spo2'],
            'airway_state': int(s['airway_state']),
            'sleep_stage': int(s['sleep_stage']),
            'is_supine': s['is_supine'],
            'resp_effort': s['resp_effort'],
            'phase_angle': s['phase_angle'],
            'snore_intensity': s['snore_intensity'],
            'intervention_level': int(s['intervention_level']),
            'total_events': s['total_events'],
            'total_arousals': s['total_arousals'],
            'episode_spo2_min': self.episode_spo2_min,
            'epoch': self.epoch,
        }
    
    def render(self):
        """Render environment state."""
        if self.render_mode == 'ansi':
            return self._render_ansi()
        return None
    
    def _render_ansi(self) -> str:
        """Text-based rendering of current state."""
        s = self.sim.state
        stage_names = {0: 'WAKE', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
        airway_names = {0: '✓OPEN', 1: '⚠PARTIAL', 2: '✗BLOCKED'}
        
        lines = [
            f"═══ Epoch {self.epoch}/{self.max_epochs} ═══",
            f"SpO2: {s['spo2']:.1f}% {'⚠' if s['spo2'] < 90 else '✓'}  "
            f"Trend: {s['spo2_trend']:+.2f}%/epoch",
            f"Airway: {airway_names.get(s['airway_state'], '?')}  "
            f"Patency: {s['airway_patency']:.0%}",
            f"Resp: {s['resp_rate']:.0f}bpm  Effort: {s['resp_effort']:.0%}  "
            f"Phase: {s['phase_angle']:.0f}°",
            f"Sleep: {stage_names.get(s['sleep_stage'], '?')}  "
            f"Depth: {s['sleep_depth']:.0%}  "
            f"Position: {'SUPINE⚠' if s['is_supine'] else 'LATERAL✓'}",
            f"Snoring: {s['snore_intensity']:.0%}  Events: {s['total_events']}  "
            f"Arousals: {s['total_arousals']}",
        ]
        
        return '\n'.join(lines)
