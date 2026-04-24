"""
Hierarchical Intervention Protocol
=====================================

Implements the clinical decision logic for OSA acoustic intervention:

Protocol Flow:
  1. MONITOR: Continuously assess OSA risk from multimodal features
  2. DETECT: When risk exceeds threshold, determine intervention strategy
  3. ASSESS POSITION: Check if patient is supine (仰卧位)
  4. DIRECTIONAL CUE: If supine, deliver spatial audio cue to encourage
     lateral position change (using ITD/ILD binaural parameters)
  5. EVALUATE: Wait for response window (30-90 seconds)
  6. ESCALATE: If no improvement, deliver short burst acoustic cue
  7. COOLDOWN: After intervention, enter cooldown to prevent habituation

Clinical rationale:
  - Positional therapy is first-line because it addresses root cause
    (supine = 2-3x OSA risk due to gravitational airway collapse)
  - Short burst cue is second-line because it causes more sleep disruption
    but is more reliable at resolving acute obstruction
  - Both are preferable to CPAP for mild-moderate OSA with good compliance
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from enum import IntEnum


class ProtocolState(IntEnum):
    """State machine states for intervention protocol."""
    MONITORING = 0           # Passive observation
    RISK_DETECTED = 1        # OSA risk identified
    DIRECTIONAL_CUE = 2     # Delivering spatial position cue
    WAITING_RESPONSE = 3     # Waiting for position change response
    ESCALATION = 4           # Escalating to short burst cue
    BURST_CUE = 5           # Delivering short burst acoustic cue
    COOLDOWN = 6            # Post-intervention cooldown
    RESOLVED = 7            # Event resolved, returning to monitoring


@dataclass
class InterventionConfig:
    """Configuration for the intervention protocol."""
    
    # Risk thresholds
    risk_threshold_low: float = 0.3      # Start monitoring closely
    risk_threshold_high: float = 0.6     # Trigger intervention
    risk_threshold_urgent: float = 0.85  # Skip directional, go to burst
    
    # Directional cue parameters
    dir_cue_loudness_start: float = 0.15    # Starting loudness (gentle)
    dir_cue_loudness_max: float = 0.45      # Maximum loudness for directional
    dir_cue_loudness_step: float = 0.05     # Increment per retry
    dir_cue_frequency: float = 250.0        # Hz (low frequency, less alerting)
    dir_cue_duration: float = 2.0           # Seconds
    dir_cue_itd: float = 1.2               # ms (strong lateralization)
    dir_cue_ild: float = 15.0              # dB (strong level difference)
    
    # Short burst cue parameters
    burst_loudness_start: float = 0.25      # Starting loudness
    burst_loudness_max: float = 0.70        # Maximum (safety limit)
    burst_loudness_step: float = 0.08       # Increment per retry
    burst_frequency: float = 1000.0         # Hz (speech frequency, alerting)
    burst_duration: float = 0.5             # Seconds (short burst)
    burst_itd: float = 0.0                 # No spatial bias
    burst_ild: float = 0.0                 # No spatial bias
    
    # Timing parameters
    response_window: int = 3               # Epochs to wait for response (90s)
    max_directional_attempts: int = 4      # Max directional cues before escalation
    max_burst_attempts: int = 3            # Max burst cues per episode
    cooldown_epochs: int = 5               # Epochs after successful intervention
    
    # Personalization parameters
    habituation_rate: float = 0.1          # How fast patient habituates
    sensitization_rate: float = 0.05       # How fast effective dose decreases


@dataclass
class InterventionRecord:
    """Record of a single intervention event."""
    epoch: int
    protocol_state: ProtocolState
    action: np.ndarray
    risk_score: float
    spo2_before: float
    spo2_after: float = 0.0
    position_before: bool = False  # is_supine before
    position_after: bool = False   # is_supine after
    was_effective: bool = False
    caused_arousal: bool = False


class InterventionProtocol:
    """
    Hierarchical intervention state machine.
    
    Manages the clinical logic layer above the RL agent:
    - When to intervene (risk threshold monitoring)
    - What type of intervention (directional vs burst)
    - Escalation and de-escalation logic
    - Cooldown and habituation management
    
    The RL agent controls the fine-grained parameters within each
    intervention type, while this protocol controls the macro strategy.
    """
    
    def __init__(self, config: InterventionConfig = None):
        self.config = config or InterventionConfig()
        self.reset()
    
    def reset(self):
        """Reset protocol state for new sleep session."""
        self.state = ProtocolState.MONITORING
        self.epoch = 0
        
        # Intervention tracking
        self.directional_attempts = 0
        self.burst_attempts = 0
        self.current_loudness = 0.0
        self.waiting_since = 0
        
        # Personalization state
        self.effective_loudness_dir = self.config.dir_cue_loudness_start
        self.effective_loudness_burst = self.config.burst_loudness_start
        self.habituation_factor = 1.0  # Increases with repeated interventions
        
        # History
        self.records: List[InterventionRecord] = []
        self.cooldown_remaining = 0
    
    def decide(
        self,
        risk_score: float,
        is_supine: bool,
        spo2: float,
        airway_state: int,
        is_aroused: bool,
        rl_action: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Make intervention decision based on current state and risk.
        
        This is the main entry point called each epoch.
        
        Args:
            risk_score: OSA risk probability (0-1) from predictor
            is_supine: Whether patient is in supine position
            spo2: Current SpO2 value
            airway_state: 0=open, 1=partial, 2=complete obstruction
            is_aroused: Whether patient was aroused this epoch
            rl_action: Optional fine-tuned action from RL agent
        
        Returns:
            action: np.ndarray of shape (6,) - acoustic intervention parameters
            info: Dict with protocol state information
        """
        self.epoch += 1
        c = self.config
        
        # Initialize action (no intervention default)
        action = np.zeros(6, dtype=np.float32)
        
        # State machine transitions
        info = {'protocol_state': self.state.name, 'risk_score': risk_score}
        
        # Handle cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.state = ProtocolState.MONITORING
                self.directional_attempts = 0
                self.burst_attempts = 0
            info['cooldown_remaining'] = self.cooldown_remaining
            return action, info
        
        # Handle arousal (intervention succeeded in opening airway)
        if is_aroused and self.state in (ProtocolState.DIRECTIONAL_CUE, 
                                          ProtocolState.BURST_CUE,
                                          ProtocolState.WAITING_RESPONSE):
            self._record_outcome(effective=True, arousal=True)
            self.state = ProtocolState.COOLDOWN
            self.cooldown_remaining = c.cooldown_epochs
            return action, info
        
        # ===== STATE: MONITORING =====
        if self.state == ProtocolState.MONITORING:
            if risk_score >= c.risk_threshold_urgent:
                # Urgent: skip directional, go straight to burst
                self.state = ProtocolState.ESCALATION
            elif risk_score >= c.risk_threshold_high:
                self.state = ProtocolState.RISK_DETECTED
            # else: continue monitoring
        
        # ===== STATE: RISK_DETECTED =====
        if self.state == ProtocolState.RISK_DETECTED:
            if is_supine:
                # Step 1: Try directional cue for position change
                self.state = ProtocolState.DIRECTIONAL_CUE
            else:
                # Not supine but still at risk → go to burst cue
                self.state = ProtocolState.ESCALATION
        
        # ===== STATE: DIRECTIONAL_CUE =====
        if self.state == ProtocolState.DIRECTIONAL_CUE:
            self.directional_attempts += 1
            
            # Generate directional cue action
            loudness = min(
                self.effective_loudness_dir + 
                (self.directional_attempts - 1) * c.dir_cue_loudness_step,
                c.dir_cue_loudness_max
            ) * self.habituation_factor
            
            # Determine direction: encourage rolling to the side
            # Use patient's current position to compute optimal direction
            direction = 1.0 if np.random.random() > 0.5 else -1.0
            
            action = np.array([
                np.clip(loudness, 0, 1),
                c.dir_cue_frequency,
                c.dir_cue_duration,
                0.5,  # Mid-epoch timing
                direction * c.dir_cue_itd,
                direction * c.dir_cue_ild,
            ], dtype=np.float32)
            
            # If RL agent provides refined parameters, blend them
            if rl_action is not None:
                action = self._blend_actions(action, rl_action, blend_factor=0.6)
            
            # Record intervention
            self.records.append(InterventionRecord(
                epoch=self.epoch,
                protocol_state=self.state,
                action=action.copy(),
                risk_score=risk_score,
                spo2_before=spo2,
                position_before=is_supine,
            ))
            
            # Transition to waiting
            self.state = ProtocolState.WAITING_RESPONSE
            self.waiting_since = self.epoch
            
            info['intervention_type'] = 'directional_cue'
            info['attempt'] = self.directional_attempts
        
        # ===== STATE: WAITING_RESPONSE =====
        elif self.state == ProtocolState.WAITING_RESPONSE:
            wait_time = self.epoch - self.waiting_since
            
            if not is_supine:
                # Position change successful!
                self._record_outcome(effective=True, arousal=False)
                self._update_personalization(effective=True)
                self.state = ProtocolState.RESOLVED
                info['position_change_detected'] = True
            
            elif airway_state == 0 and risk_score < c.risk_threshold_low:
                # Risk resolved naturally
                self._record_outcome(effective=True, arousal=False)
                self.state = ProtocolState.RESOLVED
            
            elif wait_time >= c.response_window:
                # No response within window
                if self.directional_attempts < c.max_directional_attempts:
                    # Try another directional cue (louder)
                    self.state = ProtocolState.DIRECTIONAL_CUE
                else:
                    # Exhausted directional attempts → escalate
                    self._record_outcome(effective=False, arousal=False)
                    self.state = ProtocolState.ESCALATION
        
        # ===== STATE: ESCALATION =====
        if self.state == ProtocolState.ESCALATION:
            if self.burst_attempts < c.max_burst_attempts:
                self.state = ProtocolState.BURST_CUE
            else:
                # Exhausted all attempts, enter cooldown
                self.state = ProtocolState.COOLDOWN
                self.cooldown_remaining = c.cooldown_epochs
        
        # ===== STATE: BURST_CUE =====
        if self.state == ProtocolState.BURST_CUE:
            self.burst_attempts += 1
            
            loudness = min(
                self.effective_loudness_burst +
                (self.burst_attempts - 1) * c.burst_loudness_step,
                c.burst_loudness_max
            ) * self.habituation_factor
            
            action = np.array([
                np.clip(loudness, 0, 1),
                c.burst_frequency,
                c.burst_duration,
                0.3,  # Earlier in epoch (more response time)
                c.burst_itd,
                c.burst_ild,
            ], dtype=np.float32)
            
            if rl_action is not None:
                action = self._blend_actions(action, rl_action, blend_factor=0.4)
            
            self.records.append(InterventionRecord(
                epoch=self.epoch,
                protocol_state=self.state,
                action=action.copy(),
                risk_score=risk_score,
                spo2_before=spo2,
            ))
            
            self.state = ProtocolState.WAITING_RESPONSE
            self.waiting_since = self.epoch
            
            info['intervention_type'] = 'burst_cue'
            info['attempt'] = self.burst_attempts
        
        # ===== STATE: RESOLVED =====
        if self.state == ProtocolState.RESOLVED:
            self.state = ProtocolState.COOLDOWN
            self.cooldown_remaining = c.cooldown_epochs
        
        # Clip action to valid range
        action = np.clip(action, 
                        [0, 20, 0.1, 0, -1.5, -20],
                        [1, 4000, 10, 1, 1.5, 20]).astype(np.float32)
        
        return action, info
    
    def _blend_actions(
        self, 
        protocol_action: np.ndarray, 
        rl_action: np.ndarray,
        blend_factor: float = 0.5,
    ) -> np.ndarray:
        """
        Blend protocol-derived action with RL agent's suggestion.
        
        blend_factor: 0.0 = pure RL, 1.0 = pure protocol
        
        Protocol provides the macro structure (intervention type, direction),
        RL agent fine-tunes parameters (exact loudness, timing, etc.)
        """
        return blend_factor * protocol_action + (1 - blend_factor) * rl_action
    
    def _record_outcome(self, effective: bool, arousal: bool):
        """Record outcome of most recent intervention."""
        if self.records:
            self.records[-1].was_effective = effective
            self.records[-1].caused_arousal = arousal
    
    def _update_personalization(self, effective: bool):
        """
        Update personalized intervention parameters based on outcome.
        
        If effective at current loudness → can try lower next time
        If ineffective → need higher loudness (but track habituation)
        """
        c = self.config
        
        if effective:
            # Successful: can try lower loudness next time
            self.effective_loudness_dir = max(
                c.dir_cue_loudness_start,
                self.effective_loudness_dir - c.sensitization_rate
            )
            self.effective_loudness_burst = max(
                c.burst_loudness_start,
                self.effective_loudness_burst - c.sensitization_rate
            )
        else:
            # Failed: increase effective dose, update habituation
            self.effective_loudness_dir = min(
                c.dir_cue_loudness_max,
                self.effective_loudness_dir + c.habituation_rate
            )
            self.effective_loudness_burst = min(
                c.burst_loudness_max,
                self.effective_loudness_burst + c.habituation_rate
            )
            self.habituation_factor = min(2.0, self.habituation_factor * 1.05)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics for this sleep session."""
        if not self.records:
            return {
                'total_interventions': 0,
                'directional_count': 0,
                'burst_count': 0,
                'effectiveness_rate': 0.0,
                'arousal_rate': 0.0,
            }
        
        directional = [r for r in self.records 
                      if r.protocol_state == ProtocolState.DIRECTIONAL_CUE]
        bursts = [r for r in self.records
                 if r.protocol_state == ProtocolState.BURST_CUE]
        
        effective = sum(1 for r in self.records if r.was_effective)
        arousals = sum(1 for r in self.records if r.caused_arousal)
        
        return {
            'total_interventions': len(self.records),
            'directional_count': len(directional),
            'burst_count': len(bursts),
            'effectiveness_rate': effective / max(len(self.records), 1),
            'arousal_rate': arousals / max(len(self.records), 1),
            'final_dir_loudness': self.effective_loudness_dir,
            'final_burst_loudness': self.effective_loudness_burst,
            'habituation_factor': self.habituation_factor,
        }
