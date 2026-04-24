"""
Audio Synthesis Module
=======================

Generates the actual acoustic stimuli for delivery through earphones.

Supports two main intervention types:
  1. Directional Cue (方向性提示): Binaural spatial audio using ITD/ILD
     to encourage unconscious position change during sleep
  2. Short Burst Cue (短促声音刺激): Brief tonal burst to promote
     airway muscle activation without full arousal

Audio design principles for sleep intervention:
  - Gradual onset (fade-in) to avoid startle response
  - Low frequencies (< 500 Hz) for directional cues (less alerting)
  - Natural sound characteristics (pink noise modulation)
  - Precise binaural parameter control for spatial perception
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio synthesis configuration."""
    sample_rate: int = 48000       # Output sample rate (Hz)
    max_amplitude: float = 0.7     # Maximum amplitude (< 1.0 for safety)
    fade_in_ms: float = 50.0       # Fade-in duration (ms)
    fade_out_ms: float = 100.0     # Fade-out duration (ms)
    
    # Safety limits
    max_loudness_db: float = -20.0   # Maximum loudness relative to full scale
    max_duration_sec: float = 10.0   # Maximum stimulus duration


class AudioSynthesizer:
    """
    Generates binaural acoustic stimuli for OSA intervention.
    
    Creates stereo audio with precise ITD and ILD control for spatial
    perception during sleep.
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
    
    def generate_stimulus(
        self,
        loudness: float,       # 0-1 normalized
        frequency: float,      # Hz
        duration: float,       # seconds
        timing: float,         # 0-1 within-epoch position
        itd: float,           # ms (interaural time difference)
        ild: float,           # dB (interaural level difference)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a binaural audio stimulus.
        
        Args:
            loudness: Volume level 0-1 (mapped to dB scale)
            frequency: Carrier frequency in Hz
            duration: Stimulus duration in seconds
            timing: Position within sleep epoch (0=start, 1=end)
            itd: Interaural time difference in milliseconds
                 Positive = right ear leads (sound from right)
                 Negative = left ear leads (sound from left)
            ild: Interaural level difference in dB
                 Positive = right ear louder (sound from right)
                 Negative = left ear louder (sound from left)
        
        Returns:
            left_channel: Left ear audio signal
            right_channel: Right ear audio signal
        """
        cfg = self.config
        
        # Safety: clip all parameters
        loudness = np.clip(loudness, 0.0, 1.0)
        frequency = np.clip(frequency, 20.0, 4000.0)
        duration = np.clip(duration, 0.01, cfg.max_duration_sec)
        itd = np.clip(itd, -1.5, 1.5)
        ild = np.clip(ild, -20.0, 20.0)
        
        # If loudness is essentially zero, return silence
        if loudness < 0.01:
            n_samples = int(duration * cfg.sample_rate)
            return np.zeros(n_samples), np.zeros(n_samples)
        
        # Convert loudness to linear amplitude
        # Map 0-1 to max_loudness_db..0 dB
        loudness_db = cfg.max_loudness_db * (1.0 - loudness)
        amplitude = cfg.max_amplitude * 10 ** (loudness_db / 20.0)
        
        # Generate base tone
        n_samples = int(duration * cfg.sample_rate)
        t = np.arange(n_samples) / cfg.sample_rate
        
        # Use sinusoidal tone with slight pink noise modulation for naturalness
        carrier = np.sin(2 * np.pi * frequency * t)
        
        # Add harmonic richness (more natural than pure tone)
        carrier += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)  # 2nd harmonic
        carrier += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)  # 3rd harmonic
        carrier /= np.max(np.abs(carrier) + 1e-8)
        
        # Amplitude modulation for more natural sound
        am_freq = 3.0  # Slow amplitude modulation (Hz)
        am_depth = 0.15
        am = 1.0 + am_depth * np.sin(2 * np.pi * am_freq * t)
        carrier *= am
        
        # Apply envelope (fade-in/fade-out)
        envelope = self._make_envelope(n_samples)
        carrier *= envelope * amplitude
        
        # Apply binaural parameters
        left_channel, right_channel = self._apply_binaural(carrier, t, itd, ild)
        
        return left_channel, right_channel
    
    def generate_directional_cue(
        self,
        direction: float,      # -1 (left) to +1 (right)
        loudness: float = 0.2,
        duration: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a directional cue optimized for position change.
        
        Uses low frequency and strong binaural cues to create a
        compelling spatial impression without full arousal.
        
        Args:
            direction: -1 = encourage roll left, +1 = encourage roll right
            loudness: Volume level 0-1
            duration: Duration in seconds
        """
        # Low frequency is less alerting and has better spatial resolution
        frequency = 200.0
        
        # Strong spatial parameters for clear directional perception
        itd = float(direction * 1.2)   # Near-maximum ITD
        ild = float(direction * 15.0)  # Strong level difference
        
        return self.generate_stimulus(
            loudness=loudness,
            frequency=frequency,
            duration=duration,
            timing=0.5,
            itd=itd,
            ild=ild,
        )
    
    def generate_burst_cue(
        self,
        loudness: float = 0.4,
        duration: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a short burst cue for airway muscle activation.
        
        Uses mid-frequency with no spatial bias.
        Short duration to minimize sleep disruption.
        """
        return self.generate_stimulus(
            loudness=loudness,
            frequency=1000.0,  # Speech-range frequency
            duration=duration,
            timing=0.3,       # Early delivery
            itd=0.0,          # No spatial bias (bilateral)
            ild=0.0,
        )
    
    def _make_envelope(self, n_samples: int) -> np.ndarray:
        """Create a smooth fade-in/fade-out envelope."""
        cfg = self.config
        
        fade_in_samples = int(cfg.fade_in_ms / 1000.0 * cfg.sample_rate)
        fade_out_samples = int(cfg.fade_out_ms / 1000.0 * cfg.sample_rate)
        
        envelope = np.ones(n_samples)
        
        # Raised cosine fade-in
        if fade_in_samples > 0 and fade_in_samples < n_samples:
            t_in = np.arange(fade_in_samples) / fade_in_samples
            envelope[:fade_in_samples] = 0.5 * (1 - np.cos(np.pi * t_in))
        
        # Raised cosine fade-out
        if fade_out_samples > 0 and fade_out_samples < n_samples:
            t_out = np.arange(fade_out_samples) / fade_out_samples
            envelope[-fade_out_samples:] = 0.5 * (1 + np.cos(np.pi * t_out))
        
        return envelope
    
    def _apply_binaural(
        self,
        mono: np.ndarray,
        t: np.ndarray,
        itd: float,
        ild: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ITD and ILD to create binaural stereo signal.
        
        ITD (Interaural Time Difference):
          - Delays one ear's signal relative to the other
          - Maximum natural ITD: ~0.65ms (head width / speed of sound)
          - We allow up to 1.5ms for enhanced spatial perception
        
        ILD (Interaural Level Difference):
          - Adjusts relative volume between ears
          - Frequency-dependent in reality, simplified here
          - Maximum natural ILD: ~20dB at high frequencies
        """
        cfg = self.config
        
        # ITD: apply time delay via phase shift
        itd_samples = int(abs(itd) / 1000.0 * cfg.sample_rate)
        
        if itd >= 0:
            # Right ear leads = left ear is delayed
            left = np.pad(mono, (itd_samples, 0))[:len(mono)]
            right = mono.copy()
        else:
            # Left ear leads = right ear is delayed
            left = mono.copy()
            right = np.pad(mono, (itd_samples, 0))[:len(mono)]
        
        # ILD: adjust relative level
        ild_linear = 10 ** (abs(ild) / 20.0)
        
        if ild >= 0:
            # Right ear louder
            right *= np.sqrt(ild_linear)
            left /= np.sqrt(ild_linear)
        else:
            # Left ear louder
            left *= np.sqrt(ild_linear)
            right /= np.sqrt(ild_linear)
        
        # Ensure no clipping
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > cfg.max_amplitude:
            scale = cfg.max_amplitude / max_val
            left *= scale
            right *= scale
        
        return left, right
    
    def action_to_audio(
        self, 
        action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert RL agent action vector to binaural audio.
        
        Args:
            action: np.ndarray of shape (6,)
                [loudness, frequency, duration, timing, ITD, ILD]
        
        Returns:
            (left_channel, right_channel) stereo audio arrays
        """
        return self.generate_stimulus(
            loudness=float(action[0]),
            frequency=float(action[1]),
            duration=float(action[2]),
            timing=float(action[3]),
            itd=float(action[4]),
            ild=float(action[5]),
        )
