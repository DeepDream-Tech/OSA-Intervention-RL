"""
Signal Processing Module for OSA Acoustic Intervention System
==============================================================

Extracts real-time features from four sensor modalities:
  1. RIP (Respiratory Inductance Plethysmography): thoracoabdominal bands
  2. Audio: snoring characteristics from earphone microphone
  3. IMU: 6-axis inertial measurement for body position
  4. SpO2: pulse oximetry for blood oxygen monitoring

Design based on:
  - Portiloop (arxiv:2107.13473): online EMA normalization, bandpass filtering
  - 1D-ViT multimodal sleep (arxiv:2502.17486): respiratory feature extraction
  - AASM scoring manual: standard respiratory event definitions

All processors maintain internal state for real-time streaming operation.
"""

import numpy as np
from scipy.signal import butter, sosfilt, find_peaks, welch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


# ==============================================================================
# Configuration Constants
# ==============================================================================

@dataclass
class SignalConfig:
    """Global signal processing configuration."""
    fs: int = 64                  # Sampling frequency (Hz) for physiological signals
    fs_audio: int = 16000         # Audio sampling frequency (Hz)
    epoch_sec: float = 30.0       # Sleep epoch duration (AASM standard)
    
    # RIP filtering
    rip_lowcut: float = 0.05      # Hz - captures slow breathing (3 bpm)
    rip_highcut: float = 1.0      # Hz - captures fast breathing (60 bpm)
    rip_filter_order: int = 4     # Butterworth filter order
    
    # Audio filtering
    audio_snore_lowcut: float = 30.0    # Hz - snoring fundamental lower bound
    audio_snore_highcut: float = 2000.0 # Hz - snoring harmonics upper bound
    
    # SpO2 parameters
    spo2_normal: float = 95.0     # Normal SpO2 baseline (%)
    spo2_desat_threshold: float = 3.0  # Desaturation event threshold (% drop)
    spo2_critical: float = 80.0   # Critical hypoxemia threshold (%)
    
    # Online normalization (Portiloop-style EMA)
    ema_alpha: float = 0.001      # EMA decay for online z-score


# ==============================================================================
# RIP (Respiratory Inductance Plethysmography) Processor
# ==============================================================================

class RIPProcessor:
    """
    Processes thoracic and abdominal RIP band signals.
    
    Extracts:
      - Respiratory amplitude (tidal volume proxy)
      - Respiratory rate (breaths per minute)
      - Thoracoabdominal phase angle (paradoxical breathing indicator)
      - Flow limitation index (airflow obstruction proxy)
    
    Paradoxical breathing (phase angle > 90°) is a key OSA precursor:
      Normal: chest and abdomen expand together (in-phase)
      OSA: chest expands while abdomen contracts (anti-phase) due to
           respiratory effort against collapsed airway
    """
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self._design_filters()
        self._reset_ema()
    
    def _design_filters(self):
        """Design Butterworth bandpass filters for respiratory band."""
        nyq = self.config.fs / 2.0
        self.sos_resp = butter(
            self.config.rip_filter_order,
            [self.config.rip_lowcut / nyq, self.config.rip_highcut / nyq],
            btype='band', output='sos'
        )
        # Narrower band for clean rate estimation
        self.sos_rate = butter(
            2,
            [0.1 / nyq, 0.6 / nyq],  # 6-36 bpm
            btype='band', output='sos'
        )
    
    def _reset_ema(self):
        """Reset online normalization state."""
        self.thorax_mu = 0.0
        self.thorax_var = 1.0
        self.abdomen_mu = 0.0
        self.abdomen_var = 1.0
    
    def _online_normalize(self, signal: np.ndarray, channel: str = 'thorax') -> np.ndarray:
        """
        Portiloop-style online z-score normalization using EMA.
        
        μ̂(t) = α·s(t) + (1-α)·μ̂(t-1)
        σ̂²(t) = α·(s(t)-μ̂(t))² + (1-α)·σ̂²(t-1)
        s'(t) = (s(t) - μ̂(t)) / σ̂(t)
        """
        alpha = self.config.ema_alpha
        normalized = np.zeros_like(signal)
        
        if channel == 'thorax':
            mu, var = self.thorax_mu, self.thorax_var
        else:
            mu, var = self.abdomen_mu, self.abdomen_var
        
        for i, s in enumerate(signal):
            mu = alpha * s + (1 - alpha) * mu
            var = alpha * (s - mu) ** 2 + (1 - alpha) * var
            normalized[i] = (s - mu) / (np.sqrt(var) + 1e-8)
        
        if channel == 'thorax':
            self.thorax_mu, self.thorax_var = mu, var
        else:
            self.abdomen_mu, self.abdomen_var = mu, var
        
        return normalized
    
    def extract_features(
        self, 
        thorax: np.ndarray, 
        abdomen: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract respiratory features from one epoch of RIP data.
        
        Args:
            thorax: Thoracic band signal, shape (n_samples,)
            abdomen: Abdominal band signal, shape (n_samples,)
        
        Returns:
            Dictionary with respiratory features
        """
        # 1. Bandpass filter
        thorax_filt = sosfilt(self.sos_resp, thorax)
        abdomen_filt = sosfilt(self.sos_resp, abdomen)
        
        # 2. Online normalize
        thorax_norm = self._online_normalize(thorax_filt, 'thorax')
        abdomen_norm = self._online_normalize(abdomen_filt, 'abdomen')
        
        # 3. Respiratory amplitude (tidal volume proxy)
        thorax_amplitude = np.std(thorax_filt)
        abdomen_amplitude = np.std(abdomen_filt)
        total_amplitude = thorax_amplitude + abdomen_amplitude
        
        # 4. Respiratory rate via peak detection
        rate_signal = sosfilt(self.sos_rate, thorax)
        peaks, _ = find_peaks(
            rate_signal, 
            distance=int(self.config.fs * 1.0),  # min 1s between breaths
            height=0
        )
        
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / self.config.fs
            resp_rate = 60.0 / np.mean(intervals)  # breaths per minute
            resp_rate_variability = np.std(intervals) / (np.mean(intervals) + 1e-8)
        else:
            resp_rate = 0.0  # Cannot estimate - possibly apnea
            resp_rate_variability = 0.0
        
        # 5. Thoracoabdominal phase angle (Lissajous method)
        # Cross-correlation to find phase lag
        phase_angle = self._compute_phase_angle(thorax_filt, abdomen_filt)
        
        # 6. Paradoxical breathing indicator
        # Phase angle > 90° indicates paradoxical movement (OSA hallmark)
        is_paradoxical = float(abs(phase_angle) > 90.0)
        
        # 7. Flow limitation index
        # Ratio of peak-to-mean inspiratory flow (flattened waveform = obstruction)
        flow_limitation = self._compute_flow_limitation(thorax_filt + abdomen_filt)
        
        # 8. Respiratory effort (sum of band amplitudes during attempted breathing)
        # High effort + low flow = obstructive event
        resp_effort = (thorax_amplitude + abdomen_amplitude) / (total_amplitude + 1e-8)
        
        return {
            'thorax_amplitude': float(thorax_amplitude),
            'abdomen_amplitude': float(abdomen_amplitude),
            'total_amplitude': float(total_amplitude),
            'resp_rate': float(np.clip(resp_rate, 0, 60)),
            'resp_rate_variability': float(resp_rate_variability),
            'phase_angle': float(phase_angle),
            'is_paradoxical': float(is_paradoxical),
            'flow_limitation': float(flow_limitation),
            'resp_effort': float(resp_effort),
            'thorax_normalized': thorax_norm,
            'abdomen_normalized': abdomen_norm,
        }
    
    def _compute_phase_angle(self, thorax: np.ndarray, abdomen: np.ndarray) -> float:
        """
        Compute thoracoabdominal phase angle using Hilbert transform analytic signal.
        
        Phase angle interpretation:
          0°: Perfect synchrony (normal breathing)
          90°: Quarter-cycle lag (partial obstruction)
          180°: Complete paradox (full obstruction, OSA)
        """
        from scipy.signal import hilbert
        
        analytic_t = hilbert(thorax)
        analytic_a = hilbert(abdomen)
        
        phase_t = np.angle(analytic_t)
        phase_a = np.angle(analytic_a)
        
        # Circular mean of phase difference
        phase_diff = phase_t - phase_a
        mean_phase_diff = np.arctan2(
            np.mean(np.sin(phase_diff)),
            np.mean(np.cos(phase_diff))
        )
        
        return float(np.degrees(mean_phase_diff))
    
    def _compute_flow_limitation(self, flow_signal: np.ndarray) -> float:
        """
        Flow limitation index: ratio of peak inspiratory flow to mean flow.
        
        Normal: ~2.0 (sinusoidal waveform)
        Flow-limited: >3.0 (peaked, narrow inspiratory waveform)
        Obstructed: approaching infinity (minimal flow with high effort)
        """
        # Find inspiratory phases (positive half-cycles)
        inspiratory = flow_signal[flow_signal > 0]
        
        if len(inspiratory) < 10:
            return 5.0  # Very low flow - likely apnea
        
        peak_flow = np.percentile(inspiratory, 95)
        mean_flow = np.mean(inspiratory)
        
        return float(peak_flow / (mean_flow + 1e-8))


# ==============================================================================
# Audio (Snoring) Processor
# ==============================================================================

class AudioProcessor:
    """
    Processes earphone microphone audio for snoring characteristics.
    
    Extracts:
      - Snore energy (RMS in snoring band)
      - Fundamental frequency (F0) and stability
      - Snore pattern type: Sustained vs Snore Bout
      - Spectral centroid and bandwidth
    
    Snoring taxonomy relevant to OSA:
      - Sustained snoring: continuous, stable F0 → partial obstruction
      - Snore bout: intermittent bursts → cyclical obstruction/recovery
      - Crescendo snoring: increasing amplitude → progressive collapse
    """
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self._design_filters()
        self.prev_f0 = None
        self.snore_history = []  # Track snoring pattern over time
    
    def _design_filters(self):
        """Design bandpass filter for snoring frequency band."""
        nyq = self.config.fs_audio / 2.0
        self.sos_snore = butter(
            4,
            [self.config.audio_snore_lowcut / nyq, 
             self.config.audio_snore_highcut / nyq],
            btype='band', output='sos'
        )
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract snoring features from audio segment.
        
        Args:
            audio: Raw audio signal, shape (n_samples,), fs=16000Hz
        
        Returns:
            Dictionary with audio/snoring features
        """
        # 1. Bandpass filter to snoring band
        snore_signal = sosfilt(self.sos_snore, audio)
        
        # 2. Snore energy (RMS)
        rms = float(np.sqrt(np.mean(snore_signal ** 2)))
        
        # 3. Fundamental frequency (F0) via autocorrelation
        f0, f0_confidence = self._estimate_f0(snore_signal)
        
        # 4. F0 stability (variation from previous epoch)
        if self.prev_f0 is not None and self.prev_f0 > 0 and f0 > 0:
            f0_stability = 1.0 - min(abs(f0 - self.prev_f0) / self.prev_f0, 1.0)
        else:
            f0_stability = 0.0
        self.prev_f0 = f0
        
        # 5. Spectral features
        freqs, psd = welch(snore_signal, fs=self.config.fs_audio, nperseg=1024)
        
        # Spectral centroid
        total_power = np.sum(psd) + 1e-12
        spectral_centroid = float(np.sum(freqs * psd) / total_power)
        
        # Spectral bandwidth (spread)
        spectral_bandwidth = float(np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power
        ))
        
        # 6. Snore pattern classification
        # Track RMS over time to determine sustained vs bout pattern
        self.snore_history.append(rms)
        if len(self.snore_history) > 10:
            self.snore_history = self.snore_history[-10:]
        
        snore_pattern = self._classify_snore_pattern()
        
        # 7. Snore bout detection (intermittency ratio)
        # High intermittency = snore bout; Low = sustained snoring
        if len(self.snore_history) >= 3:
            snore_threshold = np.median(self.snore_history) * 0.5
            active_epochs = sum(1 for s in self.snore_history if s > snore_threshold)
            intermittency = 1.0 - (active_epochs / len(self.snore_history))
        else:
            intermittency = 0.0
        
        # 8. Crescendo detection (trending amplitude increase)
        if len(self.snore_history) >= 5:
            recent = np.array(self.snore_history[-5:])
            slope = np.polyfit(np.arange(5), recent, 1)[0]
            is_crescendo = float(slope > 0.01)  # Positive trend
        else:
            is_crescendo = 0.0
        
        return {
            'snore_rms': rms,
            'snore_f0': float(f0),
            'f0_confidence': float(f0_confidence),
            'f0_stability': float(f0_stability),
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'snore_pattern': float(snore_pattern),  # 0=none, 1=sustained, 2=bout
            'intermittency': float(intermittency),
            'is_crescendo': float(is_crescendo),
        }
    
    def _estimate_f0(self, signal: np.ndarray) -> Tuple[float, float]:
        """
        Estimate fundamental frequency using autocorrelation method.
        
        Returns:
            (f0_hz, confidence): Fundamental frequency and estimation confidence
        """
        # Autocorrelation
        n = len(signal)
        autocorr = np.correlate(signal, signal, mode='full')[n-1:]
        autocorr = autocorr / (autocorr[0] + 1e-12)
        
        # Search for F0 in valid range (50-500 Hz for snoring)
        min_lag = int(self.config.fs_audio / 500)  # 500 Hz upper bound
        max_lag = int(self.config.fs_audio / 50)   # 50 Hz lower bound
        
        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1
        if min_lag >= max_lag:
            return 0.0, 0.0
        
        search_region = autocorr[min_lag:max_lag]
        
        if len(search_region) == 0:
            return 0.0, 0.0
        
        peak_idx = np.argmax(search_region) + min_lag
        confidence = float(autocorr[peak_idx])
        
        if confidence > 0.3:  # Minimum confidence threshold
            f0 = self.config.fs_audio / peak_idx
            return float(f0), confidence
        
        return 0.0, 0.0
    
    def _classify_snore_pattern(self) -> float:
        """
        Classify snoring pattern from recent history.
        
        Returns:
            0.0: No snoring
            1.0: Sustained snoring (continuous, stable)
            2.0: Snore bout (intermittent bursts)
        """
        if len(self.snore_history) < 3:
            return 0.0
        
        recent = np.array(self.snore_history[-5:]) if len(self.snore_history) >= 5 \
            else np.array(self.snore_history)
        
        threshold = 0.01  # Minimum RMS to count as snoring
        snoring_ratio = np.mean(recent > threshold)
        
        if snoring_ratio < 0.3:
            return 0.0  # No significant snoring
        elif snoring_ratio > 0.7:
            return 1.0  # Sustained snoring
        else:
            return 2.0  # Snore bout (intermittent)


# ==============================================================================
# IMU (Inertial Measurement Unit) Processor
# ==============================================================================

class IMUProcessor:
    """
    Processes 6-axis IMU data for body position detection.
    
    Uses gravity vector orientation to classify sleep position:
      - Supine (仰卧): Gravity along posterior axis → HIGHEST OSA risk
      - Prone (俯卧): Gravity along anterior axis → Lowest risk
      - Left lateral (左侧卧): Gravity along left axis
      - Right lateral (右侧卧): Gravity along right axis
    
    Supine position increases OSA risk 2-3x due to gravity pulling the
    tongue and soft palate posteriorly, narrowing the airway.
    """
    
    # Position codes
    SUPINE = 0      # 仰卧 - highest OSA risk
    PRONE = 1       # 俯卧
    LEFT = 2        # 左侧卧
    RIGHT = 3       # 右侧卧
    UPRIGHT = 4     # 直立
    UNKNOWN = 5
    
    POSITION_NAMES = {0: 'supine', 1: 'prone', 2: 'left', 3: 'right', 4: 'upright', 5: 'unknown'}
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self.position_history = []
        # Low-pass filter for gravity extraction (remove dynamic acceleration)
        nyq = self.config.fs / 2.0
        self.sos_gravity = butter(2, 0.5 / nyq, btype='low', output='sos')
    
    def extract_features(self, accel: np.ndarray, gyro: np.ndarray = None) -> Dict[str, float]:
        """
        Extract body position features from IMU data.
        
        Args:
            accel: Accelerometer data, shape (n_samples, 3) in [x, y, z] (m/s²)
                   Convention: x=anterior, y=left, z=superior (anatomical)
            gyro: Gyroscope data, shape (n_samples, 3) in [x, y, z] (rad/s)
                  Optional, used for movement detection
        
        Returns:
            Dictionary with position features
        """
        # 1. Extract gravity vector (low-pass filter removes motion artifacts)
        gravity = np.zeros_like(accel)
        for axis in range(3):
            gravity[:, axis] = sosfilt(self.sos_gravity, accel[:, axis])
        
        # Mean gravity vector over epoch
        g_mean = np.mean(gravity, axis=0)
        g_norm = np.linalg.norm(g_mean)
        
        if g_norm > 0:
            g_unit = g_mean / g_norm
        else:
            g_unit = np.array([0, 0, 1])
        
        # 2. Classify body position from gravity orientation
        position = self._classify_position(g_unit)
        is_supine = float(position == self.SUPINE)
        
        # 3. Track position stability (time in current position)
        self.position_history.append(position)
        if len(self.position_history) > 20:
            self.position_history = self.position_history[-20:]
        
        position_stability = self._compute_position_stability()
        
        # 4. Movement intensity (from gyroscope if available)
        if gyro is not None:
            movement_intensity = float(np.mean(np.abs(gyro)))
            # Angular velocity variance (restlessness indicator)
            movement_variability = float(np.std(np.linalg.norm(gyro, axis=1)))
        else:
            # Estimate movement from accelerometer variance
            accel_var = np.var(accel, axis=0)
            movement_intensity = float(np.sum(accel_var))
            movement_variability = float(np.std(accel_var))
        
        # 5. Tilt angles (pitch and roll)
        pitch = float(np.degrees(np.arctan2(g_unit[0], np.sqrt(g_unit[1]**2 + g_unit[2]**2))))
        roll = float(np.degrees(np.arctan2(g_unit[1], g_unit[2])))
        
        return {
            'position': float(position),
            'position_name': self.POSITION_NAMES[position],
            'is_supine': is_supine,
            'gravity_vector': g_unit.tolist(),
            'pitch': pitch,
            'roll': roll,
            'position_stability': float(position_stability),
            'movement_intensity': float(movement_intensity),
            'movement_variability': float(movement_variability),
        }
    
    def _classify_position(self, g_unit: np.ndarray) -> int:
        """
        Classify body position from gravity unit vector.
        
        Using anatomical coordinate system:
          x: anterior (+) / posterior (-)
          y: left (+) / right (-)  
          z: superior (+) / inferior (-)
        
        Supine: gravity points anteriorly (person lying on back)
          → g_x is most positive
        """
        # Angle thresholds (degrees)
        angle_threshold = 35.0  # Within 35° of primary axis
        
        # Compute angle from each axis
        angles = np.degrees(np.arccos(np.clip(np.abs(g_unit), 0, 1)))
        
        # Determine dominant axis
        dominant_axis = np.argmin(angles)
        
        if angles[dominant_axis] > angle_threshold:
            return self.UNKNOWN
        
        if dominant_axis == 0:  # x-axis dominant (anterior/posterior)
            return self.SUPINE if g_unit[0] > 0 else self.PRONE
        elif dominant_axis == 1:  # y-axis dominant (left/right)
            return self.LEFT if g_unit[1] > 0 else self.RIGHT
        else:  # z-axis dominant (superior/inferior)
            return self.UPRIGHT
    
    def _compute_position_stability(self) -> float:
        """
        Compute position stability: fraction of recent epochs in same position.
        High stability + supine = sustained OSA risk.
        """
        if len(self.position_history) < 2:
            return 1.0
        
        current = self.position_history[-1]
        same_count = sum(1 for p in self.position_history if p == current)
        return float(same_count / len(self.position_history))


# ==============================================================================
# SpO2 (Pulse Oximetry) Processor
# ==============================================================================

class SpO2Processor:
    """
    Processes real-time blood oxygen saturation data.
    
    Extracts:
      - Current SpO2 level and trend
      - Desaturation events (3%+ drops from baseline)
      - Desaturation slope (speed of oxygen decline)
      - Oxygen Desaturation Index (ODI) proxy
      - Hypoxemia risk score
    
    SpO2 dynamics during OSA:
      1. Airway obstruction → ventilation ceases
      2. SpO2 begins declining after ~15-30s lag (lung O2 stores)
      3. Desaturation slope reflects severity
      4. Arousal/intervention → airway reopens → SpO2 recovers
      5. Recovery time indicates physiological reserve
    """
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self.baseline = config.spo2_normal if config else 95.0
        self.history = []  # Rolling SpO2 history
        self.desat_events = []  # Timestamps of desaturation events
        self.epoch_count = 0
    
    def extract_features(self, spo2_values: np.ndarray) -> Dict[str, float]:
        """
        Extract SpO2 features from one epoch of pulse oximetry data.
        
        Args:
            spo2_values: SpO2 readings over epoch, shape (n_samples,), range [0, 100]
        
        Returns:
            Dictionary with SpO2 features
        """
        self.epoch_count += 1
        
        # 1. Current SpO2 (mean of epoch, excluding artifacts)
        valid_spo2 = spo2_values[(spo2_values > 50) & (spo2_values <= 100)]
        if len(valid_spo2) == 0:
            current_spo2 = self.baseline
        else:
            current_spo2 = float(np.mean(valid_spo2))
        
        # 2. Update rolling history
        self.history.append(current_spo2)
        if len(self.history) > 120:  # Keep last 60 minutes (at 30s epochs)
            self.history = self.history[-120:]
        
        # 3. Update baseline (90th percentile of recent values)
        if len(self.history) >= 10:
            self.baseline = float(np.percentile(self.history[-60:], 90))
        
        # 4. Desaturation from baseline
        desat_from_baseline = self.baseline - current_spo2
        is_desaturating = float(desat_from_baseline >= self.config.spo2_desat_threshold)
        
        # 5. Desaturation slope (rate of SpO2 decline)
        if len(self.history) >= 3:
            recent = np.array(self.history[-5:]) if len(self.history) >= 5 \
                else np.array(self.history)
            slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
            desat_slope = float(slope)  # Negative = declining
        else:
            desat_slope = 0.0
        
        # 6. Track desaturation events (for ODI calculation)
        if is_desaturating and (
            len(self.desat_events) == 0 or 
            self.epoch_count - self.desat_events[-1] >= 2  # Min 2 epochs apart
        ):
            self.desat_events.append(self.epoch_count)
        
        # 7. ODI proxy (desaturation events per hour)
        hours_elapsed = (self.epoch_count * self.config.epoch_sec) / 3600.0
        if hours_elapsed > 0:
            odi_proxy = len(self.desat_events) / hours_elapsed
        else:
            odi_proxy = 0.0
        
        # 8. Hypoxemia risk score (0-1 scale)
        # Combines current level, trend, and history
        risk_level = self._compute_hypoxemia_risk(current_spo2, desat_slope)
        
        # 9. SpO2 variability (coefficient of variation over recent history)
        if len(self.history) >= 5:
            spo2_variability = float(np.std(self.history[-10:]) / 
                                     (np.mean(self.history[-10:]) + 1e-8))
        else:
            spo2_variability = 0.0
        
        # 10. Time below 90% (critical hypoxemia time fraction)
        if len(self.history) >= 5:
            recent_arr = np.array(self.history[-10:])
            t90 = float(np.mean(recent_arr < 90.0))
        else:
            t90 = 0.0
        
        # Normalize SpO2 to [0, 1] for RL observation
        spo2_normalized = float(np.clip((current_spo2 - 70.0) / 30.0, 0.0, 1.0))
        
        return {
            'spo2_current': float(current_spo2),
            'spo2_normalized': spo2_normalized,
            'spo2_baseline': float(self.baseline),
            'desat_from_baseline': float(desat_from_baseline),
            'is_desaturating': float(is_desaturating),
            'desat_slope': float(desat_slope),
            'odi_proxy': float(odi_proxy),
            'hypoxemia_risk': float(risk_level),
            'spo2_variability': float(spo2_variability),
            'time_below_90': float(t90),
        }
    
    def _compute_hypoxemia_risk(self, current_spo2: float, slope: float) -> float:
        """
        Compute composite hypoxemia risk score (0-1).
        
        Components:
          - Current level risk: sigmoid centered at 90%
          - Slope risk: steeper decline = higher risk
          - Sustained low risk: cumulative time below threshold
        """
        # Level component: sigmoid risk
        level_risk = 1.0 / (1.0 + np.exp(0.5 * (current_spo2 - 90.0)))
        
        # Slope component: negative slope = increasing risk
        slope_risk = np.clip(-slope / 2.0, 0.0, 1.0)
        
        # Baseline deviation component
        desat_risk = np.clip((self.baseline - current_spo2) / 10.0, 0.0, 1.0)
        
        # Weighted combination
        risk = 0.4 * level_risk + 0.3 * slope_risk + 0.3 * desat_risk
        
        return float(np.clip(risk, 0.0, 1.0))


# ==============================================================================
# Integrated Feature Extractor
# ==============================================================================

class MultimodalFeatureExtractor:
    """
    Orchestrates all four sensor processors and produces a unified feature vector
    for the OSA risk predictor and RL agent.
    
    Feature vector layout (33 dimensions):
      [0:11]  RIP features (11)
      [11:20] Audio features (9) 
      [20:29] IMU features (9)
      [29:39] SpO2 features (10)
    """
    
    FEATURE_DIM = 33  # Total number of scalar features
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self.rip = RIPProcessor(self.config)
        self.audio = AudioProcessor(self.config)
        self.imu = IMUProcessor(self.config)
        self.spo2 = SpO2Processor(self.config)
    
    def extract_all(
        self,
        thorax: np.ndarray,
        abdomen: np.ndarray,
        audio: np.ndarray,
        accel: np.ndarray,
        gyro: Optional[np.ndarray],
        spo2_values: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Dict]]:
        """
        Extract features from all modalities for one epoch.
        
        Returns:
            feature_vector: np.ndarray of shape (33,) - normalized features
            raw_features: Dict of dicts from each processor
        """
        # Extract from each modality
        rip_feats = self.rip.extract_features(thorax, abdomen)
        audio_feats = self.audio.extract_features(audio)
        imu_feats = self.imu.extract_features(accel, gyro)
        spo2_feats = self.spo2.extract_features(spo2_values)
        
        # Assemble scalar feature vector
        feature_vector = np.array([
            # RIP (11 features)
            rip_feats['thorax_amplitude'],
            rip_feats['abdomen_amplitude'],
            rip_feats['total_amplitude'],
            rip_feats['resp_rate'] / 60.0,  # Normalize to [0, 1]
            rip_feats['resp_rate_variability'],
            rip_feats['phase_angle'] / 180.0,  # Normalize to [-1, 1]
            rip_feats['is_paradoxical'],
            rip_feats['flow_limitation'] / 5.0,  # Normalize
            rip_feats['resp_effort'],
            0.0,  # Reserved
            0.0,  # Reserved
            
            # Audio (9 features)
            min(audio_feats['snore_rms'] * 10, 1.0),  # Normalize
            audio_feats['snore_f0'] / 500.0,           # Normalize to ~[0, 1]
            audio_feats['f0_confidence'],
            audio_feats['f0_stability'],
            audio_feats['spectral_centroid'] / 2000.0,  # Normalize
            audio_feats['spectral_bandwidth'] / 1000.0, # Normalize
            audio_feats['snore_pattern'] / 2.0,         # Normalize to [0, 1]
            audio_feats['intermittency'],
            audio_feats['is_crescendo'],
            
            # IMU (4 scalar features - position_name excluded)
            imu_feats['is_supine'],
            imu_feats['pitch'] / 90.0,              # Normalize to [-1, 1]
            imu_feats['roll'] / 180.0,               # Normalize to [-1, 1]
            imu_feats['position_stability'],
            imu_feats['movement_intensity'],
            imu_feats['movement_variability'],
            0.0,  # Reserved
            0.0,  # Reserved
            0.0,  # Reserved
            
            # SpO2 (4 features)
            spo2_feats['spo2_normalized'],
            spo2_feats['desat_slope'],
            spo2_feats['hypoxemia_risk'],
            spo2_feats['odi_proxy'] / 30.0,  # Normalize (severe OSA = 30+/hr)
        ], dtype=np.float32)
        
        raw_features = {
            'rip': rip_feats,
            'audio': audio_feats,
            'imu': imu_feats,
            'spo2': spo2_feats,
        }
        
        return feature_vector, raw_features
