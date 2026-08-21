"""Fuzzy RIP + SpO2 controller for the pre-experiment protocol.

Inputs:
  - RIP respiratory samples
  - SpO2 samples (optional but used when available)
  - user-awake button samples, normally 0 with a 1 at the press timestamp

Output:
  - Sound loudness only. Sound selection and playback timing are managed by the
    integration layer.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


Number = Union[int, float]
SignalInput = Union[Number, Sequence[Number], Iterable[Number]]


def _load_breathfinder():
    try:
        import BreathFinder as breathfinder  # type: ignore

        return breathfinder
    except ImportError:
        pass

    local_roots: list[Path] = []
    here = Path(__file__).resolve()
    for root in (here.parent, *here.parents[:3]):
        if root not in local_roots:
            local_roots.append(root)
    for root in local_roots:
        candidate = root / "BreathFinder"
        if not candidate.is_dir():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            import BreathFinder as breathfinder  # type: ignore

            return breathfinder
        except ImportError:
            continue

    for extra_path in (
        Path("/private/tmp/breathfinder_pkg"),
        Path("/tmp/BreathFinder-main"),
    ):
        if not extra_path.is_dir():
            continue
        path_str = str(extra_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        try:
            import BreathFinder as breathfinder  # type: ignore

            return breathfinder
        except ImportError:
            continue
    return None


_BREATHFINDER = _load_breathfinder()


class InterventionPhase(str, Enum):
    """Controller state exposed for logging and debugging."""

    MONITORING = "monitoring"
    EVENT_PENDING = "event_pending"
    INTERVENING = "intervening"
    RECOVERED = "recovered"
    USER_AWAKE = "user_awake"


@dataclass(frozen=True)
class PreExperimentConfig:
    """Protocol parameters for the standalone pre-experiment controller."""

    rip_fs: float = 25.0
    rip_amplitude_window_sec: float = 2.0
    rip_period_window_sec: float = 12.0
    baseline_history_sec: float = 180.0
    spo2_baseline_history_sec: float = 120.0
    spo2_stale_after_sec: float = 10.0

    # Legacy deterministic threshold kept for compatibility and logging.
    airflow_drop_threshold_fraction: float = 0.70

    # Fuzzy trigger condition.
    trigger_duration_sec: float = 8.5
    fast_trigger_duration_sec: float = 3.0
    apnea_min_duration_sec: float = 8.5
    period_missing_spo2_rescue_threshold: float = 2.0
    # Legacy score thresholds kept for compatibility with old logs / configs.
    trigger_score_threshold: float = 0.70
    high_risk_trigger_score_threshold: float = 0.85
    trigger_score_stop_threshold: float = 0.40
    airflow_soft_gate_upper_ratio: float = 0.45
    baseline_update_trigger_score_max: float = 0.25
    baseline_update_safe_duration_sec: float = 20.0
    baseline_freeze_after_movement_sec: float = 20.0
    spo2_hypopnea_threshold_pct: float = 3.0
    rip_smoothing_window_sec: float = 0.44
    consensus_window_sec: float = 12.0
    consensus_severe_ratio_max: float = 0.30
    consensus_support_ratio_max: float = 0.45
    consensus_min_severe_hits: int = 1
    consensus_min_support_hits: int = 2
    consensus_spo2_delta_min_pct: float = 3.0
    consensus_min_spo2_hits: int = 5
    consensus_min_event_duration_sec: float = 7.0

    # Continue-score driven intervention control.
    continue_score_stop_threshold: float = 0.30
    continue_score_decay_sec: float = 8.0
    continue_score_memory_peak: float = 1.0
    continue_score_memory_decay_sec: float = 10.0
    continue_score_recovery_hold_sec: float = 10.0
    continue_score_recovery_rip_ratio_min: float = 0.72
    continue_score_recovery_period_ratio_max: float = 1.18
    continue_score_recovery_low_rip_run_max_sec: float = 2.0

    # Streaming AASM-style trigger condition.
    aasm_breath_buffer_sec: float = 16.0
    aasm_breath_edge_guard_sec: float = 3.0
    aasm_breathfinder_update_sec: float = 10.0
    aasm_rip_baseline_window_sec: float = 120.0
    aasm_rip_min_baseline_breaths: int = 8
    aasm_rip_baseline_guard_sec: float = 5.0
    aasm_rip_fallback_breath_count: int = 12
    aasm_rip_fallback_min_breaths: int = 5
    aasm_direct_trigger_ratio: float = 0.90
    aasm_drop_ratio_strong: float = 0.70
    aasm_drop_ratio_weak: float = 0.80
    aasm_pending_event_ratio_threshold: float = 0.70
    aasm_min_low_rip_duration_sec: float = 10.0
    aasm_low_rip_merge_gap_sec: float = 2.0
    aasm_direct_active_grace_sec: float = 3.3
    aasm_direct_min_breath_count: int = 3
    aasm_direct_current_rip_ratio_max: float = 0.85
    aasm_spo2_baseline_window_sec: float = 60.0
    aasm_spo2_baseline_gap_sec: float = 3.0
    aasm_spo2_min_baseline_samples: int = 5
    aasm_spo2_fallback_sample_count: int = 12
    aasm_spo2_fallback_min_samples: int = 5
    aasm_spo2_lookahead_sec: float = 60.0
    aasm_spo2_drop_threshold_pct: float = 3.0
    aasm_spo2_min_desat_duration_sec: float = 10.0
    aasm_event_merge_gap_sec: float = 8.0
    posture_baseline_reset_before_sec: float = 10.0
    posture_baseline_reset_after_sec: float = 10.0

    # Intervention schedule.
    sound_interval_sec: float = 0.6
    loudness_levels: Tuple[float, ...] = (0.20, 0.28, 0.36, 0.44, 0.52, 0.60)
    loudness_eval_window_sounds: int = 3
    loudness_recovery_windows_for_step_down: int = 2
    loudness_initial_level_index: int = 1
    loudness_high_risk_floor_index: int = 2
    loudness_very_high_floor_index: int = 3
    loudness_very_high_trigger_threshold: float = 0.92
    loudness_trigger_recovery_delta: float = 0.08
    loudness_trigger_worsen_delta: float = 0.05
    loudness_rip_ratio_recovery_delta: float = 0.08
    loudness_rip_ratio_worsen_delta: float = 0.05
    loudness_period_recovery_delta: float = 0.12
    loudness_period_worsen_delta: float = 0.10
    loudness_spo2_recovery_delta: float = 0.5
    loudness_spo2_worsen_delta: float = 0.5

    # Basic signal-quality guards for the raw RIP batch.
    min_valid_rip_span: float = 1e-3
    max_valid_rip_span: float = 10000.0

    @property
    def airflow_ratio_threshold(self) -> float:
        return max(0.0, 1.0 - self.airflow_drop_threshold_fraction)


@dataclass(frozen=True)
class SensorSnapshot:
    """Computed physiological state at one controller update."""

    timestamp: float
    rip_amplitude: float
    rip_baseline: Optional[float]
    rip_amplitude_ratio: Optional[float]
    airflow_drop_fraction: Optional[float]
    breath_period_sec: Optional[float]
    breath_period_baseline: Optional[float]
    breath_period_ratio: Optional[float]
    period_feature_available: bool
    spo2_pct: Optional[float]
    spo2_baseline: Optional[float]
    spo2_delta: Optional[float]
    artifact_like_low_amplitude: bool
    apnea_like_drop_met: bool
    apnea_like_duration_sec: float
    raw_risk_score: float
    continue_score: float
    loudness_score: float
    signal_valid: bool
    movement_detected: bool
    button_pressed: bool
    low_rip_run_active: bool
    low_rip_run_duration_sec: float
    aasm_candidate_strength: Optional[str]
    aasm_event_condition_met: bool
    aasm_event_condition_duration_sec: float
    direct_trigger_met: bool
    direct_trigger_duration_sec: float
    event_condition_met: bool
    event_condition_duration_sec: float
    high_risk_duration_sec: float

    @property
    def raw_trigger_score(self) -> float:
        return self.raw_risk_score

    @property
    def trigger_score(self) -> float:
        return self.continue_score


@dataclass(frozen=True)
class InterventionCommand:
    """Decision returned by ``PreExperimentController.update``."""

    should_play_sound: bool
    loudness: float
    phase: InterventionPhase
    reason: str
    loudness_level_index: int = 0
    snapshot: Optional[SensorSnapshot] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "should_play_sound": self.should_play_sound,
            "loudness": self.loudness,
            "phase": self.phase.value,
            "reason": self.reason,
            "loudness_level_index": self.loudness_level_index,
            "snapshot": self.snapshot,
        }


@dataclass
class _TimedValue:
    timestamp: float
    value: float


@dataclass
class _ConsensusPoint:
    timestamp: float
    rip_amplitude_ratio: Optional[float]
    spo2_delta: Optional[float]


@dataclass
class _AasmBreath:
    start: float
    end: float
    amp: float
    baseline: float
    ratio: float
    confidence: float


@dataclass
class _AasmPendingEvent:
    start: float
    end: float
    min_ratio: float
    median_ratio: float
    strength: str
    breath_count: int
    created_at: float


@dataclass
class _AasmConfirmedEvent:
    start: float
    end: float
    rip_start: float
    rip_end: float
    confirm_time: float
    min_ratio: float
    median_ratio: float
    spo2_drop: float
    spo2_baseline: float
    spo2_nadir: float
    desat_start: float
    desat_end: float
    desat_duration: float
    strength: str
    breath_count: int


@dataclass
class _DirectLowRipEvent:
    start: float
    end: float
    min_ratio: float
    median_ratio: float
    breath_count: int


@dataclass
class PreExperimentController:
    """State machine for the pre-experiment acoustic intervention rule.

    The controller is split into two layers:

    - the trigger layer follows the streaming AASM-style detector plus the
      direct low-RIP run rule
    - the loudness layer preserves the original score-driven cue sizing once an
      intervention has started
    """

    config: PreExperimentConfig = field(default_factory=PreExperimentConfig)

    def __post_init__(self) -> None:
        if not self.config.loudness_levels:
            raise ValueError("loudness_levels must contain at least one value")
        if not 0.0 <= self.config.airflow_drop_threshold_fraction <= 1.0:
            raise ValueError("airflow_drop_threshold_fraction must be in [0, 1]")
        if not 0.0 <= self.config.trigger_score_stop_threshold <= 1.0:
            raise ValueError("trigger_score_stop_threshold must be in [0, 1]")
        if not 0.0 <= self.config.trigger_score_threshold <= 1.0:
            raise ValueError("trigger_score_threshold must be in [0, 1]")
        if not 0.0 <= self.config.high_risk_trigger_score_threshold <= 1.0:
            raise ValueError("high_risk_trigger_score_threshold must be in [0, 1]")
        if not 0.0 <= self.config.airflow_soft_gate_upper_ratio <= 1.0:
            raise ValueError("airflow_soft_gate_upper_ratio must be in [0, 1]")
        if not 0.0 <= self.config.baseline_update_trigger_score_max <= 1.0:
            raise ValueError("baseline_update_trigger_score_max must be in [0, 1]")
        if not 0.0 <= self.config.continue_score_stop_threshold <= 1.0:
            raise ValueError("continue_score_stop_threshold must be in [0, 1]")
        if self.config.continue_score_decay_sec <= 0.0:
            raise ValueError("continue_score_decay_sec must be > 0")
        if not 0.0 <= self.config.continue_score_memory_peak <= 1.0:
            raise ValueError("continue_score_memory_peak must be in [0, 1]")
        if self.config.continue_score_memory_decay_sec <= 0.0:
            raise ValueError("continue_score_memory_decay_sec must be > 0")
        if self.config.continue_score_recovery_hold_sec < 0.0:
            raise ValueError("continue_score_recovery_hold_sec must be >= 0")
        if self.config.continue_score_recovery_rip_ratio_min < 0.0:
            raise ValueError("continue_score_recovery_rip_ratio_min must be >= 0")
        if self.config.continue_score_recovery_period_ratio_max <= 0.0:
            raise ValueError("continue_score_recovery_period_ratio_max must be > 0")
        if self.config.continue_score_recovery_low_rip_run_max_sec < 0.0:
            raise ValueError("continue_score_recovery_low_rip_run_max_sec must be >= 0")
        if self.config.baseline_update_safe_duration_sec < 0.0:
            raise ValueError("baseline_update_safe_duration_sec must be >= 0")
        if self.config.trigger_duration_sec <= 0.0:
            raise ValueError("trigger_duration_sec must be > 0")
        if self.config.fast_trigger_duration_sec <= 0.0:
            raise ValueError("fast_trigger_duration_sec must be > 0")
        if self.config.apnea_min_duration_sec <= 0.0:
            raise ValueError("apnea_min_duration_sec must be > 0")
        if self.config.loudness_eval_window_sounds <= 0:
            raise ValueError("loudness_eval_window_sounds must be > 0")
        if self.config.loudness_recovery_windows_for_step_down <= 0:
            raise ValueError("loudness_recovery_windows_for_step_down must be > 0")
        if self.config.loudness_initial_level_index < 0:
            raise ValueError("loudness_initial_level_index must be >= 0")
        if self.config.loudness_high_risk_floor_index < 0:
            raise ValueError("loudness_high_risk_floor_index must be >= 0")
        if self.config.loudness_very_high_floor_index < 0:
            raise ValueError("loudness_very_high_floor_index must be >= 0")
        if self.config.trigger_score_stop_threshold > self.config.trigger_score_threshold:
            raise ValueError("stop threshold must be <= start threshold")
        if self.config.airflow_soft_gate_upper_ratio < self.config.airflow_ratio_threshold:
            raise ValueError("airflow_soft_gate_upper_ratio must be >= airflow_ratio_threshold")
        if self.config.baseline_update_trigger_score_max > self.config.trigger_score_stop_threshold:
            raise ValueError(
                "baseline_update_trigger_score_max must be <= trigger_score_stop_threshold"
            )
        if self.config.consensus_window_sec <= 0.0:
            raise ValueError("consensus_window_sec must be > 0")
        if not 0.0 <= self.config.consensus_severe_ratio_max <= 1.0:
            raise ValueError("consensus_severe_ratio_max must be in [0, 1]")
        if not 0.0 <= self.config.consensus_support_ratio_max <= 1.0:
            raise ValueError("consensus_support_ratio_max must be in [0, 1]")
        if self.config.consensus_support_ratio_max < self.config.consensus_severe_ratio_max:
            raise ValueError(
                "consensus_support_ratio_max must be >= consensus_severe_ratio_max"
            )
        if self.config.consensus_min_severe_hits < 0:
            raise ValueError("consensus_min_severe_hits must be >= 0")
        if self.config.consensus_min_support_hits < 0:
            raise ValueError("consensus_min_support_hits must be >= 0")
        if self.config.consensus_min_spo2_hits < 0:
            raise ValueError("consensus_min_spo2_hits must be >= 0")
        if self.config.consensus_spo2_delta_min_pct < 0.0:
            raise ValueError("consensus_spo2_delta_min_pct must be >= 0")
        if self.config.consensus_min_event_duration_sec <= 0.0:
            raise ValueError("consensus_min_event_duration_sec must be > 0")
        if not 0.0 < self.config.aasm_drop_ratio_strong <= 1.0:
            raise ValueError("aasm_drop_ratio_strong must be in (0, 1]")
        if not 0.0 < self.config.aasm_drop_ratio_weak <= 1.0:
            raise ValueError("aasm_drop_ratio_weak must be in (0, 1]")
        if not 0.0 < self.config.aasm_pending_event_ratio_threshold <= 1.0:
            raise ValueError("aasm_pending_event_ratio_threshold must be in (0, 1]")
        if not 0.0 < self.config.aasm_direct_trigger_ratio <= 1.0:
            raise ValueError("aasm_direct_trigger_ratio must be in (0, 1]")
        if self.config.aasm_drop_ratio_weak > self.config.aasm_direct_trigger_ratio:
            raise ValueError("aasm_drop_ratio_weak must be <= aasm_direct_trigger_ratio")
        if self.config.aasm_pending_event_ratio_threshold > self.config.aasm_direct_trigger_ratio:
            raise ValueError(
                "aasm_pending_event_ratio_threshold must be <= aasm_direct_trigger_ratio"
            )
        if self.config.aasm_drop_ratio_strong > self.config.aasm_drop_ratio_weak:
            raise ValueError("aasm_drop_ratio_strong must be <= aasm_drop_ratio_weak")
        if self.config.aasm_drop_ratio_strong > self.config.aasm_pending_event_ratio_threshold:
            raise ValueError(
                "aasm_drop_ratio_strong must be <= aasm_pending_event_ratio_threshold"
            )
        if self.config.aasm_min_low_rip_duration_sec <= 0.0:
            raise ValueError("aasm_min_low_rip_duration_sec must be > 0")
        if self.config.aasm_low_rip_merge_gap_sec < 0.0:
            raise ValueError("aasm_low_rip_merge_gap_sec must be >= 0")
        if self.config.aasm_direct_active_grace_sec < 0.0:
            raise ValueError("aasm_direct_active_grace_sec must be >= 0")
        if self.config.aasm_direct_min_breath_count <= 0:
            raise ValueError("aasm_direct_min_breath_count must be > 0")
        if not 0.0 < self.config.aasm_direct_current_rip_ratio_max <= 1.0:
            raise ValueError("aasm_direct_current_rip_ratio_max must be in (0, 1]")
        if self.config.aasm_spo2_baseline_window_sec <= 0.0:
            raise ValueError("aasm_spo2_baseline_window_sec must be > 0")
        if self.config.aasm_spo2_baseline_gap_sec < 0.0:
            raise ValueError("aasm_spo2_baseline_gap_sec must be >= 0")
        if self.config.aasm_spo2_min_baseline_samples <= 0:
            raise ValueError("aasm_spo2_min_baseline_samples must be > 0")
        if self.config.aasm_spo2_fallback_sample_count <= 0:
            raise ValueError("aasm_spo2_fallback_sample_count must be > 0")
        if self.config.aasm_spo2_fallback_min_samples <= 0:
            raise ValueError("aasm_spo2_fallback_min_samples must be > 0")
        if self.config.aasm_spo2_lookahead_sec < 0.0:
            raise ValueError("aasm_spo2_lookahead_sec must be >= 0")
        if self.config.aasm_spo2_drop_threshold_pct <= 0.0:
            raise ValueError("aasm_spo2_drop_threshold_pct must be > 0")
        if self.config.aasm_spo2_min_desat_duration_sec <= 0.0:
            raise ValueError("aasm_spo2_min_desat_duration_sec must be > 0")
        if self.config.aasm_event_merge_gap_sec < 0.0:
            raise ValueError("aasm_event_merge_gap_sec must be >= 0")
        if self.config.posture_baseline_reset_before_sec < 0.0:
            raise ValueError("posture_baseline_reset_before_sec must be >= 0")
        if self.config.posture_baseline_reset_after_sec < 0.0:
            raise ValueError("posture_baseline_reset_after_sec must be >= 0")
        if self.config.min_valid_rip_span < 0.0:
            raise ValueError("min_valid_rip_span must be >= 0")
        if self.config.max_valid_rip_span <= self.config.min_valid_rip_span:
            raise ValueError("max_valid_rip_span must be > min_valid_rip_span")
        self.reset()

    def reset(self) -> None:
        self._rip: Deque[_TimedValue] = deque()
        self._rip_baseline_amplitudes: Deque[_TimedValue] = deque()
        self._rip_baseline_periods: Deque[_TimedValue] = deque()
        self._spo2_samples: Deque[_TimedValue] = deque()
        self._spo2_baselines: Deque[_TimedValue] = deque()
        self._consensus_points: Deque[_ConsensusPoint] = deque()
        self._aasm_breaths: Deque[_AasmBreath] = deque()
        self._aasm_pending_events: Deque[_AasmPendingEvent] = deque()
        self._aasm_confirmed_events: Deque[_AasmConfirmedEvent] = deque()

        self._manual_rip_baseline: Optional[float] = None
        self._manual_breath_period_baseline: Optional[float] = None
        self._manual_spo2_baseline: Optional[float] = None
        self._frozen_rip_baseline: Optional[float] = None
        self._frozen_breath_period_baseline: Optional[float] = None
        self._frozen_spo2_baseline: Optional[float] = None
        self._baseline_safe_started_at: Optional[float] = None
        self._movement_freeze_until: Optional[float] = None

        self._phase = InterventionPhase.MONITORING
        self._disturbance_started_at: Optional[float] = None
        self._condition_started_at: Optional[float] = None
        self._high_risk_started_at: Optional[float] = None
        self._last_sound_at: Optional[float] = None
        self._last_emitted_breath_start = float("-inf")
        self._last_breathfinder_update = float("-inf")
        self._aasm_current_time = 0.0
        self._active_confirmed_event: Optional[_AasmConfirmedEvent] = None
        self._pending_confirmed_trigger: Optional[_AasmConfirmedEvent] = None
        self._active_direct_event: Optional[_DirectLowRipEvent] = None
        self._pending_direct_trigger: Optional[_DirectLowRipEvent] = None
        self._direct_trigger_history: Deque[_DirectLowRipEvent] = deque()
        self._rip_baseline_reset_after = float("-inf")
        self._last_posture_change_at: Optional[float] = None
        self._continue_score_state = 0.0
        self._continue_score_last_at: Optional[float] = None
        self._continue_score_memory_at: Optional[float] = None
        self._recovery_ready_started_at: Optional[float] = None
        self._loudness_level_index = 0
        self._sounds_since_loudness_eval = 0
        self._consecutive_recovery_windows = 0
        self._last_loudness_eval_snapshot: Optional[SensorSnapshot] = None

    def set_baseline(
        self,
        rip_amplitude: Optional[float] = None,
        breath_period_sec: Optional[float] = None,
        spo2_pct: Optional[float] = None,
    ) -> None:
        """Set manual baselines from a known stable breathing period."""

        if rip_amplitude is not None:
            if rip_amplitude <= 0:
                raise ValueError("rip_amplitude baseline must be positive")
            self._manual_rip_baseline = float(rip_amplitude)

        if breath_period_sec is not None:
            if breath_period_sec <= 0:
                raise ValueError("breath_period_sec baseline must be positive")
            self._manual_breath_period_baseline = float(breath_period_sec)

        if spo2_pct is not None:
            if not 50.0 <= float(spo2_pct) <= 100.0:
                raise ValueError("spo2_pct baseline must be in [50, 100]")
            self._manual_spo2_baseline = float(spo2_pct)

    def update(
        self,
        rip: SignalInput,
        button: Optional[SignalInput] = None,
        spo2_pct: Optional[Number] = None,
        timestamp: Optional[Number] = None,
        movement_detected: bool = False,
        posture_changed: bool = False,
    ) -> InterventionCommand:
        """Feed the newest RIP/SpO2/button values and return a loudness command.

        ``rip`` may be one sample or a batch of samples. When a batch is passed,
        samples are assumed to end at ``timestamp`` and to be spaced by
        ``1 / rip_fs``.

        ``button`` follows the same streaming style: it may be a scalar 0/1
        sample or a batch of 0/1 samples. Any value >= 0.5 in the current input
        batch is treated as a user-awake button press.

        ``spo2_pct`` is the most recent SpO2 percentage aligned to the packet
        end time. It is optional; the trigger layer can still use the direct
        low-RIP rule when no recent SpO2 is available, while the loudness layer
        falls back to RIP-only scoring.
        """

        ts = float(timestamp if timestamp is not None else time.time())
        rip_values = _coerce_signal_values(rip)
        signal_valid = self._is_valid_rip_values(rip_values)
        if signal_valid:
            self._append_rip(rip_values, ts)
        else:
            self._rip.clear()

        self._append_spo2(spo2_pct, ts)
        if posture_changed:
            self._handle_posture_change(ts)
        self._trim(ts)

        button_pressed = self._button_pressed(button)
        snapshot = self._snapshot(
            ts,
            button_pressed=button_pressed,
            signal_valid=signal_valid,
            movement_detected=bool(movement_detected),
        )

        if button_pressed:
            return self._stop_for_user_awake(snapshot)

        if not snapshot.signal_valid:
            return self._stop_for_invalid_signal(snapshot)

        self._update_condition_timers(snapshot)
        self._maybe_update_baselines(snapshot)

        if self._phase == InterventionPhase.INTERVENING:
            return self._intervention_command(snapshot)

        if self._pending_direct_trigger is not None:
            direct = self._pending_direct_trigger
            self._pending_direct_trigger = None
            self._start_intervention(ts, snapshot)
            return self._play_sound(
                snapshot,
                reason=(
                    "低幅 breath run 已持续至少10秒，直接开始声音干预"
                    f" (rip={direct.start:.2f}-{direct.end:.2f}s, min_ratio={direct.min_ratio:.2f})"
                ),
            )

        if self._pending_confirmed_trigger is not None:
            confirmed = self._pending_confirmed_trigger
            self._pending_confirmed_trigger = None
            self._start_intervention(ts, snapshot)
            return self._play_sound(
                snapshot,
                reason=(
                    "AASM 候选事件已获 SpO2 确认，开始声音干预"
                    f" (rip={confirmed.rip_start:.2f}-{confirmed.rip_end:.2f}s, spo2_drop={confirmed.spo2_drop:.2f})"
                ),
            )

        if snapshot.artifact_like_low_amplitude:
            self._phase = InterventionPhase.MONITORING
            return self._no_sound(
                snapshot,
                "RIP振幅显著降低但缺少稳定周期特征，先视为可疑信号而非低通气事件，跳过本次触发",
            )

        if snapshot.event_condition_met:
            self._phase = InterventionPhase.EVENT_PENDING
            return self._no_sound(
                snapshot,
                (
                    "AASM 候选事件已确认，等待进入干预状态"
                    + f" (duration={snapshot.event_condition_duration_sec:.2f}s, continue={snapshot.continue_score:.2f})"
                ),
            )

        self._phase = InterventionPhase.MONITORING
        return self._no_sound(
            snapshot,
            f"尚未满足触发条件 (continue={snapshot.continue_score:.2f})",
        )

    def update_loudness(
        self,
        rip: SignalInput,
        button: Optional[SignalInput] = None,
        spo2_pct: Optional[Number] = None,
        timestamp: Optional[Number] = None,
        posture_changed: bool = False,
    ) -> float:
        """Feed RIP/SpO2/button samples and return only the loudness output."""

        return self.update(
            rip=rip,
            button=button,
            spo2_pct=spo2_pct,
            timestamp=timestamp,
            posture_changed=posture_changed,
        ).loudness

    def _append_rip(
        self,
        rip: SignalInput,
        timestamp: float,
    ) -> None:
        if isinstance(rip, (int, float)):
            self._rip.append(_TimedValue(timestamp, float(rip)))
            return

        values = [float(v) for v in rip]
        if not values:
            return
        fs = max(self.config.rip_fs, 1e-6)
        n = len(values)
        for idx, value in enumerate(values):
            sample_t = timestamp - (n - 1 - idx) / fs
            self._rip.append(_TimedValue(sample_t, value))

    def _append_spo2(self, spo2_pct: Optional[Number], timestamp: float) -> None:
        value = _optional_float(spo2_pct)
        if value is None:
            return
        if not 50.0 <= value <= 100.0:
            return
        self._spo2_samples.append(_TimedValue(timestamp, value))

    def _handle_posture_change(self, timestamp: float) -> None:
        reset_after = float(timestamp) + float(self.config.posture_baseline_reset_after_sec)
        self._last_posture_change_at = float(timestamp)
        self._rip_baseline_reset_after = max(self._rip_baseline_reset_after, reset_after)

        clear_before = float(timestamp) - float(self.config.posture_baseline_reset_before_sec)
        while self._aasm_breaths and self._aasm_breaths[-1].start >= clear_before:
            self._aasm_breaths.pop()
        self._aasm_pending_events.clear()
        self._active_direct_event = None
        self._pending_direct_trigger = None
        self._pending_confirmed_trigger = None

    @staticmethod
    def _button_pressed(button: Optional[SignalInput]) -> bool:
        if button is None:
            return False
        if isinstance(button, (int, float)):
            return float(button) >= 0.5
        return any(float(value) >= 0.5 for value in button)

    def _trim(self, now: float) -> None:
        rip_cutoff = now - max(
            self.config.baseline_history_sec,
            self.config.rip_amplitude_window_sec,
            self.config.rip_period_window_sec,
        )
        while self._rip and self._rip[0].timestamp < rip_cutoff:
            self._rip.popleft()

        baseline_cutoff = now - self.config.baseline_history_sec
        while self._rip_baseline_amplitudes and self._rip_baseline_amplitudes[0].timestamp < baseline_cutoff:
            self._rip_baseline_amplitudes.popleft()
        while self._rip_baseline_periods and self._rip_baseline_periods[0].timestamp < baseline_cutoff:
            self._rip_baseline_periods.popleft()

        spo2_cutoff = now - max(
            self.config.spo2_baseline_history_sec,
            self.config.spo2_stale_after_sec,
        )
        while self._spo2_samples and self._spo2_samples[0].timestamp < spo2_cutoff:
            self._spo2_samples.popleft()

        spo2_baseline_cutoff = now - self.config.spo2_baseline_history_sec
        while self._spo2_baselines and self._spo2_baselines[0].timestamp < spo2_baseline_cutoff:
            self._spo2_baselines.popleft()

        consensus_cutoff = now - self.config.consensus_window_sec
        while self._consensus_points and self._consensus_points[0].timestamp < consensus_cutoff:
            self._consensus_points.popleft()

        pending_cutoff = now - (
            self.config.aasm_spo2_baseline_window_sec
            + self.config.aasm_spo2_lookahead_sec
            + self.config.aasm_min_low_rip_duration_sec
            + self.config.aasm_event_merge_gap_sec
        )
        while self._aasm_pending_events and self._aasm_pending_events[0].end < pending_cutoff:
            self._aasm_pending_events.popleft()
        while self._aasm_confirmed_events and self._aasm_confirmed_events[0].end < pending_cutoff:
            stale = self._aasm_confirmed_events.popleft()
            if self._active_confirmed_event is stale:
                self._active_confirmed_event = None
        while self._direct_trigger_history and self._direct_trigger_history[0].end < pending_cutoff:
            self._direct_trigger_history.popleft()

    def _snapshot(
        self,
        now: float,
        button_pressed: bool,
        signal_valid: bool,
        movement_detected: bool,
    ) -> SensorSnapshot:
        self._advance_aasm_detector(now=now, signal_valid=signal_valid, movement_detected=movement_detected)
        rip_amplitude = self._current_rip_amplitude(now)
        rip_baseline = self._rip_baseline()
        rip_ratio = None
        airflow_drop = None
        if rip_baseline is not None and rip_baseline > 0:
            rip_ratio = rip_amplitude / rip_baseline
            airflow_drop = max(0.0, 1.0 - rip_ratio)

        breath_period = self._current_breath_period(now)
        breath_period_baseline = self._breath_period_baseline()
        breath_period_ratio = None
        if breath_period is not None and breath_period_baseline is not None and breath_period_baseline > 0:
            breath_period_ratio = breath_period / breath_period_baseline
        period_feature_available = breath_period is not None

        current_spo2 = self._current_spo2(now)
        spo2_baseline = self._current_trigger_spo2_baseline(now)
        spo2_delta = None
        if current_spo2 is not None and spo2_baseline is not None:
            spo2_delta = max(0.0, spo2_baseline - current_spo2)

        active_confirmed_event = self._confirmed_event_for_snapshot(now)
        active_pending_event = self._current_pending_aasm_event(now)
        active_direct_event = self._current_active_direct_event(now)
        low_rip_run_active = active_pending_event is not None
        low_rip_run_duration = 0.0
        if active_pending_event is not None:
            low_rip_run_duration = max(0.0, active_pending_event.end - active_pending_event.start)
        if active_direct_event is not None:
            low_rip_run_active = True
            low_rip_run_duration = max(
                low_rip_run_duration,
                max(0.0, active_direct_event.end - active_direct_event.start),
            )
        aasm_candidate_strength = active_pending_event.strength if active_pending_event is not None else None
        if active_confirmed_event is not None:
            aasm_candidate_strength = active_confirmed_event.strength
        elif active_direct_event is not None:
            aasm_candidate_strength = (
                "strong"
                if active_direct_event.min_ratio <= self.config.aasm_drop_ratio_strong
                else "weak"
            )

        hypopnea_spo2_met = self._meets_hypopnea_spo2_drop(spo2_delta, spo2_baseline)
        apnea_like_drop_met = self._meets_apnea_like_drop(rip_ratio) and hypopnea_spo2_met
        artifact_like_low_amplitude = (
            self._meets_apnea_like_drop(rip_ratio)
            and not period_feature_available
            and (spo2_delta is None or spo2_delta < self.config.period_missing_spo2_rescue_threshold)
        )
        aasm_event_condition_met = active_confirmed_event is not None
        direct_trigger_met = active_direct_event is not None
        direct_trigger_duration = (
            max(0.0, active_direct_event.end - active_direct_event.start)
            if active_direct_event is not None
            else 0.0
        )
        aasm_event_condition_duration = (
            max(0.0, min(now, active_confirmed_event.rip_end) - active_confirmed_event.rip_start)
            if active_confirmed_event is not None
            else 0.0
        )
        event_condition_met = aasm_event_condition_met or direct_trigger_met
        disturbance_eligible = apnea_like_drop_met and not artifact_like_low_amplitude
        apnea_like_duration = self._current_disturbance_duration(now, disturbance_eligible)
        aasm_risk_score = self._aasm_risk_score(
            rip_amplitude_ratio=rip_ratio,
            low_rip_run_duration_sec=low_rip_run_duration,
            spo2_delta=spo2_delta,
            candidate_strength=aasm_candidate_strength,
            event_confirmed=event_condition_met,
        )
        raw_risk_score = aasm_risk_score
        continue_score = self._continue_score(
            now=now,
            raw_risk_score=aasm_risk_score,
            signal_valid=signal_valid,
            rip_amplitude_ratio=rip_ratio,
            breath_period_ratio=breath_period_ratio,
            low_rip_run_duration_sec=low_rip_run_duration,
            spo2_delta=spo2_delta,
            artifact_like_low_amplitude=artifact_like_low_amplitude,
            candidate_strength=aasm_candidate_strength,
            event_condition_met=event_condition_met,
        )
        loudness_score = self._loudness_score(continue_score=continue_score)
        duration = aasm_event_condition_duration if aasm_event_condition_met else direct_trigger_duration

        high_risk_duration = (
            duration
            if (active_confirmed_event is not None and aasm_candidate_strength == "strong")
            else 0.0
        )

        return SensorSnapshot(
            timestamp=now,
            rip_amplitude=rip_amplitude,
            rip_baseline=rip_baseline,
            rip_amplitude_ratio=rip_ratio,
            airflow_drop_fraction=airflow_drop,
            breath_period_sec=breath_period,
            breath_period_baseline=breath_period_baseline,
            breath_period_ratio=breath_period_ratio,
            period_feature_available=period_feature_available,
            spo2_pct=current_spo2,
            spo2_baseline=spo2_baseline,
            spo2_delta=spo2_delta,
            artifact_like_low_amplitude=artifact_like_low_amplitude,
            apnea_like_drop_met=apnea_like_drop_met,
            apnea_like_duration_sec=apnea_like_duration,
            raw_risk_score=raw_risk_score,
            continue_score=continue_score,
            loudness_score=loudness_score,
            signal_valid=signal_valid,
            movement_detected=movement_detected,
            button_pressed=button_pressed,
            low_rip_run_active=low_rip_run_active,
            low_rip_run_duration_sec=low_rip_run_duration,
            aasm_candidate_strength=aasm_candidate_strength,
            aasm_event_condition_met=aasm_event_condition_met,
            aasm_event_condition_duration_sec=aasm_event_condition_duration,
            direct_trigger_met=direct_trigger_met,
            direct_trigger_duration_sec=direct_trigger_duration,
            event_condition_met=event_condition_met,
            event_condition_duration_sec=duration,
            high_risk_duration_sec=high_risk_duration,
        )

    def _current_rip_amplitude(self, now: float) -> float:
        cutoff = now - self.config.rip_amplitude_window_sec
        values = self._smoothed_rip_values_since(cutoff)
        if len(values) < 2:
            return 0.0
        return _percentile(values, 95.0) - _percentile(values, 5.0)

    def _current_breath_period(self, now: float) -> Optional[float]:
        cutoff = now - self.config.rip_period_window_sec
        points = [item for item in self._rip if item.timestamp >= cutoff]
        if len(points) < 5:
            return None

        values = _smooth_series(
            [item.value for item in points],
            window=self._rip_smoothing_window_points(),
        )
        times = [item.timestamp for item in points]

        peak_periods = _candidate_cycle_periods(times, values)
        trough_periods = _candidate_cycle_periods(times, [-value for value in values])
        periods = peak_periods if len(peak_periods) >= len(trough_periods) else trough_periods
        if not periods:
            return None
        return max(_percentile(periods, 50.0), 1e-6)

    def _current_spo2(self, now: float) -> Optional[float]:
        if not self._spo2_samples:
            return None
        latest = self._spo2_samples[-1]
        if now - latest.timestamp > self.config.spo2_stale_after_sec:
            return None
        return latest.value

    def _rip_baseline(self) -> Optional[float]:
        return self._current_aasm_rip_baseline(self._aasm_current_time)

    def _breath_period_baseline(self) -> Optional[float]:
        if self._manual_breath_period_baseline is not None:
            return self._manual_breath_period_baseline
        if self._frozen_breath_period_baseline is not None:
            return self._frozen_breath_period_baseline

        values = [item.value for item in self._rip_baseline_periods if item.value > 0]
        if len(values) < 2:
            return None
        return max(_percentile(values, 50.0), 1e-6)

    def _spo2_baseline(self) -> Optional[float]:
        return self._stream_like_spo2_baseline(self._aasm_current_time)

    def _current_rip_baseline_value(self) -> Optional[float]:
        baseline = self._current_aasm_rip_baseline(self._aasm_current_time)
        if baseline is None or not _isfinite(baseline) or baseline <= 0.0:
            return None
        return float(baseline)

    def _current_breath_period_baseline_value(self) -> Optional[float]:
        values = [item.value for item in self._rip_baseline_periods if item.value > 0]
        if len(values) < 2:
            return None
        return max(_percentile(values, 50.0), 1e-6)

    def _current_spo2_baseline_value(self) -> Optional[float]:
        return self._stream_like_spo2_baseline(self._aasm_current_time)

    def _set_baseline_frozen(self, frozen: bool) -> None:
        if frozen:
            if self._frozen_rip_baseline is None:
                self._frozen_rip_baseline = self._current_rip_baseline_value()
            if self._frozen_breath_period_baseline is None:
                self._frozen_breath_period_baseline = self._current_breath_period_baseline_value()
            if self._frozen_spo2_baseline is None:
                self._frozen_spo2_baseline = self._current_spo2_baseline_value()
            return

        self._frozen_rip_baseline = None
        self._frozen_breath_period_baseline = None
        self._frozen_spo2_baseline = None

    def _mark_baseline_unsafe(self) -> None:
        self._baseline_safe_started_at = None
        self._set_baseline_frozen(True)

    def _current_disturbance_duration(self, now: float, suspicious: bool) -> float:
        if not suspicious or self._disturbance_started_at is None:
            return 0.0
        return max(0.0, now - self._disturbance_started_at)

    def _advance_aasm_detector(
        self,
        *,
        now: float,
        signal_valid: bool,
        movement_detected: bool,
    ) -> None:
        self._aasm_current_time = float(now)
        if not signal_valid or movement_detected or _BREATHFINDER is None:
            return
        if now - self._last_breathfinder_update >= self.config.aasm_breathfinder_update_sec:
            self._update_aasm_breaths(now)
            self._last_breathfinder_update = float(now)
        self._update_pending_aasm_from_breaths(now)
        self._update_direct_low_rip_trigger(now)
        self._confirm_pending_with_spo2(now)

    def _current_rip_ratio(self, now: float) -> Optional[float]:
        baseline = self._current_aasm_rip_baseline(now)
        if baseline is None or baseline <= 0.0:
            return None
        amplitude = self._current_rip_amplitude(now)
        return amplitude / baseline

    def _update_aasm_breaths(self, now: float) -> None:
        if _BREATHFINDER is None:
            return

        signal = np.asarray([item.value for item in self._rip], dtype=np.float64)
        if len(signal) < int(round(max(12.0 * self.config.rip_fs, 1.0))):
            return

        rip_items = list(self._rip)
        buffer_samples = min(
            len(signal),
            int(round(self.config.aasm_breath_buffer_sec * self.config.rip_fs)),
        )
        if buffer_samples <= 0:
            return

        buffer = signal[-buffer_samples:]
        buffer_start_abs = rip_items[-buffer_samples].timestamp
        safe_until_abs = now - self.config.aasm_breath_edge_guard_sec
        if safe_until_abs <= self._last_emitted_breath_start:
            return

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                detected = _BREATHFINDER.find_breaths(buffer, self.config.rip_fs)
        except Exception:
            return

        for raw in detected:
            start = buffer_start_abs + float(raw[0])
            duration = float(raw[1])
            end = start + duration
            if start <= self._last_emitted_breath_start + 0.25:
                continue
            if end > safe_until_abs:
                continue

            amp = self._robust_breath_amp(start, end)
            baseline = self._aasm_rip_baseline_for_breath(start)
            ratio = (
                amp / baseline
                if _isfinite(amp) and _isfinite(baseline) and baseline > 0.0
                else float("nan")
            )
            confidence = float(raw[2]) if len(raw) > 2 else float("nan")
            self._aasm_breaths.append(
                _AasmBreath(
                    start=float(start),
                    end=float(end),
                    amp=float(amp),
                    baseline=float(baseline),
                    ratio=float(ratio),
                    confidence=float(confidence),
                )
            )
            self._last_emitted_breath_start = float(start)

        cutoff = now - (
            self.config.aasm_rip_baseline_window_sec
            + self.config.aasm_spo2_lookahead_sec
            + 300.0
        )
        while self._aasm_breaths and self._aasm_breaths[0].end < cutoff:
            self._aasm_breaths.popleft()

    def _robust_breath_amp(self, start: float, end: float) -> float:
        samples = [
            item.value
            for item in self._rip
            if start <= item.timestamp <= end
        ]
        if len(samples) < 3:
            return float("nan")
        arr = np.asarray(samples, dtype=np.float64)
        return float(np.percentile(arr, 95.0) - np.percentile(arr, 5.0))

    def _aasm_rip_baseline_for_breath(self, at_time: float) -> float:
        vals = [
            breath.amp
            for breath in self._aasm_breaths
            if (
                breath.start >= self._rip_baseline_reset_after
                and
                at_time - self.config.aasm_rip_baseline_window_sec <= breath.start
                <= at_time - self.config.aasm_rip_baseline_guard_sec
                and _isfinite(breath.amp)
            )
        ]
        if len(vals) >= self.config.aasm_rip_min_baseline_breaths:
            return float(np.median(np.asarray(vals, dtype=np.float64)))

        fallback = [
            breath.amp
            for breath in list(self._aasm_breaths)[-self.config.aasm_rip_fallback_breath_count :]
            if _isfinite(breath.amp) and breath.start >= self._rip_baseline_reset_after
        ]
        if len(fallback) >= self.config.aasm_rip_fallback_min_breaths:
            return float(np.median(np.asarray(fallback, dtype=np.float64)))
        return float("nan")

    def _current_aasm_rip_baseline(self, now: float) -> Optional[float]:
        del now
        if self._manual_rip_baseline is not None:
            return self._manual_rip_baseline

        if self._aasm_breaths:
            baseline = self._aasm_rip_baseline_for_breath(self._aasm_current_time)
            if _isfinite(baseline) and baseline > 0.0:
                return float(baseline)
        return None

    def _update_pending_aasm_from_breaths(self, now: float) -> None:
        del now
        low_breaths = [
            breath
            for breath in self._aasm_breaths
            if _isfinite(breath.ratio)
            and breath.ratio <= self.config.aasm_pending_event_ratio_threshold
        ]
        if not low_breaths:
            return

        runs: list[list[_AasmBreath]] = []
        current: list[_AasmBreath] = []
        for breath in low_breaths:
            if not current or breath.start - current[-1].end <= self.config.aasm_low_rip_merge_gap_sec:
                current.append(breath)
            else:
                runs.append(current)
                current = [breath]
        if current:
            runs.append(current)

        existing = (
            [(pending.start, pending.end) for pending in self._aasm_pending_events]
            + [(event.rip_start, event.rip_end) for event in self._aasm_confirmed_events]
        )
        for run in runs:
            start = run[0].start
            end = run[-1].end
            if end - start < self.config.aasm_min_low_rip_duration_sec:
                continue
            if any(
                abs(start - event_start) <= self.config.aasm_event_merge_gap_sec
                or (start < event_end and end > event_start)
                for event_start, event_end in existing
            ):
                continue

            ratios = np.asarray([breath.ratio for breath in run], dtype=np.float64)
            min_ratio = float(np.nanmin(ratios))
            median_ratio = float(np.nanmedian(ratios))
            strength = "strong" if min_ratio <= self.config.aasm_drop_ratio_strong else "weak"
            self._aasm_pending_events.append(
                _AasmPendingEvent(
                    start=float(start),
                    end=float(end),
                    min_ratio=min_ratio,
                    median_ratio=median_ratio,
                    strength=strength,
                    breath_count=len(run),
                    created_at=self._aasm_current_time,
                )
            )
            existing.append((start, end))

    def _update_direct_low_rip_trigger(self, now: float) -> None:
        del now
        low_breaths = [
            breath
            for breath in self._aasm_breaths
            if _isfinite(breath.ratio) and breath.ratio <= self.config.aasm_direct_trigger_ratio
        ]
        if not low_breaths:
            self._active_direct_event = None
            return

        runs: list[list[_AasmBreath]] = []
        current: list[_AasmBreath] = []
        for breath in low_breaths:
            if not current or breath.start - current[-1].end <= self.config.aasm_low_rip_merge_gap_sec:
                current.append(breath)
            else:
                runs.append(current)
                current = [breath]
        if current:
            runs.append(current)

        active_run: Optional[list[_AasmBreath]] = None
        for run in reversed(runs):
            if self._aasm_current_time - run[-1].end <= self.config.aasm_direct_active_grace_sec:
                active_run = run
                break
        if active_run is None:
            self._active_direct_event = None
            return

        if len(active_run) < self.config.aasm_direct_min_breath_count:
            self._active_direct_event = None
            return

        start = active_run[0].start
        end = active_run[-1].end
        if end - start < self.config.aasm_min_low_rip_duration_sec:
            self._active_direct_event = None
            return

        current_rip_ratio = self._current_rip_ratio(self._aasm_current_time)
        if (
            current_rip_ratio is None
            or not _isfinite(current_rip_ratio)
            or current_rip_ratio > self.config.aasm_direct_current_rip_ratio_max
        ):
            self._active_direct_event = None
            return

        ratios = np.asarray([breath.ratio for breath in active_run], dtype=np.float64)
        event = _DirectLowRipEvent(
            start=float(start),
            end=float(end),
            min_ratio=float(np.nanmin(ratios)),
            median_ratio=float(np.nanmedian(ratios)),
            breath_count=len(active_run),
        )

        already_triggered = any(
            not (
                event.end < past.start - self.config.aasm_event_merge_gap_sec
                or event.start > past.end + self.config.aasm_event_merge_gap_sec
            )
            for past in self._direct_trigger_history
        )
        if not already_triggered:
            self._pending_direct_trigger = event
            self._direct_trigger_history.append(event)
        self._active_direct_event = event

    def _confirm_pending_with_spo2(self, now: float) -> None:
        if not self._aasm_pending_events:
            return

        remaining: Deque[_AasmPendingEvent] = deque()
        for pending in self._aasm_pending_events:
            if now < pending.end + self.config.aasm_spo2_lookahead_sec:
                remaining.append(pending)
                continue

            baseline = self._aasm_spo2_baseline_for_event(pending.start)
            ok, nadir, desat_start, desat_end, desat_duration = self._sustained_desat_for_event(
                event_start=pending.start,
                event_end=pending.end,
                baseline=baseline,
            )
            if not ok:
                continue

            spo2_drop = baseline - nadir
            confirmed = _AasmConfirmedEvent(
                start=pending.start,
                end=pending.end + self.config.aasm_spo2_lookahead_sec,
                rip_start=pending.start,
                rip_end=pending.end,
                confirm_time=now,
                min_ratio=pending.min_ratio,
                median_ratio=pending.median_ratio,
                spo2_drop=float(spo2_drop),
                spo2_baseline=float(baseline),
                spo2_nadir=float(nadir),
                desat_start=float(desat_start),
                desat_end=float(desat_end),
                desat_duration=float(desat_duration),
                strength=pending.strength,
                breath_count=pending.breath_count,
            )
            self._aasm_confirmed_events.append(confirmed)
            self._active_confirmed_event = confirmed
            if self._phase != InterventionPhase.INTERVENING:
                self._pending_confirmed_trigger = confirmed

        self._aasm_pending_events = remaining

    def _current_pending_aasm_event(self, now: float) -> Optional[_AasmPendingEvent]:
        active: Optional[_AasmPendingEvent] = None
        for pending in reversed(self._aasm_pending_events):
            if pending.start <= now <= pending.end:
                active = pending
                break
        return active

    def _current_active_direct_event(self, now: float) -> Optional[_DirectLowRipEvent]:
        active = self._active_direct_event
        if active is None:
            return None
        if now - active.end <= self.config.aasm_direct_active_grace_sec:
            return active
        self._active_direct_event = None
        return None

    def _aasm_spo2_baseline_for_event(self, event_start: float) -> Optional[float]:
        return self._stream_like_spo2_baseline(event_start)

    def _current_trigger_spo2_baseline(self, now: float) -> Optional[float]:
        if self._manual_spo2_baseline is not None:
            return self._manual_spo2_baseline
        return self._stream_like_spo2_baseline(now)

    def _stream_like_spo2_baseline(self, at_time: float) -> Optional[float]:
        lower = at_time - self.config.aasm_spo2_baseline_window_sec
        upper = at_time - self.config.aasm_spo2_baseline_gap_sec
        values = [
            sample.value
            for sample in self._spo2_samples
            if lower <= sample.timestamp <= upper
        ]
        if len(values) >= self.config.aasm_spo2_min_baseline_samples:
            return _percentile(values, 50.0)

        fallback = [
            sample.value
            for sample in list(self._spo2_samples)[-self.config.aasm_spo2_fallback_sample_count :]
            if sample.timestamp <= upper
        ]
        if len(fallback) >= self.config.aasm_spo2_fallback_min_samples:
            return _percentile(fallback, 50.0)
        return None

    def _sustained_desat_for_event(
        self,
        *,
        event_start: float,
        event_end: float,
        baseline: Optional[float],
    ) -> tuple[bool, float, float, float, float]:
        if baseline is None or not _isfinite(baseline):
            return False, float("nan"), float("nan"), float("nan"), float("nan")

        window_end = event_end + self.config.aasm_spo2_lookahead_sec
        samples = [
            sample
            for sample in self._spo2_samples
            if event_start <= sample.timestamp <= window_end
        ]
        if not samples:
            return False, float("nan"), float("nan"), float("nan"), float("nan")

        threshold = baseline - self.config.aasm_spo2_drop_threshold_pct
        nadir = min(sample.value for sample in samples)
        below = [sample for sample in samples if sample.value <= threshold]
        if not below:
            return False, float(nadir), float("nan"), float("nan"), 0.0

        best_start = below[0].timestamp
        best_end = below[0].timestamp
        current_start = below[0].timestamp
        prev_time = below[0].timestamp

        for sample in below[1:]:
            if sample.timestamp - prev_time <= 1.5:
                prev_time = sample.timestamp
                continue
            if (prev_time - current_start) > (best_end - best_start):
                best_start = current_start
                best_end = prev_time
            current_start = sample.timestamp
            prev_time = sample.timestamp
        if (prev_time - current_start) > (best_end - best_start):
            best_start = current_start
            best_end = prev_time

        desat_duration = max(0.0, best_end - best_start) + 1.0
        ok = desat_duration >= self.config.aasm_spo2_min_desat_duration_sec
        return ok, float(nadir), float(best_start), float(best_end), float(desat_duration)

    def _current_active_confirmed_event(self, now: float) -> Optional[_AasmConfirmedEvent]:
        active = self._active_confirmed_event
        if active is not None and now <= active.end:
            return active

        self._active_confirmed_event = None
        for event in reversed(self._aasm_confirmed_events):
            if event.start <= now <= event.end:
                self._active_confirmed_event = event
                return event
        return None

    def _confirmed_event_for_snapshot(self, now: float) -> Optional[_AasmConfirmedEvent]:
        active = self._current_active_confirmed_event(now)
        if active is not None:
            return active
        pending = self._pending_confirmed_trigger
        if pending is None:
            return None
        if abs(float(now) - float(pending.confirm_time)) <= max(self.config.sound_interval_sec, 1e-3):
            return pending
        return None

    def _meets_apnea_like_drop(self, rip_amplitude_ratio: Optional[float]) -> bool:
        if rip_amplitude_ratio is None:
            return False
        return rip_amplitude_ratio <= self.config.airflow_ratio_threshold

    def _meets_hypopnea_spo2_drop(
        self,
        spo2_delta: Optional[float],
        spo2_baseline: Optional[float],
    ) -> bool:
        del spo2_baseline
        if spo2_delta is None:
            return False
        return float(spo2_delta) >= float(self.config.spo2_hypopnea_threshold_pct)

    def _aasm_risk_score(
        self,
        *,
        rip_amplitude_ratio: Optional[float],
        low_rip_run_duration_sec: float,
        spo2_delta: Optional[float],
        candidate_strength: Optional[str],
        event_confirmed: bool,
    ) -> float:
        ratio = None
        if rip_amplitude_ratio is not None and _isfinite(rip_amplitude_ratio):
            ratio = max(float(rip_amplitude_ratio), 0.0)
        duration = max(float(low_rip_run_duration_sec), 0.0)
        desat = 0.0 if spo2_delta is None else max(float(spo2_delta), 0.0)

        score = 0.0
        if ratio is not None:
            if ratio <= self.config.aasm_drop_ratio_strong:
                score = max(score, 0.82)
            elif ratio <= self.config.aasm_drop_ratio_weak:
                progress = (self.config.aasm_drop_ratio_weak - ratio) / max(
                    self.config.aasm_drop_ratio_weak - self.config.aasm_drop_ratio_strong,
                    1e-6,
                )
                score = max(score, 0.58 + 0.20 * progress)
            elif ratio <= self.config.aasm_direct_trigger_ratio:
                progress = (self.config.aasm_direct_trigger_ratio - ratio) / max(
                    self.config.aasm_direct_trigger_ratio - self.config.aasm_drop_ratio_weak,
                    1e-6,
                )
                score = max(score, 0.35 + 0.20 * progress)

        if duration >= self.config.aasm_min_low_rip_duration_sec:
            duration_progress = min(
                (duration - self.config.aasm_min_low_rip_duration_sec)
                / max(self.config.fast_trigger_duration_sec, 1e-6),
                1.0,
            )
            if ratio is not None and ratio <= self.config.aasm_direct_trigger_ratio:
                score = max(score, 0.68 + 0.14 * max(duration_progress, 0.0))

        if candidate_strength == "strong":
            score = max(score, 0.85)

        if desat >= self.config.aasm_spo2_drop_threshold_pct:
            extra = min((desat - self.config.aasm_spo2_drop_threshold_pct) / 2.0, 1.0)
            score = max(score, 0.80 + 0.15 * max(extra, 0.0))

        if event_confirmed:
            score = max(score, 0.85 if candidate_strength == "strong" else 0.75)

        return min(max(score, 0.0), 1.0)

    def _windowed_consensus_met(
        self,
        *,
        now: float,
        signal_valid: bool,
        artifact_like_low_amplitude: bool,
        rip_amplitude_ratio: Optional[float],
        spo2_delta: Optional[float],
    ) -> bool:
        if not signal_valid or artifact_like_low_amplitude:
            self._consensus_points.clear()
            return False

        self._consensus_points.append(
            _ConsensusPoint(
                timestamp=now,
                rip_amplitude_ratio=rip_amplitude_ratio,
                spo2_delta=spo2_delta,
            )
        )
        cutoff = now - self.config.consensus_window_sec
        while self._consensus_points and self._consensus_points[0].timestamp < cutoff:
            self._consensus_points.popleft()

        severe_hits = 0
        support_hits = 0
        spo2_hits = 0
        for point in self._consensus_points:
            if (
                point.rip_amplitude_ratio is not None
                and point.rip_amplitude_ratio <= self.config.consensus_severe_ratio_max
            ):
                severe_hits += 1
            if (
                point.rip_amplitude_ratio is not None
                and point.rip_amplitude_ratio <= self.config.consensus_support_ratio_max
            ):
                support_hits += 1
            if (
                point.spo2_delta is not None
                and point.spo2_delta >= self.config.consensus_spo2_delta_min_pct
            ):
                spo2_hits += 1

        return (
            severe_hits >= self.config.consensus_min_severe_hits
            and support_hits >= self.config.consensus_min_support_hits
            and spo2_hits >= self.config.consensus_min_spo2_hits
        )

    def _continue_score(
        self,
        *,
        now: float,
        raw_risk_score: float,
        signal_valid: bool,
        rip_amplitude_ratio: Optional[float],
        breath_period_ratio: Optional[float],
        low_rip_run_duration_sec: float,
        spo2_delta: Optional[float],
        artifact_like_low_amplitude: bool,
        candidate_strength: Optional[str],
        event_condition_met: bool,
    ) -> float:
        if event_condition_met:
            self._continue_score_memory_at = float(now)

        if not signal_valid:
            self._continue_score_state = 0.0
            self._continue_score_last_at = float(now)
            return 0.0

        amp_need = self._continue_amp_need(rip_amplitude_ratio)
        period_need = self._continue_period_need(breath_period_ratio)
        run_need = self._continue_run_need(low_rip_run_duration_sec)
        spo2_need = self._continue_spo2_need(spo2_delta)
        memory_need = self._continue_memory_need(now)

        target = (
            0.40 * amp_need
            + 0.15 * period_need
            + 0.15 * run_need
            + 0.15 * spo2_need
            + 0.15 * memory_need
        )
        target = max(target, 0.25 * min(max(float(raw_risk_score), 0.0), 1.0))

        if event_condition_met:
            floor = 0.82 if candidate_strength == "strong" else 0.72
            target = max(target, floor)
        elif low_rip_run_duration_sec >= self.config.aasm_min_low_rip_duration_sec:
            target = max(target, 0.55)

        if artifact_like_low_amplitude:
            cap = max(0.0, self.config.continue_score_stop_threshold - 1e-3)
            target = min(target, cap)

        target = min(max(float(target), 0.0), 1.0)
        previous = min(max(float(self._continue_score_state), 0.0), 1.0)
        if self._continue_score_last_at is None or target >= previous:
            score = target
        else:
            elapsed = max(float(now) - float(self._continue_score_last_at), 0.0)
            alpha = 1.0 - np.exp(-elapsed / max(self.config.continue_score_decay_sec, 1e-6))
            score = previous + alpha * (target - previous)

        self._continue_score_state = min(max(float(score), 0.0), 1.0)
        self._continue_score_last_at = float(now)
        return self._continue_score_state

    def _continue_amp_need(self, rip_amplitude_ratio: Optional[float]) -> float:
        if rip_amplitude_ratio is None or not _isfinite(rip_amplitude_ratio):
            return 0.0
        ratio = max(float(rip_amplitude_ratio), 0.0)
        if ratio <= 0.30:
            return 1.0
        if ratio <= 0.45:
            return _lerp(ratio, 0.30, 0.45, 1.00, 0.80)
        if ratio <= 0.60:
            return _lerp(ratio, 0.45, 0.60, 0.80, 0.50)
        if ratio <= 0.75:
            return _lerp(ratio, 0.60, 0.75, 0.50, 0.20)
        if ratio <= 0.90:
            return _lerp(ratio, 0.75, 0.90, 0.20, 0.00)
        return 0.0

    def _continue_period_need(self, breath_period_ratio: Optional[float]) -> float:
        if breath_period_ratio is None or not _isfinite(breath_period_ratio):
            return 0.0
        ratio = max(float(breath_period_ratio), 0.0)
        if ratio <= 1.05:
            return 0.0
        if ratio <= 1.20:
            return _lerp(ratio, 1.05, 1.20, 0.0, 0.35)
        if ratio <= 1.35:
            return _lerp(ratio, 1.20, 1.35, 0.35, 0.70)
        if ratio <= 1.55:
            return _lerp(ratio, 1.35, 1.55, 0.70, 1.00)
        return 1.0

    def _continue_run_need(self, low_rip_run_duration_sec: float) -> float:
        duration = max(float(low_rip_run_duration_sec), 0.0)
        if duration <= 2.0:
            return 0.0
        if duration <= 4.0:
            return _lerp(duration, 2.0, 4.0, 0.0, 0.30)
        if duration <= self.config.aasm_min_low_rip_duration_sec:
            return _lerp(duration, 4.0, self.config.aasm_min_low_rip_duration_sec, 0.30, 0.70)
        if duration <= self.config.aasm_min_low_rip_duration_sec + 6.0:
            return _lerp(
                duration,
                self.config.aasm_min_low_rip_duration_sec,
                self.config.aasm_min_low_rip_duration_sec + 6.0,
                0.70,
                1.00,
            )
        return 1.0

    def _continue_spo2_need(self, spo2_delta: Optional[float]) -> float:
        if spo2_delta is None or not _isfinite(spo2_delta):
            return 0.0
        delta = max(float(spo2_delta), 0.0)
        if delta <= 1.0:
            return 0.0
        if delta <= 3.0:
            return _lerp(delta, 1.0, 3.0, 0.0, 0.60)
        if delta <= 5.0:
            return _lerp(delta, 3.0, 5.0, 0.60, 1.00)
        return 1.0

    def _continue_memory_need(self, now: float) -> float:
        if self._continue_score_memory_at is None:
            return 0.0
        elapsed = max(float(now) - float(self._continue_score_memory_at), 0.0)
        decay = np.exp(-elapsed / max(self.config.continue_score_memory_decay_sec, 1e-6))
        return min(
            max(self.config.continue_score_memory_peak * float(decay), 0.0),
            1.0,
        )

    def _soft_gate_cap_for_ratio(self, rip_amplitude_ratio: Optional[float]) -> float:
        stop_cap = max(0.0, self.config.trigger_score_stop_threshold - 1e-3)
        start_cap = max(stop_cap, self.config.trigger_score_threshold - 1e-3)
        if rip_amplitude_ratio is None:
            return stop_cap

        ratio = max(float(rip_amplitude_ratio), 0.0)
        lower = self.config.airflow_ratio_threshold
        upper = self.config.airflow_soft_gate_upper_ratio
        if ratio >= upper:
            return stop_cap
        if ratio <= lower:
            return start_cap

        progress = (upper - ratio) / max(upper - lower, 1e-6)
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        return stop_cap + smooth_progress * (start_cap - stop_cap)

    @property
    def last_confirmed_aasm_event(self) -> Optional[dict[str, object]]:
        event = self._aasm_confirmed_events[-1] if self._aasm_confirmed_events else None
        if event is None:
            return None
        return {
            "start": event.start,
            "end": event.end,
            "rip_start": event.rip_start,
            "rip_end": event.rip_end,
            "confirm_time": event.confirm_time,
            "min_ratio": event.min_ratio,
            "median_ratio": event.median_ratio,
            "spo2_drop": event.spo2_drop,
            "spo2_baseline": event.spo2_baseline,
            "spo2_nadir": event.spo2_nadir,
            "desat_start": event.desat_start,
            "desat_end": event.desat_end,
            "desat_duration": event.desat_duration,
            "strength": event.strength,
        }

    def _update_condition_timers(self, snapshot: SensorSnapshot) -> None:
        if not snapshot.signal_valid:
            self._disturbance_started_at = None
            self._condition_started_at = None
            self._high_risk_started_at = None
            return

        if snapshot.apnea_like_drop_met and not snapshot.artifact_like_low_amplitude:
            if self._disturbance_started_at is None:
                self._disturbance_started_at = snapshot.timestamp
        else:
            self._disturbance_started_at = None

        self._high_risk_started_at = (
            snapshot.timestamp - snapshot.high_risk_duration_sec
            if snapshot.high_risk_duration_sec > 0.0
            else None
        )
        self._condition_started_at = (
            snapshot.timestamp - snapshot.event_condition_duration_sec
            if snapshot.event_condition_duration_sec > 0.0
            else None
        )

    def _maybe_update_baselines(self, snapshot: SensorSnapshot) -> None:
        self._set_baseline_frozen(False)
        if snapshot.breath_period_sec is not None and snapshot.breath_period_sec > 0:
            self._rip_baseline_periods.append(
                _TimedValue(snapshot.timestamp, snapshot.breath_period_sec)
            )
        if snapshot.spo2_pct is not None:
            self._spo2_baselines.append(_TimedValue(snapshot.timestamp, snapshot.spo2_pct))

    def _start_intervention(self, timestamp: float, snapshot: SensorSnapshot) -> None:
        self._phase = InterventionPhase.INTERVENING
        self._last_sound_at = None
        self._pending_direct_trigger = None
        self._pending_confirmed_trigger = None
        self._continue_score_memory_at = float(timestamp)
        self._recovery_ready_started_at = None
        self._initialize_intervention_loudness(snapshot)
        if self._condition_started_at is None:
            self._condition_started_at = timestamp

    def _intervention_command(self, snapshot: SensorSnapshot) -> InterventionCommand:
        if not snapshot.signal_valid:
            self._phase = InterventionPhase.RECOVERED
            self._disturbance_started_at = None
            self._condition_started_at = None
            self._high_risk_started_at = None
            return self._no_sound(snapshot, "RIP信号无效，停止声音干预")

        if snapshot.artifact_like_low_amplitude:
            self._phase = InterventionPhase.RECOVERED
            self._disturbance_started_at = None
            self._condition_started_at = None
            self._high_risk_started_at = None
            self._recovery_ready_started_at = None
            return self._no_sound(
                snapshot,
                "RIP振幅极低但缺少稳定周期特征，视为可疑信号，停止声音干预",
            )

        if self._recovery_ready(snapshot):
            if self._recovery_ready_started_at is None:
                self._recovery_ready_started_at = snapshot.timestamp
            elif (
                snapshot.timestamp - self._recovery_ready_started_at
                >= self.config.continue_score_recovery_hold_sec
            ):
                self._phase = InterventionPhase.RECOVERED
                self._disturbance_started_at = None
                self._condition_started_at = None
                self._high_risk_started_at = None
                self._recovery_ready_started_at = None
                return self._no_sound(
                    snapshot,
                    (
                        "呼吸已连续恢复稳定，停止声音干预"
                        f" (continue={snapshot.continue_score:.2f})"
                    ),
                )
        else:
            self._recovery_ready_started_at = None

        if not snapshot.event_condition_met and snapshot.continue_score <= self.config.continue_score_stop_threshold:
            self._phase = InterventionPhase.RECOVERED
            self._disturbance_started_at = None
            self._condition_started_at = None
            self._high_risk_started_at = None
            self._recovery_ready_started_at = None
            return self._no_sound(
                snapshot,
                (
                    "干预需求已明显回落，停止声音干预"
                    f" (continue={snapshot.continue_score:.2f})"
                ),
            )

        due = (
            self._last_sound_at is None
            or snapshot.timestamp - self._last_sound_at >= self.config.sound_interval_sec
        )
        if not due:
            return self._no_sound(snapshot, "等待固定0.6秒播放间隔")

        self._update_intervention_loudness(snapshot)
        return self._play_sound(
            snapshot,
            (
                "按固定间隔继续声音干预，当前响度由瞬时风险和恢复趋势共同决定"
                f" (continue={snapshot.continue_score:.2f}, loudness={snapshot.loudness_score:.2f})"
            ),
        )

    def _recovery_ready(self, snapshot: SensorSnapshot) -> bool:
        if snapshot.continue_score > self.config.continue_score_stop_threshold:
            return False

        amp_ok = (
            snapshot.rip_amplitude_ratio is not None
            and snapshot.rip_amplitude_ratio >= self.config.continue_score_recovery_rip_ratio_min
        )
        if not amp_ok:
            return False

        if snapshot.low_rip_run_duration_sec > self.config.continue_score_recovery_low_rip_run_max_sec:
            return False

        if (
            snapshot.breath_period_ratio is not None
            and snapshot.breath_period_ratio > self.config.continue_score_recovery_period_ratio_max
        ):
            return False

        return True

    def _stop_for_user_awake(self, snapshot: SensorSnapshot) -> InterventionCommand:
        self._phase = InterventionPhase.USER_AWAKE
        self._disturbance_started_at = None
        self._condition_started_at = None
        self._high_risk_started_at = None
        self._last_sound_at = None
        self._active_direct_event = None
        self._pending_confirmed_trigger = None
        self._pending_direct_trigger = None
        self._movement_freeze_until = None
        self._continue_score_state = 0.0
        self._continue_score_last_at = None
        self._continue_score_memory_at = None
        self._recovery_ready_started_at = None
        self._set_baseline_frozen(False)
        return self._no_sound(snapshot, "用户按按钮确认清醒，停止声音干预")

    def _stop_for_invalid_signal(self, snapshot: SensorSnapshot) -> InterventionCommand:
        self._disturbance_started_at = None
        self._condition_started_at = None
        self._high_risk_started_at = None
        self._last_sound_at = None
        self._active_direct_event = None
        self._pending_confirmed_trigger = None
        self._pending_direct_trigger = None
        self._movement_freeze_until = None
        self._continue_score_state = 0.0
        self._continue_score_last_at = None
        self._continue_score_memory_at = None
        self._recovery_ready_started_at = None
        self._set_baseline_frozen(False)
        if self._phase == InterventionPhase.INTERVENING:
            self._phase = InterventionPhase.RECOVERED
            return self._no_sound(snapshot, "RIP信号无效，停止声音干预")
        self._phase = InterventionPhase.MONITORING
        return self._no_sound(snapshot, "RIP信号无效，跳过本次干预判定")

    def _is_valid_rip_values(self, values: Sequence[float]) -> bool:
        if not values:
            return False
        if any(not _isfinite(value) for value in values):
            return False
        if len(values) == 1:
            return True
        span = _percentile(
            _smooth_series(values, window=self._rip_smoothing_window_points()),
            95.0,
        ) - _percentile(
            _smooth_series(values, window=self._rip_smoothing_window_points()),
            5.0,
        )
        if span < self.config.min_valid_rip_span:
            return False
        if span > self.config.max_valid_rip_span:
            return False
        return True

    def _smoothed_rip_values_since(self, cutoff: float) -> list[float]:
        points = [item for item in self._rip if item.timestamp >= cutoff]
        if not points:
            return []
        values = [item.value for item in points]
        return _smooth_series(values, window=self._rip_smoothing_window_points())

    def _rip_smoothing_window_points(self) -> int:
        return _window_points(self.config.rip_fs, self.config.rip_smoothing_window_sec, minimum=3)

    def _play_sound(self, snapshot: SensorSnapshot, reason: str) -> InterventionCommand:
        loudness_index = self._clamp_loudness_level_index(self._loudness_level_index)
        self._loudness_level_index = loudness_index
        loudness = self.config.loudness_levels[loudness_index]

        self._last_sound_at = snapshot.timestamp
        self._sounds_since_loudness_eval += 1

        return InterventionCommand(
            should_play_sound=True,
            loudness=float(loudness),
            phase=self._phase,
            reason=reason,
            loudness_level_index=self._loudness_level_index,
            snapshot=snapshot,
        )

    def _no_sound(self, snapshot: SensorSnapshot, reason: str) -> InterventionCommand:
        return InterventionCommand(
            should_play_sound=False,
            loudness=0.0,
            phase=self._phase,
            reason=reason,
            loudness_level_index=self._loudness_level_index,
            snapshot=snapshot,
        )

    def _loudness_index_from_score(self, score: float) -> int:
        if not self.config.loudness_levels:
            return 0
        clipped = min(max(float(score), 0.0), 1.0)
        idx = int(clipped * len(self.config.loudness_levels))
        return min(max(idx, 0), len(self.config.loudness_levels) - 1)

    def _clamp_loudness_level_index(self, index: int) -> int:
        if not self.config.loudness_levels:
            return 0
        return min(max(int(index), 0), len(self.config.loudness_levels) - 1)

    def _instantaneous_loudness_floor_index(self, snapshot: SensorSnapshot) -> int:
        floor = self._clamp_loudness_level_index(self.config.loudness_initial_level_index)
        spo2_delta = 0.0 if snapshot.spo2_delta is None else float(snapshot.spo2_delta)

        if (
            snapshot.continue_score >= self.config.loudness_very_high_trigger_threshold
            or spo2_delta >= 3.5
        ):
            return max(
                floor,
                self._clamp_loudness_level_index(self.config.loudness_very_high_floor_index),
            )

        if (
            snapshot.continue_score >= self.config.high_risk_trigger_score_threshold
            or spo2_delta >= 2.5
        ):
            return max(
                floor,
                self._clamp_loudness_level_index(self.config.loudness_high_risk_floor_index),
            )

        return floor

    def _instantaneous_loudness_ceiling_index(self, snapshot: SensorSnapshot) -> int:
        floor = self._instantaneous_loudness_floor_index(snapshot)
        ceiling = self._loudness_index_from_score(snapshot.loudness_score)
        return max(floor, ceiling)

    def _initialize_intervention_loudness(self, snapshot: SensorSnapshot) -> None:
        start_index = self._instantaneous_loudness_floor_index(snapshot)
        self._loudness_level_index = min(
            start_index,
            self._instantaneous_loudness_ceiling_index(snapshot),
        )
        self._sounds_since_loudness_eval = 0
        self._consecutive_recovery_windows = 0
        self._last_loudness_eval_snapshot = snapshot

    def _update_intervention_loudness(self, snapshot: SensorSnapshot) -> None:
        floor = self._instantaneous_loudness_floor_index(snapshot)
        if self._loudness_level_index < floor:
            self._loudness_level_index = floor
            self._consecutive_recovery_windows = 0

        if self._sounds_since_loudness_eval < self.config.loudness_eval_window_sounds:
            return

        anchor = self._last_loudness_eval_snapshot
        self._last_loudness_eval_snapshot = snapshot
        self._sounds_since_loudness_eval = 0
        if anchor is None:
            return

        status = self._loudness_recovery_status(anchor, snapshot)
        if status == "recovering":
            self._consecutive_recovery_windows += 1
            if (
                self._consecutive_recovery_windows
                >= self.config.loudness_recovery_windows_for_step_down
            ):
                initial = self._clamp_loudness_level_index(self.config.loudness_initial_level_index)
                self._loudness_level_index = max(initial, self._loudness_level_index - 1)
                self._consecutive_recovery_windows = 0
            return

        if status == "no_recovery":
            self._consecutive_recovery_windows = 0
            ceiling = self._instantaneous_loudness_ceiling_index(snapshot)
            if self._loudness_level_index < ceiling:
                self._loudness_level_index += 1
            return

        self._consecutive_recovery_windows = 0

    def _loudness_recovery_status(
        self,
        anchor: SensorSnapshot,
        snapshot: SensorSnapshot,
    ) -> str:
        recovery_signs = 0
        worsening_signs = 0

        if snapshot.loudness_score <= anchor.loudness_score - self.config.loudness_trigger_recovery_delta:
            recovery_signs += 1
        elif snapshot.loudness_score >= anchor.loudness_score + self.config.loudness_trigger_worsen_delta:
            worsening_signs += 1

        if (
            anchor.rip_amplitude_ratio is not None
            and snapshot.rip_amplitude_ratio is not None
        ):
            if (
                snapshot.rip_amplitude_ratio
                >= anchor.rip_amplitude_ratio + self.config.loudness_rip_ratio_recovery_delta
            ):
                recovery_signs += 1
            elif (
                snapshot.rip_amplitude_ratio
                <= anchor.rip_amplitude_ratio - self.config.loudness_rip_ratio_worsen_delta
            ):
                worsening_signs += 1

        if anchor.spo2_delta is not None and snapshot.spo2_delta is not None:
            if snapshot.spo2_delta <= anchor.spo2_delta - self.config.loudness_spo2_recovery_delta:
                recovery_signs += 1
            elif snapshot.spo2_delta >= anchor.spo2_delta + self.config.loudness_spo2_worsen_delta:
                worsening_signs += 1

        if recovery_signs >= 2 and worsening_signs == 0:
            return "recovering"
        if recovery_signs == 0 or worsening_signs > 0:
            return "no_recovery"
        return "hold"

    def _trigger_score(
        self,
        rip_amplitude_ratio: Optional[float],
        breath_period_ratio: Optional[float],
        event_persistence_sec: float,
        spo2_delta: Optional[float],
    ) -> float:
        amplitude = _amplitude_memberships(rip_amplitude_ratio)
        period = _period_memberships(breath_period_ratio)
        persistence = _persistence_memberships(event_persistence_sec)
        desaturation = _desaturation_memberships(spo2_delta)

        outputs: Dict[str, float] = {}

        def fire(label: str, strength: float) -> None:
            outputs[label] = max(outputs.get(label, 0.0), min(max(strength, 0.0), 1.0))

        fire("high", amplitude["severe"])
        fire("very_high", min(amplitude["severe"], persistence["prolonged"]))
        fire("very_high", min(amplitude["severe"], desaturation["high"]))
        fire("high", min(amplitude["moderate"], period["very_long"]))
        fire("medium", min(amplitude["moderate"], persistence["sustained"]))
        fire("high", min(amplitude["moderate"], desaturation["high"]))
        fire("medium", min(amplitude["moderate"], desaturation["moderate"]))
        fire("medium", min(amplitude["mild"], desaturation["moderate"]))
        fire("high", min(amplitude["mild"], desaturation["high"]))
        fire("low", min(amplitude["mild"], period["long"]))
        fire("low", min(amplitude["mild"], persistence["sustained"]))
        fire("none", min(amplitude["normal"], desaturation["none"]))

        # When the respiratory amplitude is already poor, a long event should
        # gradually push the score upward even if SpO2 has not fallen yet.
        fire("medium", min(amplitude["severe"], persistence["sustained"]))

        return _defuzzify_labels(
            outputs,
            values={
                "none": 0.0,
                "low": 0.25,
                "medium": 0.50,
                "high": 0.75,
                "very_high": 1.0,
            },
            default=0.0,
        )

    def _loudness_score(self, continue_score: float) -> float:
        return min(max(float(continue_score), 0.0), 1.0)


def _optional_float(value: Optional[Number]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 <= x0:
        return float(y1)
    clamped = min(max(float(x), float(x0)), float(x1))
    progress = (clamped - float(x0)) / (float(x1) - float(x0))
    return float(y0) + progress * (float(y1) - float(y0))


def _coerce_signal_values(signal: SignalInput) -> Sequence[float]:
    if isinstance(signal, (int, float)):
        return [float(signal)]
    return [float(value) for value in signal]


def _isfinite(value: float) -> bool:
    return isfinite(float(value))


def _window_points(fs: float, seconds: float, minimum: int = 1) -> int:
    points = max(int(round(max(float(fs), 1e-6) * max(float(seconds), 0.0))), int(minimum))
    if points % 2 == 0:
        points += 1
    return points


def _smooth_series(values: Sequence[float], window: int) -> list[float]:
    arr = [float(v) for v in values]
    if len(arr) < 3 or window <= 1:
        return arr
    window = min(int(window), len(arr))
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        return arr
    radius = window // 2
    out: list[float] = []
    for idx in range(len(arr)):
        start = max(0, idx - radius)
        end = min(len(arr), idx + radius + 1)
        chunk = arr[start:end]
        out.append(sum(chunk) / float(len(chunk)))
    return out


def _amplitude_memberships(value: Optional[float]) -> Dict[str, float]:
    x = 1.0 if value is None else max(float(value), 0.0)
    return {
        "normal": _trapezoid_membership(x, 0.60, 0.72, 2.0, 2.0),
        "mild": _triangle_membership(x, 0.40, 0.58, 0.78),
        "moderate": _triangle_membership(x, 0.18, 0.36, 0.55),
        "severe": _trapezoid_membership(x, 0.0, 0.0, 0.18, 0.30),
    }


def _period_memberships(value: Optional[float]) -> Dict[str, float]:
    x = 1.0 if value is None else max(float(value), 0.0)
    return {
        "normal": _trapezoid_membership(x, 0.0, 0.0, 1.08, 1.18),
        "long": _triangle_membership(x, 1.05, 1.28, 1.50),
        "very_long": _trapezoid_membership(x, 1.35, 1.50, 6.0, 6.0),
    }


def _persistence_memberships(value: float) -> Dict[str, float]:
    x = max(float(value), 0.0)
    return {
        "brief": _trapezoid_membership(x, 0.0, 0.0, 2.0, 5.0),
        "sustained": _triangle_membership(x, 3.0, 7.0, 11.0),
        "prolonged": _trapezoid_membership(x, 8.0, 10.0, 60.0, 60.0),
    }


def _desaturation_memberships(value: Optional[float]) -> Dict[str, float]:
    x = 0.0 if value is None else max(float(value), 0.0)
    return {
        "none": _trapezoid_membership(x, 0.0, 0.0, 0.5, 1.0),
        "mild": _triangle_membership(x, 0.5, 1.5, 2.5),
        "moderate": _triangle_membership(x, 1.5, 2.75, 3.75),
        "high": _trapezoid_membership(x, 2.5, 3.5, 12.0, 12.0),
    }


def _risk_memberships(value: float) -> Dict[str, float]:
    x = min(max(float(value), 0.0), 1.0)
    return {
        "low": _trapezoid_membership(x, 0.05, 0.18, 0.32, 0.45),
        "medium": _triangle_membership(x, 0.30, 0.50, 0.70),
        "high": _triangle_membership(x, 0.60, 0.76, 0.92),
        "very_high": _trapezoid_membership(x, 0.82, 0.92, 1.0, 1.0),
    }


def _triangle_membership(x: float, left: float, center: float, right: float) -> float:
    if left == center == right:
        return 1.0 if x == center else 0.0
    if x <= left or x >= right:
        return 0.0
    if x == center:
        return 1.0
    if x < center:
        denom = center - left
        return 0.0 if denom <= 0 else (x - left) / denom
    denom = right - center
    return 0.0 if denom <= 0 else (right - x) / denom


def _trapezoid_membership(x: float, left: float, left_top: float, right_top: float, right: float) -> float:
    if x < left or x > right:
        return 0.0
    if left_top <= x <= right_top:
        return 1.0
    if x < left_top:
        denom = left_top - left
        return 0.0 if denom <= 0 else (x - left) / denom
    denom = right - right_top
    return 0.0 if denom <= 0 else (right - x) / denom


def _defuzzify_labels(outputs: Dict[str, float], values: Dict[str, float], default: float) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for label, strength in outputs.items():
        if strength <= 0:
            continue
        total_weight += strength
        weighted_sum += strength * values[label]
    if total_weight <= 0:
        return float(default)
    return min(max(weighted_sum / total_weight, 0.0), 1.0)


def _candidate_cycle_periods(times: Sequence[float], values: Sequence[float]) -> Sequence[float]:
    if len(times) != len(values) or len(times) < 5:
        return []

    span = max(values) - min(values)
    if span <= 1e-6:
        return []

    baseline = _percentile(values, 50.0)
    height_threshold = baseline + span * 0.12
    min_period_sec = 0.8
    max_period_sec = 10.0

    peak_indices = []
    for idx in range(1, len(values) - 1):
        value = values[idx]
        if value < values[idx - 1] or value < values[idx + 1]:
            continue
        if value < height_threshold:
            continue

        if peak_indices:
            prev_idx = peak_indices[-1]
            if times[idx] - times[prev_idx] < min_period_sec:
                if value > values[prev_idx]:
                    peak_indices[-1] = idx
                continue

        peak_indices.append(idx)

    periods = []
    for prev_idx, next_idx in zip(peak_indices, peak_indices[1:]):
        period = times[next_idx] - times[prev_idx]
        if min_period_sec <= period <= max_period_sec:
            periods.append(period)
    return periods


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    p = min(max(float(percentile), 0.0), 100.0)
    pos = (len(ordered) - 1) * p / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
