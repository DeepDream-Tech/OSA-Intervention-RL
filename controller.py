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

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple, Union


Number = Union[int, float]
SignalInput = Union[Number, Sequence[Number], Iterable[Number]]


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
    airflow_drop_threshold_fraction: float = 0.90

    # Fuzzy trigger condition.
    trigger_duration_sec: float = 6.0
    fast_trigger_duration_sec: float = 3.0
    trigger_score_threshold: float = 0.70
    high_risk_trigger_score_threshold: float = 0.85
    trigger_score_stop_threshold: float = 0.40

    # Intervention schedule.
    sound_interval_sec: float = 0.6
    loudness_levels: Tuple[float, ...] = (0.20, 0.28, 0.36, 0.44, 0.52, 0.60)

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
    spo2_pct: Optional[float]
    spo2_baseline: Optional[float]
    spo2_delta: Optional[float]
    trigger_score: float
    loudness_score: float
    button_pressed: bool
    event_condition_met: bool
    event_condition_duration_sec: float
    high_risk_duration_sec: float


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
class PreExperimentController:
    """State machine for the fuzzy pre-experiment acoustic intervention rule.

    Fuzzy-v1 keeps the fixed 0.6 s playback cadence but replaces:

    - deterministic RIP threshold triggering
    - deterministic "two sounds then increase loudness" logic

    with RIP + SpO2 fuzzy scoring:

    - ``trigger_score`` decides whether an event is severe enough to start or
      continue intervention
    - ``loudness_score`` decides which configured loudness level to use for the
      next scheduled sound
    """

    config: PreExperimentConfig = field(default_factory=PreExperimentConfig)

    def __post_init__(self) -> None:
        if not self.config.loudness_levels:
            raise ValueError("loudness_levels must contain at least one value")
        if not 0.0 <= self.config.trigger_score_stop_threshold <= 1.0:
            raise ValueError("trigger_score_stop_threshold must be in [0, 1]")
        if not 0.0 <= self.config.trigger_score_threshold <= 1.0:
            raise ValueError("trigger_score_threshold must be in [0, 1]")
        if not 0.0 <= self.config.high_risk_trigger_score_threshold <= 1.0:
            raise ValueError("high_risk_trigger_score_threshold must be in [0, 1]")
        if self.config.trigger_score_stop_threshold > self.config.trigger_score_threshold:
            raise ValueError("stop threshold must be <= start threshold")
        self.reset()

    def reset(self) -> None:
        self._rip: Deque[_TimedValue] = deque()
        self._rip_baseline_amplitudes: Deque[_TimedValue] = deque()
        self._rip_baseline_periods: Deque[_TimedValue] = deque()
        self._spo2_samples: Deque[_TimedValue] = deque()
        self._spo2_baselines: Deque[_TimedValue] = deque()

        self._manual_rip_baseline: Optional[float] = None
        self._manual_breath_period_baseline: Optional[float] = None
        self._manual_spo2_baseline: Optional[float] = None

        self._phase = InterventionPhase.MONITORING
        self._disturbance_started_at: Optional[float] = None
        self._condition_started_at: Optional[float] = None
        self._high_risk_started_at: Optional[float] = None
        self._last_sound_at: Optional[float] = None
        self._loudness_level_index = 0

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
    ) -> InterventionCommand:
        """Feed the newest RIP/SpO2/button values and return a loudness command.

        ``rip`` may be one sample or a batch of samples. When a batch is passed,
        samples are assumed to end at ``timestamp`` and to be spaced by
        ``1 / rip_fs``.

        ``button`` follows the same streaming style: it may be a scalar 0/1
        sample or a batch of 0/1 samples. Any value >= 0.5 in the current input
        batch is treated as a user-awake button press.

        ``spo2_pct`` is the most recent SpO2 percentage aligned to the packet
        end time. It is optional; the fuzzy controller falls back to RIP-only
        scoring when no recent SpO2 is available.
        """

        ts = float(timestamp if timestamp is not None else time.time())
        self._append_rip(rip, ts)
        self._append_spo2(spo2_pct, ts)
        self._trim(ts)

        button_pressed = self._button_pressed(button)
        snapshot = self._snapshot(ts, button_pressed=button_pressed)

        if button_pressed:
            return self._stop_for_user_awake(snapshot)

        self._update_condition_timers(snapshot)
        self._maybe_update_baselines(snapshot)

        if self._phase == InterventionPhase.INTERVENING:
            return self._intervention_command(snapshot)

        if snapshot.high_risk_duration_sec >= self.config.fast_trigger_duration_sec:
            self._start_intervention(ts)
            return self._play_sound(
                snapshot,
                reason=(
                    "高风险模糊触发分数持续达到快速阈值，开始声音干预"
                    f" (score={snapshot.trigger_score:.2f})"
                ),
            )

        if snapshot.event_condition_duration_sec >= self.config.trigger_duration_sec:
            self._start_intervention(ts)
            return self._play_sound(
                snapshot,
                reason=(
                    "模糊触发分数持续达到起始阈值，开始声音干预"
                    f" (score={snapshot.trigger_score:.2f})"
                ),
            )

        if snapshot.event_condition_met:
            self._phase = InterventionPhase.EVENT_PENDING
            return self._no_sound(
                snapshot,
                (
                    "模糊触发分数已超过起始阈值，等待持续时间满足开始条件"
                    f" (score={snapshot.trigger_score:.2f})"
                ),
            )

        self._phase = InterventionPhase.MONITORING
        return self._no_sound(
            snapshot,
            f"模糊触发分数未达到开始条件 (score={snapshot.trigger_score:.2f})",
        )

    def update_loudness(
        self,
        rip: SignalInput,
        button: Optional[SignalInput] = None,
        spo2_pct: Optional[Number] = None,
        timestamp: Optional[Number] = None,
    ) -> float:
        """Feed RIP/SpO2/button samples and return only the loudness output."""

        return self.update(
            rip=rip,
            button=button,
            spo2_pct=spo2_pct,
            timestamp=timestamp,
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

    def _snapshot(self, now: float, button_pressed: bool) -> SensorSnapshot:
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

        current_spo2 = self._current_spo2(now)
        spo2_baseline = self._spo2_baseline()
        spo2_delta = None
        if current_spo2 is not None and spo2_baseline is not None:
            spo2_delta = max(0.0, spo2_baseline - current_spo2)

        suspicious = self._is_suspicious(
            rip_amplitude_ratio=rip_ratio,
            breath_period_ratio=breath_period_ratio,
            spo2_delta=spo2_delta,
        )

        trigger_score = self._trigger_score(
            rip_amplitude_ratio=rip_ratio,
            breath_period_ratio=breath_period_ratio,
            event_persistence_sec=self._current_disturbance_duration(now, suspicious),
            spo2_delta=spo2_delta,
        )
        loudness_score = self._loudness_score(trigger_score=trigger_score, spo2_delta=spo2_delta)

        event_condition_met = trigger_score >= self.config.trigger_score_threshold
        duration = 0.0
        if event_condition_met and self._condition_started_at is not None:
            duration = max(0.0, now - self._condition_started_at)

        high_risk_duration = 0.0
        if (
            trigger_score >= self.config.high_risk_trigger_score_threshold
            and self._high_risk_started_at is not None
        ):
            high_risk_duration = max(0.0, now - self._high_risk_started_at)

        return SensorSnapshot(
            timestamp=now,
            rip_amplitude=rip_amplitude,
            rip_baseline=rip_baseline,
            rip_amplitude_ratio=rip_ratio,
            airflow_drop_fraction=airflow_drop,
            breath_period_sec=breath_period,
            breath_period_baseline=breath_period_baseline,
            breath_period_ratio=breath_period_ratio,
            spo2_pct=current_spo2,
            spo2_baseline=spo2_baseline,
            spo2_delta=spo2_delta,
            trigger_score=trigger_score,
            loudness_score=loudness_score,
            button_pressed=button_pressed,
            event_condition_met=event_condition_met,
            event_condition_duration_sec=duration,
            high_risk_duration_sec=high_risk_duration,
        )

    def _current_rip_amplitude(self, now: float) -> float:
        cutoff = now - self.config.rip_amplitude_window_sec
        values = [item.value for item in self._rip if item.timestamp >= cutoff]
        if len(values) < 2:
            return 0.0
        return _percentile(values, 95.0) - _percentile(values, 5.0)

    def _current_breath_period(self, now: float) -> Optional[float]:
        cutoff = now - self.config.rip_period_window_sec
        points = [item for item in self._rip if item.timestamp >= cutoff]
        if len(points) < 5:
            return None

        values = [item.value for item in points]
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
        if self._manual_rip_baseline is not None:
            return self._manual_rip_baseline

        values = [item.value for item in self._rip_baseline_amplitudes if item.value > 0]
        if len(values) < 3:
            return None

        # Use a high percentile so short low-amplitude events do not pull the
        # baseline down while the subject is obstructed.
        return max(_percentile(values, 80.0), 1e-6)

    def _breath_period_baseline(self) -> Optional[float]:
        if self._manual_breath_period_baseline is not None:
            return self._manual_breath_period_baseline

        values = [item.value for item in self._rip_baseline_periods if item.value > 0]
        if len(values) < 2:
            return None
        return max(_percentile(values, 50.0), 1e-6)

    def _spo2_baseline(self) -> Optional[float]:
        if self._manual_spo2_baseline is not None:
            return self._manual_spo2_baseline

        values = [item.value for item in self._spo2_baselines]
        if len(values) < 3:
            return None
        return _percentile(values, 80.0)

    def _current_disturbance_duration(self, now: float, suspicious: bool) -> float:
        if not suspicious or self._disturbance_started_at is None:
            return 0.0
        return max(0.0, now - self._disturbance_started_at)

    @staticmethod
    def _is_suspicious(
        rip_amplitude_ratio: Optional[float],
        breath_period_ratio: Optional[float],
        spo2_delta: Optional[float],
    ) -> bool:
        return bool(
            (rip_amplitude_ratio is not None and rip_amplitude_ratio < 0.80)
            or (breath_period_ratio is not None and breath_period_ratio > 1.10)
            or (spo2_delta is not None and spo2_delta >= 1.0)
        )

    def _update_condition_timers(self, snapshot: SensorSnapshot) -> None:
        if self._is_suspicious(
            rip_amplitude_ratio=snapshot.rip_amplitude_ratio,
            breath_period_ratio=snapshot.breath_period_ratio,
            spo2_delta=snapshot.spo2_delta,
        ):
            if self._disturbance_started_at is None:
                self._disturbance_started_at = snapshot.timestamp
        else:
            self._disturbance_started_at = None

        if snapshot.trigger_score >= self.config.high_risk_trigger_score_threshold:
            if self._high_risk_started_at is None:
                self._high_risk_started_at = snapshot.timestamp
        else:
            self._high_risk_started_at = None

        if snapshot.event_condition_met:
            if self._condition_started_at is None:
                self._condition_started_at = snapshot.timestamp
            return

        self._condition_started_at = None

    def _maybe_update_baselines(self, snapshot: SensorSnapshot) -> None:
        if snapshot.button_pressed:
            return
        if self._phase == InterventionPhase.INTERVENING:
            return
        if snapshot.trigger_score >= self.config.trigger_score_stop_threshold:
            return

        if snapshot.rip_amplitude > 0:
            self._rip_baseline_amplitudes.append(
                _TimedValue(snapshot.timestamp, snapshot.rip_amplitude)
            )
        if snapshot.breath_period_sec is not None and snapshot.breath_period_sec > 0:
            self._rip_baseline_periods.append(
                _TimedValue(snapshot.timestamp, snapshot.breath_period_sec)
            )
        if snapshot.spo2_pct is not None:
            self._spo2_baselines.append(_TimedValue(snapshot.timestamp, snapshot.spo2_pct))

    def _start_intervention(self, timestamp: float) -> None:
        self._phase = InterventionPhase.INTERVENING
        self._last_sound_at = None
        self._loudness_level_index = 0
        if self._condition_started_at is None:
            self._condition_started_at = timestamp

    def _intervention_command(self, snapshot: SensorSnapshot) -> InterventionCommand:
        if snapshot.trigger_score < self.config.trigger_score_stop_threshold:
            self._phase = InterventionPhase.RECOVERED
            self._disturbance_started_at = None
            self._condition_started_at = None
            self._high_risk_started_at = None
            return self._no_sound(
                snapshot,
                f"模糊触发分数已降到停止阈值以下，停止声音干预 (score={snapshot.trigger_score:.2f})",
            )

        due = (
            self._last_sound_at is None
            or snapshot.timestamp - self._last_sound_at >= self.config.sound_interval_sec
        )
        if not due:
            return self._no_sound(snapshot, "等待固定0.6秒播放间隔")

        return self._play_sound(
            snapshot,
            (
                "按固定间隔继续声音干预，当前响度由模糊强度分数决定"
                f" (trigger={snapshot.trigger_score:.2f}, loudness={snapshot.loudness_score:.2f})"
            ),
        )

    def _stop_for_user_awake(self, snapshot: SensorSnapshot) -> InterventionCommand:
        self._phase = InterventionPhase.USER_AWAKE
        self._disturbance_started_at = None
        self._condition_started_at = None
        self._high_risk_started_at = None
        self._last_sound_at = None
        return self._no_sound(snapshot, "用户按按钮确认清醒，停止声音干预")

    def _play_sound(self, snapshot: SensorSnapshot, reason: str) -> InterventionCommand:
        loudness_index = self._loudness_index_from_score(snapshot.loudness_score)
        self._loudness_level_index = loudness_index
        loudness = self.config.loudness_levels[loudness_index]

        self._last_sound_at = snapshot.timestamp

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

    def _loudness_score(self, trigger_score: float, spo2_delta: Optional[float]) -> float:
        risk = _risk_memberships(trigger_score)
        desaturation = _desaturation_memberships(spo2_delta)

        outputs: Dict[str, float] = {}

        def fire(label: str, strength: float) -> None:
            outputs[label] = max(outputs.get(label, 0.0), min(max(strength, 0.0), 1.0))

        fire("very_soft", risk["low"])
        fire("soft", risk["medium"])
        fire("medium", risk["high"])
        fire("strong", min(risk["high"], desaturation["moderate"]))
        fire("strong", min(risk["high"], desaturation["high"]))
        fire("strong", risk["very_high"])

        return _defuzzify_labels(
            outputs,
            values={
                "very_soft": 0.18,
                "soft": 0.35,
                "medium": 0.60,
                "strong": 0.90,
            },
            default=min(max(float(trigger_score), 0.0), 1.0),
        )


def _optional_float(value: Optional[Number]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
