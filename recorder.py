"""Disk recorder for intervention-v0 signals and cue triggers.

The main project already has ``pipeline.recorder.SessionRecorder`` for full
system sessions. This module is intentionally independent: it records the
pre-experiment controller's DataPacket-derived input and controller output
without requiring any change to the main sound/playback system.
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"


CSV_COLUMNS = [
    "ts",
    "packet_sn",
    "rip_n",
    "rip_fs",
    "spo2_pct",
    "pulse_rate",
    "awake_pressed",
    "cue_triggered",
    "loudness",
    "loudness_level_index",
    "phase",
    "reason",
    "rip_amplitude",
    "rip_baseline",
    "rip_amplitude_ratio",
    "airflow_drop_fraction",
    "breath_period_sec",
    "breath_period_baseline",
    "breath_period_ratio",
    "spo2_baseline",
    "spo2_delta",
    "trigger_score",
    "loudness_score",
    "event_condition_duration_sec",
    "high_risk_duration_sec",
]


def new_session_id(tag: str = "preexp") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tag}" if tag else ts


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class PreExperimentSessionMeta:
    """Metadata saved to ``meta.json`` for one pre-experiment recording."""

    session_id: str = field(default_factory=lambda: new_session_id("preexp"))
    started_at: str = field(default_factory=_now_iso)
    subject_id: str = ""
    note: str = ""
    protocol: str = "intervention-v0"
    config: Dict[str, Any] = field(default_factory=dict)


class PreExperimentRecorder:
    """Record RIP/SpO2/pulse inputs and controller cue decisions.

    Files written under ``intervention-v0/sessions/<session_id>/`` by default:

    - ``meta.json``: subject/session/config metadata.
    - ``signals.csv``: one row per DataPacket/controller update.
    - ``signals_####.npz``: chunked arrays for training/replay.
    - ``cue_events.jsonl``: one JSON object for each sound cue trigger.
    - ``summary.json``: counts written on close.
    """

    def __init__(
        self,
        meta: Optional[PreExperimentSessionMeta] = None,
        root: Optional[Path] = None,
        block_size_packets: int = 60,
    ) -> None:
        self.meta = meta or PreExperimentSessionMeta()
        self.dir = (Path(root) if root is not None else SESSIONS_DIR) / self.meta.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.block_size_packets = max(1, int(block_size_packets))

        with open(self.dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.meta), f, ensure_ascii=False, indent=2, default=_json_default)

        self._signals_f = open(self.dir / "signals.csv", "a", newline="", encoding="utf-8")
        self._signals_w = csv.DictWriter(self._signals_f, fieldnames=CSV_COLUMNS)
        if self._signals_f.tell() == 0:
            self._signals_w.writeheader()

        self._cue_f = open(self.dir / "cue_events.jsonl", "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False
        self._block_idx = 0
        self._block = _empty_block()

        self.sample_count = 0
        self.cue_count = 0

    def __enter__(self) -> "PreExperimentRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def record_sample(self, sample: Any) -> None:
        """Record a signal sample without a controller command."""

        self.record_step(sample=sample, command=None)

    def record_step(
        self,
        sample: Any,
        command: Optional[Any] = None,
        cue_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one aligned DataPacket sample and controller output.

        Call this once after ``controller.update(...)``. If ``command`` says a
        sound should be played, the trigger is also appended to
        ``cue_events.jsonl``.
        """

        with self._lock:
            self._raise_if_closed()
            row = _sample_row(sample, command)
            self._signals_w.writerow(row)
            self.sample_count += 1

            if _command_triggered(command):
                self._write_cue_event_locked(command, sample=sample, cue_params=cue_params)

            self._append_block_locked(sample, command)
            if len(self._block["ts"]) >= self.block_size_packets:
                self._flush_block_locked()

    def record_cue_event(
        self,
        command: Any,
        sample: Optional[Any] = None,
        cue_params: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        """Record a cue event separately, usually after the sound system plays it.

        ``record_step`` is the preferred API. This method is useful if the
        playback layer wants to add actual cue parameters such as calibrated dB,
        device name, or file path after playback succeeds.
        """

        if not force and not _command_triggered(command):
            return
        with self._lock:
            self._raise_if_closed()
            self._write_cue_event_locked(command, sample=sample, cue_params=cue_params)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_block_locked()
            self._signals_f.flush()
            self._signals_f.close()
            self._cue_f.flush()
            self._cue_f.close()
            ended_at = _now_iso()
            with open(self.dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_id": self.meta.session_id,
                        "started_at": self.meta.started_at,
                        "ended_at": ended_at,
                        "samples": self.sample_count,
                        "cue_events": self.cue_count,
                        "signal_blocks": self._block_idx,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self._closed = True

    def _write_cue_event_locked(
        self,
        command: Any,
        sample: Optional[Any] = None,
        cue_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = {
            "t": _event_time(command, sample),
            "packet_sn": getattr(sample, "packet_sn", None) if sample is not None else None,
            "spo2_pct": getattr(sample, "spo2_pct", None) if sample is not None else None,
            "pulse_rate": getattr(sample, "pulse_rate", None) if sample is not None else None,
            "awake_pressed": _awake_pressed(getattr(sample, "awake_button", None))
            if sample is not None
            else None,
            "command": _command_dict(command),
        }
        if cue_params:
            rec["cue_params"] = cue_params
        self._cue_f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        self.cue_count += 1

    def _append_block_locked(self, sample: Any, command: Optional[Any]) -> None:
        snapshot = getattr(command, "snapshot", None) if command is not None else None
        rip = np.asarray(getattr(sample, "rip", []), dtype=np.float32).reshape(-1)
        awake = np.asarray(getattr(sample, "awake_button", []), dtype=np.float32).reshape(-1)
        block = self._block
        block["ts"].append(float(getattr(sample, "timestamp", np.nan)))
        block["packet_sn"].append(_optional_int(getattr(sample, "packet_sn", None), missing=-1))
        block["rip"].append(rip)
        block["rip_len"].append(int(rip.size))
        block["awake_button"].append(awake)
        block["awake_len"].append(int(awake.size))
        block["rip_fs"].append(_optional_float(getattr(sample, "rip_fs", None), missing=np.nan))
        block["spo2_pct"].append(_optional_float(getattr(sample, "spo2_pct", None), missing=np.nan))
        block["pulse_rate"].append(_optional_float(getattr(sample, "pulse_rate", None), missing=np.nan))
        block["cue_triggered"].append(1 if _command_triggered(command) else 0)
        block["loudness"].append(_optional_float(getattr(command, "loudness", None), missing=0.0))
        block["loudness_level_index"].append(
            _optional_int(getattr(command, "loudness_level_index", None), missing=-1)
        )
        block["rip_amplitude"].append(_snapshot_float(snapshot, "rip_amplitude"))
        block["rip_baseline"].append(_snapshot_float(snapshot, "rip_baseline"))
        block["rip_amplitude_ratio"].append(_snapshot_float(snapshot, "rip_amplitude_ratio"))
        block["airflow_drop_fraction"].append(_snapshot_float(snapshot, "airflow_drop_fraction"))
        block["breath_period_sec"].append(_snapshot_float(snapshot, "breath_period_sec"))
        block["breath_period_baseline"].append(_snapshot_float(snapshot, "breath_period_baseline"))
        block["breath_period_ratio"].append(_snapshot_float(snapshot, "breath_period_ratio"))
        block["spo2_baseline"].append(_snapshot_float(snapshot, "spo2_baseline"))
        block["spo2_delta"].append(_snapshot_float(snapshot, "spo2_delta"))
        block["trigger_score"].append(_snapshot_float(snapshot, "trigger_score"))
        block["loudness_score"].append(_snapshot_float(snapshot, "loudness_score"))
        block["event_condition_duration_sec"].append(
            _snapshot_float(snapshot, "event_condition_duration_sec")
        )
        block["high_risk_duration_sec"].append(
            _snapshot_float(snapshot, "high_risk_duration_sec")
        )

    def _flush_block_locked(self) -> None:
        if not self._block["ts"]:
            return
        block = self._block
        self._block = _empty_block()

        out = {
            "ts": np.asarray(block["ts"], dtype=np.float64),
            "packet_sn": np.asarray(block["packet_sn"], dtype=np.int64),
            "rip": _pad_2d(block["rip"], fill_value=np.nan),
            "rip_len": np.asarray(block["rip_len"], dtype=np.int32),
            "awake_button": _pad_2d(block["awake_button"], fill_value=0.0),
            "awake_len": np.asarray(block["awake_len"], dtype=np.int32),
            "rip_fs": np.asarray(block["rip_fs"], dtype=np.float32),
            "spo2_pct": np.asarray(block["spo2_pct"], dtype=np.float32),
            "pulse_rate": np.asarray(block["pulse_rate"], dtype=np.float32),
            "cue_triggered": np.asarray(block["cue_triggered"], dtype=np.uint8),
            "loudness": np.asarray(block["loudness"], dtype=np.float32),
            "loudness_level_index": np.asarray(block["loudness_level_index"], dtype=np.int32),
            "rip_amplitude": np.asarray(block["rip_amplitude"], dtype=np.float32),
            "rip_baseline": np.asarray(block["rip_baseline"], dtype=np.float32),
            "rip_amplitude_ratio": np.asarray(block["rip_amplitude_ratio"], dtype=np.float32),
            "airflow_drop_fraction": np.asarray(block["airflow_drop_fraction"], dtype=np.float32),
            "breath_period_sec": np.asarray(block["breath_period_sec"], dtype=np.float32),
            "breath_period_baseline": np.asarray(block["breath_period_baseline"], dtype=np.float32),
            "breath_period_ratio": np.asarray(block["breath_period_ratio"], dtype=np.float32),
            "spo2_baseline": np.asarray(block["spo2_baseline"], dtype=np.float32),
            "spo2_delta": np.asarray(block["spo2_delta"], dtype=np.float32),
            "trigger_score": np.asarray(block["trigger_score"], dtype=np.float32),
            "loudness_score": np.asarray(block["loudness_score"], dtype=np.float32),
            "event_condition_duration_sec": np.asarray(
                block["event_condition_duration_sec"], dtype=np.float32
            ),
            "high_risk_duration_sec": np.asarray(
                block["high_risk_duration_sec"], dtype=np.float32
            ),
        }
        np.savez_compressed(self.dir / f"signals_{self._block_idx:04d}.npz", **out)
        self._block_idx += 1

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("PreExperimentRecorder is closed")


def _empty_block() -> Dict[str, list]:
    return {
        "ts": [],
        "packet_sn": [],
        "rip": [],
        "rip_len": [],
        "awake_button": [],
        "awake_len": [],
        "rip_fs": [],
        "spo2_pct": [],
        "pulse_rate": [],
        "cue_triggered": [],
        "loudness": [],
        "loudness_level_index": [],
        "rip_amplitude": [],
        "rip_baseline": [],
        "rip_amplitude_ratio": [],
        "airflow_drop_fraction": [],
        "breath_period_sec": [],
        "breath_period_baseline": [],
        "breath_period_ratio": [],
        "spo2_baseline": [],
        "spo2_delta": [],
        "trigger_score": [],
        "loudness_score": [],
        "event_condition_duration_sec": [],
        "high_risk_duration_sec": [],
    }


def _sample_row(sample: Any, command: Optional[Any]) -> Dict[str, Any]:
    snapshot = getattr(command, "snapshot", None) if command is not None else None
    rip = np.asarray(getattr(sample, "rip", []), dtype=np.float32).reshape(-1)
    return {
        "ts": f"{float(getattr(sample, 'timestamp', np.nan)):.3f}",
        "packet_sn": _csv_value(getattr(sample, "packet_sn", None)),
        "rip_n": int(rip.size),
        "rip_fs": _csv_value(getattr(sample, "rip_fs", None)),
        "spo2_pct": _csv_value(getattr(sample, "spo2_pct", None)),
        "pulse_rate": _csv_value(getattr(sample, "pulse_rate", None)),
        "awake_pressed": int(_awake_pressed(getattr(sample, "awake_button", None))),
        "cue_triggered": int(_command_triggered(command)),
        "loudness": _csv_value(getattr(command, "loudness", None)),
        "loudness_level_index": _csv_value(getattr(command, "loudness_level_index", None)),
        "phase": _csv_value(_enum_value(getattr(command, "phase", None))),
        "reason": _csv_value(getattr(command, "reason", None)),
        "rip_amplitude": _csv_value(getattr(snapshot, "rip_amplitude", None)),
        "rip_baseline": _csv_value(getattr(snapshot, "rip_baseline", None)),
        "rip_amplitude_ratio": _csv_value(getattr(snapshot, "rip_amplitude_ratio", None)),
        "airflow_drop_fraction": _csv_value(getattr(snapshot, "airflow_drop_fraction", None)),
        "breath_period_sec": _csv_value(getattr(snapshot, "breath_period_sec", None)),
        "breath_period_baseline": _csv_value(getattr(snapshot, "breath_period_baseline", None)),
        "breath_period_ratio": _csv_value(getattr(snapshot, "breath_period_ratio", None)),
        "spo2_baseline": _csv_value(getattr(snapshot, "spo2_baseline", None)),
        "spo2_delta": _csv_value(getattr(snapshot, "spo2_delta", None)),
        "trigger_score": _csv_value(getattr(snapshot, "trigger_score", None)),
        "loudness_score": _csv_value(getattr(snapshot, "loudness_score", None)),
        "event_condition_duration_sec": _csv_value(
            getattr(snapshot, "event_condition_duration_sec", None)
        ),
        "high_risk_duration_sec": _csv_value(
            getattr(snapshot, "high_risk_duration_sec", None)
        ),
    }


def _command_triggered(command: Optional[Any]) -> bool:
    return bool(getattr(command, "should_play_sound", False)) if command is not None else False


def _event_time(command: Any, sample: Optional[Any]) -> Optional[float]:
    if sample is not None and getattr(sample, "timestamp", None) is not None:
        return float(getattr(sample, "timestamp"))
    snapshot = getattr(command, "snapshot", None)
    if snapshot is not None and getattr(snapshot, "timestamp", None) is not None:
        return float(getattr(snapshot, "timestamp"))
    return None


def _command_dict(command: Any) -> Dict[str, Any]:
    if command is None:
        return {}
    if hasattr(command, "as_dict") and callable(command.as_dict):
        data = command.as_dict()
    elif is_dataclass(command):
        data = asdict(command)
    else:
        data = {
            "should_play_sound": getattr(command, "should_play_sound", None),
            "loudness": getattr(command, "loudness", None),
            "phase": getattr(command, "phase", None),
            "reason": getattr(command, "reason", None),
            "loudness_level_index": getattr(command, "loudness_level_index", None),
            "snapshot": getattr(command, "snapshot", None),
        }
    return _normalize_json(data)


def _normalize_json(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(v) for v in value]
    enum_value = _enum_value(value)
    if enum_value is not value:
        return enum_value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _json_default(obj: Any) -> Any:
    normalized = _normalize_json(obj)
    if normalized is not obj:
        return normalized
    return str(obj)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _csv_value(value: Any) -> Any:
    value = _enum_value(value)
    return "" if value is None else value


def _optional_float(value: Any, missing: float) -> float:
    if value is None:
        return float(missing)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(missing)


def _optional_int(value: Any, missing: int) -> int:
    if value is None:
        return int(missing)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(missing)


def _snapshot_float(snapshot: Optional[Any], name: str) -> float:
    if snapshot is None:
        return float("nan")
    return _optional_float(getattr(snapshot, name, None), missing=np.nan)


def _awake_pressed(button: Any) -> bool:
    if button is None:
        return False
    try:
        arr = np.asarray(button, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(arr.size and np.any(arr >= 0.5))


def _pad_2d(arrays: list[np.ndarray], fill_value: float) -> np.ndarray:
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    max_len = max((int(np.asarray(a).size) for a in arrays), default=0)
    out = np.full((len(arrays), max_len), fill_value, dtype=np.float32)
    for idx, arr in enumerate(arrays):
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        out[idx, : flat.size] = flat
    return out
