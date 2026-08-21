from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from controller import PreExperimentConfig, PreExperimentController
from data_reader import detect_packet_movement


@dataclass
class InterventionEvent:
    t: float
    loudness: float
    trigger_score: float | None
    phase: str
    reason: str
    strategy: str
    direction: str


@dataclass
class InterventionEpisode:
    start_t: float
    end_t: float
    n_sounds: int
    max_loudness: float
    max_trigger_score: float | None
    start_reason: str


@dataclass
class ReplayTrace:
    t: np.ndarray
    rip_amplitude: np.ndarray
    rip_baseline: np.ndarray
    rip_amplitude_ratio: np.ndarray
    breath_period_sec: np.ndarray
    breath_period_baseline: np.ndarray
    breath_period_ratio: np.ndarray
    spo2_pct: np.ndarray
    spo2_baseline: np.ndarray
    spo2_delta: np.ndarray
    trigger_score: np.ndarray
    loudness_score: np.ndarray
    should_play_sound: np.ndarray
    phase: list[str]
    strategy_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a session overview plot with intervention timing.")
    parser.add_argument("session_dir", type=Path, help="Path to a recorded session directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output artifacts. Defaults to <session_dir>/viz",
    )
    parser.add_argument(
        "--start-clock",
        type=str,
        default=None,
        help="Optional display window start clock time in HH:MM or HH:MM:SS.",
    )
    parser.add_argument(
        "--end-clock",
        type=str,
        default=None,
        help="Optional display window end clock time in HH:MM or HH:MM:SS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir.resolve()
    out_dir = (args.out_dir or session_dir / "viz").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((session_dir / "meta.json").read_text())
    csv_rows = _read_chestband_csv(session_dir / "chestband.csv")
    interventions = _read_interventions(session_dir / "interventions.jsonl")
    episodes = _build_episodes(interventions)
    rip_t, rip_y = _read_rip_waveform(session_dir)
    replay_trace = _replay_fuzzy_trace(session_dir, csv_rows)
    started_dt = datetime.fromisoformat(meta["started_at"])
    first_ts = float(csv_rows[0]["ts"]) if csv_rows else 0.0
    window_bounds = _resolve_window_bounds(
        started_dt=started_dt,
        first_ts=first_ts,
        start_clock=args.start_clock,
        end_clock=args.end_clock,
    )

    display_csv_rows = csv_rows
    display_interventions = interventions
    display_episodes = episodes
    display_rip_t = rip_t
    display_rip_y = rip_y
    display_replay_trace = replay_trace
    file_suffix = ""
    if window_bounds is not None:
        window_start_ts, window_end_ts, window_label = window_bounds
        display_csv_rows = [row for row in csv_rows if window_start_ts <= float(row["ts"]) <= window_end_ts]
        display_interventions = [
            event for event in interventions if window_start_ts <= event.t <= window_end_ts
        ]
        display_episodes = _build_episodes(display_interventions)
        if rip_t.size:
            rip_mask = (rip_t >= window_start_ts) & (rip_t <= window_end_ts)
            display_rip_t = rip_t[rip_mask]
            display_rip_y = rip_y[rip_mask]
        replay_mask = (replay_trace.t >= window_start_ts) & (replay_trace.t <= window_end_ts)
        display_replay_trace = _slice_replay_trace(replay_trace, replay_mask)
        file_suffix = f"_{window_label}"

    fig_path = out_dir / f"{session_dir.name}{file_suffix}_overview.png"
    csv_path = out_dir / f"{session_dir.name}{file_suffix}_intervention_episodes.csv"
    trace_path = out_dir / f"{session_dir.name}{file_suffix}_fuzzy_trace.csv"

    _write_episode_csv(csv_path, display_episodes, first_ts=first_ts, started_dt=started_dt)
    _write_trace_csv(trace_path, display_replay_trace, first_ts=first_ts, started_dt=started_dt)
    _render_figure(
        fig_path=fig_path,
        session_name=session_dir.name,
        meta=meta,
        csv_rows=display_csv_rows,
        rip_t=display_rip_t,
        rip_y=display_rip_y,
        replay_trace=display_replay_trace,
        interventions=display_interventions,
        episodes=display_episodes,
        first_ts=first_ts,
        window_bounds=None if window_bounds is None else (window_start_ts, window_end_ts),
    )

    print(fig_path)
    print(csv_path)
    print(trace_path)


def _read_chestband_csv(path: Path) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "ts": float(row["ts"]),
                    "packet_sn": int(row["packet_sn"]) if row["packet_sn"] else None,
                    "spo2_pct": _float_or_none(row["spo2_pct"]),
                    "pulse_rate": _float_or_none(row["pulse_rate"]),
                    "gesture": _float_or_none(row.get("gesture", "")),
                }
            )
    return rows


def _read_interventions(path: Path) -> list[InterventionEvent]:
    events: list[InterventionEvent] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "played" in row and not row.get("played", False):
                continue
            events.append(
                InterventionEvent(
                    t=float(row["t"]),
                    loudness=float(row.get("loudness", 0.0)),
                    trigger_score=_float_or_none(row.get("trigger_score")),
                    phase=str(row.get("phase", "")),
                    reason=str(row.get("reason", "")),
                    strategy=str(row.get("strategy", "")),
                    direction=str(row.get("direction", "")),
                )
            )
    return events


def _build_episodes(
    interventions: Sequence[InterventionEvent],
    split_gap_s: float = 8.0,
) -> list[InterventionEpisode]:
    if not interventions:
        return []

    episodes: list[InterventionEpisode] = []
    current: list[InterventionEvent] = [interventions[0]]

    for event in interventions[1:]:
        if event.t - current[-1].t > split_gap_s:
            episodes.append(_episode_from_events(current))
            current = [event]
            continue
        current.append(event)

    episodes.append(_episode_from_events(current))
    return episodes


def _episode_from_events(events: Sequence[InterventionEvent]) -> InterventionEpisode:
    scores = [event.trigger_score for event in events if event.trigger_score is not None]
    return InterventionEpisode(
        start_t=events[0].t,
        end_t=events[-1].t,
        n_sounds=len(events),
        max_loudness=max(event.loudness for event in events),
        max_trigger_score=max(scores) if scores else None,
        start_reason=events[0].reason,
    )


def _read_rip_waveform(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    all_t: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for npz_path in _sorted_chestband_npz_paths(session_dir):
        data = np.load(npz_path)
        ts = np.asarray(data["ts"], dtype=np.float64)
        chest = np.asarray(data["chest_resp"], dtype=np.float32)
        if chest.ndim != 2 or not len(ts):
            continue
        n = chest.shape[1]
        fs = float(n)
        offsets = (np.arange(n, dtype=np.float64) - (n - 1)) / fs
        sample_t = ts[:, None] + offsets[None, :]
        all_t.append(sample_t.reshape(-1))
        all_y.append(chest.reshape(-1))

    if not all_t:
        return np.asarray([]), np.asarray([])
    return np.concatenate(all_t), np.concatenate(all_y)


def _read_rip_packets(session_dir: Path) -> list[tuple[float, np.ndarray, bool]]:
    packets: list[tuple[float, np.ndarray, bool]] = []
    prev_orientation: np.ndarray | None = None

    for npz_path in _sorted_chestband_npz_paths(session_dir):
        data = np.load(npz_path)
        ts = np.asarray(data["ts"], dtype=np.float64)

        if "chest_resp" in data.files:
            rip = np.asarray(data["chest_resp"], dtype=np.float32)
            rip_len = np.full(len(ts), rip.shape[1], dtype=np.int32)
        elif "rip" in data.files:
            rip = np.asarray(data["rip"], dtype=np.float32)
            if "rip_len" in data.files:
                rip_len = np.asarray(data["rip_len"], dtype=np.int32)
            else:
                rip_len = np.full(len(ts), rip.shape[1], dtype=np.int32)
        else:
            continue

        if rip.ndim != 2:
            continue

        count = min(len(ts), rip.shape[0], len(rip_len))
        for idx in range(count):
            n = int(max(0, min(rip.shape[1], rip_len[idx])))
            if n <= 0:
                continue
            accel_x = data["accel_x"][idx] if "accel_x" in data.files and idx < len(data["accel_x"]) else None
            accel_y = data["accel_y"][idx] if "accel_y" in data.files and idx < len(data["accel_y"]) else None
            accel_z = data["accel_z"][idx] if "accel_z" in data.files and idx < len(data["accel_z"]) else None
            movement_detected, prev_orientation = detect_packet_movement(
                accel_x,
                accel_y,
                accel_z,
                prev_orientation=prev_orientation,
            )
            packets.append(
                (
                    float(ts[idx]),
                    np.asarray(rip[idx, :n], dtype=np.float32),
                    movement_detected,
                )
            )

    return packets


def _sorted_chestband_npz_paths(session_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        suffix = path.stem.rsplit("_", 1)[-1]
        try:
            return int(suffix), path.name
        except ValueError:
            return 10**12, path.name

    return sorted(session_dir.glob("chestband_*.npz"), key=sort_key)


def _read_posture_change_times(session_dir: Path) -> list[float]:
    path = session_dir / "events.jsonl"
    if not path.exists():
        return []
    times: list[float] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("kind") != "posture.change":
                continue
            payload = obj.get("payload", {})
            try:
                times.append(float(payload.get("t", obj.get("t"))))
            except (TypeError, ValueError):
                continue
    times.sort()
    return times


def _replay_fuzzy_trace(
    session_dir: Path,
    csv_rows: Sequence[dict[str, float | int | None]],
) -> ReplayTrace:
    packets = _read_rip_packets(session_dir)
    posture_change_times = _read_posture_change_times(session_dir)
    if len(packets) != len(csv_rows):
        raise ValueError(
            f"RIP packets ({len(packets)}) do not match chestband rows ({len(csv_rows)})"
        )

    # Reconstruct the fuzzy internals using the legacy experiment behavior:
    # keep the original scoring path without the later raw-RIP validity guard.
    controller = PreExperimentController(
        PreExperimentConfig(
            min_valid_rip_span=0.0,
            max_valid_rip_span=1e12,
        )
    )

    values: dict[str, list[float]] = {
        "t": [],
        "rip_amplitude": [],
        "rip_baseline": [],
        "rip_amplitude_ratio": [],
        "breath_period_sec": [],
        "breath_period_baseline": [],
        "breath_period_ratio": [],
        "spo2_pct": [],
        "spo2_baseline": [],
        "spo2_delta": [],
        "trigger_score": [],
        "loudness_score": [],
        "should_play_sound": [],
        "strategy_index": [],
    }
    phase: list[str] = []
    strategy_pool = [
        str(item)
        for item in json.loads((session_dir / "meta.json").read_text()).get("config", {}).get("strategy_pool", [])
        if str(item)
    ]
    if not strategy_pool:
        strategy_pool = ["M1", "M2", "M3", "M4", "M5", "M6"]
    strategy_idx = 0
    posture_idx = 0
    prev_packet_ts = float(csv_rows[0]["ts"]) if csv_rows else 0.0

    for row, (packet_ts, rip_batch, movement_detected) in zip(csv_rows, packets):
        row_ts = float(row["ts"])
        if abs(packet_ts - row_ts) > 0.02:
            raise ValueError(f"Packet timestamp mismatch: {packet_ts} vs {row_ts}")

        posture_changed = False
        while posture_idx < len(posture_change_times) and posture_change_times[posture_idx] <= packet_ts:
            if posture_change_times[posture_idx] > prev_packet_ts:
                posture_changed = True
            posture_idx += 1

        spo2 = _clean_spo2(row["spo2_pct"])
        spo2_input = None if np.isnan(spo2) else float(spo2)
        command = controller.update(
            rip=rip_batch,
            button=None,
            spo2_pct=spo2_input,
            timestamp=packet_ts,
            movement_detected=movement_detected,
            posture_changed=posture_changed,
        )
        prev_packet_ts = packet_ts
        snapshot = command.snapshot
        if snapshot is None:
            continue

        values["t"].append(packet_ts)
        values["rip_amplitude"].append(float(snapshot.rip_amplitude))
        values["rip_baseline"].append(_float_or_nan(snapshot.rip_baseline))
        values["rip_amplitude_ratio"].append(_float_or_nan(snapshot.rip_amplitude_ratio))
        values["breath_period_sec"].append(_float_or_nan(snapshot.breath_period_sec))
        values["breath_period_baseline"].append(_float_or_nan(snapshot.breath_period_baseline))
        values["breath_period_ratio"].append(_float_or_nan(snapshot.breath_period_ratio))
        values["spo2_pct"].append(_float_or_nan(snapshot.spo2_pct))
        values["spo2_baseline"].append(_float_or_nan(snapshot.spo2_baseline))
        values["spo2_delta"].append(_float_or_nan(snapshot.spo2_delta))
        values["trigger_score"].append(float(snapshot.trigger_score))
        values["loudness_score"].append(float(snapshot.loudness_score))
        values["should_play_sound"].append(1.0 if command.should_play_sound else 0.0)
        values["strategy_index"].append(float(strategy_idx if command.should_play_sound else np.nan))
        if command.should_play_sound:
            strategy_idx = (strategy_idx + 1) % len(strategy_pool)
        phase.append(command.phase.value if hasattr(command.phase, "value") else str(command.phase))

    return ReplayTrace(
        t=np.asarray(values["t"], dtype=np.float64),
        rip_amplitude=np.asarray(values["rip_amplitude"], dtype=np.float64),
        rip_baseline=np.asarray(values["rip_baseline"], dtype=np.float64),
        rip_amplitude_ratio=np.asarray(values["rip_amplitude_ratio"], dtype=np.float64),
        breath_period_sec=np.asarray(values["breath_period_sec"], dtype=np.float64),
        breath_period_baseline=np.asarray(values["breath_period_baseline"], dtype=np.float64),
        breath_period_ratio=np.asarray(values["breath_period_ratio"], dtype=np.float64),
        spo2_pct=np.asarray(values["spo2_pct"], dtype=np.float64),
        spo2_baseline=np.asarray(values["spo2_baseline"], dtype=np.float64),
        spo2_delta=np.asarray(values["spo2_delta"], dtype=np.float64),
        trigger_score=np.asarray(values["trigger_score"], dtype=np.float64),
        loudness_score=np.asarray(values["loudness_score"], dtype=np.float64),
        should_play_sound=np.asarray(values["should_play_sound"], dtype=np.float64),
        strategy_index=np.asarray(values["strategy_index"], dtype=np.float64),
        phase=phase,
    )


def _write_episode_csv(
    path: Path,
    episodes: Sequence[InterventionEpisode],
    first_ts: float,
    started_dt: datetime,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_index",
                "start_time",
                "end_time",
                "duration_s",
                "n_sounds",
                "max_loudness",
                "max_continue_score",
                "start_reason",
            ]
        )
        for idx, episode in enumerate(episodes, start=1):
            writer.writerow(
                [
                    idx,
                    _fmt_clock(episode.start_t, first_ts, started_dt),
                    _fmt_clock(episode.end_t, first_ts, started_dt),
                    f"{episode.end_t - episode.start_t:.1f}",
                    episode.n_sounds,
                    f"{episode.max_loudness:.2f}",
                    "" if episode.max_trigger_score is None else f"{episode.max_trigger_score:.3f}",
                    episode.start_reason,
                ]
            )


def _write_trace_csv(
    path: Path,
    replay_trace: ReplayTrace,
    first_ts: float,
    started_dt: datetime,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "rip_amplitude",
                "rip_baseline",
                "rip_amplitude_ratio",
                "breath_period_sec",
                "breath_period_baseline",
                "breath_period_ratio",
                "spo2_pct",
                "spo2_baseline",
                "spo2_delta",
                "continue_score",
                "loudness_score",
                "phase",
                "should_play_sound",
            ]
        )
        for idx, epoch_s in enumerate(replay_trace.t):
            writer.writerow(
                [
                    _fmt_clock(epoch_s, first_ts, started_dt),
                    _csv_float(replay_trace.rip_amplitude[idx]),
                    _csv_float(replay_trace.rip_baseline[idx]),
                    _csv_float(replay_trace.rip_amplitude_ratio[idx]),
                    _csv_float(replay_trace.breath_period_sec[idx]),
                    _csv_float(replay_trace.breath_period_baseline[idx]),
                    _csv_float(replay_trace.breath_period_ratio[idx]),
                    _csv_float(replay_trace.spo2_pct[idx]),
                    _csv_float(replay_trace.spo2_baseline[idx]),
                    _csv_float(replay_trace.spo2_delta[idx]),
                    _csv_float(replay_trace.trigger_score[idx]),
                    _csv_float(replay_trace.loudness_score[idx]),
                    replay_trace.phase[idx],
                    int(replay_trace.should_play_sound[idx] > 0.5),
                ]
            )


def _render_figure(
    fig_path: Path,
    session_name: str,
    meta: dict,
    csv_rows: Sequence[dict[str, float | int | None]],
    rip_t: np.ndarray,
    rip_y: np.ndarray,
    replay_trace: ReplayTrace,
    interventions: Sequence[InterventionEvent],
    episodes: Sequence[InterventionEpisode],
    first_ts: float,
    window_bounds: tuple[float, float] | None = None,
) -> None:
    started_dt = datetime.fromisoformat(meta["started_at"])
    times = np.asarray([row["ts"] for row in csv_rows], dtype=np.float64)
    local_times = _to_local_datetimes(times, first_ts, started_dt)
    replay_times = _to_local_datetimes(replay_trace.t, first_ts, started_dt)
    spo2 = np.asarray([_clean_spo2(row["spo2_pct"]) for row in csv_rows], dtype=float)
    pulse = np.asarray([_clean_pulse(row["pulse_rate"]) for row in csv_rows], dtype=float)

    episode_spans = [
        (
            started_dt + timedelta(seconds=ep.start_t - first_ts),
            started_dt + timedelta(seconds=ep.end_t - first_ts),
        )
        for ep in episodes
    ]
    intervention_times = np.asarray(_to_local_datetimes([event.t for event in interventions], first_ts, started_dt))
    intervention_loudness = np.asarray([event.loudness for event in interventions], dtype=float)

    plt.close("all")
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(21, 15.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.7, 1.5, 1.3, 1.1]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.06, right=0.80, top=0.96, bottom=0.06, hspace=0.07)

    ax_rip, ax_baseline, ax_vitals, ax_interventions, ax_strategy = axes

    if rip_t.size:
        ax_rip.plot_date(
            _to_local_datetimes(rip_t, first_ts, started_dt),
            rip_y,
            "-",
            linewidth=0.6,
            color="#0d4d7a",
            alpha=0.9,
            label="Chest RIP",
        )
        rip_ylim = _robust_ylim(rip_y, lower_q=0.5, upper_q=99.5, pad_ratio=0.08)
        if rip_ylim is not None:
            ax_rip.set_ylim(*rip_ylim)
    ax_rip.set_ylabel("Chest RIP")
    ax_rip.grid(alpha=0.25)
    ax_rip.legend(loc="upper right")

    ax_baseline.plot_date(
        replay_times,
        replay_trace.rip_amplitude,
        "-",
        linewidth=1.2,
        color="#3b6ea5",
        alpha=0.85,
        label="RIP amplitude",
    )
    ax_baseline.plot_date(
        replay_times,
        replay_trace.rip_baseline,
        "--",
        linewidth=1.8,
        color="#083d77",
        label="RIP baseline",
    )
    rip_change_mask = _change_mask(replay_trace.rip_baseline, atol=1.0)
    ax_baseline.scatter(
        [dt for dt, keep in zip(replay_times, rip_change_mask) if keep],
        replay_trace.rip_baseline[rip_change_mask],
        s=14,
        color="#083d77",
        alpha=0.9,
        label="RIP baseline change",
    )
    ax_baseline.set_ylabel("RIP Amp")
    ax_baseline.grid(alpha=0.25)

    ax_period = ax_baseline.twinx()
    ax_period.plot_date(
        replay_times,
        replay_trace.breath_period_sec,
        "-",
        linewidth=1.0,
        color="#d97706",
        alpha=0.80,
        label="Breath period (s)",
    )
    ax_period.plot_date(
        replay_times,
        replay_trace.breath_period_baseline,
        "--",
        linewidth=1.6,
        color="#92400e",
        alpha=0.95,
        label="Breath baseline (s)",
    )
    period_change_mask = _change_mask(replay_trace.breath_period_baseline, atol=0.02)
    ax_period.scatter(
        [dt for dt, keep in zip(replay_times, period_change_mask) if keep],
        replay_trace.breath_period_baseline[period_change_mask],
        s=14,
        color="#92400e",
        alpha=0.9,
        label="Breath baseline change",
    )
    ax_period.set_ylabel("Breath Period (s)", color="#92400e")
    ax_period.tick_params(axis="y", colors="#92400e")

    handles1, labels1 = ax_baseline.get_legend_handles_labels()
    handles2, labels2 = ax_period.get_legend_handles_labels()
    ax_baseline.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8, ncol=2)

    ax_vitals.plot_date(
        local_times,
        spo2,
        "-",
        linewidth=1.8,
        color="#c0392b",
        label="SpO2 (%)",
    )
    ax_vitals.plot_date(
        replay_times,
        replay_trace.spo2_baseline,
        "--",
        linewidth=1.6,
        color="#7f1d1d",
        alpha=0.90,
        label="SpO2 baseline",
    )
    spo2_change_mask = _change_mask(replay_trace.spo2_baseline, atol=0.01)
    ax_vitals.scatter(
        [dt for dt, keep in zip(replay_times, spo2_change_mask) if keep],
        replay_trace.spo2_baseline[spo2_change_mask],
        s=14,
        color="#7f1d1d",
        alpha=0.9,
        label="SpO2 baseline change",
    )
    ax_vitals.set_ylabel("SpO2 (%)", color="#c0392b")
    ax_vitals.tick_params(axis="y", colors="#c0392b")
    if np.isfinite(spo2).any():
        ax_vitals.set_ylim(min(90, np.nanmin(spo2) - 1), max(100, np.nanmax(spo2) + 0.5))
    ax_vitals.grid(alpha=0.25)

    ax_pulse = ax_vitals.twinx()
    ax_pulse.plot_date(
        local_times,
        pulse,
        "-",
        linewidth=1.4,
        color="#2e7d32",
        label="Pulse Rate",
    )
    ax_pulse.set_ylabel("Pulse Rate", color="#2e7d32")
    ax_pulse.tick_params(axis="y", colors="#2e7d32")

    handles1, labels1 = ax_vitals.get_legend_handles_labels()
    handles2, labels2 = ax_pulse.get_legend_handles_labels()
    ax_vitals.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)

    ax_interventions.plot_date(
        replay_times,
        replay_trace.trigger_score,
        "-",
        linewidth=1.2,
        color="#444444",
        alpha=0.9,
        label="Continue score (replayed)",
    )
    ax_interventions.axhline(0.70, color="#7f8c8d", linewidth=1.0, linestyle="--", alpha=0.8, label="Legacy start ref")
    ax_interventions.axhline(0.85, color="#a04000", linewidth=1.0, linestyle="--", alpha=0.8, label="Legacy fast ref")
    ax_interventions.axhline(0.30, color="#2c7a7b", linewidth=1.0, linestyle=":", alpha=0.8, label="Continue stop ref")

    if intervention_times.size:
        markerline, stemlines, _ = ax_interventions.stem(
            intervention_times,
            intervention_loudness,
            linefmt="#8e1b1b",
            markerfmt="o",
            basefmt=" ",
        )
        plt.setp(markerline, markersize=4, markerfacecolor="#8e1b1b", markeredgecolor="#8e1b1b")
        plt.setp(stemlines, linewidth=1.2, color="#8e1b1b", alpha=0.9)
    ax_interventions.set_ylim(-0.02, 1.02)
    ax_interventions.set_ylabel("Score / Loudness")
    ax_interventions.set_xlabel("Clock Time")
    ax_interventions.grid(alpha=0.25)
    ax_interventions.legend(loc="upper right", fontsize=8, ncol=2)

    _plot_strategy_panel(
        ax=ax_strategy,
        meta=meta,
        interventions=interventions,
        replay_trace=replay_trace,
        first_ts=first_ts,
        started_dt=started_dt,
    )

    for ax in axes:
        for start_num, end_num in episode_spans:
            ax.axvspan(start_num, end_num, color="#f4b2b2", alpha=0.20, lw=0)

    for idx, episode in enumerate(episodes, start=1):
        x = started_dt + timedelta(seconds=episode.start_t - first_ts)
        ax_interventions.axvline(x, color="#8e1b1b", linewidth=1.0, alpha=0.55)
        ax_interventions.text(
            x,
            episode.max_loudness + 0.03,
            f"E{idx}",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
            color="#7a1010",
        )

    if window_bounds is not None:
        window_start_ts, window_end_ts = window_bounds
        x_start = started_dt + timedelta(seconds=window_start_ts - first_ts)
        x_end = started_dt + timedelta(seconds=window_end_ts - first_ts)
    elif len(times):
        x_start = started_dt + timedelta(seconds=float(times[0]) - first_ts)
        x_end = started_dt + timedelta(seconds=float(times[-1]) - first_ts)
    else:
        x_start = started_dt
        x_end = started_dt

    for ax in axes:
        ax.set_xlim(x_start, x_end)

    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    started_at = meta.get("started_at", "")
    replay_sound_count = int(np.sum(replay_trace.should_play_sound > 0.5))
    subtitle = (
        f"session={session_name} | start={started_at} | "
        f"duration={((times[-1] - times[0]) if len(times) else 0):.1f}s | "
        f"historical sounds={len(interventions)} | historical episodes={len(episodes)} | "
        f"current-rule replay sounds={replay_sound_count} | baselines=replayed"
    )
    fig.suptitle("Night Session Overview", fontsize=18, fontweight="bold", y=0.995)
    fig.text(0.5, 0.975, subtitle, ha="center", va="top", fontsize=11, color="#444444")

    lines = [
        "Dashed lines = fuzzy baselines",
        "Dots = baseline value changed",
        f"Historical sounds={len(interventions)}",
        f"Historical episodes={len(episodes)}",
        f"Current-rule replay sounds={replay_sound_count}",
    ]
    for idx, episode in enumerate(episodes[:8], start=1):
        lines.append(
            f"E{idx}  {_fmt_clock(episode.start_t, first_ts, started_dt)}-"
            f"{_fmt_clock(episode.end_t, first_ts, started_dt)}  "
            f"{episode.n_sounds} sounds"
        )
    if len(episodes) > 8:
        lines.append(f"... {len(episodes) - 8} more episodes")
    summary_text = "\n".join(lines) if lines else "No played interventions"
    fig.text(
        0.815,
        0.95,
        summary_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.savefig(fig_path, dpi=180)


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_local_datetimes(
    epoch_values: Sequence[float],
    first_epoch_s: float,
    started_dt: datetime,
) -> list[datetime]:
    return [started_dt + timedelta(seconds=float(value) - first_epoch_s) for value in epoch_values]


def _clean_spo2(value: float | None) -> float:
    if value is None or value < 50.0 or value > 100.0:
        return np.nan
    return float(value)


def _clean_pulse(value: float | None) -> float:
    if value is None or value <= 0.0 or value > 180.0:
        return np.nan
    return float(value)


def _float_or_nan(value: float | None) -> float:
    if value is None:
        return np.nan
    return float(value)


def _csv_float(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.6f}"


def _change_mask(values: np.ndarray, atol: float) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    prev: float | None = None
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if prev is None or abs(float(value) - prev) > atol:
            mask[idx] = True
        prev = float(value)
    return mask


def _robust_ylim(
    values: np.ndarray,
    *,
    lower_q: float = 0.5,
    upper_q: float = 99.5,
    pad_ratio: float = 0.08,
) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return None

    lo = float(np.percentile(finite, lower_q))
    hi = float(np.percentile(finite, upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None

    raw_span = hi - lo
    fallback_span = max(float(np.std(finite)) * 2.0, 1.0)
    span = max(raw_span, fallback_span)
    center = (lo + hi) / 2.0
    pad = max(span * pad_ratio, fallback_span * 0.05)
    half = span / 2.0 + pad
    return center - half, center + half


def _parse_clock_time(clock_text: str) -> tuple[int, int, int]:
    parts = clock_text.strip().split(":")
    if len(parts) == 2:
        hour, minute = parts
        second = "0"
    elif len(parts) == 3:
        hour, minute, second = parts
    else:
        raise ValueError(f"Invalid clock time: {clock_text!r}")
    return int(hour), int(minute), int(second)


def _resolve_window_bounds(
    started_dt: datetime,
    first_ts: float,
    start_clock: str | None,
    end_clock: str | None,
) -> tuple[float, float, str] | None:
    if not start_clock and not end_clock:
        return None
    if not start_clock or not end_clock:
        raise ValueError("Both --start-clock and --end-clock must be provided together")

    start_h, start_m, start_s = _parse_clock_time(start_clock)
    end_h, end_m, end_s = _parse_clock_time(end_clock)
    start_dt = started_dt.replace(hour=start_h, minute=start_m, second=start_s)
    end_dt = started_dt.replace(hour=end_h, minute=end_m, second=end_s)
    if end_dt <= start_dt:
        end_dt = end_dt + timedelta(days=1)

    start_ts = first_ts + (start_dt - started_dt).total_seconds()
    end_ts = first_ts + (end_dt - started_dt).total_seconds()
    label = f"{start_dt.strftime('%H%M%S')}_{end_dt.strftime('%H%M%S')}"
    return start_ts, end_ts, label


def _slice_replay_trace(replay_trace: ReplayTrace, mask: np.ndarray) -> ReplayTrace:
    return ReplayTrace(
        t=replay_trace.t[mask],
        rip_amplitude=replay_trace.rip_amplitude[mask],
        rip_baseline=replay_trace.rip_baseline[mask],
        rip_amplitude_ratio=replay_trace.rip_amplitude_ratio[mask],
        breath_period_sec=replay_trace.breath_period_sec[mask],
        breath_period_baseline=replay_trace.breath_period_baseline[mask],
        breath_period_ratio=replay_trace.breath_period_ratio[mask],
        spo2_pct=replay_trace.spo2_pct[mask],
        spo2_baseline=replay_trace.spo2_baseline[mask],
        spo2_delta=replay_trace.spo2_delta[mask],
        trigger_score=replay_trace.trigger_score[mask],
        loudness_score=replay_trace.loudness_score[mask],
        should_play_sound=replay_trace.should_play_sound[mask],
        phase=[phase for phase, keep in zip(replay_trace.phase, mask) if keep],
        strategy_index=replay_trace.strategy_index[mask],
    )


def _plot_strategy_panel(
    ax: plt.Axes,
    meta: dict,
    interventions: Sequence[InterventionEvent],
    replay_trace: ReplayTrace,
    first_ts: float,
    started_dt: datetime,
) -> None:
    strategy_pool = [
        str(item)
        for item in meta.get("config", {}).get("strategy_pool", [])
        if str(item)
    ]
    if not strategy_pool:
        strategy_pool = []
        for event in interventions:
            if event.strategy and event.strategy not in strategy_pool:
                strategy_pool.append(event.strategy)

    if not strategy_pool:
        ax.text(0.5, 0.5, "No strategy interventions found", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Strategy")
        ax.grid(alpha=0.25)
        return

    y_map = {strategy: idx for idx, strategy in enumerate(strategy_pool)}
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(strategy_pool)
    ax.set_ylim(-0.6, len(strategy_pool) - 0.4)
    ax.set_ylabel("Strategy")
    ax.set_xlabel("Clock Time")
    ax.grid(alpha=0.25, axis="x")
    ax.set_title("Historical vs Current-Rule Intervention Track", fontsize=11, loc="left")

    for strategy, y in y_map.items():
        ax.axhline(y, color="#e6e6e6", linewidth=0.8, zorder=0)

    direction_markers = {
        "left": "<",
        "right": ">",
        "center": "o",
        "unknown": "s",
    }
    scatter_ref = None

    history_points = [event for event in interventions if event.strategy in y_map]
    if history_points:
        x_hist = _to_local_datetimes([event.t for event in history_points], first_ts, started_dt)
        y_hist = [y_map[event.strategy] for event in history_points]
        ax.scatter(
            x_hist,
            y_hist,
            s=26,
            marker="o",
            facecolors="none",
            edgecolors="#9aa0a6",
            linewidths=0.8,
            alpha=0.9,
            zorder=1,
        )

    replay_mask = np.isfinite(replay_trace.strategy_index) & (replay_trace.should_play_sound > 0.5)
    if np.any(replay_mask):
        x_replay = [dt for dt, keep in zip(_to_local_datetimes(replay_trace.t, first_ts, started_dt), replay_mask) if keep]
        strategy_indices = replay_trace.strategy_index[replay_mask].astype(int)
        y_replay = [strategy_indices[idx] for idx in range(len(strategy_indices))]
        loudness = replay_trace.loudness_score[replay_mask]
        sizes = 55.0 + loudness * 220.0
        scatter_ref = ax.scatter(
            x_replay,
            y_replay,
            c=loudness,
            s=sizes,
            cmap="OrRd",
            vmin=0.0,
            vmax=max(0.6, float(np.nanmax(loudness)) if loudness.size else 0.6),
            marker="o",
            edgecolors="#7a1f1f",
            linewidths=0.45,
            alpha=0.95,
            zorder=3,
        )
    elif not history_points:
        ax.text(0.5, 0.5, "No played interventions found", ha="center", va="center", transform=ax.transAxes)
        return

    if scatter_ref is not None:
        cbar = plt.colorbar(scatter_ref, ax=ax, pad=0.01, fraction=0.035)
        cbar.set_label("Loudness", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#9aa0a6",
            markeredgewidth=0.9,
            markersize=6.5,
            label="Historical logged intervention",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d95f0e",
            markeredgecolor="#7a1f1f",
            markeredgewidth=0.5,
            markersize=7,
            label="Current-rule replay intervention",
        ),
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=1)


def _fmt_clock(epoch_s: float, first_epoch_s: float, started_dt: datetime) -> str:
    return (started_dt + timedelta(seconds=epoch_s - first_epoch_s)).strftime("%H:%M:%S")


if __name__ == "__main__":
    main()
