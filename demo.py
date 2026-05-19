"""Interactive synthetic OSA demo for the fuzzy intervention controller.

This script fabricates a short physiological session with:

- stable breathing baseline
- one OSA-like event with reduced RIP amplitude and falling SpO2
- optional user wake-up via Space

Run in a terminal so the spacebar listener can capture key presses:

    python demo.py

Press Space after you see intervention start to test the awake-button path.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from controller import PreExperimentConfig, PreExperimentController
from data_reader import DataPacketSample, SpacebarAwakeButtonSource
from recorder import PreExperimentRecorder, PreExperimentSessionMeta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an interactive synthetic OSA demo.")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=0.5,
        help="Real seconds per simulated second. Default: 0.5",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=36.0,
        help="Total simulated duration in seconds. Default: 36",
    )
    parser.add_argument(
        "--record-root",
        type=Path,
        default=Path("tmp_demo_sessions"),
        help="Directory for demo recorder output. Default: tmp_demo_sessions",
    )
    parser.add_argument(
        "--auto-press-at",
        type=float,
        default=None,
        help="Inject a synthetic Space press at the given simulated second.",
    )
    return parser.parse_args()


class SyntheticOsaSession:
    """Fabricate a normal -> OSA -> recovery trajectory."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.awakened_at: Optional[float] = None

    def mark_awake(self, t_sec: float) -> None:
        if self.awakened_at is None:
            self.awakened_at = t_sec

    def rip_batch(self, start_t: float, end_t: float, fs: float) -> np.ndarray:
        n = max(1, int(round((end_t - start_t) * fs)))
        dt = 1.0 / max(fs, 1e-6)
        values = []
        for idx in range(n):
            t = start_t + idx * dt
            amp, period, offset = self._state_at(t)
            phase = 2.0 * math.pi * (t + offset) / max(period, 1e-6)
            noise = self.rng.uniform(-0.03, 0.03)
            values.append(amp * math.sin(phase) + noise)
        return np.asarray(values, dtype=np.float32)

    def spo2_at(self, t_sec: float) -> float:
        _, _, spo2 = self._state_at(t_sec)
        return round(spo2, 2)

    def _state_at(self, t_sec: float) -> tuple[float, float, float]:
        if self.awakened_at is not None and t_sec >= self.awakened_at:
            recover_t = t_sec - self.awakened_at
            amp = _lerp(0.12, 1.75, min(recover_t / 5.0, 1.0))
            period = _lerp(5.4, 3.2, min(recover_t / 5.0, 1.0))
            spo2 = _lerp(93.2, 97.2, min(recover_t / 10.0, 1.0))
            return amp, period, spo2

        if t_sec < 10.0:
            return 2.0, 3.0, 98.0
        if t_sec < 15.0:
            progress = (t_sec - 10.0) / 5.0
            amp = _lerp(2.0, 0.45, progress)
            period = _lerp(3.0, 4.8, progress)
            spo2 = _lerp(98.0, 96.2, progress)
            return amp, period, spo2
        if t_sec < 24.0:
            progress = (t_sec - 15.0) / 9.0
            amp = _lerp(0.45, 0.10, progress)
            period = _lerp(4.8, 5.4, progress)
            spo2 = _lerp(96.2, 93.2, progress)
            return amp, period, spo2
        if t_sec < 32.0:
            progress = (t_sec - 24.0) / 8.0
            amp = _lerp(0.10, 1.45, progress)
            period = _lerp(5.4, 3.4, progress)
            spo2 = _lerp(93.2, 96.6, progress)
            return amp, period, spo2
        return 1.7, 3.2, 97.0


def main() -> None:
    args = parse_args()

    config = PreExperimentConfig(rip_fs=25.0)
    controller = PreExperimentController(config)
    controller.set_baseline(rip_amplitude=2.0, breath_period_sec=3.0, spo2_pct=98.0)

    button_source = SpacebarAwakeButtonSource()
    recorder = PreExperimentRecorder(
        PreExperimentSessionMeta(
            subject_id="synthetic_demo",
            note="interactive synthetic OSA session",
            config={
                "rip_fs": config.rip_fs,
                "time_scale": args.time_scale,
                "duration_sec": args.duration_sec,
            },
        ),
        root=args.record_root,
    )
    scenario = SyntheticOsaSession(random.Random(42))

    batch_sec = 1.0
    sim_t = 0.0
    packet_sn = 0
    auto_pressed = False

    print("交互仿真开始")
    print(f"记录目录: {recorder.dir}")
    if button_source.available:
        print("空格键监听已启用。看到进入 intervening 后，按空格测试用户清醒中断。")
    else:
        print("当前环境没有可用 TTY，空格键监听不可用。可以改用 --auto-press-at 做自动测试。")
    print("-" * 88)
    print(
        " sim_t | phase         | trig | loud | spo2 | amp_ratio | period_ratio | play | reason"
    )
    print("-" * 88)

    try:
        while sim_t < args.duration_sec:
            start_t = sim_t
            end_t = min(sim_t + batch_sec, args.duration_sec)
            sim_t = end_t

            if args.auto_press_at is not None and not auto_pressed and sim_t >= args.auto_press_at:
                button_source.inject_press()
                auto_pressed = True

            rip = scenario.rip_batch(start_t, end_t, fs=config.rip_fs)
            spo2_pct = scenario.spo2_at(end_t)
            button = button_source.read_batch(len(rip), timestamp=end_t)

            sample = DataPacketSample(
                timestamp=end_t,
                rip=rip,
                awake_button=button,
                packet_sn=packet_sn,
                rip_fs=config.rip_fs,
                spo2_pct=spo2_pct,
            )
            packet_sn += 1

            command = controller.update(
                rip=sample.rip,
                button=sample.awake_button,
                spo2_pct=sample.spo2_pct,
                timestamp=sample.timestamp,
            )
            recorder.record_step(sample, command, cue_params={"loudness": command.loudness} if command.should_play_sound else None)

            snapshot = command.snapshot
            amp_ratio = _fmt(snapshot.rip_amplitude_ratio)
            period_ratio = _fmt(snapshot.breath_period_ratio)
            print(
                f"{end_t:6.1f} | "
                f"{command.phase.value:12s} | "
                f"{snapshot.trigger_score:4.2f} | "
                f"{snapshot.loudness_score:4.2f} | "
                f"{spo2_pct:4.1f} | "
                f"{amp_ratio:9s} | "
                f"{period_ratio:12s} | "
                f"{'yes' if command.should_play_sound else ' no'}  | "
                f"{command.reason}"
            )

            if snapshot.button_pressed:
                scenario.mark_awake(end_t)
                print(f"[{end_t:4.1f}s] 检测到空格键，后续生理数据切换到清醒后恢复轨迹。")

            if args.time_scale > 0:
                time.sleep(batch_sec * args.time_scale)
    finally:
        button_source.close()
        recorder.close()

    print("-" * 88)
    print("仿真结束")
    print(f"结果已写入: {recorder.dir}")


def _lerp(start: float, end: float, progress: float) -> float:
    p = min(max(progress, 0.0), 1.0)
    return start + (end - start) * p


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


if __name__ == "__main__":
    main()
