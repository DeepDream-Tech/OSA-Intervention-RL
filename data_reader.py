"""Realtime DataPacket reader for the intervention-v0 pre-experiment.

This module is intentionally small and dependency-light. It does not import the
main chest-band classes directly; instead it accepts DataPacket-like objects
emitted by /home/osa-main as ``chestband.data`` EventBus payloads.

Output is shaped for ``PreExperimentController.update``:

    controller.update(
        rip=sample.rip,
        button=sample.awake_button,
        spo2_pct=sample.spo2_pct,
        timestamp=sample.timestamp,
    )

The awake-button stream is implemented as a keyboard spacebar listener. When the
participant wakes up and presses Space in the terminal running this process, the
next button batch contains a single 1 aligned to the latest RIP sample. If no
terminal input is available, it safely falls back to all-zero button batches.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass(frozen=True)
class DataPacketSample:
    """One controller-ready sample batch derived from a chest-band DataPacket."""

    timestamp: float
    rip: np.ndarray
    awake_button: np.ndarray
    packet_sn: Optional[int] = None
    rip_fs: float = 25.0
    spo2_pct: Optional[float] = None
    pulse_rate: Optional[float] = None


class AwakeButtonPlaceholder:
    """Manual/fallback awake-button timing source.

    Every RIP batch is aligned with a same-length button array. Values are 0 by
    default. ``inject_press`` makes the next non-empty batch contain one 1 at
    the final sample; this is useful for tests and as a fallback when keyboard
    input is unavailable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_presses = 0

    def inject_press(self) -> None:
        """Make the next ``read_batch`` contain one press sample."""

        with self._lock:
            self._pending_presses += 1

    def _consume_press(self) -> bool:
        with self._lock:
            if self._pending_presses <= 0:
                return False
            self._pending_presses -= 1
            return True

    def read_batch(self, n_samples: int, timestamp: Optional[float] = None) -> np.ndarray:
        del timestamp
        button = np.zeros(max(0, int(n_samples)), dtype=np.float32)
        if len(button) > 0 and self._consume_press():
            button[-1] = 1.0
        return button

    def close(self) -> None:
        """Release resources. Fallback source has nothing to release."""


class SpacebarAwakeButtonSource(AwakeButtonPlaceholder):
    """Awake-button source backed by the computer keyboard Space key.

    This source starts a daemon thread that watches ``stdin`` without blocking
    the DataPacket path. It works when the process is run in an interactive
    terminal. In services without a TTY, ``available`` is False and batches stay
    all-zero, preserving the old placeholder behavior.
    """

    def __init__(self, stream: Optional[Any] = None, autostart: bool = True) -> None:
        super().__init__()
        self.stream = stream if stream is not None else sys.stdin
        self.available = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None
        self._old_termios = None
        self._atexit_registered = False
        if autostart:
            self.start()

    def start(self) -> bool:
        """Start listening for Space. Returns True when a listener is active."""

        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()

        if os.name == 'nt':
            self.available = True
            self._thread = threading.Thread(
                target=self._windows_loop,
                name='awake-spacebar-listener',
                daemon=True,
            )
            self._thread.start()
            return True

        if not self._prepare_posix_terminal():
            self.available = False
            return False

        self.available = True
        self._thread = threading.Thread(
            target=self._posix_loop,
            name='awake-spacebar-listener',
            daemon=True,
        )
        self._thread.start()
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=0.2)
        self._restore_posix_terminal()
        self.available = False

    def _prepare_posix_terminal(self) -> bool:
        try:
            if not hasattr(self.stream, 'isatty') or not self.stream.isatty():
                return False
            import termios
            import tty
            self._fd = self.stream.fileno()
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            return True
        except Exception:
            self._fd = None
            self._old_termios = None
            return False

    def _restore_posix_terminal(self) -> None:
        if self._fd is None or self._old_termios is None:
            return
        try:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass
        finally:
            self._fd = None
            self._old_termios = None

    def _posix_loop(self) -> None:
        try:
            import select
            while not self._stop.is_set():
                readable, _, _ = select.select([self.stream], [], [], 0.05)
                if not readable:
                    continue
                ch = self.stream.read(1)
                if ch == ' ':
                    self.inject_press()
        except Exception:
            self.available = False
        finally:
            self._restore_posix_terminal()

    def _windows_loop(self) -> None:
        try:
            import msvcrt
            while not self._stop.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == ' ':
                        self.inject_press()
                time.sleep(0.05)
        except Exception:
            self.available = False


class ChestbandDataPacketReader:
    """Convert realtime chest-band DataPackets into controller input batches."""

    def __init__(
        self,
        rip_fs: float = 25.0,
        button_source: Optional[AwakeButtonPlaceholder] = None,
    ) -> None:
        self.rip_fs = float(rip_fs)
        self.button_source = button_source or SpacebarAwakeButtonSource()
        self.last_sample: Optional[DataPacketSample] = None

    def close(self) -> None:
        close = getattr(self.button_source, 'close', None)
        if callable(close):
            close()

    def attach_event_bus(self, bus: Any, callback: Callable[[DataPacketSample], None]) -> Callable[[], None]:
        """Subscribe to /home/osa-main ``chestband.data`` events.

        ``callback`` receives a ``DataPacketSample`` for every packet that has
        a non-empty ``chest_resp`` RIP waveform. The return value is the
        EventBus unsubscribe function.
        """

        def _on_event(ev: Any) -> None:
            sample = self.from_event(ev)
            if sample is not None:
                callback(sample)

        return bus.subscribe("chestband.data", _on_event)

    def from_event(self, ev: Any) -> Optional[DataPacketSample]:
        """Read an EventBus event whose payload is a DataPacket-like object."""

        return self.from_datapacket(
            getattr(ev, "payload", ev),
            timestamp=getattr(ev, "t", None),
        )

    def from_datapacket(
        self,
        dp: Any,
        timestamp: Optional[float] = None,
    ) -> Optional[DataPacketSample]:
        """Read one DataPacket-like object.

        Returns ``None`` when the packet has no chest RIP samples. This keeps
        callers simple: only non-empty RIP batches are forwarded to the
        controller.
        """

        rip = getattr(dp, "chest_resp", None)
        rip_arr = np.asarray(rip if rip is not None else [], dtype=np.float32).reshape(-1)
        if rip_arr.size == 0:
            return None

        ts = float(timestamp if timestamp is not None else time.time())
        button = self.button_source.read_batch(len(rip_arr), timestamp=ts)
        vitals = getattr(dp, "vitals", None)
        sample = DataPacketSample(
            timestamp=ts,
            rip=rip_arr,
            awake_button=button,
            packet_sn=getattr(dp, "packet_sn", None),
            rip_fs=self.rip_fs,
            spo2_pct=_optional_float(getattr(vitals, "spo2_pct", None) if vitals else None),
            pulse_rate=_optional_float(getattr(vitals, "pulse_rate", None) if vitals else None),
        )
        self.last_sample = sample
        return sample


def update_controller_from_datapacket(
    controller: Any,
    dp: Any,
    reader: Optional[ChestbandDataPacketReader] = None,
    timestamp: Optional[float] = None,
) -> Optional[Any]:
    """Convenience helper: DataPacket -> reader sample -> controller command."""

    reader = reader or ChestbandDataPacketReader(
        rip_fs=getattr(getattr(controller, "config", None), "rip_fs", 25.0)
    )
    sample = reader.from_datapacket(dp, timestamp=timestamp)
    if sample is None:
        return None
    return controller.update(
        rip=sample.rip,
        button=sample.awake_button,
        spo2_pct=sample.spo2_pct,
        timestamp=sample.timestamp,
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
