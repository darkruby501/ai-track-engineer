"""Text-to-speech for the coach.

On Windows a single long-lived ``powershell.exe`` process hosts the built-in
SAPI synthesiser and reads lines from stdin, so speaking starts immediately
(no per-message process start-up). On macOS the ``say`` command is used for
local testing. Every speaker also logs what it said.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Protocol

import structlog

_log = structlog.get_logger("coach.voice")

# Lines starting with "!" interrupt whatever is being said.
_PS_SPEAKER = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = {rate}
$s.Volume = 100
while (($line = [Console]::In.ReadLine()) -ne $null) {{
  if ($line.StartsWith('!')) {{ $s.SpeakAsyncCancelAll(); $line = $line.Substring(1) }}
  if ($line.Trim()) {{ [void]$s.SpeakAsync($line) }}
}}
"""


class Speaker(Protocol):
    def say(self, text: str, *, urgent: bool = False) -> None: ...

    def close(self) -> None: ...


class LogSpeaker:
    """Only logs (and remembers) what would have been said."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str, *, urgent: bool = False) -> None:
        self.spoken.append(text)
        _log.info("say", text=text, urgent=urgent)

    def close(self) -> None:
        return None


class WindowsSpeaker(LogSpeaker):  # pragma: no cover - needs Windows SAPI
    def __init__(self, rate: int = 2) -> None:
        super().__init__()
        self._proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _PS_SPEAKER.format(rate=rate),
            ],
            stdin=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def say(self, text: str, *, urgent: bool = False) -> None:
        super().say(text, urgent=urgent)
        if self._proc.stdin is None or self._proc.poll() is not None:
            return
        clean = " ".join(text.split())
        self._proc.stdin.write(("!" if urgent else "") + clean + "\n")
        self._proc.stdin.flush()

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.terminate()


class MacSpeaker(LogSpeaker):  # pragma: no cover - local testing only
    def say(self, text: str, *, urgent: bool = False) -> None:
        super().say(text, urgent=urgent)
        subprocess.Popen(["say", "-r", "210", text])


def make_speaker(*, enabled: bool, rate: int = 2) -> LogSpeaker:
    """The best available speaker for this platform (or a silent logger)."""
    if not enabled:
        return LogSpeaker()
    if sys.platform == "win32":  # pragma: no cover - platform-specific
        return WindowsSpeaker(rate=rate)
    if sys.platform == "darwin":  # pragma: no cover - platform-specific
        return MacSpeaker()
    return LogSpeaker()  # pragma: no cover
