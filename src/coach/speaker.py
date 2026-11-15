"""Text-to-speech for the coach.

* :class:`ElevenLabsSpeaker` - natural voice from ElevenLabs. Audio is cached
  on disk by text, and recurring phrases (cue words, every corner note) are
  generated ahead of time with :meth:`prefetch`, so position cues play
  instantly. Anything that fails falls back to the system voice.
* :class:`WindowsSpeaker` - the built-in SAPI voice in one long-lived
  ``powershell.exe`` (no start-up delay per message).
* :class:`MacSpeaker` - ``say``, for local testing.

Every speaker also logs what it said. ``urgent`` speech interrupts whatever is
playing and drops anything queued.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

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

CUE_WORDS = ("Brake", "Turn in", "Throttle", "Coach ready.")


class Speaker(Protocol):
    def say(self, text: str, *, urgent: bool = False) -> None: ...

    def prefetch(self, texts: Iterable[str]) -> None: ...

    def close(self) -> None: ...


class LogSpeaker:
    """Only logs (and remembers) what would have been said."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str, *, urgent: bool = False) -> None:
        self.spoken.append(text)
        _log.info("say", text=text, urgent=urgent)

    def prefetch(self, texts: Iterable[str]) -> None:
        return None

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


# -- ElevenLabs -------------------------------------------------------------------
PCM_RATE = 22050


def pcm_to_wav(pcm: bytes, rate: int = PCM_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())


class Player(Protocol):
    def play(self, path: Path) -> None: ...

    def stop(self) -> None: ...


class WinsoundPlayer:  # pragma: no cover - Windows only
    def play(self, path: Path) -> None:
        if sys.platform == "win32":
            import winsound

            flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
            winsound.PlaySound(str(path), flags)

    def stop(self) -> None:
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(None, 0)


class AfplayPlayer:  # pragma: no cover - macOS testing only
    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None

    def play(self, path: Path) -> None:
        self.stop()
        self._proc = subprocess.Popen(["afplay", str(path)])

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


HttpPost = Callable[[str, dict[str, str], dict[str, Any]], bytes]
HttpGet = Callable[[str, dict[str, str]], dict[str, Any]]


def _httpx_post(
    url: str, headers: dict[str, str], body: dict[str, Any]
) -> bytes:  # pragma: no cover
    import httpx

    resp = httpx.post(url, headers=headers, json=body, timeout=8.0)
    resp.raise_for_status()
    return resp.content


def _httpx_get(url: str, headers: dict[str, str]) -> dict[str, Any]:  # pragma: no cover
    import httpx

    resp = httpx.get(url, headers=headers, timeout=8.0)
    resp.raise_for_status()
    return dict(resp.json())


PREFERRED_VOICES = ("Brian", "Adam", "George", "Daniel", "Chris", "Eric")


class ElevenLabsSpeaker(LogSpeaker):
    """ElevenLabs voice with an on-disk cache and a fallback speaker."""

    API = "https://api.elevenlabs.io/v1"

    def __init__(
        self,
        api_key: str,
        *,
        voice: str = "",
        model: str = "eleven_flash_v2_5",
        speed: float = 1.1,
        cache_dir: Path = Path("data/local/tts_cache"),
        fallback: LogSpeaker | None = None,
        player: Player | None = None,
        http_post: HttpPost | None = None,
        http_get: HttpGet | None = None,
        start_threads: bool = True,
    ) -> None:
        super().__init__()
        self._headers = {"xi-api-key": api_key}
        self._voice_pref = voice
        self._voice_id: str | None = voice if len(voice) >= 20 else None
        self._model = model
        self._speed = speed
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)
        self._fallback = fallback or LogSpeaker()
        self._player = player or (WinsoundPlayer() if sys.platform == "win32" else AfplayPlayer())
        self._post = http_post or _httpx_post
        self._get = http_get or _httpx_get
        self._queue: deque[str] = deque()
        self._prefetch_queue: deque[str] = deque()
        self._cv = threading.Condition()
        self._busy_until = 0.0
        self._closed = False
        self.failures = 0
        if start_threads:
            threading.Thread(target=self._play_loop, daemon=True).start()
            threading.Thread(target=self._prefetch_loop, daemon=True).start()

    # -- voice and audio -------------------------------------------------------
    def voice_id(self) -> str:
        if self._voice_id is None:
            voices = self._get(f"{self.API}/voices", self._headers).get("voices", [])
            wanted = [self._voice_pref] if self._voice_pref else list(PREFERRED_VOICES)
            chosen = next(
                (
                    v
                    for name in wanted
                    for v in voices
                    if str(v.get("name", "")).lower().startswith(name.lower())
                ),
                voices[0] if voices else None,
            )
            if chosen is None:
                raise RuntimeError("no ElevenLabs voices available on this account")
            self._voice_id = str(chosen["voice_id"])
            _log.info("elevenlabs-voice", name=chosen.get("name"), voice_id=self._voice_id)
        return self._voice_id

    def _cache_path(self, text: str) -> Path:
        key = f"{self._voice_pref}|{self._model}|{self._speed}|{text}"
        return self._cache / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".wav")

    def audio(self, text: str) -> Path:
        """Path to a WAV of *text*, synthesising (and caching) it if needed."""
        path = self._cache_path(text)
        if path.is_file():
            return path
        url = f"{self.API}/text-to-speech/{self.voice_id()}?output_format=pcm_{PCM_RATE}"
        body = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": self._speed},
        }
        pcm = self._post(url, self._headers, body)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pcm_to_wav(pcm))
        os.replace(tmp, path)
        return path

    def cached(self, text: str) -> bool:
        return self._cache_path(text).is_file()

    # -- speaking ----------------------------------------------------------------
    def say(self, text: str, *, urgent: bool = False) -> None:
        super().say(text, urgent=urgent)
        clean = " ".join(text.split())
        if not clean:
            return
        with self._cv:
            if urgent:
                self._queue.clear()
                self._player.stop()
                self._busy_until = 0.0
                self._queue.appendleft(clean)
            else:
                self._queue.append(clean)
            self._cv.notify_all()

    def prefetch(self, texts: Iterable[str]) -> None:
        with self._cv:
            for t in texts:
                clean = " ".join(t.split())
                if clean and not self.cached(clean) and clean not in self._prefetch_queue:
                    self._prefetch_queue.append(clean)
            self._cv.notify_all()

    def play_next(self) -> bool:
        """Play the next queued phrase if the player is free (the loop calls this)."""
        with self._cv:
            if not self._queue or time.monotonic() < self._busy_until:
                return False
            text = self._queue.popleft()
        try:
            path = self.audio(text)
            seconds = wav_seconds(path.read_bytes())
            self._player.play(path)
            with self._cv:
                self._busy_until = time.monotonic() + seconds
        except Exception as exc:  # noqa: BLE001 - never lose a cue: use the system voice
            self.failures += 1
            _log.warning("elevenlabs-failed", text=text, error=repr(exc))
            self._fallback.say(text, urgent=True)
        return True

    def _play_loop(self) -> None:  # pragma: no cover - thread body
        while not self._closed:
            with self._cv:
                wait = max(0.01, self._busy_until - time.monotonic()) if self._queue else 0.5
                self._cv.wait(timeout=wait)
            self.play_next()

    def prefetch_next(self) -> bool:
        with self._cv:
            if not self._prefetch_queue:
                return False
            text = self._prefetch_queue.popleft()
        try:
            self.audio(text)
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            _log.warning("elevenlabs-prefetch-failed", text=text, error=repr(exc))
        return True

    def _prefetch_loop(self) -> None:  # pragma: no cover - thread body
        while not self._closed:
            if not self.prefetch_next():
                with self._cv:
                    self._cv.wait(timeout=1.0)

    def close(self) -> None:
        self._closed = True
        with self._cv:
            self._cv.notify_all()
        self._fallback.close()


def make_speaker(
    *,
    enabled: bool,
    rate: int = 2,
    engine: str = "auto",
    voice: str = "",
    model: str = "eleven_flash_v2_5",
    speed: float = 1.1,
    cache_dir: str = "data/local/tts_cache",
) -> LogSpeaker:
    """The best available speaker (or a silent logger when disabled)."""
    if not enabled:
        return LogSpeaker()
    system: LogSpeaker
    if sys.platform == "win32":  # pragma: no cover - platform-specific
        system = WindowsSpeaker(rate=rate)
    elif sys.platform == "darwin":  # pragma: no cover - platform-specific
        system = MacSpeaker()
    else:  # pragma: no cover
        system = LogSpeaker()
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if engine in ("auto", "elevenlabs") and key:  # pragma: no cover - needs a key
        speaker = ElevenLabsSpeaker(
            key, voice=voice, model=model, speed=speed, cache_dir=Path(cache_dir), fallback=system
        )
        speaker.prefetch(CUE_WORDS)
        return speaker
    return system
