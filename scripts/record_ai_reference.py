"""Record AI-driven reference laps from a live Assetto Corsa session (Windows only).

Optionally launches ``acs.exe`` (using whatever ``cfg/race.ini`` currently
describes), waits for the shared-memory pages, hands the player's car to the
AI by injecting AC's "AI drives my car" shortcut (Ctrl+C), then records every
physics frame to CSV until the requested number of laps is complete.

It must run inside the interactive desktop session that owns AC (shared
memory and keyboard input are per-session), e.g. from a scheduled task with
an interactive logon, not from an SSH shell.

    uv run python scripts/record_ai_reference.py --launch --laps 5
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import subprocess
import sys
import time
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path

from src.config import find_ac_install, get_settings
from src.storage.session_recorder import _COLUMNS, _row
from src.telemetry import SharedMemoryTelemetrySource
from src.telemetry.models import TelemetryFrame
from src.telemetry.shm_reader import SharedMemoryUnavailableError
from src.telemetry.shm_structs import ACStatus

EXTRA_COLUMNS = ["current_time_ms", "last_time_ms", "in_pit", "compound", "status"]

_log_file = None


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


# ---------------------------------------------------------------------------
# Windows focus + keyboard injection (scan codes, so DirectInput sees them)
# ---------------------------------------------------------------------------
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
VK_MENU = 0x12
SCAN = {"ctrl": 0x1D, "c": 0x2E}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _user32():  # type: ignore[no-untyped-def]
    return ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]


def windows_for_pid(pid: int) -> list[int]:
    user32 = _user32()
    found: list[int] = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)  # type: ignore[attr-defined]

    def callback(hwnd: int, _lparam: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    return found


def focus_window(hwnd: int) -> bool:
    user32 = _user32()
    # Tapping Alt lifts the foreground lock so SetForegroundWindow is honoured.
    user32.keybd_event(VK_MENU, 0, 0, 0)
    ok = bool(user32.SetForegroundWindow(hwnd))
    user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
    return ok


def send_chord(keys: list[str]) -> None:
    user32 = _user32()
    events = [(SCAN[k], False) for k in keys] + [(SCAN[k], True) for k in reversed(keys)]
    for scan, up in events:
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
        inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, scan, flags, 0, 0))
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.06)


def find_ac_pid() -> int | None:
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq acs.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, check=False,
    ).stdout
    for row in csv.reader(out.splitlines()):
        if len(row) > 1 and row[0].lower() == "acs.exe":
            return int(row[1])
    return None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def wait_for(predicate: Callable[[], bool], timeout_s: float, poll_s: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def extra(frame: TelemetryFrame) -> list[object]:
    g = frame.graphics
    return [g.current_time_ms, g.last_time_ms, int(g.is_in_pit), g.tyre_compound, int(g.status)]


def main() -> int:
    global _log_file
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--laps", type=int, default=5, help="flying laps to record after the first")
    ap.add_argument("--launch", action="store_true", help="start acs.exe first")
    ap.add_argument("--no-ai-key", action="store_true", help="don't send Ctrl+C (record a human)")
    ap.add_argument("--timeout-min", type=float, default=30.0)
    ap.add_argument("--out-dir", default="data/sessions")
    ap.add_argument("--keep-running", action="store_true", help="leave AC running afterwards")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    _log_file = open(out_dir / f"ai_ref_{stamp}.log", "w", encoding="utf-8")  # noqa: SIM115

    proc: subprocess.Popen[bytes] | None = None
    if args.launch:
        ac_dir = find_ac_install(get_settings().app.ac_install_path)
        if ac_dir is None:
            log("ERROR: Assetto Corsa install not found")
            return 2
        log(f"launching {Path(ac_dir) / 'acs.exe'}")
        proc = subprocess.Popen([str(Path(ac_dir) / "acs.exe")], cwd=str(ac_dir))

    source = SharedMemoryTelemetrySource()
    static = None

    def try_connect() -> bool:
        nonlocal static
        try:
            static = source.connect()
            return True
        except SharedMemoryUnavailableError:
            return False

    if not wait_for(try_connect, 240, 1.0) or static is None:
        log("ERROR: shared memory never appeared")
        return 3
    log(
        f"connected: car={static.car_model} track={static.track} "
        f"layout={static.track_configuration}"
    )

    def live() -> bool:
        f = source.read_frame()
        return f.graphics.status == ACStatus.LIVE and f.physics.packet_id > 0

    if not wait_for(live, 240, 1.0):
        log("ERROR: session never went live")
        return 4
    time.sleep(4.0)
    first = source.read_frame()
    log(
        f"live: session={first.graphics.session_type.name} "
        f"compound={first.graphics.tyre_compound!r} in_pit={first.graphics.is_in_pit} "
        f"pos={first.graphics.normalized_car_position:.4f}"
    )

    if not args.no_ai_key:
        pid = find_ac_pid()
        hwnds = windows_for_pid(pid) if pid else []
        log(f"acs.exe pid={pid} windows={hwnds}")
        moving = False
        for attempt, chord in enumerate((["ctrl", "c"], ["ctrl", "c"], ["c"]), start=1):
            if hwnds:
                log(f"focus window -> {focus_window(hwnds[0])}")
            time.sleep(0.5)
            log(f"attempt {attempt}: sending {'+'.join(chord)}")
            send_chord(chord)
            moving = wait_for(lambda: source.read_frame().physics.speed_kmh > 15.0, 20, 0.25)
            if moving:
                log("AI is driving (speed > 15 km/h)")
                break
        if not moving:
            log("ERROR: car did not start moving after the AI shortcut")
            if proc is not None and not args.keep_running:
                proc.terminate()
            return 5

    name = f"ai_ref_{static.track}_{static.track_configuration}_{static.car_model}_{stamp}"
    csv_path = out_dir / f"{name}.csv"
    laps: list[dict[str, object]] = []
    deadline = time.monotonic() + args.timeout_min * 60
    last_packet = -1
    rows = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_COLUMNS + EXTRA_COLUMNS)
        frame = source.read_frame()
        start_laps = frame.graphics.completed_laps
        prev_laps = start_laps
        log(f"recording to {csv_path} (completed_laps at start = {start_laps})")
        while time.monotonic() < deadline:
            try:
                frame = source.read_frame()
            except Exception as exc:  # noqa: BLE001 - AC closed or memory went away
                log(f"stopping: read failed ({exc})")
                break
            if frame.graphics.status == ACStatus.OFF:
                log("stopping: AC status OFF")
                break
            if frame.physics.packet_id != last_packet:
                last_packet = frame.physics.packet_id
                writer.writerow(_row(frame) + extra(frame))
                rows += 1
                if rows % 120 == 0:
                    fh.flush()
            done = frame.graphics.completed_laps
            if done != prev_laps:
                lap = {"lap_index": done - start_laps, "time_ms": frame.graphics.last_time_ms}
                laps.append(lap)
                log(f"lap {lap['lap_index']} complete: {frame.graphics.last_time_ms / 1000:.3f}s")
                prev_laps = done
                if done - start_laps >= args.laps + 1:
                    log("requested laps complete")
                    break
            time.sleep(1 / 120)
    source.close()

    summary = {
        "csv": str(csv_path),
        "rows": rows,
        "car": static.car_model,
        "track": static.track,
        "layout": static.track_configuration,
        "compound": first.graphics.tyre_compound,
        "laps": laps,
        "note": "lap_index 1 includes the standing start; later laps are flying laps",
    }
    (out_dir / f"ai_ref_{stamp}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"wrote {rows} rows; summary {json.dumps(laps)}")

    if proc is not None and not args.keep_running:
        log("closing AC")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
