"""Voice input for the cobot-claw REPL, with speech-to-text running locally.

- Push-to-talk: press the voice key (default F9) at the prompt and speak.
  - Pause: the transcript is put in the prompt to check and send with Enter
    (or sent at once with voice.auto_send).
  - Press the key again instead: recording stops and the transcript is sent
    at once, as if Enter was pressed. This also works while it is still
    transcribing.
- Spoken stop: while a task is running, a listener waits for "stop" (or
  "halt" / "freeze"). It cancels the task and brakes the arm (stop_motion: URScript stopj),
  without going through the AI. This is a convenience, not a safety function.
  The robot's E-stop is the safety stop.

Speech recognition is faster-whisper (CTranslate2, int8 on the CPU), fully
offline once the model has been downloaded. Audio comes from `arecord` (ALSA),
so no PortAudio install is needed. Install with: pip install -e ".[voice]"
"""

from __future__ import annotations

import collections
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Callable

import numpy as np

from cobot.config import VoiceConfig

SAMPLE_RATE = 16000
FRAME = 480                      # 30 ms at 16 kHz
PREROLL_FRAMES = 20              # keep 0.6 s from before speech starts

_STOP_WORDS = {"stop", "halt", "freeze"}
_STOP_FILLER = {"stop", "halt", "freeze", "emergency", "please", "now", "robot", "cobot",
                "arm", "the", "it", "everything", "right", "immediately"}


def is_stop_command(text: str) -> bool:
    """True for short commands like "Stop!", "halt now" or "robot stop".
    Longer sentences ("don't stop until...") are not stop commands."""
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and len(words) <= 4 and bool(_STOP_WORDS & set(words)) \
        and set(words) <= _STOP_FILLER


class MicStream:
    """One `arecord` capture kept open for the whole session.

    Opening the microphone per key press lost the first words: from cold, the
    built-in mic delivers ~0.5 s pinned at full scale plus a burst before it
    settles. The stream is read continuously; audio is only kept or
    transcribed while a recording or the stop listener has subscribed.
    """

    def __init__(self, device: str = "") -> None:
        self.device = device
        self.noise: float | None = None          # background level (AC RMS)
        self.error = ""
        self._subs: set[queue.Queue] = set()
        self._recent = collections.deque(maxlen=PREROLL_FRAMES)
        self._quiet = collections.deque(maxlen=66)  # ~2 s of non-speech levels
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1"]
        if self.device:
            cmd += ["-D", self.device]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._run, daemon=True, name="voice-mic").start()

    def _run(self) -> None:
        settled = 0
        while True:
            raw = self._proc.stdout.read(FRAME * 2)
            if not raw or len(raw) < FRAME * 2:
                self.error = "microphone stopped delivering audio"
                return
            x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
            ac = x - x.mean()                       # ignore DC offset
            rms = float(np.sqrt(np.mean(ac * ac)))
            if settled < 10:                        # skip the start-up transient
                settled = settled + 1 if abs(float(x.mean())) < 0.05 and rms < 0.2 else 0
                continue
            if self.noise is None or rms < self.noise * 2.5:
                self._quiet.append(rms)
                self.noise = float(np.median(self._quiet))
            frame = (ac, rms)
            with self._lock:
                self._recent.append(frame)
                for q in self._subs:
                    q.put(frame)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            for f in self._recent:                  # pre-roll from before the key press
                q.put(f)
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)


class VoiceInput:
    def __init__(self, cfg: VoiceConfig) -> None:
        self.cfg = cfg
        self.state = "loading"           # loading | ready | listening | transcribing | error
        self.message = ""                # last notice shown in the toolbar
        self.cancel = threading.Event()  # set to stop the current recording
        self.send_now = False            # key pressed again: send the transcript at once
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()
        self._loaded = threading.Event()
        self._mic_stream: MicStream | None = None

    # ── models ───────────────────────────────────────────────────────────────
    def preload(self) -> None:
        """Load the speech models in a background thread (the first run downloads them)."""
        def _load() -> None:
            try:
                if shutil.which("arecord") is None:
                    raise RuntimeError("arecord not found (install alsa-utils)")
                self._mic()
                self._model(self.cfg.model)
                if self.cfg.stop_while_running and self.cfg.stop_model:
                    self._model(self.cfg.stop_model)
                self.state = "ready"
            except Exception as e:  # noqa: BLE001
                self.state = "error"
                self.message = f"voice unavailable: {e}"
            finally:
                self._loaded.set()
        threading.Thread(target=_load, daemon=True, name="voice-load").start()

    def _model(self, name: str):
        with self._lock:
            if name not in self._models:
                from faster_whisper import WhisperModel
                self._models[name] = WhisperModel(
                    name, device="cpu", compute_type="int8", cpu_threads=self.cfg.cpu_threads)
            return self._models[name]

    @property
    def ready(self) -> bool:
        return self.state not in ("loading", "error")

    # ── audio ────────────────────────────────────────────────────────────────
    def _utterances(self, stop: threading.Event, silence_sec: float, max_sec: float,
                    wait_sec: float | None, on_pause: Callable[[np.ndarray], object] | None = None,
                    pause_sec: float = 0.0):
        """Yield (audio, token) per utterance (audio: float32 array), split on
        pauses. Ends when `stop` is set, or after the first utterance /
        `wait_sec` of no speech if wait_sec is given.

        With on_pause: called with the audio so far once `pause_sec` of a
        pause has passed, before the pause is long enough to end the
        utterance. Its return value comes back as the token if the utterance
        then ends without more speech, and None otherwise."""
        mic = self._mic()
        q = mic.subscribe()
        pause_frames = max(1, round(pause_sec * SAMPLE_RATE / FRAME))
        try:
            frames: list[np.ndarray] = []
            speaking = False
            draining = False   # after a cut at max_sec: wait for a pause before listening again
            loud_run = 0
            silent = 0
            token = None       # on_pause result for the current pause
            started = time.monotonic()
            while not stop.is_set():
                if mic.error:
                    raise RuntimeError(mic.error)
                try:
                    x, rms = q.get(timeout=0.1)
                except queue.Empty:
                    if wait_sec is not None and not speaking and time.monotonic() - started > wait_sec:
                        return
                    continue
                loud = mic.noise is not None and rms > max(self.cfg.min_level, mic.noise * 2.5)
                if draining:
                    # Don't start mid-sentence: a cut-off tail on its own made
                    # Whisper hallucinate words (e.g. "Freeze.").
                    silent = 0 if loud else silent + 1
                    if silent * FRAME / SAMPLE_RATE >= silence_sec:
                        draining, silent = False, 0
                    continue
                frames.append(x)
                if not speaking:
                    loud_run = loud_run + 1 if loud else 0
                    if loud_run >= 3:                   # 90 ms above the threshold
                        speaking, silent = True, 0
                    else:
                        frames = frames[-PREROLL_FRAMES:]
                elif loud:
                    silent, token = 0, None             # speech resumed: early result is stale
                else:
                    silent += 1
                    if on_pause is not None and silent == pause_frames:
                        token = on_pause(np.concatenate(frames))
                cut = len(frames) * FRAME / SAMPLE_RATE >= max_sec
                if speaking and (silent * FRAME / SAMPLE_RATE >= silence_sec or cut):
                    yield np.concatenate(frames), (None if cut else token)
                    frames, speaking, silent, loud_run, token = [], False, 0, 0, None
                    draining = cut
                    if wait_sec is not None:
                        return
                if not speaking and wait_sec is not None and time.monotonic() - started > wait_sec:
                    return
            if speaking and frames and wait_sec is not None:
                yield np.concatenate(frames), token     # key pressed again mid-sentence
        finally:
            mic.unsubscribe(q)

    def _mic(self) -> "MicStream":
        with self._lock:
            if self._mic_stream is None:
                self._mic_stream = MicStream(self.cfg.device)
                self._mic_stream.start()
            return self._mic_stream

    def record_utterance(self) -> np.ndarray | None:
        """Record one utterance for push-to-talk (blocking; run in a thread)."""
        self.cancel.clear()
        for audio, _ in self._utterances(self.cancel, self.cfg.silence_sec,
                                         self.cfg.max_record_sec, wait_sec=self.cfg.wait_for_speech_sec):
            return audio
        return None

    def listen_and_transcribe(self, on_recorded: Callable[[], None] | None = None) -> str | None:
        """Record one utterance and return its transcript, or None if nothing
        was said (blocking; run in a thread). on_recorded() is called when
        recording ends.

        Transcription starts early, at the first short pause
        (early_start_sec), while the recorder keeps waiting for the full
        silence_sec. If speech doesn't resume, that result is used, so the
        text is ready sooner. If it does resume, the early result is dropped
        and the whole utterance is transcribed once it ends."""
        from concurrent.futures import ThreadPoolExecutor

        self.cancel.clear()
        early = self.cfg.early_start_sec
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-stt") as pool:
            on_pause = (lambda a: pool.submit(self.transcribe, a)) \
                if 0 < early < self.cfg.silence_sec else None
            for audio, future in self._utterances(
                    self.cancel, self.cfg.silence_sec, self.cfg.max_record_sec,
                    wait_sec=self.cfg.wait_for_speech_sec, on_pause=on_pause, pause_sec=early):
                if on_recorded:
                    on_recorded()
                if future is None:
                    future = pool.submit(self.transcribe, audio)
                return future.result()
        return None

    # ── speech to text ───────────────────────────────────────────────────────
    def transcribe(self, audio: np.ndarray, model: str | None = None) -> str:
        m = self._model(model or self.cfg.model)
        segments, _ = m.transcribe(
            audio, language="en", beam_size=1, vad_filter=True,
            without_timestamps=True, initial_prompt=self.cfg.vocabulary or None)
        return " ".join(s.text.strip() for s in segments).strip()

    # ── spoken stop while a task runs ────────────────────────────────────────
    def listen_for_stop(self, done: threading.Event, on_stop: Callable[[str], None]) -> None:
        """Blocking loop (run in a thread) until `done` is set. Calls
        on_stop(transcript) when a stop command is heard."""
        model = self.cfg.stop_model or self.cfg.model
        try:
            for audio, _ in self._utterances(done, silence_sec=0.4, max_sec=3.0, wait_sec=None):
                if len(audio) >= 3.0 * SAMPLE_RATE - FRAME:
                    continue                            # cut at the cap: a long sentence, not "stop"
                text = self.transcribe(audio, model)
                if is_stop_command(text):
                    on_stop(text)
                    return
        except Exception as e:  # noqa: BLE001 — never break the running task
            self.message = f"stop listener failed: {e}"

    # ── toolbar ──────────────────────────────────────────────────────────────
    def key_names(self) -> str:
        names = {"c-space": "Ctrl+Space"}
        keys = [k for k in (self.cfg.key, self.cfg.alt_key) if k]
        return " / ".join(names.get(k, k.replace("c-", "Ctrl+").upper() if k.startswith("c-") else k.upper())
                          for k in keys)

    def rprompt(self) -> str:
        """Right-hand side of the prompt line (prompt_toolkit HTML)."""
        if self.state == "listening":
            return '<style bg="ansired" fg="ansiwhite"><b> ● listening </b></style>'
        if self.state == "transcribing":
            return '<style bg="ansiyellow" fg="ansiblack"><b> … transcribing </b></style>'
        return ""

    def toolbar(self) -> str:
        key = self.key_names()
        if self.state == "loading":
            return " 🎙 loading speech model…"
        if self.state == "error":
            return f" 🎙 {self.message}"
        if self.state == "listening":
            if self.send_now:
                return " 🎙 finishing… will send"
            return f" 🎙 listening… pause to check the text, or press {key} again to send it"
        if self.state == "transcribing":
            if self.send_now:
                return " 🎙 transcribing… will send"
            return f" 🎙 transcribing… press {key} to send it when ready"
        note = f"   {self.message}" if self.message else ""
        return f" 🎙 {key}: speak a command{note}"
