"""Tests for voice input helpers (no microphone or speech model needed)."""
import pytest

from cobot.config import VoiceConfig
from cobot.cli.voice import is_stop_command


@pytest.mark.parametrize("text", [
    "Stop!", "Stop.", "stop", "Halt now", "Robot, stop!", "please stop", "Freeze!",
    "stop stop stop", "Emergency stop.",
])
def test_stop_commands(text):
    assert is_stop_command(text)


@pytest.mark.parametrize("text", [
    "", "Don't stop until the bin.", "Stop the robot and pick the block",
    "Close the gripper.", "Pick up the blue cylinder and put it in the grey bin.",
    "Put it on the stop sign",
])
def test_not_stop_commands(text):
    assert not is_stop_command(text)


def test_voice_disabled_by_default():
    cfg = VoiceConfig()
    assert cfg.enabled is False
    assert cfg.auto_send is False       # transcripts are checked before sending
    assert cfg.key == "f9"
    assert cfg.alt_key == "c-space"     # works when F-keys are media keys


def test_voice_speed_defaults():
    from cobot.config import VoiceConfig
    cfg = VoiceConfig()
    assert cfg.model == "distil-small.en"
    assert 0 < cfg.early_start_sec < cfg.silence_sec


class _FakeMic:
    """Feeds pre-made (frame, rms) pairs; 'loud' frames count as speech."""
    def __init__(self, pattern):
        import numpy as np, queue
        from cobot.cli import voice as V
        self.noise, self.error = 0.001, ""
        self.q = queue.Queue()
        for loud in pattern:
            self.q.put((np.full(V.FRAME, 0.1 if loud else 0.0, np.float32), 0.1 if loud else 0.0))

    def subscribe(self):
        return self.q

    def unsubscribe(self, q):
        pass


def _run_listen(pattern):
    from cobot.config import VoiceConfig
    from cobot.cli.voice import VoiceInput
    vi = VoiceInput(VoiceConfig(silence_sec=0.6, early_start_sec=0.25, wait_for_speech_sec=1))
    vi._mic_stream = _FakeMic(pattern)
    lengths = []
    vi.transcribe = lambda a, m=None: (lengths.append(len(a)), f"t{len(lengths)}")[1]
    return vi.listen_and_transcribe(), lengths


def test_early_transcript_used_when_speech_does_not_resume():
    text, lengths = _run_listen([1] * 10 + [0] * 30)
    assert text == "t1" and len(lengths) == 1       # one early transcription, reused


def test_early_transcript_dropped_when_speech_resumes():
    # 10 loud, 12 quiet (0.36 s: past early start, short of the end), 10 loud, then silence
    text, lengths = _run_listen([1] * 10 + [0] * 12 + [1] * 10 + [0] * 30)
    assert text == "t2" and len(lengths) == 2       # early result discarded
    assert lengths[1] > lengths[0]                  # final covers the whole command


class _FakeBuffer:
    def __init__(self):
        self.text, self.cursor_position, self.sent = "", 0, False

    def validate_and_handle(self):
        self.sent = True


class _FakeApp:
    def __init__(self):
        self.current_buffer = _FakeBuffer()

    def invalidate(self):
        pass


def _voice_round(send_now):
    import asyncio
    from cobot.config import VoiceConfig
    from cobot.cli.repl import _voice_to_prompt
    from cobot.cli.voice import VoiceInput

    vi = VoiceInput(VoiceConfig())
    vi.state = "ready"

    def listen(on_recorded):
        vi.send_now = send_now          # as if the key was pressed again while listening
        on_recorded()
        return "pick up the red block"
    vi.listen_and_transcribe = listen
    app = _FakeApp()
    asyncio.run(_voice_to_prompt(app, vi, session=None))
    return app.current_buffer


def test_pause_puts_text_in_prompt_without_sending():
    buf = _voice_round(send_now=False)
    assert buf.text == "pick up the red block" and not buf.sent


def test_second_key_press_sends_the_command():
    buf = _voice_round(send_now=True)
    assert buf.text == "pick up the red block" and buf.sent
