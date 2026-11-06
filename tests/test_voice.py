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
