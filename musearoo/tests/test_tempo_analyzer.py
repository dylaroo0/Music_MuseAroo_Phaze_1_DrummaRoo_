# File: tests/test_tempo_analyzer.py

import tempfile
import numpy as np
import soundfile as sf
import pytest

from analysis.audio.beat import TempoAnalyzer, TempoFeatureSet

def generate_click_track(bpm: float, duration_s: float, sr: int = 22050) -> np.ndarray:
    """
    Generate a click track (impulse clicks) at the given BPM for the specified duration.
    Returns a waveform array of shape (n_samples,).
    """
    seconds_per_beat = 60.0 / bpm
    total_samples = int(duration_s * sr)
    waveform = np.zeros(total_samples, dtype=np.float32)
    click_samples = np.arange(0, total_samples, int(seconds_per_beat * sr))
    for idx in click_samples:
        if idx < total_samples:
            waveform[idx] = 1.0  # impulse
    return waveform

@pytest.mark.parametrize("bpm", [60, 120, 90])
def test_tempo_estimation(bpm):
    # generate 10-second click track
    duration = 10.0
    sr = 22050
    signal = generate_click_track(bpm, duration, sr)

    # write to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, signal, sr)

        analyzer = TempoAnalyzer()
        result: TempoFeatureSet = analyzer.analyze(tmp.name)

        # check type
        assert isinstance(result, TempoFeatureSet)
        # BPM should be within ±2 BPM of expected
        assert pytest.approx(bpm, abs=2) == result.bpm
        # ensure beat_times length roughly matches expected count
        expected_beats = int(duration * bpm / 60)
        assert abs(len(result.beat_times) - expected_beats) <= 1

def test_zero_length_input():
    # create an empty file
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        # write zero samples
        sf.write(tmp.name, np.array([], dtype=np.float32), 22050)

        analyzer = TempoAnalyzer()
        result = analyzer.analyze(tmp.name)

        # empty input should yield bpm=0 and no beat_times
        assert pytest.approx(0.0, abs=0.1) == result.bpm
        assert result.beat_times == []
