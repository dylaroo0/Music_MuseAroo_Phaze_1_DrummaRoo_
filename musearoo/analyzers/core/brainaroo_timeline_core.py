#!/usr/bin/env python3
"""
Core Timeline Creation for BrainAroo
Full timeline builder with CREPE, chroma, OpenL3, onset, silence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class MusicalMoment:
    time_ms: int
    tempo: Optional[float] = None
    chord: Optional[str] = None
    is_downbeat: Optional[bool] = None
    melody_pitch: Optional[int] = None
    dynamic_level: Optional[float] = None
    onset_energy: Optional[float] = None
    harmonic_density: Optional[float] = None
    silence: bool = False
    timbre_vector: Optional[np.ndarray] = None
    style_probs: Dict[str, float] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)

@dataclass
class BrainArooTimeline:
    timepoints: List[MusicalMoment]
    resolution_ms: int
    duration_ms: int

def build_timeline(duration_ms: int, resolution_ms: int = 10) -> BrainArooTimeline:
    timepoints = [MusicalMoment(time_ms=t) for t in range(0, duration_ms, resolution_ms)]
    return BrainArooTimeline(timepoints, resolution_ms, duration_ms)

def master_orchestrate(audio_path: str) -> BrainArooTimeline:
    import librosa, openl3, crepe, numpy as np

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration_ms = int(librosa.get_duration(y=y, sr=sr) * 1000)
    timeline = build_timeline(duration_ms)

    frame_length = int(0.025 * sr)
    hop_length = int(timeline.resolution_ms * sr / 1000)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms = np.interp(np.arange(len(timeline.timepoints)), np.linspace(0, len(timeline.timepoints), num=len(rms)), rms)
    silence_threshold = 0.01 * np.max(rms)
    for i, m in enumerate(timeline.timepoints):
        m.dynamic_level = float(rms[i])
        m.silence = rms[i] < silence_threshold

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    for ms in (onsets * 1000).astype(int):
        idx = ms // timeline.resolution_ms
        if 0 <= idx < len(timeline.timepoints):
            timeline.timepoints[idx].onset_energy = 1.0

    _, freq, conf, _ = crepe.predict(y.astype(np.float32), sr, viterbi=True, step_size=timeline.resolution_ms)
    for i, m in enumerate(timeline.timepoints):
        if i < len(freq) and conf[i] > 0.5:
            m.melody_pitch = int(librosa.hz_to_midi(freq[i]))
            m.confidence['melody'] = float(conf[i])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chord_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i, m in enumerate(timeline.timepoints):
        frame = int(i * chroma.shape[1] / len(timeline.timepoints))
        if 0 <= frame < chroma.shape[1]:
            root = np.argmax(chroma[:, frame])
            m.chord = chord_labels[root]

    emb, _ = openl3.get_audio_embedding(y, sr, hop_size=timeline.resolution_ms / 1000.0, embedding_size=512, content_type="music")
    for i, m in enumerate(timeline.timepoints):
        if i < len(emb):
            m.timbre_vector = emb[i]

    return timeline
