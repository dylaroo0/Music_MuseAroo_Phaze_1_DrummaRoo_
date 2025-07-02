Enter your prompt here

# Project Structure

├─ 📁 core
  └─ __init__.py
  └─ master_orchestrator.py
  └─ confidence_scorer.py
  └─ brainaroo_timeline_core.py
├─ 📁 rhythm
  └─ __init__.py
  └─ madmom_suite.py
└─ __init__.py
└─ rhythm_analyzer.py
└─ musical_analyzer.py
└─ ai_analyzer.py


# Project Files

- __init__.py
- rhythm_analyzer.py
- musical_analyzer.py
- rhythm\__init__.py
- rhythm\madmom_suite.py
- ai_analyzer.py
- core\__init__.py
- core\master_orchestrator.py
- core\confidence_scorer.py
- core\brainaroo_timeline_core.py

## __init__.py
```

```

## rhythm_analyzer.py
```
import asyncio
import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

async def analyze_tempo(audio_path: str) -> dict:
    """Analyzes the tempo of an audio file."""
    logger.info(f"Analyzing tempo for: {audio_path}")
    try:
        # Use a thread pool to avoid blocking the asyncio event loop
        loop = asyncio.get_running_loop()
        y, sr = await loop.run_in_executor(
            None, lambda: librosa.load(audio_path, sr=None)
        )

        # Get tempo
        onset_env = librosa.onset.onset_detect(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]

        analysis_data = {
            "tempo": round(float(tempo), 2),
            "confidence": 1.0  # Placeholder confidence
        }

        logger.info(f"Detected tempo: {analysis_data['tempo']} BPM")

        return {
            'status': 'success',
            'data': analysis_data
        }

    except Exception as e:
        logger.error(f"Error analyzing tempo for {audio_path}: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

```

## musical_analyzer.py
```
#!/usr/bin/env python3
"""
Phase 1: ANALYZE - Complete Musical Intelligence
===============================================
Revolutionary analysis phase that combines basic + AI analysis in parallel.
All analysis happens here - no artificial separation.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# from core.advanced_plugin_architecture import register_plugin
from musearoo.utils.timing import PrecisionTimingHandler

logger = logging.getLogger(__name__)


# @register_plugin(...)
async def unified_musical_analysis(
    input_path: str,
    output_dir: str = "reports",
    analysis_context: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Revolutionary unified analysis that extracts complete musical intelligence.
    Combines tempo, key, style, AI brain analysis into one comprehensive result.
    """
    
    logger.info(f"🧠 Starting unified musical analysis for {input_path}")
    
    try:
        # Initialize components
        timing_handler = PrecisionTimingHandler()
        
        # Get precise timing metadata first
        timing_metadata = timing_handler.analyze_input_timing(input_path)
        
        # Run all analysis methods in parallel
        loop = asyncio.get_running_loop()
        
        # Create analysis tasks
        tasks = [
            loop.run_in_executor(None, _analyze_basic_features, input_path),
            loop.run_in_executor(None, _analyze_harmonic_content, input_path),
            loop.run_in_executor(None, _analyze_rhythmic_content, input_path),
            loop.run_in_executor(None, _analyze_structural_content, input_path),
            loop.run_in_executor(None, _analyze_ai_intelligence, input_path)
        ]
        
        # Execute all analyses in parallel
        basic_features, harmonic_content, rhythmic_content, structural_content, ai_intelligence = \
            await asyncio.gather(*tasks)
        
        # Synthesize complete musical intelligence
        musical_intelligence = _synthesize_complete_intelligence(
            basic_features,
            harmonic_content, 
            rhythmic_content,
            structural_content,
            ai_intelligence,
            timing_metadata
        )
        
        # Save comprehensive analysis
        output_file = Path(output_dir) / f"{Path(input_path).stem}_musical_intelligence.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(musical_intelligence, f, indent=2, default=str)
        
        logger.info(f"✅ Musical intelligence extracted: {musical_intelligence['style']} in {musical_intelligence['key']}")
        logger.info(f"🎯 Energy: {musical_intelligence['energy']:.2f}, Complexity: {musical_intelligence['complexity']:.2f}")
        
        return {
            'status': 'success',
            'output_file': str(output_file),
            'data': musical_intelligence,
            'analysis_quality': musical_intelligence.get('analysis_confidence', 0.8)
        }
        
    except Exception as e:
        logger.error(f"Unified musical analysis failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'input_file': input_path
        }


def _analyze_basic_features(input_path: str) -> Dict[str, Any]:
    """Extract basic musical features (tempo, key, time signature)."""
    
    try:
        if input_path.lower().endswith(('.mid', '.midi')):
            return _analyze_midi_basic_features(input_path)
        else:
            return _analyze_audio_basic_features(input_path)
    except Exception as e:
        logger.error(f"Basic feature analysis failed: {e}")
        return {'error': str(e)}


def _analyze_midi_basic_features(midi_path: str) -> Dict[str, Any]:
    """Extract basic features from MIDI."""
    
    import pretty_midi
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Tempo analysis
    tempo_changes = pm.get_tempo_changes()
    if len(tempo_changes[1]) > 0:
        avg_tempo = np.mean(tempo_changes[1])
    else:
        avg_tempo = 120.0
    
    # Key analysis using pitch class histogram
    pitch_classes = np.zeros(12)
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch_classes[note.pitch % 12] += note.end - note.start
    
    if np.sum(pitch_classes) > 0:
        pitch_classes /= np.sum(pitch_classes)
    
    # Simple key detection
    key = _detect_key_from_pitch_classes(pitch_classes)
    
    # Time signature
    time_signature = "4/4"  # Default
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        time_signature = f"{ts.numerator}/{ts.denominator}"
    
    return {
        'tempo': float(avg_tempo),
        'key': key,
        'time_signature': time_signature,
        'duration': float(pm.get_end_time()),
        'pitch_class_distribution': pitch_classes.tolist()
    }


def _analyze_audio_basic_features(audio_path: str) -> Dict[str, Any]:
    """Extract basic features from audio."""
    
    import librosa
    
    # Load audio
    y, sr = librosa.load(audio_path)
    
    # Tempo detection
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Key detection via chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean /= np.sum(chroma_mean)
    
    key = _detect_key_from_pitch_classes(chroma_mean)
    
    return {
        'tempo': float(tempo),
        'key': key,
        'time_signature': "4/4",  # Default for audio
        'duration': float(len(y) / sr),
        'pitch_class_distribution': chroma_mean.tolist()
    }


def _detect_key_from_pitch_classes(pitch_classes: np.ndarray) -> str:
    """Detect musical key from pitch class distribution."""
    
    # Krumhansl-Schmuckler profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles
    major_profile /= np.sum(major_profile)
    minor_profile /= np.sum(minor_profile)
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_correlation = -1
    best_key = "C major"
    
    # Test all keys
    for i in range(12):
        # Major
        rotated_major = np.roll(major_profile, i)
        correlation = np.corrcoef(pitch_classes, rotated_major)[0, 1]
        if not np.isnan(correlation) and correlation > best_correlation:
            best_correlation = correlation
            best_key = f"{note_names[i]} major"
        
        # Minor
        rotated_minor = np.roll(minor_profile, i)
        correlation = np.corrcoef(pitch_classes, rotated_minor)[0, 1]
        if not np.isnan(correlation) and correlation > best_correlation:
            best_correlation = correlation
            best_key = f"{note_names[i]} minor"
    
    return best_key


def _analyze_harmonic_content(input_path: str) -> Dict[str, Any]:
    """Analyze harmonic complexity and chord progressions."""
    
    try:
        if input_path.lower().endswith(('.mid', '.midi')):
            return _analyze_midi_harmony(input_path)
        else:
            return _analyze_audio_harmony(input_path)
    except Exception as e:
        logger.error(f"Harmonic analysis failed: {e}")
        return {'harmonic_complexity': 0.5, 'error': str(e)}


def _analyze_midi_harmony(midi_path: str) -> Dict[str, Any]:
    """Analyze harmony from MIDI."""
    
    import pretty_midi
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Extract simultaneous note combinations
    time_step = 0.5  # 500ms resolution
    total_time = pm.get_end_time()
    
    harmonies = []
    for t in np.arange(0, total_time, time_step):
        active_notes = []
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                if note.start <= t < note.end:
                    active_notes.append(note.pitch % 12)
        
        if len(active_notes) > 1:
            harmonies.append(sorted(set(active_notes)))
    
    # Calculate complexity metrics
    if harmonies:
        unique_harmonies = len(set(tuple(h) for h in harmonies))
        harmonic_diversity = unique_harmonies / len(harmonies)
        
        # Average chord size
        avg_chord_size = np.mean([len(h) for h in harmonies])
        
        # Dissonance calculation
        dissonance_scores = []
        for harmony in harmonies:
            dissonance = 0.0
            for i in range(len(harmony)):
                for j in range(i + 1, len(harmony)):
                    interval = abs(harmony[i] - harmony[j]) % 12
                    if interval in [1, 2, 6, 10, 11]:  # Dissonant intervals
                        dissonance += 1
            if len(harmony) > 1:
                dissonance_scores.append(dissonance / (len(harmony) * (len(harmony) - 1) / 2))
        
        avg_dissonance = np.mean(dissonance_scores) if dissonance_scores else 0.0
        
        # Overall harmonic complexity
        harmonic_complexity = (harmonic_diversity + min(avg_chord_size / 5, 1.0) + avg_dissonance) / 3
    else:
        harmonic_complexity = 0.0
        harmonic_diversity = 0.0
        avg_chord_size = 0.0
        avg_dissonance = 0.0
    
    return {
        'harmonic_complexity': float(harmonic_complexity),
        'harmonic_diversity': float(harmonic_diversity),
        'average_chord_size': float(avg_chord_size),
        'average_dissonance': float(avg_dissonance),
        'chord_changes': len(harmonies)
    }


def _analyze_audio_harmony(audio_path: str) -> Dict[str, Any]:
    """Analyze harmony from audio."""
    
    import librosa
    
    y, sr = librosa.load(audio_path)
    
    # Chromagram for harmonic analysis
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=2048)
    
    # Calculate harmonic complexity from chroma
    chroma_var = np.var(chroma, axis=1)
    harmonic_complexity = np.mean(chroma_var)
    
    # Estimate chord changes
    chroma_diff = np.diff(chroma, axis=1)
    chord_changes = np.sum(np.linalg.norm(chroma_diff, axis=0) > 0.5)
    
    return {
        'harmonic_complexity': float(min(harmonic_complexity * 5, 1.0)),  # Normalize
        'chord_changes': int(chord_changes),
        'harmonic_diversity': float(np.std(chroma.flatten()))
    }


def _analyze_rhythmic_content(input_path: str) -> Dict[str, Any]:
    """Analyze rhythmic complexity and patterns."""
    
    try:
        if input_path.lower().endswith(('.mid', '.midi')):
            return _analyze_midi_rhythm(input_path)
        else:
            return _analyze_audio_rhythm(input_path)
    except Exception as e:
        logger.error(f"Rhythmic analysis failed: {e}")
        return {'rhythmic_complexity': 0.5, 'error': str(e)}


def _analyze_midi_rhythm(midi_path: str) -> Dict[str, Any]:
    """Analyze rhythm from MIDI."""
    
    import pretty_midi
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Collect all note onsets
    all_onsets = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            all_onsets.append(note.start)
    
    if not all_onsets:
        return {'rhythmic_complexity': 0.0}
    
    all_onsets.sort()
    
    # Calculate inter-onset intervals
    intervals = np.diff(all_onsets)
    
    # Syncopation detection
    beat_duration = 60.0 / 120.0  # Assume 120 BPM for analysis
    beat_positions = [(onset % beat_duration) / beat_duration for onset in all_onsets]
    
    # Count off-beat notes
    on_beat_tolerance = 0.1
    syncopated_notes = sum(1 for pos in beat_positions 
                          if on_beat_tolerance < pos < 1 - on_beat_tolerance)
    syncopation_index = syncopated_notes / len(beat_positions) if beat_positions else 0
    
    # Rhythmic complexity
    if len(intervals) > 1:
        interval_variety = np.std(intervals) / np.mean(intervals)
        rhythmic_complexity = (syncopation_index + min(interval_variety, 1.0)) / 2
    else:
        rhythmic_complexity = 0.0
    
    return {
        'rhythmic_complexity': float(rhythmic_complexity),
        'syncopation_index': float(syncopation_index),
        'note_density': float(len(all_onsets) / pm.get_end_time()),
        'interval_variety': float(interval_variety) if len(intervals) > 1 else 0.0
    }


def _analyze_audio_rhythm(audio_path: str) -> Dict[str, Any]:
    """Analyze rhythm from audio."""
    
    import librosa
    
    y, sr = librosa.load(audio_path)
    
    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Tempo and beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Calculate rhythmic complexity
    if len(onset_times) > 1:
        # Inter-onset intervals
        intervals = np.diff(onset_times)
        interval_variety = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        # Note density
        note_density = len(onset_times) / (len(y) / sr)
        
        rhythmic_complexity = min((interval_variety + note_density / 10), 1.0)
    else:
        rhythmic_complexity = 0.0
        interval_variety = 0.0
        note_density = 0.0
    
    return {
        'rhythmic_complexity': float(rhythmic_complexity),
        'note_density': float(note_density),
        'interval_variety': float(interval_variety),
        'detected_tempo': float(tempo)
    }


def _analyze_structural_content(input_path: str) -> Dict[str, Any]:
    """Analyze musical structure and form."""
    
    try:
        if input_path.lower().endswith(('.mid', '.midi')):
            return _analyze_midi_structure(input_path)
        else:
            return _analyze_audio_structure(input_path)
    except Exception as e:
        logger.error(f"Structural analysis failed: {e}")
        return {'form_complexity': 0.5, 'error': str(e)}


def _analyze_midi_structure(midi_path: str) -> Dict[str, Any]:
    """Analyze structure from MIDI."""
    
    import pretty_midi
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    total_duration = pm.get_end_time()
    
    # Simple structure analysis based on note density
    segment_duration = 8.0  # 8-second segments
    num_segments = max(1, int(total_duration / segment_duration))
    
    segment_densities = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)
        
        notes_in_segment = 0
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                if start_time <= note.start < end_time:
                    notes_in_segment += 1
        
        density = notes_in_segment / segment_duration
        segment_densities.append(density)
    
    # Calculate form complexity
    if len(segment_densities) > 1:
        density_variance = np.var(segment_densities)
        form_complexity = min(density_variance / 10, 1.0)  # Normalize
    else:
        form_complexity = 0.0
    
    return {
        'form_complexity': float(form_complexity),
        'num_segments': num_segments,
        'segment_densities': segment_densities,
        'total_duration': float(total_duration)
    }


def _analyze_audio_structure(audio_path: str) -> Dict[str, Any]:
    """Analyze structure from audio."""
    
    import librosa
    
    y, sr = librosa.load(audio_path)
    
    # Spectral features for structure
    hop_length = 512
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=12)
    
    # Self-similarity for structure detection
    features = np.vstack([chroma, mfcc])
    
    # Calculate structure complexity
    feature_variance = np.var(features, axis=1)
    form_complexity = min(np.mean(feature_variance), 1.0)
    
    total_duration = len(y) / sr
    
    return {
        'form_complexity': float(form_complexity),
        'total_duration': float(total_duration),
        'spectral_complexity': float(np.mean(feature_variance))
    }


def _analyze_ai_intelligence(input_path: str) -> Dict[str, Any]:
    """Run AI brain analysis for advanced musical intelligence."""
    
    try:
        # Import AI brain if available
        from BrainAroo import AIBrainOrchestrator
        
        brain = AIBrainOrchestrator()
        ai_results = brain.analyze_file(input_path)
        
        if 'error' in ai_results:
            logger.warning(f"AI brain analysis failed: {ai_results['error']}")
            return _fallback_ai_analysis(input_path)
        
        # Extract key insights
        insights = ai_results.get('insights', {})
        style_analysis = ai_results.get('style_analysis', {})
        
        return {
            'primary_style': style_analysis.get('primary_style', 'unknown'),
            'style_confidence': style_analysis.get('style_confidence', 0.5),
            'complexity_level': insights.get('summary', {}).get('complexity_level', 'medium'),
            'originality_score': insights.get('summary', {}).get('originality', 0.5),
            'emotional_profile': insights.get('summary', {}).get('emotional_character', {}),
            'ai_analysis_available': True
        }
        
    except ImportError:
        logger.info("AI brain not available, using fallback analysis")
        return _fallback_ai_analysis(input_path)
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        return _fallback_ai_analysis(input_path)


def _fallback_ai_analysis(input_path: str) -> Dict[str, Any]:
    """Fallback AI analysis when full AI brain is not available."""
    
    # Simple heuristic-based style detection
    try:
        if input_path.lower().endswith(('.mid', '.midi')):
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(input_path)
            
            # Simple style heuristics
            has_drums = any(inst.is_drum for inst in pm.instruments)
            num_instruments = len([inst for inst in pm.instruments if not inst.is_drum])
            
            if has_drums and num_instruments <= 4:
                style = 'rock'
            elif num_instruments > 8:
                style = 'classical'
            elif has_drums:
                style = 'pop'
            else:
                style = 'acoustic'
        else:
            style = 'unknown'
        
        return {
            'primary_style': style,
            'style_confidence': 0.6,
            'complexity_level': 'medium',
            'originality_score': 0.5,
            'emotional_profile': {'valence': 0.0, 'arousal': 0.5, 'tension': 0.0},
            'ai_analysis_available': False
        }
        
    except Exception:
        return {
            'primary_style': 'unknown',
            'style_confidence': 0.5,
            'complexity_level': 'medium',
            'originality_score': 0.5,
            'emotional_profile': {'valence': 0.0, 'arousal': 0.5, 'tension': 0.0},
            'ai_analysis_available': False
        }


def _synthesize_complete_intelligence(
    basic_features: Dict[str, Any],
    harmonic_content: Dict[str, Any],
    rhythmic_content: Dict[str, Any],
    structural_content: Dict[str, Any],
    ai_intelligence: Dict[str, Any],
    timing_metadata: Any
) -> Dict[str, Any]:
    """Synthesize all analysis into complete musical intelligence."""
    
    # Extract key metrics
    tempo = basic_features.get('tempo', 120.0)
    key = basic_features.get('key', 'C major')
    style = ai_intelligence.get('primary_style', 'unknown')
    style_confidence = ai_intelligence.get('style_confidence', 0.5)
    
    # Calculate composite scores
    harmonic_complexity = harmonic_content.get('harmonic_complexity', 0.5)
    rhythmic_complexity = rhythmic_content.get('rhythmic_complexity', 0.5)
    form_complexity = structural_content.get('form_complexity', 0.5)
    
    overall_complexity = (harmonic_complexity + rhythmic_complexity + form_complexity) / 3
    
    # Energy calculation
    note_density = rhythmic_content.get('note_density', 5.0)
    energy = min(note_density / 10.0 + rhythmic_complexity, 1.0)
    
    # Missing instrument detection
    missing_instruments = _detect_missing_instruments(basic_features, harmonic_content)
    
    # Analysis confidence
    analysis_confidence = 0.8 if ai_intelligence.get('ai_analysis_available', False) else 0.6
    
    return {
        # Core musical properties
        'tempo': float(tempo),
        'key': key,
        'time_signature': basic_features.get('time_signature', '4/4'),
        'style': style,
        'style_confidence': float(style_confidence),
        
        # Complexity metrics
        'complexity': float(overall_complexity),
        'harmonic_complexity': float(harmonic_complexity),
        'rhythmic_complexity': float(rhythmic_complexity),
        'form_complexity': float(form_complexity),
        
        # Energy and emotion
        'energy': float(energy),
        'emotional_profile': ai_intelligence.get('emotional_profile', {}),
        
        # Structural info
        'duration': float(structural_content.get('total_duration', 0)),
        'missing_instruments': missing_instruments,
        
        # Technical details
        'analysis_confidence': float(analysis_confidence),
        'timing_metadata': {
            'total_duration': float(timing_metadata.total_duration_seconds),
            'leading_silence': float(timing_metadata.leading_silence_seconds),
            'sample_rate': timing_metadata.sample_rate
        },
        
        # Raw analysis data
        'raw_analysis': {
            'basic_features': basic_features,
            'harmonic_content': harmonic_content,
            'rhythmic_content': rhythmic_content,
            'structural_content': structural_content,
            'ai_intelligence': ai_intelligence
        }
    }


def _detect_missing_instruments(basic_features: Dict[str, Any], harmonic_content: Dict[str, Any]) -> List[str]:
    """Detect missing instruments based on analysis."""
    
    missing = []
    
    # Simple heuristics for missing instruments
    # This would be much more sophisticated in the full system
    
    # Check for drums (basic heuristic)
    if basic_features.get('rhythmic_prominence', 0) < 0.3:
        missing.append('drums')
    
    # Check for bass (low frequency content)
    if harmonic_content.get('low_frequency_content', 0.5) < 0.3:
        missing.append('bass')
    
    # Check for harmonic content
    if harmonic_content.get('harmonic_complexity', 0.5) < 0.2:
        missing.append('harmony')
    
    return missing


# @register_plugin(...)
def instrumentation_detector(
    input_path: str,
    output_dir: str = "reports",
    analysis_context: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Advanced instrumentation detection and missing instrument analysis.
    """
    
    try:
        import pretty_midi
        
        pm = pretty_midi.PrettyMIDI(input_path)
        
        # Analyze each instrument
        detected_instruments = []
        instrument_roles = set()
        
        for i, instrument in enumerate(pm.instruments):
            if instrument.is_drum:
                role = 'drums'
                detected_instruments.append({
                    'index': i,
                    'role': role,
                    'program': instrument.program,
                    'name': instrument.name or f"Drums {i}",
                    'note_count': len(instrument.notes),
                    'is_drum': True
                })
                instrument_roles.add(role)
            else:
                # Determine role from pitch range and patterns
                role = _determine_instrument_role(instrument)
                detected_instruments.append({
                    'index': i,
                    'role': role,
                    'program': instrument.program,
                    'name': instrument.name or f"Instrument {i}",
                    'note_count': len(instrument.notes),
                    'is_drum': False,
                    'pitch_range': _get_pitch_range(instrument)
                })
                instrument_roles.add(role)
        
        # Detect missing roles
        essential_roles = {'drums', 'bass', 'harmony'}
        missing_roles = list(essential_roles - instrument_roles)
        
        # Arrangement analysis
        arrangement_density = len(detected_instruments)
        arrangement_balance = _analyze_arrangement_balance(detected_instruments)
        
        result = {
            'detected_instruments': detected_instruments,
            'instrument_roles': list(instrument_roles),
            'missing_roles': missing_roles,
            'arrangement_density': arrangement_density,
            'arrangement_balance': arrangement_balance,
            'total_instruments': len(pm.instruments),
            'melodic_instruments': len([i for i in pm.instruments if not i.is_drum])
        }
        
        # Save results
        output_file = Path(output_dir) / f"{Path(input_path).stem}_instrumentation.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return {
            'status': 'success',
            'output_file': str(output_file),
            'data': result
        }
        
    except Exception as e:
        logger.error(f"Instrumentation detection failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'input_file': input_path
        }


def _determine_instrument_role(instrument) -> str:
    """Determine the musical role of an instrument."""
    
    if not instrument.notes:
        return 'empty'
    
    pitches = [note.pitch for note in instrument.notes]
    avg_pitch = np.mean(pitches)
    pitch_range = max(pitches) - min(pitches)
    note_count = len(instrument.notes)
    
    # Role determination logic
    if avg_pitch < 50:
        return 'bass'
    elif pitch_range > 24 and note_count > 20:
        return 'harmony'  # Wide range, many notes
    elif avg_pitch > 70:
        return 'lead'
    else:
        return 'melody'


def _get_pitch_range(instrument) -> Dict[str, int]:
    """Get pitch range for an instrument."""
    
    if not instrument.notes:
        return {'min': 0, 'max': 0, 'span': 0}
    
    pitches = [note.pitch for note in instrument.notes]
    return {
        'min': int(min(pitches)),
        'max': int(max(pitches)),
        'span': int(max(pitches) - min(pitches))
    }


def _analyze_arrangement_balance(instruments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the balance of the arrangement."""
    
    # Count notes by frequency range
    bass_notes = 0
    mid_notes = 0
    treble_notes = 0
    
    for inst in instruments:
        if inst['is_drum']:
            continue
        
        pitch_range = inst.get('pitch_range', {})
        avg_pitch = (pitch_range.get('min', 60) + pitch_range.get('max', 60)) / 2
        
        if avg_pitch < 50:
            bass_notes += inst['note_count']
        elif avg_pitch < 70:
            mid_notes += inst['note_count']
        else:
            treble_notes += inst['note_count']
    
    total_notes = bass_notes + mid_notes + treble_notes
    
    if total_notes > 0:
        return {
            'bass_ratio': bass_notes / total_notes,
            'mid_ratio': mid_notes / total_notes,
            'treble_ratio': treble_notes / total_notes,
            'balance_score': 1.0 - np.std([bass_notes, mid_notes, treble_notes]) / np.mean([bass_notes, mid_notes, treble_notes]) if np.mean([bass_notes, mid_notes, treble_notes]) > 0 else 0.0
        }
    else:
        return {
            'bass_ratio': 0.0,
            'mid_ratio': 0.0,
            'treble_ratio': 0.0,
            'balance_score': 0.0
        }

```

## rhythm\__init__.py
```

```

## rhythm\madmom_suite.py
```
import madmom
import numpy as np
from typing import Dict, List, Tuple

class MadmomRhythmAnalyzer:
    """State-of-the-art beat and downbeat tracking"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.beat_processor = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        self.downbeat_processor = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4])
        
    def analyze(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract beats, downbeats, tempo"""
        # Get beat activations
        act = madmom.features.beats.RNNBeatProcessor()(audio)
        
        # Track beats
        beats = self.beat_processor(act)
        
        # Get downbeats
        downbeats = self.downbeat_processor(act)
        
        # Calculate tempo
        if len(beats) > 1:
            intervals = np.diff(beats)
            tempo = 60.0 / np.median(intervals)
        else:
            tempo = 120.0  # fallback
            
        return {
            'beats': beats.tolist(),
            'downbeats': downbeats[:, 0].tolist(),  # just times
            'tempo': float(tempo),
            'time_signature': 4  # TODO: detect from downbeats
        }

```

## ai_analyzer.py
```
#!/usr/bin/env python3
"""
BrainAroo v5 Complete — Ultimate Clean Musical Intelligence Engine
================================================================
- Microsecond-perfect timing via PrecisionTimingHandler (single source of truth)
- Audio-to-MIDI (Omnizart/CREPE/librosa) with quality scoring  
- 200+ comprehensive features (symbolic, ML, emotional, form, complexity, etc.)
- Advanced structural, rhythmic, and harmonic analysis
- Performance difficulty and originality scoring
- No Style Detection Specialists (kept simple heuristic classifier)
- Clean, production-ready architecture
"""

import json
import logging
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque

import numpy as np
import pretty_midi
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from context.precision_timing_handler import PrecisionTimingHandler
from core.plugin_registry import register_plugin

# Optional advanced libraries
try:
    import omnizart
    OMNIZART_AVAILABLE = True
except ImportError:
    OMNIZART_AVAILABLE = False

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from music21 import converter, analysis, key, meter, roman, chord
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrainAroo")


@register_plugin(
    name="brainaroo_complete",
    phase=1,
    description="Complete musical intelligence engine with all advanced features",
    input_types=["midi", "audio"],
    capabilities=["audio_to_midi", "symbolic_analysis", "timing_analysis", "style_detection", "emotional_analysis", "difficulty_estimation"],
    parallel_safe=True,
    estimated_time=8.0,
    author="MuseAroo Team",
    version="5.0",
    tags=["intelligence", "analysis", "complete", "advanced", "robust"]
)
async def brainaroo_complete_analysis(
    input_path: str,
    output_dir: str = "reports",
    analysis_context: Optional[Dict[str, Any]] = None,
    extract_advanced_features: bool = True,
    include_ml_analysis: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete musical intelligence analysis with all advanced features.
    
    Args:
        input_path: MIDI or audio file path
        output_dir: Output directory for reports
        analysis_context: Previous analysis context
        extract_advanced_features: Whether to extract full feature set
        include_ml_analysis: Whether to include ML clustering and analysis
        
    Returns:
        Comprehensive analysis results
    """
    
    logger.info(f"🧠 BrainAroo Complete analysis: {input_path}")
    
    try:
        # 1. Timing is ALWAYS via PrecisionTimingHandler (single source of truth)
        timing_handler = PrecisionTimingHandler()
        timing_metadata = timing_handler.analyze_input_timing(input_path)
        
        logger.info(f"⏱️  Timing analyzed: {timing_metadata.total_duration_seconds}s total")
        
        # 2. Audio-to-MIDI conversion if needed (best available method)
        midi_path, conversion_metadata = _audio_to_midi_if_needed(input_path)
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        # 3. Extract comprehensive features
        logger.info("🔍 Extracting comprehensive musical features...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_running_loop()
            
            # Schedule feature extraction tasks
            basic_features_task = loop.run_in_executor(executor, _extract_basic_features, midi, timing_metadata)
            comprehensive_features_task = loop.run_in_executor(executor, _extract_comprehensive_features, midi, timing_metadata)
            advanced_structure_task = loop.run_in_executor(executor, _extract_advanced_structural_features, midi)
            advanced_rhythm_task = loop.run_in_executor(executor, _extract_advanced_rhythmic_features, midi)
            emotional_features_task = loop.run_in_executor(executor, _extract_emotional_features, midi, comprehensive_features_task)
            difficulty_features_task = loop.run_in_executor(executor, _estimate_performance_difficulty, comprehensive_features_task, midi)
            originality_features_task = loop.run_in_executor(executor, _calculate_originality_score, comprehensive_features_task, midi)
            
            # Wait for all tasks to complete
            basic_features = await basic_features_task
            comprehensive_features = await comprehensive_features_task
            advanced_structure = await advanced_structure_task
            advanced_rhythm = await advanced_rhythm_task
            emotional_features = await emotional_features_task
            difficulty_features = await difficulty_features_task
            originality_features = await originality_features_task
            
            # Combine features
            features = basic_features
            features.update(comprehensive_features)
            features.update(advanced_structure)
            features.update(advanced_rhythm)
            features.update(emotional_features)
            features.update(difficulty_features)
            features.update(originality_features)
        
        # Always include timing data
        features.update({
            'timing_total_duration': float(timing_metadata.total_duration_seconds),
            'timing_leading_silence': float(timing_metadata.leading_silence_seconds),
            'timing_trailing_silence': float(timing_metadata.trailing_silence_seconds),
            'timing_sample_rate': timing_metadata.sample_rate
        })
        
        # 4. Style classification (simple heuristic, no specialists)
        logger.info("🎵 Classifying musical style...")
        style_info = _classify_musical_style(features, midi)
        
        # 5. Music21 symbolic analysis (if available)
        symbolic_features = {}
        if MUSIC21_AVAILABLE:
            try:
                symbolic_features = _extract_symbolic_features(midi_path)
                logger.info("🎼 Symbolic analysis complete")
            except Exception as e:
                logger.warning(f"Symbolic analysis failed: {e}")
        
        # 6. Generate comprehensive insights and recommendations
        insights = _generate_comprehensive_insights(features, style_info, symbolic_features)
        
        # 7. Assess analysis quality
        quality_score = _assess_analysis_quality(features, conversion_metadata, symbolic_features)
        
        # 8. Compose the comprehensive report
        report = {
            "input_file": input_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "midi_path_used": midi_path if midi_path != input_path else None,
            "timing_metadata": {
                "total_duration": float(timing_metadata.total_duration_seconds),
                "leading_silence": float(timing_metadata.leading_silence_seconds),
                "trailing_silence": float(timing_metadata.trailing_silence_seconds),
                "sample_rate": timing_metadata.sample_rate,
                "file_format": timing_metadata.file_format
            },
            "conversion_metadata": conversion_metadata,
            "features": features,
            "style_analysis": style_info,
            "symbolic_analysis": symbolic_features,
            "insights": insights,
            "analysis_quality": quality_score,
            "context_used": analysis_context is not None,
            "feature_categories": {
                "basic": True,
                "advanced": extract_advanced_features,
                "ml_analysis": include_ml_analysis,
                "emotional": extract_advanced_features,
                "difficulty": extract_advanced_features,
                "originality": extract_advanced_features
            }
        }
        
        # 9. Save comprehensive JSON report
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / f"{Path(input_path).stem}_brainaroo_complete.json"
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # 10. Clean up temporary files
        if midi_path != input_path and Path(midi_path).exists():
            try:
                Path(midi_path).unlink()
            except:
                pass
        
        logger.info(f"✅ BrainAroo Complete: {style_info['primary_style']} style detected")
        logger.info(f"📊 Quality: {quality_score:.2f}, Features: {len(features)}")
        logger.info(f"🎯 Complexity: {features.get('overall_complexity', 0.5):.2f}")
        logger.info(f"🏆 Difficulty: {features.get('performance_difficulty', 0.5):.2f}")
        
        try:
            from core.config_module import get_config
            config = get_config()
            ENABLE_JSYMBOLIC = getattr(config.features, 'ENABLE_JSYMBOLIC', True)
        except ImportError:
            ENABLE_JSYMBOLIC = True
        
        # after existing feature extraction
        if ENABLE_JSYMBOLIC:
            try:
                from utils.jsymbolic_bridge import merge_with_brainaroo_features
                features = merge_with_brainaroo_features(features, midi_path)
            except ImportError:
                pass
        
        return {
            "status": "success",
            "output_file": str(output_file),
            "data": {
                "style": style_info['primary_style'],
                "style_confidence": style_info['confidence'],
                "tempo": features.get('tempo_estimated', 120.0),
                "key": symbolic_features.get('key', 'unknown'),
                "complexity": features.get('overall_complexity', 0.5),
                "energy": features.get('emotional_energy', 0.5),
                "difficulty": features.get('performance_difficulty', 0.5),
                "originality": features.get('originality_score', 0.5),
                "duration": float(timing_metadata.total_duration_seconds),
                "analysis_quality": quality_score
            },
            "features": features,
            "style_analysis": style_info,
            "insights": insights,
            "timing_metadata": timing_metadata.__dict__
        }
        
    except Exception as e:
        logger.error(f"BrainAroo Complete analysis failed: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e),
            "input_file": input_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "error_traceback": traceback.format_exc()
        }


# ============================================================================
# AUDIO-TO-MIDI CONVERSION (Same as before)
# ============================================================================

def _audio_to_midi_if_needed(input_path: str) -> Tuple[str, Dict[str, Any]]:
    """Convert audio to MIDI if needed, using best available method."""
    
    input_path_obj = Path(input_path)
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a'}
    
    if input_path_obj.suffix.lower() not in audio_extensions:
        return input_path, {"method": "midi_original", "is_conversion": False}
    
    logger.info("🎵 Converting audio to MIDI...")
    
    # Try methods in order of preference
    conversion_methods = []
    
    if OMNIZART_AVAILABLE:
        conversion_methods.append(("omnizart", _convert_with_omnizart))
    if CREPE_AVAILABLE and LIBROSA_AVAILABLE:
        conversion_methods.append(("crepe", _convert_with_crepe))
    if LIBROSA_AVAILABLE:
        conversion_methods.append(("librosa_onset", _convert_with_librosa))
    
    if not conversion_methods:
        raise RuntimeError("No audio-to-MIDI conversion methods available. Install librosa, omnizart, or crepe.")
    
    # Try each method and return the first successful one
    for method_name, convert_func in conversion_methods:
        try:
            output_path = f"{input_path_obj.stem}_{method_name}.mid"
            success = convert_func(input_path, output_path)
            
            if success and Path(output_path).exists():
                logger.info(f"✅ Audio converted using {method_name}")
                return output_path, {
                    "method": method_name,
                    "is_conversion": True,
                    "conversion_quality": _assess_conversion_quality(output_path)
                }
        except Exception as e:
            logger.warning(f"Conversion method {method_name} failed: {e}")
            continue
    
    raise RuntimeError("All audio-to-MIDI conversion methods failed")


def _convert_with_omnizart(audio_path: str, output_path: str) -> bool:
    """Convert using Omnizart melody transcription."""
    try:
        omnizart.melody.app.transcribe(audio_path, output=output_path)
        return True
    except Exception as e:
        logger.error(f"Omnizart conversion failed: {e}")
        return False


def _convert_with_crepe(audio_path: str, output_path: str) -> bool:
    """Convert using CREPE pitch tracking."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True)
        
        confident_mask = confidence > 0.5
        time = time[confident_mask]
        frequency = frequency[confident_mask]
        
        if len(frequency) == 0:
            return False
        
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        midi_pitches = librosa.hz_to_midi(frequency)
        current_pitch = midi_pitches[0]
        note_start = time[0]
        
        for i in range(1, len(midi_pitches)):
            pitch_diff = abs(midi_pitches[i] - current_pitch)
            time_gap = time[i] - time[i-1]
            
            if pitch_diff > 1 or time_gap > 0.2:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=int(np.clip(current_pitch, 0, 127)),
                    start=note_start,
                    end=time[i-1]
                )
                instrument.notes.append(note)
                current_pitch = midi_pitches[i]
                note_start = time[i]
        
        if len(time) > 0:
            note = pretty_midi.Note(
                velocity=80,
                pitch=int(np.clip(current_pitch, 0, 127)),
                start=note_start,
                end=time[-1] + 0.1
            )
            instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        midi.write(output_path)
        
        return len(instrument.notes) > 0
        
    except Exception as e:
        logger.error(f"CREPE conversion failed: {e}")
        return False


def _convert_with_librosa(audio_path: str, output_path: str) -> bool:
    """Convert using librosa onset detection and pitch tracking."""
    try:
        y, sr = librosa.load(audio_path)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        if len(onset_times) == 0:
            return False
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        for i, onset_time in enumerate(onset_times):
            onset_frame = onset_frames[i]
            
            if onset_frame < pitches.shape[1]:
                pitch_strengths = magnitudes[:, onset_frame]
                if np.max(pitch_strengths) > 0:
                    pitch_idx = np.argmax(pitch_strengths)
                    pitch_hz = pitches[pitch_idx, onset_frame]
                    
                    if pitch_hz > 0:
                        midi_pitch = librosa.hz_to_midi(pitch_hz)
                        
                        if i < len(onset_times) - 1:
                            duration = min(onset_times[i+1] - onset_time, 2.0)
                        else:
                            duration = 0.5
                        
                        note = pretty_midi.Note(
                            velocity=80,
                            pitch=int(np.clip(midi_pitch, 0, 127)),
                            start=onset_time,
                            end=onset_time + duration
                        )
                        instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        midi.write(output_path)
        
        return len(instrument.notes) > 0
        
    except Exception as e:
        logger.error(f"Librosa conversion failed: {e}")
        return False


def _assess_conversion_quality(midi_path: str) -> float:
    """Assess the quality of audio-to-MIDI conversion."""
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        note_count = sum(len(inst.notes) for inst in midi.instruments)
        duration = midi.get_end_time()
        
        if duration == 0 or note_count == 0:
            return 0.0
        
        note_density = note_count / duration
        density_score = min(1.0, note_density / 5.0)
        
        all_pitches = [note.pitch for inst in midi.instruments for note in inst.notes]
        if all_pitches:
            pitch_range = max(all_pitches) - min(all_pitches)
            range_score = min(1.0, pitch_range / 36.0)
        else:
            range_score = 0.0
        
        quality = (density_score * 0.7 + range_score * 0.3)
        return float(quality)
        
    except Exception:
        return 0.0


# ============================================================================
# BASIC FEATURE EXTRACTION
# ============================================================================

def _extract_basic_features(midi: pretty_midi.PrettyMIDI, timing_metadata: Any) -> Dict[str, Any]:
    """Extract essential musical features (fast, always works)."""
    
    melodic_notes = []
    drum_notes = []
    
    for inst in midi.instruments:
        if inst.is_drum:
            drum_notes.extend(inst.notes)
        else:
            melodic_notes.extend(inst.notes)
    
    all_notes = melodic_notes + drum_notes
    
    if not all_notes:
        return {"error": "no_notes_found"}
    
    features = {
        "total_notes": len(all_notes),
        "melodic_notes": len(melodic_notes),
        "drum_notes": len(drum_notes),
        "num_instruments": len(midi.instruments),
        "num_melodic_instruments": len([i for i in midi.instruments if not i.is_drum]),
        "num_drum_instruments": len([i for i in midi.instruments if i.is_drum]),
        "duration": float(midi.get_end_time()),
        "has_drums": len(drum_notes) > 0
    }
    
    if melodic_notes:
        pitches = [n.pitch for n in melodic_notes]
        features.update({
            "mean_pitch": float(np.mean(pitches)),
            "pitch_range": max(pitches) - min(pitches),
            "lowest_pitch": min(pitches),
            "highest_pitch": max(pitches),
            "pitch_variety": len(set(pitches))
        })
        
        onsets = sorted([n.start for n in melodic_notes])
        durations = [n.end - n.start for n in melodic_notes]
        
        features.update({
            "note_density": len(melodic_notes) / max(features["duration"], 0.1),
            "avg_note_duration": float(np.mean(durations)),
            "duration_variance": float(np.std(durations)),
            "tempo_estimated": _estimate_tempo_from_onsets(onsets)
        })
        
        velocities = [n.velocity for n in melodic_notes]
        features.update({
            "avg_velocity": float(np.mean(velocities)),
            "velocity_range": max(velocities) - min(velocities),
            "velocity_variance": float(np.std(velocities))
        })
        
        features["max_polyphony"] = _calculate_max_polyphony(midi)
        features["rhythmic_complexity"] = _calculate_rhythmic_complexity(onsets)
        features["pitch_complexity"] = float(np.std(pitches)) / 12.0
        features["overall_complexity"] = (features["rhythmic_complexity"] + features["pitch_complexity"]) / 2
        
        tempo_energy = min(1.0, (features["tempo_estimated"] - 60) / 140.0)
        velocity_energy = (features["avg_velocity"] - 64) / 63.0
        features["energy_level"] = max(0.0, (tempo_energy + velocity_energy) / 2)
    
    return features


def _extract_comprehensive_features(midi: pretty_midi.PrettyMIDI, timing_metadata: Any) -> Dict[str, Any]:
    """Extract comprehensive musical features (includes basic + comprehensive analysis)."""
    
    features = _extract_basic_features(midi, timing_metadata)
    
    if "error" in features:
        return features
    
    melodic_notes = [n for inst in midi.instruments for n in inst.notes if not inst.is_drum]
    
    if melodic_notes:
        # Advanced pitch analysis
        pitches = [n.pitch for n in melodic_notes]
        pitch_classes = np.zeros(12)
        
        for note in melodic_notes:
            pitch_classes[note.pitch % 12] += note.end - note.start
        
        if pitch_classes.sum() > 0:
            pitch_classes = pitch_classes / pitch_classes.sum()
            
            # Pitch class features
            for i, pc_name in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
                features[f'pitch_class_{pc_name}'] = float(pitch_classes[i])
            
            features["pitch_class_entropy"] = float(-np.sum(pitch_classes * np.log(pitch_classes + 1e-10)))
            features["tonal_clarity"] = _calculate_tonal_clarity(pitch_classes)
            features["chromaticism"] = _calculate_chromaticism(pitch_classes)
        
        # Melodic intervals and voice leading
        features.update(_extract_melodic_interval_features(midi))
        
        # Harmonic analysis
        features.update(_analyze_harmony_comprehensive(midi))
        
        # Texture and voice leading
        features.update(_analyze_texture_and_voice_leading(midi))
        
        # Dynamics and expression
        features.update(_extract_dynamics_features(midi))
        
        # Form analysis
        features.update(_analyze_musical_form(midi))
    
    return features


# ============================================================================
# ADVANCED FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def _calculate_tonal_clarity(pitch_classes: np.ndarray) -> float:
    """Calculate how clearly a tonal center is established."""
    sorted_pcs = np.sort(pitch_classes)[::-1]
    if len(sorted_pcs) > 1:
        return float(sorted_pcs[0] - sorted_pcs[1])
    return 0.0


def _calculate_chromaticism(pitch_classes: np.ndarray) -> float:
    """Calculate chromaticism level using diatonic template matching."""
    try:
        # Major scale template
        diatonic_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        
        max_diatonic_match = 0.0
        for shift in range(12):
            template = np.roll(diatonic_template, shift)
            diatonic_weight = np.sum(pitch_classes * template)
            max_diatonic_match = max(max_diatonic_match, diatonic_weight)
        
        chromaticism = 1.0 - max_diatonic_match
        return float(chromaticism)
    except Exception:
        return 0.5


def _extract_melodic_interval_features(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Extract detailed melodic interval features."""
    
    features = {}
    all_intervals = []
    
    for inst in midi.instruments:
        if not inst.is_drum and len(inst.notes) > 1:
            inst_notes = sorted(inst.notes, key=lambda n: n.start)
            intervals = [inst_notes[i+1].pitch - inst_notes[i].pitch for i in range(len(inst_notes) - 1)]
            all_intervals.extend(intervals)
    
    if all_intervals:
        features["avg_melodic_interval"] = float(np.mean(np.abs(all_intervals)))
        features["interval_variety"] = float(np.std(all_intervals))
        features["large_leaps_ratio"] = sum(1 for i in all_intervals if abs(i) > 4) / len(all_intervals)
        features["stepwise_motion_ratio"] = sum(1 for i in all_intervals if abs(i) <= 2) / len(all_intervals)
        
        # Direction changes (melodic contour)
        direction_changes = 0
        for i in range(1, len(all_intervals)):
            if all_intervals[i] * all_intervals[i-1] < 0:  # Sign change
                direction_changes += 1
        features["direction_changes_ratio"] = direction_changes / max(len(all_intervals) - 1, 1)
        
        # Upward vs downward tendency
        upward = sum(1 for i in all_intervals if i > 0)
        features["upward_motion_ratio"] = upward / len(all_intervals)
    else:
        features.update({
            "avg_melodic_interval": 0.0,
            "interval_variety": 0.0,
            "large_leaps_ratio": 0.0,
            "stepwise_motion_ratio": 0.0,
            "direction_changes_ratio": 0.0,
            "upward_motion_ratio": 0.5
        })
    
    return features


def _analyze_harmony_comprehensive(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Comprehensive harmonic analysis."""
    
    features = {}
    
    # Extract simultaneous note combinations with high resolution
    time_resolution = 0.25
    total_time = midi.get_end_time()
    
    harmonies = []
    harmony_times = []
    
    for t in np.arange(0, total_time, time_resolution):
        active_pitches = []
        for inst in midi.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    if note.start <= t < note.end:
                        active_pitches.append(note.pitch % 12)
        
        if len(active_pitches) > 1:
            harmony = tuple(sorted(set(active_pitches)))
            harmonies.append(harmony)
            harmony_times.append(t)
    
    if harmonies:
        # Harmonic diversity and complexity
        unique_harmonies = len(set(harmonies))
        features["harmonic_diversity"] = unique_harmonies / len(harmonies)
        features["avg_chord_size"] = float(np.mean([len(h) for h in harmonies]))
        features["max_chord_size"] = max(len(h) for h in harmonies)
        
        # Dissonance analysis
        dissonance_scores = []
        for harmony in harmonies:
            dissonance = _calculate_chord_dissonance(harmony)
            dissonance_scores.append(dissonance)
        
        features["avg_dissonance"] = float(np.mean(dissonance_scores))
        features["dissonance_variety"] = float(np.std(dissonance_scores))
        features["max_dissonance"] = float(np.max(dissonance_scores))
        
        # Harmonic rhythm (chord change frequency)
        chord_changes = 0
        for i in range(1, len(harmonies)):
            if harmonies[i] != harmonies[i-1]:
                chord_changes += 1
        
        features["harmonic_rhythm"] = chord_changes / max(len(harmonies) - 1, 1)
        features["chord_change_frequency"] = chord_changes / max(total_time, 1)
        
        # Voice leading analysis
        features.update(_analyze_voice_leading(harmonies))
        
    else:
        features.update({
            "harmonic_diversity": 0.0,
            "avg_chord_size": 1.0,
            "max_chord_size": 1,
            "avg_dissonance": 0.0,
            "dissonance_variety": 0.0,
            "max_dissonance": 0.0,
            "harmonic_rhythm": 0.0,
            "chord_change_frequency": 0.0
        })
    
    return features


def _calculate_chord_dissonance(chord: Tuple[int, ...]) -> float:
    """Calculate dissonance level of a chord."""
    if len(chord) < 2:
        return 0.0
    
    dissonance = 0.0
    for i in range(len(chord)):
        for j in range(i + 1, len(chord)):
            interval = abs(chord[i] - chord[j]) % 12
            # Dissonant intervals: minor 2nd, major 2nd, tritone, minor 7th, major 7th
            if interval in [1, 2, 6, 10, 11]:
                dissonance += 1
    
    # Normalize by number of possible intervals
    max_intervals = len(chord) * (len(chord) - 1) / 2
    return dissonance / max_intervals if max_intervals > 0 else 0.0


def _analyze_voice_leading(harmonies: List[Tuple]) -> Dict[str, float]:
    """Analyze voice leading between chords."""
    
    if len(harmonies) < 2:
        return {"voice_leading_smoothness": 1.0, "voice_crossings": 0.0}
    
    voice_movements = []
    crossings = 0
    
    for i in range(len(harmonies) - 1):
        current_chord = sorted(harmonies[i])
        next_chord = sorted(harmonies[i + 1])
        
        # Calculate voice movements (simplified)
        min_len = min(len(current_chord), len(next_chord))
        if min_len > 0:
            movements = []
            for j in range(min_len):
                movement = abs(next_chord[j] - current_chord[j])
                movements.append(movement)
            voice_movements.extend(movements)
            
            # Detect voice crossings (simplified)
            for j in range(min_len - 1):
                if (current_chord[j] > current_chord[j+1]) != (next_chord[j] > next_chord[j+1]):
                    crossings += 1
    
    if voice_movements:
        avg_movement = np.mean(voice_movements)
        smoothness = 1.0 / (1.0 + avg_movement)  # Inverse relationship
    else:
        smoothness = 1.0
    
    return {
        "voice_leading_smoothness": float(smoothness),
        "voice_crossings": float(crossings / max(len(harmonies) - 1, 1))
    }


def _analyze_texture_and_voice_leading(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Analyze texture density and voice independence."""
    
    melodic_instruments = [inst for inst in midi.instruments if not inst.is_drum]
    
    features = {
        "num_voices": len(melodic_instruments),
        "texture_density": 0.0,
        "voice_independence": 1.0
    }
    
    if not melodic_instruments:
        return features
    
    # Calculate texture density
    total_notes = sum(len(inst.notes) for inst in melodic_instruments)
    features["texture_density"] = total_notes / max(len(melodic_instruments), 1)
    
    # Voice range analysis
    voice_ranges = []
    voice_tessituras = []  # Average pitch per voice
    
    for inst in melodic_instruments:
        if inst.notes:
            pitches = [note.pitch for note in inst.notes]
            voice_ranges.append(max(pitches) - min(pitches))
            voice_tessituras.append(np.mean(pitches))
    
    if voice_ranges:
        features["avg_voice_range"] = float(np.mean(voice_ranges))
        features["voice_range_variety"] = float(np.std(voice_ranges))
    
    # Voice independence (based on tessitura separation)
    if len(voice_tessituras) > 1:
        tessitura_spread = max(voice_tessituras) - min(voice_tessituras)
        features["voice_independence"] = min(1.0, tessitura_spread / 24.0)  # Normalize by 2 octaves
    
    return features


def _extract_dynamics_features(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Extract detailed dynamics and expression features."""
    
    all_velocities = []
    for inst in midi.instruments:
        for note in inst.notes:
            all_velocities.append(note.velocity)
    
    if not all_velocities:
        return {"dynamic_range": 0.0, "dynamic_contrast": 0.0}
    
    features = {
        "dynamic_range": max(all_velocities) - min(all_velocities),
        "dynamic_contrast": _calculate_dynamic_contrast(all_velocities),
        "soft_notes_ratio": sum(1 for v in all_velocities if v < 50) / len(all_velocities),
        "loud_notes_ratio": sum(1 for v in all_velocities if v > 100) / len(all_velocities),
        "velocity_skewness": float(np.mean([(v - 64)**3 for v in all_velocities]) / (np.std(all_velocities)**3 + 1e-10))
    }
    
    return features


def _calculate_dynamic_contrast(velocities: List[int]) -> float:
    """Calculate dynamic contrast with sophisticated measure."""
    if len(velocities) < 2:
        return 0.0
    
    # Calculate local velocity differences
    sorted_vels = sorted(velocities)
    velocity_changes = []
    
    # Look at velocity changes over time (if we had timing, but we'll use sorted for now)
    for i in range(len(sorted_vels) - 1):
        change = abs(sorted_vels[i+1] - sorted_vels[i])
        velocity_changes.append(change)
    
    if velocity_changes:
        contrast = np.mean(velocity_changes) / 127.0  # Normalize
    else:
        contrast = 0.0
    
    return float(contrast)


def _analyze_musical_form(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Analyze musical form and structure."""
    
    # Segment analysis with multiple resolutions
    segment_lengths = [4.0, 8.0, 16.0]  # Different time scales
    total_time = midi.get_end_time()
    
    features = {}
    
    for segment_length in segment_lengths:
        num_segments = max(1, int(total_time / segment_length))
        
        # Extract features for each segment
        segment_features = []
        for i in range(num_segments):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, total_time)
            segment_data = _extract_segment_features(midi, start_time, end_time)
            segment_features.append(segment_data)
        
        # Calculate similarity matrix
        similarity_matrix = _calculate_segment_similarity(segment_features)
        
        # Form analysis metrics
        key_prefix = f"form_{int(segment_length)}s"
        features[f"{key_prefix}_repetition_ratio"] = _calculate_repetition_ratio(similarity_matrix)
        features[f"{key_prefix}_structural_coherence"] = float(np.mean(similarity_matrix))
        features[f"{key_prefix}_contrast"] = 1.0 - float(np.mean(similarity_matrix))
    
    # Overall form complexity
    form_complexities = [features.get(f"form_{int(sl)}s_contrast", 0.0) for sl in segment_lengths]
    features["overall_form_complexity"] = float(np.mean(form_complexities))
    
    return features


def _extract_segment_features(midi: pretty_midi.PrettyMIDI, start_time: float, end_time: float) -> np.ndarray:
    """Extract features for a specific time segment."""
    
    # Count notes and analyze basic properties in this segment
    note_count = 0
    total_velocity = 0
    pitch_classes = np.zeros(12)
    total_duration = 0
    
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if start_time <= note.start < end_time:
                note_count += 1
                total_velocity += note.velocity
                pitch_classes[note.pitch % 12] += note.end - note.start
                total_duration += note.end - note.start
    
    segment_duration = end_time - start_time
    
    # Normalize pitch classes
    if pitch_classes.sum() > 0:
        pitch_classes = pitch_classes / pitch_classes.sum()
    
    # Create feature vector
    features = [
        note_count / segment_duration,  # Note density
        total_velocity / max(note_count, 1),  # Average velocity
        total_duration / segment_duration,  # Duration ratio
    ]
    
    # Add pitch class distribution
    features.extend(pitch_classes.tolist())
    
    return np.array(features)


def _calculate_segment_similarity(segment_features: List[np.ndarray]) -> np.ndarray:
    """Calculate similarity matrix between segments."""
    
    if not segment_features:
        return np.array([[1.0]])
    
    n_segments = len(segment_features)
    similarity_matrix = np.zeros((n_segments, n_segments))
    
    for i in range(n_segments):
        for j in range(n_segments):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Cosine similarity
                feat_i = segment_features[i]
                feat_j = segment_features[j]
                
                norm_i = np.linalg.norm(feat_i)
                norm_j = np.linalg.norm(feat_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(feat_i, feat_j) / (norm_i * norm_j)
                else:
                    similarity = 0.0
                
                similarity_matrix[i, j] = max(0.0, similarity)  # Ensure non-negative
    
    return similarity_matrix


def _calculate_repetition_ratio(similarity_matrix: np.ndarray) -> float:
    """Calculate how much repetition exists in the structure."""
    
    if similarity_matrix.shape[0] < 2:
        return 0.0
    
    # Count highly similar pairs (excluding diagonal)
    threshold = 0.8
    n = similarity_matrix.shape[0]
    similar_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] > threshold:
                similar_pairs += 1
    
    max_pairs = n * (n - 1) / 2
    return similar_pairs / max_pairs if max_pairs > 0 else 0.0


# ============================================================================
# ADVANCED STRUCTURAL FEATURES
# ============================================================================

def _extract_advanced_structural_features(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Extract advanced structural and formal features."""
    
    features = {}
    
    # Multi-scale structural analysis
    total_time = midi.get_end_time()
    
    # Phrase-level analysis (4-8 second phrases)
    phrase_features = _analyze_phrase_structure(midi, phrase_length=6.0)
    features.update({f"phrase_{k}": v for k, v in phrase_features.items()})
    
    # Section-level analysis (16-32 second sections)
    section_features = _analyze_section_structure(midi, section_length=20.0)
    features.update({f"section_{k}": v for k, v in section_features.items()})
    
    # Overall structural coherence
    features["structural_coherence"] = _calculate_structural_coherence(midi)
    
    # Information content and predictability
    features["information_content"] = _calculate_information_content(midi)
    
    return features


def _analyze_phrase_structure(midi: pretty_midi.PrettyMIDI, phrase_length: float) -> Dict[str, float]:
    """Analyze phrase-level structure."""
    
    total_time = midi.get_end_time()
    num_phrases = max(1, int(total_time / phrase_length))
    
    phrase_densities = []
    phrase_pitch_centers = []
    
    for i in range(num_phrases):
        start_time = i * phrase_length
        end_time = min((i + 1) * phrase_length, total_time)
        
        # Count notes in phrase
        notes_in_phrase = 0
        pitches_in_phrase = []
        
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if start_time <= note.start < end_time:
                    notes_in_phrase += 1
                    pitches_in_phrase.append(note.pitch)
        
        phrase_density = notes_in_phrase / phrase_length
        phrase_densities.append(phrase_density)
        
        if pitches_in_phrase:
            phrase_pitch_centers.append(np.mean(pitches_in_phrase))
        else:
            phrase_pitch_centers.append(60.0)  # Middle C default
    
    features = {
        "density_variation": float(np.std(phrase_densities)) if phrase_densities else 0.0,
        "pitch_arch": _calculate_pitch_arch(phrase_pitch_centers),
        "phrase_count": num_phrases
    }
    
    return features


def _analyze_section_structure(midi: pretty_midi.PrettyMIDI, section_length: float) -> Dict[str, float]:
    """Analyze section-level structure."""
    
    total_time = midi.get_end_time()
    num_sections = max(1, int(total_time / section_length))
    
    section_complexities = []
    
    for i in range(num_sections):
        start_time = i * section_length
        end_time = min((i + 1) * section_length, total_time)
        
        # Calculate complexity for this section
        complexity = _calculate_section_complexity(midi, start_time, end_time)
        section_complexities.append(complexity)
    
    features = {
        "complexity_trajectory": _analyze_complexity_trajectory(section_complexities),
        "section_count": num_sections,
        "max_section_complexity": float(np.max(section_complexities)) if section_complexities else 0.0
    }
    
    return features


def _calculate_pitch_arch(pitch_centers: List[float]) -> float:
    """Calculate how arch-like the pitch contour is."""
    
    if len(pitch_centers) < 3:
        return 0.0
    
    # Simple arch detection: high point in the middle
    mid_point = len(pitch_centers) // 2
    first_half_avg = np.mean(pitch_centers[:mid_point])
    second_half_avg = np.mean(pitch_centers[mid_point:])
    middle_value = pitch_centers[mid_point]
    
    # Arch score: how much the middle exceeds the ends
    arch_score = (middle_value - (first_half_avg + second_half_avg) / 2) / 12.0  # Normalize by octave
    
    return float(max(0.0, arch_score))


def _calculate_section_complexity(midi: pretty_midi.PrettyMIDI, start_time: float, end_time: float) -> float:
    """Calculate complexity of a specific section."""
    
    # Extract notes in this section
    section_notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if start_time <= note.start < end_time:
                section_notes.append(note)
    
    if not section_notes:
        return 0.0
    
    # Multiple complexity measures
    pitch_variety = len(set(note.pitch for note in section_notes)) / 12.0  # Normalize by octave
    rhythm_complexity = _calculate_local_rhythm_complexity([note.start for note in section_notes])
    velocity_variety = np.std([note.velocity for note in section_notes]) / 64.0  # Normalize
    
    # Combine complexity measures
    complexity = (pitch_variety + rhythm_complexity + velocity_variety) / 3.0
    
    return float(complexity)


def _calculate_local_rhythm_complexity(onsets: List[float]) -> float:
    """Calculate rhythmic complexity for a local section."""
    
    if len(onsets) < 3:
        return 0.0
    
    onsets = sorted(onsets)
    iois = np.diff(onsets)
    
    if len(iois) == 0:
        return 0.0
    
    # Coefficient of variation
    mean_ioi = np.mean(iois)
    if mean_ioi <= 0:
        return 0.0
    
    complexity = np.std(iois) / mean_ioi
    return float(min(complexity, 2.0))  # Cap at 2.0


def _analyze_complexity_trajectory(complexities: List[float]) -> str:
    """Analyze how complexity changes over time."""
    
    if len(complexities) < 3:
        return "stable"
    
    # Simple trajectory analysis
    first_third = np.mean(complexities[:len(complexities)//3])
    last_third = np.mean(complexities[-len(complexities)//3:])
    
    difference = last_third - first_third
    
    if difference > 0.2:
        return "increasing"
    elif difference < -0.2:
        return "decreasing"
    else:
        return "stable"


def _calculate_structural_coherence(midi: pretty_midi.PrettyMIDI) -> float:
    """Calculate overall structural coherence."""
    
    # This is a simplified measure - could be much more sophisticated
    total_time = midi.get_end_time()
    
    if total_time < 8.0:
        return 1.0  # Short pieces are considered coherent
    
    # Analyze consistency of musical elements over time
    segment_length = total_time / 8  # 8 segments
    segment_features = []
    
    for i in range(8):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        features = _extract_segment_features(midi, start_time, end_time)
        segment_features.append(features)
    
    # Calculate average similarity between adjacent segments
    coherence_scores = []
    for i in range(len(segment_features) - 1):
        similarity = cosine_similarity(
            segment_features[i].reshape(1, -1),
            segment_features[i+1].reshape(1, -1)
        )[0, 0]
        coherence_scores.append(max(0.0, similarity))
    
    return float(np.mean(coherence_scores)) if coherence_scores else 1.0


def _calculate_information_content(midi: pretty_midi.PrettyMIDI) -> float:
    """Calculate information content using entropy measures."""
    
    # Collect all musical events
    all_pitches = []
    all_intervals = []
    all_durations = []
    
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        
        inst_notes = sorted(inst.notes, key=lambda n: n.start)
        for note in inst_notes:
            all_pitches.append(note.pitch % 12)  # Pitch classes
            all_durations.append(note.end - note.start)
        
        # Calculate intervals
        for i in range(len(inst_notes) - 1):
            interval = inst_notes[i+1].pitch - inst_notes[i].pitch
            all_intervals.append(interval % 12)  # Interval classes
    
    if not all_pitches:
        return 0.0
    
    # Calculate entropies
    pitch_entropy = _calculate_entropy(all_pitches)
    interval_entropy = _calculate_entropy(all_intervals) if all_intervals else 0.0
    duration_entropy = _calculate_entropy(_quantize_durations(all_durations))
    
    # Combine entropies
    information_content = (pitch_entropy + interval_entropy + duration_entropy) / 3.0
    
    return float(information_content)


def _calculate_entropy(data: List) -> float:
    """Calculate entropy of a data sequence."""
    
    if not data:
        return 0.0
    
    # Count occurrences
    from collections import Counter
    counts = Counter(data)
    total = len(data)
    
    # Calculate entropy
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def _quantize_durations(durations: List[float]) -> List[int]:
    """Quantize durations into discrete categories."""
    
    quantized = []
    for duration in durations:
        if duration < 0.25:
            quantized.append(0)  # Very short
        elif duration < 0.5:
            quantized.append(1)  # Short
        elif duration < 1.0:
            quantized.append(2)  # Medium
        elif duration < 2.0:
            quantized.append(3)  # Long
        else:
            quantized.append(4)  # Very long
    
    return quantized


# ============================================================================
# ADVANCED RHYTHMIC FEATURES
# ============================================================================

def _extract_advanced_rhythmic_features(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Extract advanced rhythmic features including microtiming and groove."""
    
    features = {}
    
    # Extract all onset times
    all_onsets = []
    for inst in midi.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                all_onsets.append(note.start)
    
    if len(all_onsets) < 4:
        return {"microtiming_variance": 0.0, "groove_strength": 0.0}
    
    all_onsets = sorted(all_onsets)
    
    # Microtiming analysis
    features.update(_extract_microtiming_features(all_onsets))
    
    # Groove analysis
    features.update(_extract_groove_features(all_onsets))
    
    # Harmonic rhythm
    features.update(_extract_harmonic_rhythm_features(midi))
    
    # Syncopation analysis
    features.update(_extract_syncopation_features(all_onsets))
    
    return features


def _extract_microtiming_features(onsets: List[float]) -> Dict[str, float]:
    """Extract microtiming features."""
    
    # Estimate grid and quantization
    tempo = _estimate_tempo_from_onsets(onsets)
    beat_duration = 60.0 / tempo
    grid_resolutions = [beat_duration/4, beat_duration/8, beat_duration/16]  # 16th, 32nd, 64th notes
    
    features = {}
    
    for i, grid_size in enumerate(grid_resolutions):
        deviations = []
        for onset in onsets:
            # Find nearest grid point
            grid_position = round(onset / grid_size) * grid_size
            deviation = abs(onset - grid_position)
            deviations.append(deviation)
        
        grid_name = ["16th", "32nd", "64th"][i]
        features[f"microtiming_{grid_name}_deviation"] = float(np.mean(deviations))
        features[f"microtiming_{grid_name}_consistency"] = 1.0 / (1.0 + np.std(deviations))
    
    # Overall microtiming variance
    all_deviations = []
    for grid_size in grid_resolutions:
        for onset in onsets:
            grid_position = round(onset / grid_size) * grid_size
            deviation = abs(onset - grid_position)
            all_deviations.append(deviation)
    
    features["microtiming_variance"] = float(np.var(all_deviations))
    
    return features


def _extract_groove_features(onsets: List[float]) -> Dict[str, float]:
    """Extract groove-related features."""
    
    tempo = _estimate_tempo_from_onsets(onsets)
    beat_duration = 60.0 / tempo
    
    # Analyze timing relative to beat grid
    beat_positions = [(onset % beat_duration) / beat_duration for onset in onsets]
    
    # Groove strength (consistency of timing patterns)
    position_hist, _ = np.histogram(beat_positions, bins=16, range=(0, 1))
    position_hist = position_hist / len(beat_positions)  # Normalize
    
    # Calculate groove strength as the non-uniformity of the histogram
    uniform_distribution = 1.0 / 16
    groove_strength = np.sum(np.abs(position_hist - uniform_distribution))
    
    features = {
        "groove_strength": float(groove_strength),
        "beat_emphasis": _calculate_beat_emphasis(beat_positions),
        "off_beat_ratio": sum(1 for pos in beat_positions if 0.25 < pos < 0.75) / len(beat_positions)
    }
    
    return features


def _calculate_beat_emphasis(beat_positions: List[float]) -> float:
    """Calculate how much emphasis is placed on strong beats."""
    
    # Count notes on/near strong beats (0.0, 0.25, 0.5, 0.75)
    strong_beat_threshold = 0.1
    strong_beat_positions = [0.0, 0.5]  # Beats 1 and 3
    
    on_strong_beats = 0
    for pos in beat_positions:
        for strong_pos in strong_beat_positions:
            if abs(pos - strong_pos) < strong_beat_threshold:
                on_strong_beats += 1
                break
    
    return on_strong_beats / len(beat_positions)


def _extract_harmonic_rhythm_features(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Extract harmonic rhythm features."""
    
    # Sample harmonic content at regular intervals
    time_resolution = 0.5
    total_time = midi.get_end_time()
    
    harmonic_changes = []
    previous_harmony = None
    
    for t in np.arange(0, total_time, time_resolution):
        current_harmony = set()
        for inst in midi.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    if note.start <= t < note.end:
                        current_harmony.add(note.pitch % 12)
        
        current_harmony = tuple(sorted(current_harmony))
        
        if previous_harmony is not None and current_harmony != previous_harmony:
            harmonic_changes.append(t)
        
        previous_harmony = current_harmony
    
    if len(harmonic_changes) < 2:
        return {"harmonic_rhythm_regularity": 1.0, "harmonic_change_rate": 0.0}
    
    # Analyze harmonic change timing
    change_intervals = np.diff(harmonic_changes)
    
    features = {
        "harmonic_rhythm_regularity": 1.0 / (1.0 + np.std(change_intervals)),
        "harmonic_change_rate": len(harmonic_changes) / total_time,
        "avg_harmonic_duration": float(np.mean(change_intervals))
    }
    
    return features


def _extract_syncopation_features(onsets: List[float]) -> Dict[str, float]:
    """Extract detailed syncopation features."""
    
    tempo = _estimate_tempo_from_onsets(onsets)
    beat_duration = 60.0 / tempo
    
    # Analyze syncopation at different metric levels
    syncopation_levels = {}
    
    # Beat level syncopation
    beat_positions = [(onset % beat_duration) / beat_duration for onset in onsets]
    beat_syncopation = sum(1 for pos in beat_positions if 0.2 < pos < 0.8) / len(beat_positions)
    syncopation_levels["beat_syncopation"] = beat_syncopation
    
    # Measure level syncopation (4 beats)
    measure_duration = beat_duration * 4
    measure_positions = [(onset % measure_duration) / measure_duration for onset in onsets]
    
    # Count notes on weak parts of the measure
    weak_beat_syncopation = 0
    for pos in measure_positions:
        beat_in_measure = pos * 4
        if (0.5 < beat_in_measure < 1.0) or (1.5 < beat_in_measure < 2.0) or \
           (2.5 < beat_in_measure < 3.0) or (3.5 < beat_in_measure < 4.0):
            weak_beat_syncopation += 1
    
    syncopation_levels["measure_syncopation"] = weak_beat_syncopation / len(onsets)
    
    # Overall syncopation index
    syncopation_levels["overall_syncopation"] = (beat_syncopation + syncopation_levels["measure_syncopation"]) / 2
    
    return syncopation_levels


# ============================================================================
# EMOTIONAL AND COMPLEXITY FEATURES
# ============================================================================

def _extract_emotional_features(midi: pretty_midi.PrettyMIDI, basic_features: Dict[str, Any]) -> Dict[str, float]:
    """Extract emotional characteristics from musical features."""
    
    features = {}
    
    # Valence (positive/negative emotion) from mode and harmony
    valence = _calculate_valence(midi, basic_features)
    features["emotional_valence"] = valence
    
    # Arousal (energy level) from tempo and dynamics
    arousal = _calculate_arousal(midi, basic_features)
    features["emotional_arousal"] = arousal
    
    # Tension from dissonance and complexity
    tension = _calculate_tension(midi, basic_features)
    features["emotional_tension"] = tension
    
    # Energy (combination of arousal and dynamics)
    energy = _calculate_energy(midi, basic_features)
    features["emotional_energy"] = energy
    
    # Emotional complexity (how much emotions vary)
    features["emotional_complexity"] = float(np.std([valence, arousal, tension]))
    
    # Mood classification
    features["mood_category"] = _classify_mood(valence, arousal, tension)
    
    return features


def _calculate_valence(midi: pretty_midi.PrettyMIDI, basic_features: Dict[str, Any]) -> float:
    """Calculate emotional valence (positive/negative)."""
    
    # Mode analysis (major/minor tendencies)
    pitch_classes = np.zeros(12)
    for inst in midi.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                pitch_classes[note.pitch % 12] += note.end - note.start
    
    if pitch_classes.sum() == 0:
        return 0.0
    
    pitch_classes = pitch_classes / pitch_classes.sum()
    
    # Major vs minor analysis using simple templates
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # C major scale
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # C minor scale
    
    # Find best match across all keys
    max_major_correlation = 0
    max_minor_correlation = 0
    
    for shift in range(12):
        major_corr = np.corrcoef(pitch_classes, np.roll(major_template, shift))[0, 1]
        minor_corr = np.corrcoef(pitch_classes, np.roll(minor_template, shift))[0, 1]
        
        if not np.isnan(major_corr):
            max_major_correlation = max(max_major_correlation, major_corr)
        if not np.isnan(minor_corr):
            max_minor_correlation = max(max_minor_correlation, minor_corr)
    
    # Calculate valence from mode
    mode_valence = max_major_correlation - max_minor_correlation
    
    # Additional factors
    tempo = basic_features.get('tempo_estimated', 120.0)
    tempo_valence = (tempo - 100) / 100.0  # Faster = more positive
    
    # Combine factors
    valence = (mode_valence * 0.7 + tempo_valence * 0.3)
    return float(np.clip(valence, -1.0, 1.0))


def _calculate_arousal(midi: pretty_midi.PrettyMIDI, basic_features: Dict[str, Any]) -> float:
    """Calculate emotional arousal (energy level)."""
    
    # Tempo contribution
    tempo = basic_features.get('tempo_estimated', 120.0)
    tempo_arousal = (tempo - 60) / 140.0  # Normalize 60-200 BPM to 0-1
    
    # Dynamic range contribution
    avg_velocity = basic_features.get('avg_velocity', 64.0)
    velocity_arousal = (avg_velocity - 64) / 63.0  # Normalize around middle velocity
    
    # Note density contribution
    note_density = basic_features.get('note_density', 5.0)
    density_arousal = min(1.0, note_density / 10.0)  # Normalize
    
    # Combine factors
    arousal = (tempo_arousal * 0.4 + velocity_arousal * 0.3 + density_arousal * 0.3)
    return float(np.clip(arousal, 0.0, 1.0))


def _calculate_tension(midi: pretty_midi.PrettyMIDI, basic_features: Dict[str, Any]) -> float:
    """Calculate emotional tension from dissonance and complexity."""
    
    # Harmonic tension from dissonance
    harmonic_tension = basic_features.get('avg_dissonance', 0.0)
    
    # Rhythmic tension from complexity
    rhythmic_complexity = basic_features.get('rhythmic_complexity', 0.0)
    rhythmic_tension = min(1.0, rhythmic_complexity)
    
    # Melodic tension from large intervals
    large_leaps_ratio = basic_features.get('large_leaps_ratio', 0.0)
    melodic_tension = large_leaps_ratio
    
    # Combine tensions
    tension = (harmonic_tension * 0.4 + rhythmic_tension * 0.3 + melodic_tension * 0.3)
    return float(np.clip(tension, 0.0, 1.0))


def _calculate_energy(midi: pretty_midi.PrettyMIDI, basic_features: Dict[str, Any]) -> float:
    """Calculate overall energy level."""
    
    arousal = basic_features.get('emotional_arousal', 0.5)
    velocity_range = basic_features.get('velocity_range', 50.0)
    note_density = basic_features.get('note_density', 5.0)
    
    # Normalize velocity range
    velocity_energy = min(1.0, velocity_range / 100.0)
    
    # Normalize note density
    density_energy = min(1.0, note_density / 15.0)
    
    # Combine factors
    energy = (arousal * 0.5 + velocity_energy * 0.25 + density_energy * 0.25)
    return float(np.clip(energy, 0.0, 1.0))


def _classify_mood(valence: float, arousal: float, tension: float) -> str:
    """Classify mood based on emotional dimensions."""
    
    if valence > 0.2 and arousal > 0.6:
        return "energetic_positive"
    elif valence > 0.2 and arousal < 0.4:
        return "calm_positive"
    elif valence < -0.2 and arousal > 0.6:
        return "dramatic_intense"
    elif valence < -0.2 and arousal < 0.4:
        return "melancholy_subdued"
    elif tension > 0.7:
        return "tense_suspenseful"
    elif arousal > 0.8:
        return "high_energy"
    elif arousal < 0.2:
        return "ambient_calm"
    else:
        return "neutral_balanced"


def _estimate_performance_difficulty(features: Dict[str, Any], midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Estimate performance difficulty from multiple factors."""
    
    difficulty_factors = {}
    
    # Technical difficulty factors
    tempo = features.get('tempo_estimated', 120.0)
    tempo_difficulty = max(0.0, (tempo - 120) / 80.0)  # Faster = harder
    difficulty_factors["tempo_difficulty"] = float(np.clip(tempo_difficulty, 0.0, 1.0))
    
    # Range difficulty
    pitch_range = features.get('pitch_range', 24.0)
    range_difficulty = min(1.0, pitch_range / 48.0)  # 4 octaves = very difficult
    difficulty_factors["range_difficulty"] = float(range_difficulty)
    
    # Polyphonic difficulty
    max_polyphony = features.get('max_polyphony', 1)
    polyphony_difficulty = min(1.0, max_polyphony / 8.0)  # 8 voices = very difficult
    difficulty_factors["polyphony_difficulty"] = float(polyphony_difficulty)
    
    # Rhythmic difficulty
    rhythmic_complexity = features.get('rhythmic_complexity', 0.0)
    difficulty_factors["rhythmic_difficulty"] = float(min(1.0, rhythmic_complexity))
    
    # Melodic difficulty (large leaps)
    large_leaps_ratio = features.get('large_leaps_ratio', 0.0)
    difficulty_factors["melodic_difficulty"] = float(large_leaps_ratio)
    
    # Harmonic difficulty
    harmonic_complexity = features.get('avg_dissonance', 0.0)
    difficulty_factors["harmonic_difficulty"] = float(harmonic_complexity)
    
    # Dynamic difficulty
    velocity_range = features.get('velocity_range', 50.0)
    dynamic_difficulty = min(1.0, velocity_range / 100.0)
    difficulty_factors["dynamic_difficulty"] = float(dynamic_difficulty)
    
    # Overall difficulty (weighted average)
    weights = {
        "tempo_difficulty": 0.2,
        "range_difficulty": 0.15,
        "polyphony_difficulty": 0.2,
        "rhythmic_difficulty": 0.2,
        "melodic_difficulty": 0.1,
        "harmonic_difficulty": 0.1,
        "dynamic_difficulty": 0.05
    }
    
    overall_difficulty = sum(difficulty_factors[factor] * weights[factor] 
                           for factor in weights.keys())
    
    difficulty_factors["performance_difficulty"] = float(overall_difficulty)
    
    # Difficulty category
    if overall_difficulty > 0.8:
        difficulty_factors["difficulty_category"] = "virtuosic"
    elif overall_difficulty > 0.6:
        difficulty_factors["difficulty_category"] = "advanced"
    elif overall_difficulty > 0.4:
        difficulty_factors["difficulty_category"] = "intermediate"
    else:
        difficulty_factors["difficulty_category"] = "beginner"
    
    return difficulty_factors


def _calculate_originality_score(features: Dict[str, Any], midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Calculate originality and uniqueness scores."""
    
    originality_factors = {}
    
    # Harmonic originality (uncommon chord progressions)
    harmonic_diversity = features.get('harmonic_diversity', 0.0)
    originality_factors["harmonic_originality"] = float(harmonic_diversity)
    
    # Rhythmic originality (unusual patterns)
    syncopation = features.get('overall_syncopation', 0.0)
    rhythmic_complexity = features.get('rhythmic_complexity', 0.0)
    rhythmic_originality = (syncopation + rhythmic_complexity) / 2
    originality_factors["rhythmic_originality"] = float(min(1.0, rhythmic_originality))
    
    # Melodic originality (interval patterns and contour)
    interval_variety = features.get('interval_variety', 0.0)
    pitch_entropy = features.get('pitch_class_entropy', 0.0)
    melodic_originality = (interval_variety / 5.0 + pitch_entropy / np.log(12)) / 2
    originality_factors["melodic_originality"] = float(min(1.0, melodic_originality))
    
    # Structural originality (form complexity)
    form_complexity = features.get('overall_form_complexity', 0.0)
    originality_factors["structural_originality"] = float(form_complexity)
    
    # Timbral originality (instrument combinations)
    num_instruments = features.get('num_instruments', 1)
    has_drums = features.get('has_drums', False)
    
    # Simple timbral diversity measure
    if num_instruments > 6:
        timbral_originality = 0.8
    elif num_instruments > 3:
        timbral_originality = 0.6
    elif has_drums and num_instruments > 1:
        timbral_originality = 0.4
    else:
        timbral_originality = 0.2
    
    originality_factors["timbral_originality"] = float(timbral_originality)
    
    # Overall originality (weighted combination)
    weights = {
        "harmonic_originality": 0.3,
        "rhythmic_originality": 0.25,
        "melodic_originality": 0.25,
        "structural_originality": 0.15,
        "timbral_originality": 0.05
    }
    
    overall_originality = sum(originality_factors[factor] * weights[factor] 
                            for factor in weights.keys())
    
    originality_factors["originality_score"] = float(overall_originality)
    
    # Creativity category
    if overall_originality > 0.8:
        originality_factors["creativity_level"] = "highly_original"
    elif overall_originality > 0.6:
        originality_factors["creativity_level"] = "creative"
    elif overall_originality > 0.4:
        originality_factors["creativity_level"] = "moderately_original"
    else:
        originality_factors["creativity_level"] = "conventional"
    
    return originality_factors


def _perform_ml_analysis(features: Dict[str, Any], midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Perform ML-based clustering and analysis."""
    
    ml_features = {}
    
    # Extract numerical features for ML
    numerical_features = []
    feature_names = []
    
    # Select key features for ML analysis
    key_ml_features = [
        'tempo_estimated', 'pitch_range', 'note_density', 'avg_velocity',
        'velocity_range', 'max_polyphony', 'rhythmic_complexity', 'pitch_complexity',
        'harmonic_diversity', 'avg_dissonance', 'emotional_valence', 'emotional_arousal',
        'emotional_tension', 'performance_difficulty', 'originality_score'
    ]
    
    for feature_name in key_ml_features:
        if feature_name in features and isinstance(features[feature_name], (int, float)):
            numerical_features.append(features[feature_name])
            feature_names.append(feature_name)
    
    if len(numerical_features) < 5:
        return {"ml_cluster": 0, "ml_analysis_possible": False}
    
    # Prepare feature vector
    feature_vector = np.array(numerical_features).reshape(1, -1)
    
    # Normalize features
    try:
        scaler = StandardScaler()
        # For single sample, we'll use simple normalization
        normalized_features = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-10)
        
        # Simple clustering (we'd need multiple samples for real clustering)
        # For now, we'll create a synthetic cluster assignment based on feature characteristics
        cluster_assignment = _assign_synthetic_cluster(features)
        ml_features["ml_cluster"] = cluster_assignment
        ml_features["ml_analysis_possible"] = True
        
        # Feature importance (simplified)
        feature_magnitudes = np.abs(normalized_features[0])
        most_important_idx = np.argmax(feature_magnitudes)
        ml_features["most_important_feature"] = feature_names[most_important_idx]
        ml_features["feature_importance_score"] = float(feature_magnitudes[most_important_idx])
        
    except Exception as e:
        logger.warning(f"ML analysis failed: {e}")
        ml_features["ml_cluster"] = 0
        ml_features["ml_analysis_possible"] = False
    
    return ml_features


def _assign_synthetic_cluster(features: Dict[str, Any]) -> int:
    """Assign a synthetic cluster based on musical characteristics."""
    
    # Simple rule-based clustering
    tempo = features.get('tempo_estimated', 120.0)
    complexity = features.get('overall_complexity', 0.5)
    energy = features.get('emotional_energy', 0.5)
    has_drums = features.get('has_drums', False)
    
    if tempo > 140 and energy > 0.7:
        return 0  # High-energy cluster
    elif complexity > 0.7:
        return 1  # Complex/sophisticated cluster
    elif not has_drums and complexity < 0.5:
        return 2  # Simple/acoustic cluster
    elif tempo < 80:
        return 3  # Slow/ambient cluster
    else:
        return 4  # General/pop cluster


# ============================================================================
# HELPER FUNCTIONS (Continued from basic version)
# ============================================================================

def _estimate_tempo_from_onsets(onsets: List[float]) -> float:
    """Estimate tempo from note onsets."""
    if len(onsets) < 4:
        return 120.0
    
    iois = np.diff(sorted(onsets))
    median_ioi = np.median(iois)
    if median_ioi <= 0:
        return 120.0
    
    estimated_tempo = 60.0 / median_ioi
    
    while estimated_tempo > 200:
        estimated_tempo /= 2
    while estimated_tempo < 60:
        estimated_tempo *= 2
    
    return float(estimated_tempo)


def _calculate_max_polyphony(midi: pretty_midi.PrettyMIDI) -> int:
    """Calculate maximum number of simultaneous notes."""
    if not midi.instruments:
        return 0
    
    total_time = midi.get_end_time()
    sample_times = np.arange(0, total_time, 0.1)
    
    max_poly = 0
    for t in sample_times:
        current_poly = 0
        for inst in midi.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    if note.start <= t < note.end:
                        current_poly += 1
        max_poly = max(max_poly, current_poly)
    
    return max_poly


def _calculate_rhythmic_complexity(onsets: List[float]) -> float:
    """Calculate rhythmic complexity from onset patterns."""
    if len(onsets) < 3:
        return 0.0
    
    iois = np.diff(sorted(onsets))
    
    if len(iois) == 0:
        return 0.0
    
    mean_ioi = np.mean(iois)
    if mean_ioi <= 0:
        return 0.0
    
    complexity = np.std(iois) / mean_ioi
    return float(min(complexity, 2.0))


# ============================================================================
# SYMBOLIC ANALYSIS AND STYLE CLASSIFICATION (Same as before)
# ============================================================================

def _extract_symbolic_features(midi_path: str) -> Dict[str, Any]:
    """Extract symbolic features using music21."""
    
    try:
        score = converter.parse(midi_path)
        features = {}
        
        # Key analysis
        try:
            detected_key = score.analyze('key')
            features['key'] = str(detected_key)
            features['key_confidence'] = float(detected_key.correlationCoefficient)
            features['mode'] = detected_key.mode
        except Exception:
            features['key'] = 'unknown'
            features['key_confidence'] = 0.0
        
        # Time signature
        try:
            time_sigs = score.getTimeSignatures()
            if time_sigs:
                ts = time_sigs[0]
                features['time_signature'] = f"{ts.numerator}/{ts.denominator}"
            else:
                features['time_signature'] = '4/4'
        except Exception:
            features['time_signature'] = '4/4'
        
        # Basic harmonic analysis
        try:
            chordified = score.chordify()
            chords = chordified.flatten().getElementsByClass('Chord')
            
            if chords:
                chord_count = len(chords)
                features['chord_count'] = min(chord_count, 32)
                
                major_chords = 0
                minor_chords = 0
                
                for chord_obj in list(chords)[:16]:
                    try:
                        if chord_obj.isMajorTriad():
                            major_chords += 1
                        elif chord_obj.isMinorTriad():
                            minor_chords += 1
                    except:
                        continue
                
                total_analyzed = major_chords + minor_chords
                if total_analyzed > 0:
                    features['major_chord_ratio'] = major_chords / total_analyzed
                    features['minor_chord_ratio'] = minor_chords / total_analyzed
        
        except Exception as e:
            logger.warning(f"Harmonic analysis failed: {e}")
        
        return features
        
    except Exception as e:
        logger.warning(f"Music21 analysis failed: {e}")
        return {'error': str(e)}


def _classify_musical_style(features: Dict[str, Any], midi: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
    """Classify musical style using enhanced heuristics (no specialists)."""
    
    # Extract key indicators
    tempo = features.get('tempo_estimated', 120.0)
    has_drums = features.get('has_drums', False)
    complexity = features.get('overall_complexity', 0.5)
    energy = features.get('emotional_energy', 0.5)
    polyphony = features.get('max_polyphony', 1)
    harmonic_diversity = features.get('harmonic_diversity', 0.0)
    syncopation = features.get('overall_syncopation', 0.0)
    
    # Instrument analysis
    programs = [inst.program for inst in midi.instruments if not inst.is_drum]
    
    # Style scoring
    style_scores = {}
    
    # Jazz indicators
    jazz_score = 0.0
    if syncopation > 0.4:
        jazz_score += 0.3
    if harmonic_diversity > 0.6:
        jazz_score += 0.3
    if complexity > 0.6:
        jazz_score += 0.2
    if 0 in programs:  # Piano
        jazz_score += 0.2
    style_scores['jazz'] = jazz_score
    
    # Rock indicators
    rock_score = 0.0
    if has_drums:
        rock_score += 0.4
    if 100 <= tempo <= 160:
        rock_score += 0.3
    if any(24 <= p <= 31 for p in programs):  # Guitar
        rock_score += 0.3
    style_scores['rock'] = rock_score
    
    # Classical indicators
    classical_score = 0.0
    if polyphony > 3:
        classical_score += 0.3
    if any(40 <= p <= 55 for p in programs):  # Strings
        classical_score += 0.3
    if complexity > 0.5:
        classical_score += 0.2
    if not has_drums:
        classical_score += 0.2
    style_scores['classical'] = classical_score
    
    # Electronic indicators
    electronic_score = 0.0
    if any(p >= 80 for p in programs):  # Synth sounds
        electronic_score += 0.4
    if tempo > 120:
        electronic_score += 0.2
    if energy > 0.7:
        electronic_score += 0.2
    if has_drums:
        electronic_score += 0.2
    style_scores['electronic'] = electronic_score
    
    # Pop indicators
    pop_score = 0.0
    if 90 <= tempo <= 130:
        pop_score += 0.3
    if has_drums:
        pop_score += 0.2
    if complexity < 0.6:
        pop_score += 0.3
    if 2 <= len(programs) <= 6:  # Typical band size
        pop_score += 0.2
    style_scores['pop'] = pop_score
    
    # Acoustic/Folk indicators
    acoustic_score = 0.0
    if all(p < 80 for p in programs):  # All acoustic instruments
        acoustic_score += 0.4
    if not has_drums:
        acoustic_score += 0.2
    if complexity < 0.5:
        acoustic_score += 0.2
    if tempo < 120:
        acoustic_score += 0.2
    style_scores['acoustic'] = acoustic_score
    
    # Normalize scores
    total_score = sum(style_scores.values())
    if total_score > 0:
        style_scores = {k: v / total_score for k, v in style_scores.items()}
    
    # Determine primary style
    if style_scores:
        primary_style = max(style_scores, key=style_scores.get)
        confidence = style_scores[primary_style]
    else:
        primary_style = 'unknown'
        confidence = 0.0
    
    return {
        'primary_style': primary_style,
        'confidence': float(confidence),
        'style_scores': style_scores,
        'is_fusion': confidence < 0.6,
        'secondary_style': sorted(style_scores.items(), key=lambda x: x[1], reverse=True)[1][0] if len(style_scores) > 1 else None
    }


def _generate_comprehensive_insights(features: Dict[str, Any], style_info: Dict[str, Any], symbolic_features: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive musical insights and recommendations."""
    
    insights = {}
    
    # Complexity assessment
    complexity = features.get('overall_complexity', 0.5)
    if complexity > 0.8:
        insights['complexity_description'] = 'Highly complex with sophisticated musical elements'
    elif complexity > 0.6:
        insights['complexity_description'] = 'Moderately complex with interesting development'
    elif complexity > 0.3:
        insights['complexity_description'] = 'Accessible complexity with clear structure'
    else:
        insights['complexity_description'] = 'Simple and straightforward musical approach'
    
    # Performance assessment
    difficulty = features.get('performance_difficulty', 0.5)
    difficulty_category = features.get('difficulty_category', 'intermediate')
    insights['performance_assessment'] = f"{difficulty_category.title()} level (difficulty: {difficulty:.2f})"
    
    # Emotional character
    valence = features.get('emotional_valence', 0.0)
    arousal = features.get('emotional_arousal', 0.5)
    mood = features.get('mood_category', 'neutral_balanced')
    insights['emotional_character'] = f"{mood.replace('_', ' ').title()}"
    
    # Originality assessment
    originality = features.get('originality_score', 0.5)
    creativity_level = features.get('creativity_level', 'moderately_original')
    insights['originality_assessment'] = f"{creativity_level.replace('_', ' ').title()} (score: {originality:.2f})"
    
    # Comprehensive recommendations
    recommendations = []
    
    if complexity < 0.3:
        recommendations.append('Consider adding harmonic or rhythmic complexity to increase musical interest')
    if features.get('harmonic_diversity', 0) < 0.2:
        recommendations.append('Explore more diverse chord progressions and harmonic content')
    if features.get('velocity_range', 0) < 20:
        recommendations.append('Add dynamic expression through velocity changes and articulation')
    if features.get('emotional_complexity', 0) < 0.2:
        recommendations.append('Develop more emotional contrast between sections')
    if features.get('structural_originality', 0) < 0.3:
        recommendations.append('Experiment with more unique formal structures')
    if not features.get('has_drums', False) and style_info['primary_style'] in ['rock', 'pop', 'electronic']:
        recommendations.append('Consider adding rhythmic percussion elements for the style')
    
    insights['recommendations'] = recommendations
    
    # Technical analysis summary
    insights['technical_summary'] = {
        'tempo': f"{features.get('tempo_estimated', 120):.0f} BPM",
        'key': symbolic_features.get('key', 'unknown'),
        'polyphony': f"Up to {features.get('max_polyphony', 1)} simultaneous voices",
        'duration': f"{features.get('duration', 0):.1f} seconds",
        'instruments': features.get('num_instruments', 1)
    }
    
    # Strengths and areas for improvement
    strengths = []
    improvements = []
    
    if features.get('harmonic_diversity', 0) > 0.6:
        strengths.append('Rich harmonic content')
    else:
        improvements.append('Harmonic diversity')
    
    if features.get('rhythmic_complexity', 0) > 0.5:
        strengths.append('Engaging rhythmic patterns')
    else:
        improvements.append('Rhythmic interest')
    
    if features.get('emotional_energy', 0) > 0.6:
        strengths.append('Good energy and momentum')
    elif features.get('emotional_energy', 0) < 0.3:
        improvements.append('Energy and drive')
    
    if originality > 0.6:
        strengths.append('Creative and original ideas')
    else:
        improvements.append('Originality and uniqueness')
    
    insights['strengths'] = strengths
    insights['areas_for_improvement'] = improvements
    
    return insights


def _assess_analysis_quality(features: Dict[str, Any], conversion_metadata: Dict[str, Any], symbolic_features: Dict[str, Any]) -> float:
    """Assess overall quality of the analysis."""
    
    quality_factors = []
    
    # Feature completeness
    expected_categories = ['basic', 'pitch', 'harmony', 'rhythm', 'emotional', 'difficulty', 'originality']
    category_completeness = 0
    
    for category in expected_categories:
        category_features = [k for k in features.keys() if category in k.lower()]
        if category_features:
            category_completeness += 1
    
    completeness_score = category_completeness / len(expected_categories)
    quality_factors.append(completeness_score)
    
    # Conversion quality (if applicable)
    if conversion_metadata.get('is_conversion', False):
        conv_quality = conversion_metadata.get('conversion_quality', 0.5)
        quality_factors.append(conv_quality)
    else:
        quality_factors.append(1.0)
    
    # Symbolic analysis success
    if symbolic_features and 'error' not in symbolic_features:
        quality_factors.append(1.0)
    else:
        quality_factors.append(0.5)
    
    # Data validity
    validity_checks = [
        features.get('total_notes', 0) > 0,
        features.get('duration', 0) > 0,
        features.get('tempo_estimated', 0) > 0,
        'error' not in features
    ]
    
    validity_score = sum(validity_checks) / len(validity_checks)
    quality_factors.append(validity_score)
    
    # Feature consistency
    consistency_score = 1.0
    if features.get('tempo_estimated', 120) < 30 or features.get('tempo_estimated', 120) > 300:
        consistency_score -= 0.2
    if features.get('overall_complexity', 0.5) < 0 or features.get('overall_complexity', 0.5) > 1:
        consistency_score -= 0.2
    
    quality_factors.append(max(0.0, consistency_score))
    
    return float(np.mean(quality_factors))


# CLI for standalone testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BrainAroo v5 Complete — Ultimate Musical Intelligence")
    parser.add_argument("input_file", help="Input MIDI or audio file")
    parser.add_argument("--output_dir", "-o", default="reports", help="Output directory")
    parser.add_argument("--basic", action="store_true", help="Extract only basic features")
    parser.add_argument("--no_ml", action="store_true", help="Skip ML analysis")
    
    args = parser.parse_args()
    
    result = brainaroo_complete_analysis(
        args.input_file, 
        output_dir=args.output_dir,
        extract_advanced_features=not args.basic,
        include_ml_analysis=not args.no_ml
    )
    
    print(json.dumps(result, indent=2, default=str))
```

## core\__init__.py
```

```

## core\master_orchestrator.py
```
import logging

logger = logging.getLogger(__name__)

class MasterOrchestrator:
    """Runs all 250+ analyzers and fuses their results."""

    def __init__(self):
        logger.info("Initializing Master Orchestrator...")
        # In the future, this will discover and load all analyzer plugins
        self.analyzers = []

    async def analyze(self, audio_path: str):
        """The main entry point to run the full analysis pipeline."""
        logger.info(f"Starting full BrainAroo analysis for {audio_path}")
        # 1. Create a timeline
        # 2. Run all analyzers in parallel
        # 3. Fuse results with confidence scores
        # 4. Return the complete musical context timeline
        pass

```

## core\confidence_scorer.py
```
import logging

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """Fuses results from multiple analyzers using confidence scores."""

    def __init__(self):
        logger.info("Initializing Confidence Scorer.")

    def fuse(self, results: list) -> dict:
        """
        Merges multiple analysis results based on their confidence.
        Example: one pitch tracker is good for highs, another for lows.
        This function will weigh them accordingly.
        """
        # Placeholder: just return the first result for now
        if not results:
            return {}
        
        # A more advanced implementation will weigh and merge results
        highest_confidence_result = max(results, key=lambda r: r.get('confidence', 0))
        return highest_confidence_result

```

## core\brainaroo_timeline_core.py
```
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

```

