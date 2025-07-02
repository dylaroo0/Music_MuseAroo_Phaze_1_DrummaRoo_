# 🎼 MuseAroo: The AI Music Co-Creator

MuseAroo is a revolutionary music intelligence system that transforms audio or MIDI input into intelligent, expressive musical accompaniment. It includes BrainAroo (musical analyzer), DrummaRoo (AI drummer), and a timeline architecture for real-time performance.

---

## 🧠 Components Overview

### 🔍 BrainAroo (`musearoo/analyzers`)
- Extracts 250+ musical features from audio (WAV, MP3, FLAC)
- Features: pitch, chords, melody, dynamics, structure, emotion
- Timeline built with millisecond precision

### 🥁 DrummaRoo (`musearoo/generators/drummaroo`)
- Uses BrainAroo timeline to generate harmonically aware, phrased drums
- Respects silence, accents, fills, ghost notes, groove
- Exports high-quality MIDI grooves

### 📊 Output (`musearoo/output`)
- MIDI files
- Saved session data
- Visual plots (planned)

---

## 🧪 Quickstart Instructions

### 1. Install dependencies (recommended in virtualenv)
```bash
pip install -r requirements.txt
```

Minimum needed:
```
librosa, openl3, crepe, numpy, matplotlib, mido, pretty_midi, jupyterlab
```

### 2. Run from CLI
```bash
python musearoo/analyzers/core/brainaroo_timeline_core.py data/your_clip.wav
```

Then generate drums:
```bash
python musearoo/generators/drummaroo/drummaroo_generate.py
```

### 3. Launch Interactive Jupyter Playground
```bash
jupyter lab
```
Then open:
```
notebooks/DrummaRoo_Interactive_Playground.ipynb
```

---

## 🗂 Folder Structure

```
MuseAroo/
├── musearoo/
│   ├── analyzers/            # BrainAroo analyzers
│   │   └── core/             # master_orchestrate and timeline
│   ├── generators/
│   │   └── drummaroo/        # DrummaRoo engine
│   ├── output/               # MIDI + session data
│   ├── interface/            # CLI / Jupyter / Ableton (future)
│   ├── context/              # Shared musical context (planned)
│   ├── utils/                # Helper tools
├── data/                     # Input audio
├── notebooks/                # Jupyter playground
├── scripts/                  # Entrypoint scripts
└── requirements.txt          # Dependencies
```

---

## ✅ Status

| Feature        | Status        |
|----------------|---------------|
| Audio Analysis | ✅ Works       |
| MIDI Drums     | ✅ Works       |
| Harmony        | ⏳ Planned     |
| Melody         | ⏳ Planned     |
| Real-time      | 🔜 Experimental|
| Ableton Bridge | 🔜 Future      |

---

## 🧠 Philosophy

MuseAroo isn't about loops—it's about understanding your music and co-creating with you. Every groove, fill, or rest is based on **your phrasing, emotion, and structure**.

DrummaRoo doesn't guess. It listens.
