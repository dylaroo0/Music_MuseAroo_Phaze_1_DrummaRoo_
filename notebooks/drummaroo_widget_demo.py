from __future__ import annotations

import asyncio
import tempfile
import os
import sys
import dataclasses
from pathlib import Path
from typing import Dict, Tuple, Union

import ipywidgets as wd
from IPython.display import Audio, display, clear_output
import pretty_midi

# Add project root to path to allow importing musearoo modules
if '..' not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from musearoo.pro.manager import DrumRooGenerationManager
from musearoo.engine.drummaroo import AlgorithmicDrummaroo, DrummarooUIControls

# ---------------------------------------------------------------------------
# 1. Instantiate engine and manager
# ---------------------------------------------------------------------------
drum_engine = AlgorithmicDrummaroo()
manager = DrumRooGenerationManager(drum_engine)

output_dir = Path(tempfile.gettempdir()) / "drummaroo_notebook_sessions"
output_dir.mkdir(exist_ok=True)
session_id = manager.start_new_session(output_dir=str(output_dir))

# ---------------------------------------------------------------------------
# 2. Define parameter metadata from the dataclass
# ---------------------------------------------------------------------------
PARAM_META: Dict[str, Tuple[Union[int, float], Union[int, float], Union[int, float]]] = {}
for field in dataclasses.fields(DrummarooUIControls):
    if field.type == float:
        PARAM_META[field.name] = (0.0, 1.0, field.default)
    elif field.type == int:
        is_seed = 'seed' in field.name
        max_val = 100000 if is_seed else 10
        min_val = 0 if is_seed else 1
        PARAM_META[field.name] = (min_val, max_val, field.default)

# ---------------------------------------------------------------------------
# 3. Build widgets dynamically
# ---------------------------------------------------------------------------
control_widgets: Dict[str, wd.Widget] = {}
for name, (min_v, max_v, default_v) in PARAM_META.items():
    widget_class = wd.FloatSlider if isinstance(default_v, float) else wd.IntSlider
    step = 0.01 if isinstance(default_v, float) else 1
    control_widgets[name] = widget_class(
        value=default_v,
        min=min_v,
        max=max_v,
        step=step,
        description=name.replace('_', ' ').title(),
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=wd.Layout(width='95%')
    )

# Session-level controls
section_dd = wd.Dropdown(options=["intro", "main", "fill", "outro"], value="main", description="Section")
length_slider = wd.FloatSlider(value=8.0, min=1.0, max=32.0, step=0.5, description="Length (s)")
generate_btn = wd.Button(description="Generate Drums 🚀", button_style="success")
output_area = wd.Output()

controls_box = wd.VBox(list(control_widgets.values()) + [section_dd, length_slider, generate_btn, output_area])

# ---------------------------------------------------------------------------
# 4. Callback to generate drums & preview
# ---------------------------------------------------------------------------
def on_generate_clicked(b):
    async def generate_task():
        with output_area:
            clear_output(wait=True)
            print('Gathering parameters...')
            
            # 1. Collect current parameter values from widgets
            current_params = {name: widget.value for name, widget in control_widgets.items()}
            manager.update_ui_controls(current_params)
            
            print('Parameters updated. Generating new drum pattern...')
            
            # 2. Generate the new version
            version = await manager.generate_new_version(length_seconds=length_slider.value)
            if not version or not version.midi_file_path:
                print('Generation failed. Please check the console logs.')
                return
                
            print(f'✅ Generation complete! MIDI saved to: {version.midi_file_path}')
            
            # 3. Synthesize MIDI to audio and display player
            try:
                print('Synthesizing audio for playback...')
                midi_obj = pretty_midi.PrettyMIDI(version.midi_file_path)
                audio_data = midi_obj.synthesize(fs=44100)
                display(Audio(audio_data, rate=44100, autoplay=True))
                print('Audio ready.')
            except Exception as e:
                print(f'\n⚠️ Could not synthesize audio for playback: {e}')
                print('Please ensure FluidSynth is installed and accessible in your system path.')

    # In a notebook context, we need to ensure an event loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(generate_task())
    else:
        asyncio.run(generate_task())

generate_btn.on_click(on_generate_clicked)

# ---------------------------------------------------------------------------
# 5. Show the UI
# ---------------------------------------------------------------------------
print("⚙️ DrummaRoo Widget Panel ready — tweak & hit ‘Generate’. ✅")
display(controls_box)
