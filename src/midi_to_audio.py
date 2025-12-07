
import pretty_midi
import numpy as np
import soundfile as sf


def midi_to_audio(midi_file_path, output_audio_path, sample_rate=44100):
    """Convert MIDI to WAV using pretty_midi + pyfluidsynth if available."""
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    raise RuntimeError("Audio synthesis is disabled; use MIDI output instead.")
