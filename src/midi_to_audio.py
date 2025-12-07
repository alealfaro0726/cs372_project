
import pretty_midi
import numpy as np
from scipy.io import wavfile
import soundfile as sf


def midi_to_audio(midi_file_path, output_audio_path, sample_rate=44100):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    audio_data = midi_data.fluidsynth(fs=sample_rate)

    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9)

    sf.write(output_audio_path, audio_data, sample_rate)

    return output_audio_path
