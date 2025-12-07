
import pretty_midi
import numpy as np
from typing import List


class MusicalExpressionEnhancer:

    EXPRESSION_PARAMS = {
        'happy': {
            'articulation': 'staccato',
            'velocity_range': (80, 110),
            'timing_humanization': 0.02,
            'sustain_amount': 0.3,
            'dynamics_variation': 0.25,
        },
        'sad': {
            'articulation': 'legato',
            'velocity_range': (50, 75),
            'timing_humanization': 0.03,
            'sustain_amount': 0.8,
            'dynamics_variation': 0.35,
        },
        'calm': {
            'articulation': 'legato',
            'velocity_range': (60, 85),
            'timing_humanization': 0.015,
            'sustain_amount': 0.6,
            'dynamics_variation': 0.2,
        },
        'angry': {
            'articulation': 'marcato',
            'velocity_range': (100, 127),
            'timing_humanization': 0.01,
            'sustain_amount': 0.2,
            'dynamics_variation': 0.4,
        },
        'surprised': {
            'articulation': 'mixed',
            'velocity_range': (70, 105),
            'timing_humanization': 0.025,
            'sustain_amount': 0.4,
            'dynamics_variation': 0.45,
        }
    }

    def __init__(self, emotion: str = 'happy'):
        self.emotion = emotion if emotion in self.EXPRESSION_PARAMS else 'happy'
        self.params = self.EXPRESSION_PARAMS[self.emotion]

    def enhance_midi(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            self._apply_dynamic_curve(instrument)
            self._apply_articulation(instrument)
            self._humanize_timing(instrument)
            self._add_sustain_pedal(instrument)
            self._add_expression_controller(instrument)

        return midi

    def _apply_dynamic_curve(self, instrument: pretty_midi.Instrument):
        if len(instrument.notes) == 0:
            return

        min_vel, max_vel = self.params['velocity_range']
        variation = self.params['dynamics_variation']

        n_notes = len(instrument.notes)

        time_points = np.array([note.start for note in instrument.notes])
        duration = time_points[-1] - time_points[0] if n_notes > 1 else 1.0

        n_phrases = max(2, int(duration / 8))
        dynamic_curve = np.zeros(n_notes)

        for i in range(n_notes):
            t = (time_points[i] - time_points[0]) / duration if duration > 0 else 0
            phrase_wave = np.sin(t * n_phrases * 2 * np.pi) * 0.3
            overall_arc = np.sin(t * np.pi) * 0.3
            dynamic_curve[i] = 0.5 + phrase_wave + overall_arc

        dynamic_curve = (dynamic_curve - dynamic_curve.min()) / (dynamic_curve.max() - dynamic_curve.min() + 1e-6)

        for i, note in enumerate(instrument.notes):
            base_velocity = min_vel + (max_vel - min_vel) * dynamic_curve[i]

            local_variation = np.random.uniform(-variation, variation) * (max_vel - min_vel)

            new_velocity = int(base_velocity + local_variation)
            note.velocity = max(20, min(127, new_velocity))

    def _apply_articulation(self, instrument: pretty_midi.Instrument):
        if len(instrument.notes) == 0:
            return

        articulation = self.params['articulation']

        for i, note in enumerate(instrument.notes):
            duration = note.end - note.start

            if articulation == 'staccato':
                new_duration = duration * np.random.uniform(0.5, 0.7)
                note.end = note.start + new_duration

            elif articulation == 'legato':
                if i < len(instrument.notes) - 1:
                    next_note = instrument.notes[i + 1]
                    gap = next_note.start - note.end
                    if gap < 0.1:
                        note.end = next_note.start + 0.01

            elif articulation == 'marcato':
                note.velocity = min(127, int(note.velocity * 1.1))

            elif articulation == 'mixed':
                if i % 3 == 0:
                    new_duration = duration * np.random.uniform(0.5, 0.7)
                    note.end = note.start + new_duration

    def _humanize_timing(self, instrument: pretty_midi.Instrument):
        if len(instrument.notes) == 0:
            return

        humanization = self.params['timing_humanization']

        for note in instrument.notes:
            timing_offset = np.random.uniform(-humanization, humanization)
            note.start = max(0, note.start + timing_offset)
            note.end = max(note.start + 0.05, note.end + timing_offset)

        instrument.notes.sort(key=lambda x: x.start)

    def _add_sustain_pedal(self, instrument: pretty_midi.Instrument):
        if len(instrument.notes) == 0:
            return

        sustain_amount = self.params['sustain_amount']

        if sustain_amount < 0.2:
            return

        sustain_value = int(64 + sustain_amount * 63)

        notes_sorted = sorted(instrument.notes, key=lambda x: x.start)

        phrase_starts = [notes_sorted[0].start]
        phrase_ends = []

        for i in range(1, len(notes_sorted)):
            gap = notes_sorted[i].start - notes_sorted[i-1].end
            if gap > 0.5:
                phrase_ends.append(notes_sorted[i-1].end)
                phrase_starts.append(notes_sorted[i].start)

        phrase_ends.append(notes_sorted[-1].end)

        for start, end in zip(phrase_starts, phrase_ends):
            instrument.control_changes.append(
                pretty_midi.ControlChange(64, sustain_value, max(0, start - 0.05))
            )
            instrument.control_changes.append(
                pretty_midi.ControlChange(64, 0, end + 0.1)
            )

        instrument.control_changes.sort(key=lambda x: x.time)

    def _add_expression_controller(self, instrument: pretty_midi.Instrument):
        if len(instrument.notes) == 0:
            return

        duration = max(note.end for note in instrument.notes)

        n_points = max(4, int(duration / 4))

        for i in range(n_points):
            time = (i / (n_points - 1)) * duration if n_points > 1 else 0

            t = i / (n_points - 1) if n_points > 1 else 0.5
            expression_value = int(64 + 32 * np.sin(t * 2 * np.pi))
            expression_value = max(40, min(100, expression_value))

            instrument.control_changes.append(
                pretty_midi.ControlChange(11, expression_value, time)
            )

        instrument.control_changes.sort(key=lambda x: x.time)


def enhance_musical_expression(midi: pretty_midi.PrettyMIDI, emotion: str = 'happy') -> pretty_midi.PrettyMIDI:
    enhancer = MusicalExpressionEnhancer(emotion=emotion)
    return enhancer.enhance_midi(midi)
