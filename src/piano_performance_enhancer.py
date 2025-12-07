
import pretty_midi
import numpy as np
import random
from typing import List, Tuple, Dict


class PianoPerformanceEnhancer:
    CHORD_VOICINGS = {
        'major': [
            [0, 4, 7],
            [0, 4, 7, 11],
            [0, 2, 7],
            [0, 5, 7],
            [4, 7, 12],
        ],
        'minor': [
            [0, 3, 7],
            [0, 3, 7, 10],
            [0, 3, 7, 11],
            [3, 7, 12],
        ],
        'diminished': [[0, 3, 6]],
        'augmented': [[0, 4, 8]],
    }

    LEFT_HAND_PATTERNS = [
        'broken_chord',
        'alberti_bass',
        'stride',
        'walking_bass',
        'octave_bass',
        'arpeggiated',
    ]

    def __init__(self, key_root: int = 60, scale_type: str = 'major', tempo: int = 120, variation_seed=None):
        self.key_root = key_root
        self.scale_type = scale_type
        self.tempo = tempo
        self.beat_duration = 60.0 / tempo

        if variation_seed is not None:
            random.seed(variation_seed)
            np.random.seed(variation_seed)

        if scale_type == 'major':
            self.scale = [0, 2, 4, 5, 7, 9, 11]
            self.chord_quality = 'major'
        else:
            self.scale = [0, 2, 3, 5, 7, 8, 10]
            self.chord_quality = 'minor'

    def enhance_piano_melody(self, piano_instrument: pretty_midi.Instrument) -> pretty_midi.Instrument:
        if len(piano_instrument.notes) == 0:
            return piano_instrument

        enhanced = pretty_midi.Instrument(
            program=0,
            name="Piano"
        )

        melody_notes = sorted(piano_instrument.notes, key=lambda n: n.start)
        duration = max(n.end for n in melody_notes)

        enhanced_melody = self._enhance_melody_dynamics(melody_notes)
        enhanced.notes.extend(enhanced_melody)

        if random.random() < 0.2:
            harmony_notes = self._add_harmony(melody_notes)
            enhanced.notes.extend(harmony_notes)

        if random.random() < 0.3:
            left_hand_pattern = random.choice(['simple_bass', 'octave_bass'])
            bass_notes = self._add_left_hand(melody_notes, duration, left_hand_pattern)
            enhanced.notes.extend(bass_notes)

        enhanced = self._humanize_performance(enhanced)

        return enhanced

    def _enhance_melody_dynamics(self, melody_notes: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
        enhanced = []
        base_velocity = 72

        for i, note in enumerate(melody_notes):
            velocity = base_velocity + random.randint(-8, 12)
            velocity = max(60, min(85, velocity))

            enhanced_note = pretty_midi.Note(
                velocity=velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end
            )
            enhanced.append(enhanced_note)

        return enhanced

    def _add_harmony(self, melody_notes: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
        harmony = []
        phrase_duration = self.beat_duration * 8
        num_phrases = int(max(n.end for n in melody_notes) / phrase_duration) + 1

        for phrase_idx in range(num_phrases):
            if random.random() > 0.6:
                continue

            phrase_start = phrase_idx * phrase_duration
            phrase_end = phrase_start + phrase_duration

            phrase_notes = [n for n in melody_notes
                          if n.start >= phrase_start and n.start < phrase_end]

            if not phrase_notes:
                continue

            melody_pitch = phrase_notes[0].pitch
            chord_root = self._find_chord_root(melody_pitch)
            voicing = random.choice(self.CHORD_VOICINGS[self.chord_quality])

            interval = voicing[1] if len(voicing) > 1 else voicing[0]
            pitch = chord_root + interval - 12
            if pitch < 48:
                pitch += 12

            note = pretty_midi.Note(
                velocity=random.randint(40, 55),
                pitch=pitch,
                start=phrase_start,
                end=min(phrase_end, phrase_start + phrase_duration * 0.7)
            )
            harmony.append(note)

        return harmony

    def _add_left_hand(self, melody_notes: List[pretty_midi.Note],
                       duration: float, pattern: str) -> List[pretty_midi.Note]:
        bass = []
        current_time = 0.0

        while current_time < duration:
            current_melody = [n for n in melody_notes
                            if n.start <= current_time < n.end]

            if current_melody:
                melody_pitch = current_melody[0].pitch
            else:
                melody_pitch = min(melody_notes,
                                 key=lambda n: abs(n.start - current_time)).pitch

            chord_root = self._find_chord_root(melody_pitch)
            bass_root = chord_root - 24
            if bass_root < 28:
                bass_root += 12

            if pattern == 'simple_bass':
                bass.append(pretty_midi.Note(
                    velocity=random.randint(50, 65),
                    pitch=bass_root,
                    start=current_time,
                    end=current_time + self.beat_duration * 1.8
                ))
                current_time += self.beat_duration * 2

            elif pattern == 'broken_chord':
                voicing = self.CHORD_VOICINGS[self.chord_quality][0]

                bass.append(pretty_midi.Note(
                    velocity=random.randint(50, 65),
                    pitch=bass_root,
                    start=current_time,
                    end=current_time + self.beat_duration * 0.4
                ))

                bass.append(pretty_midi.Note(
                    velocity=random.randint(45, 60),
                    pitch=bass_root + voicing[2],
                    start=current_time + self.beat_duration,
                    end=current_time + self.beat_duration * 1.4
                ))

                current_time += self.beat_duration * 2

            elif pattern == 'octave_bass':
                bass.append(pretty_midi.Note(
                    velocity=random.randint(55, 70),
                    pitch=bass_root,
                    start=current_time,
                    end=current_time + self.beat_duration * 0.8
                ))
                current_time += self.beat_duration * 2

            else:
                bass.append(pretty_midi.Note(
                    velocity=random.randint(50, 65),
                    pitch=bass_root,
                    start=current_time,
                    end=current_time + self.beat_duration * 1.5
                ))
                current_time += self.beat_duration * 2

        return bass

    def _find_chord_root(self, melody_pitch: int) -> int:
        pitch_class = melody_pitch % 12
        root_class = self.key_root % 12

        relative_pitch = (pitch_class - root_class) % 12

        if relative_pitch in [0, 2, 4]:
            chord_root = self.key_root
        elif relative_pitch in [5, 7, 9]:
            chord_root = self.key_root + 5
        elif relative_pitch in [7, 11, 2]:
            chord_root = self.key_root + 7
        else:
            chord_root = self.key_root

        while chord_root < 36:
            chord_root += 12
        while chord_root > 60:
            chord_root -= 12

        return chord_root

    def _humanize_performance(self, instrument: pretty_midi.Instrument) -> pretty_midi.Instrument:
        for note in instrument.notes:
            timing_var = random.uniform(-0.01, 0.01)
            note.start = max(0, note.start + timing_var)
            note.end = max(note.start + 0.1, note.end + timing_var)

            velocity_var = random.randint(-3, 3)
            note.velocity = max(40, min(100, note.velocity + velocity_var))

        return instrument
