
import pretty_midi
import numpy as np
import random
from typing import List, Tuple


class ImprovedComposer:

    CHORD_PROGRESSIONS = {
        'major': [
            ['I', 'V', 'vi', 'IV'],
            ['I', 'IV', 'V', 'I'],
            ['I', 'vi', 'IV', 'V'],
            ['I', 'V', 'IV', 'I'],
            ['vi', 'IV', 'I', 'V'],
            ['I', 'IV', 'vi', 'V'],
            ['IV', 'I', 'V', 'vi'],
            ['I', 'vi', 'ii', 'V'],
            ['I', 'iii', 'IV', 'V'],
            ['vi', 'V', 'IV', 'V'],
        ],
        'minor': [
            ['i', 'VI', 'III', 'VII'],
            ['i', 'iv', 'VII', 'i'],
            ['i', 'VII', 'VI', 'VII'],
            ['i', 'iv', 'v', 'i'],
            ['i', 'VI', 'VII', 'i'],
            ['i', 'III', 'VII', 'VI'],
            ['i', 'iv', 'VI', 'VII'],
        ]
    }

    SCALE_DEGREES = {
        'major': {
            'I': [0, 4, 7],
            'ii': [2, 5, 9],
            'iii': [4, 7, 11],
            'IV': [5, 9, 12],
            'V': [7, 11, 14],
            'vi': [9, 12, 16],
            'vii': [11, 14, 17],
        },
        'minor': {
            'i': [0, 3, 7],
            'ii': [2, 5, 8],
            'III': [3, 7, 10],
            'iv': [5, 8, 12],
            'v': [7, 10, 14],
            'VI': [8, 12, 15],
            'VII': [10, 14, 17],
        }
    }

    def __init__(self, key_root: int = 60, scale_type: str = 'major', tempo: int = 120, variation_seed=None):
        self.key_root = key_root
        self.scale_type = scale_type
        self.tempo = tempo
        self.beat_duration = 60.0 / tempo

        if variation_seed is None:
            import time
            variation_seed = int(time.time() * 1000) % 1000000

        random.seed(variation_seed)
        np.random.seed(variation_seed)
        self.variation_id = variation_seed

        if scale_type == 'major':
            self.scale = [0, 2, 4, 5, 7, 9, 11]
        else:
            self.scale = [0, 2, 3, 5, 7, 8, 10]

    def compose_song(self, duration: float = 35.0, instruments: dict = None) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)

        if instruments is None:
            instruments = {
                'melody': (0, 'Piano'),
                'harmony': (48, 'Strings'),
                'bass': (32, 'Bass'),
            }

        chord_progression = random.choice(self.CHORD_PROGRESSIONS[self.scale_type])

        bars_per_section = 4
        total_bars = int(duration / (self.beat_duration * 4))

        structure = self._create_song_structure(total_bars, bars_per_section)

        melody_notes = self._create_cohesive_melody(structure, chord_progression)
        harmony_notes = self._create_harmony(structure, chord_progression)
        bass_notes = self._create_bass_line(structure, chord_progression)

        if 'melody' in instruments:
            melody_program, melody_name = instruments['melody']
            melody_inst = pretty_midi.Instrument(program=melody_program, name=melody_name)
            melody_inst.notes.extend(melody_notes)

            if melody_program == 0:
                try:
                    from piano_performance_enhancer import PianoPerformanceEnhancer
                except ImportError:
                    import sys
                    sys.path.append('src')
                    from piano_performance_enhancer import PianoPerformanceEnhancer

                enhancer = PianoPerformanceEnhancer(
                    key_root=self.key_root,
                    scale_type=self.scale_type,
                    tempo=self.tempo,
                    variation_seed=self.variation_id
                )
                melody_inst = enhancer.enhance_piano_melody(melody_inst)

            midi.instruments.append(melody_inst)

        if 'harmony' in instruments and harmony_notes:
            harmony_program, harmony_name = instruments['harmony']
            harmony_inst = pretty_midi.Instrument(program=harmony_program, name=harmony_name)
            harmony_inst.notes.extend(harmony_notes)
            midi.instruments.append(harmony_inst)

        if 'bass' in instruments and bass_notes:
            bass_program, bass_name = instruments['bass']
            bass_inst = pretty_midi.Instrument(program=bass_program, name=bass_name)
            bass_inst.notes.extend(bass_notes)
            midi.instruments.append(bass_inst)

        return midi

    def _create_song_structure(self, total_bars: int, bars_per_section: int) -> List[str]:
        structure = []
        remaining_bars = total_bars

        if remaining_bars >= 2:
            structure.append('intro')
            remaining_bars -= 2

        while remaining_bars >= bars_per_section:
            structure.append('verse')
            remaining_bars -= bars_per_section

            if remaining_bars >= bars_per_section:
                structure.append('chorus')
                remaining_bars -= bars_per_section

        if remaining_bars >= 2:
            structure.append('outro')

        return structure

    def _create_cohesive_melody(self, structure: List[str], chord_progression: List[str]) -> List[pretty_midi.Note]:
        notes = []
        current_time = 0.0
        bar_duration = self.beat_duration * 4

        phrase_motif = None

        for section in structure:
            if section == 'intro':
                section_notes = self._create_intro_melody(current_time, bar_duration * 2)
            elif section == 'verse':
                if phrase_motif is None:
                    phrase_motif = self._generate_motif()
                section_notes = self._create_verse_melody(current_time, bar_duration * 4, chord_progression, phrase_motif)
            elif section == 'chorus':
                section_notes = self._create_chorus_melody(current_time, bar_duration * 4, chord_progression)
            else:
                section_notes = self._create_outro_melody(current_time, bar_duration * 2)

            notes.extend(section_notes)
            section_duration = max([n.end for n in section_notes]) - current_time if section_notes else bar_duration * 2
            current_time += section_duration

        return notes

    def _generate_motif(self) -> List[int]:
        motif_length = random.choice([3, 4, 5])
        motif = []

        motif_style = random.choice(['stepwise', 'skipwise', 'mixed'])

        for i in range(motif_length):
            if motif_style == 'stepwise':
                degree = random.choice([0, 1, 2, 3, 4])
            elif motif_style == 'skipwise':
                degree = random.choice([0, 2, 4, 5])
            else:
                degree = random.choice([0, 1, 2, 3, 4, 5, 6])

            motif.append(self.scale[degree % len(self.scale)])

        return motif

    def _create_intro_melody(self, start_time: float, duration: float) -> List[pretty_midi.Note]:
        notes = []
        note_duration = self.beat_duration / 2
        current_time = start_time

        intro_patterns = [
            [0, 2, 4, 5, 4, 2, 0, 2],
            [4, 2, 0, 2, 4, 5, 7, 5],
            [0, 4, 2, 5, 4, 2, 0, 0],
            [2, 4, 5, 4, 2, 0, 2, 4],
            [0, 0, 2, 4, 2, 4, 5, 4],
            [4, 5, 4, 2, 4, 2, 0, 2],
        ]

        pattern = random.choice(intro_patterns)
        octave_shift = random.choice([0, 12])

        for i, scale_degree in enumerate(pattern):
            pitch = self.key_root + self.scale[scale_degree % len(self.scale)] + octave_shift
            velocity = random.randint(60, 80)
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=current_time,
                end=current_time + note_duration * 0.9
            )
            notes.append(note)
            current_time += note_duration

        return notes

    def _create_verse_melody(self, start_time: float, duration: float,
                            chord_progression: List[str], motif: List[int]) -> List[pretty_midi.Note]:
        notes = []
        beats_per_chord = 4
        chord_duration = self.beat_duration * beats_per_chord
        current_time = start_time

        scale_degrees = self.SCALE_DEGREES[self.scale_type]

        for chord_symbol in chord_progression:
            chord_tones = scale_degrees[chord_symbol]

            for interval in motif:
                pitch = self.key_root + 12 + (chord_tones[0] + interval) % 12

                octave_variation = random.choice([-12, 0, 0, 12])
                pitch += octave_variation

                while pitch < self.key_root:
                    pitch += 12
                while pitch > self.key_root + 36:
                    pitch -= 12

                velocity = random.randint(65, 85)
                note_length = self.beat_duration * random.choice([0.5, 0.75, 1.0, 1.5])

                note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(pitch),
                    start=current_time,
                    end=min(current_time + note_length, current_time + chord_duration / len(motif))
                )
                notes.append(note)
                current_time += chord_duration / len(motif)

        return notes

    def _create_chorus_melody(self, start_time: float, duration: float,
                             chord_progression: List[str]) -> List[pretty_midi.Note]:
        notes = []
        beats_per_chord = 4
        chord_duration = self.beat_duration * beats_per_chord
        current_time = start_time

        scale_degrees = self.SCALE_DEGREES[self.scale_type]

        chorus_rhythms = [
            [0, 1, 2, 3],
            [0, 0.5, 2, 2.5],
            [0, 2, 3],
            [0, 1.5, 2, 3.5],
        ]
        rhythm = random.choice(chorus_rhythms)

        for chord_symbol in chord_progression:
            chord_tones = scale_degrees[chord_symbol]

            for beat_offset in rhythm:
                tone_choice = random.choice([0, 1, 2])
                pitch = self.key_root + 12 + chord_tones[tone_choice]

                if random.random() < 0.3:
                    pitch += 12

                velocity = random.randint(70, 90)
                note_length = self.beat_duration * random.choice([0.75, 1.0, 1.25])

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=current_time + beat_offset * self.beat_duration,
                    end=current_time + beat_offset * self.beat_duration + note_length
                )
                notes.append(note)

            current_time += chord_duration

        return notes

    def _create_outro_melody(self, start_time: float, duration: float) -> List[pretty_midi.Note]:
        notes = []
        note_duration = self.beat_duration
        current_time = start_time

        final_notes = [
            self.key_root + self.scale[4] + 12,
            self.key_root + self.scale[2] + 12,
            self.key_root + 12,
        ]

        for i, pitch in enumerate(final_notes):
            velocity = 80 - i * 15
            note = pretty_midi.Note(
                velocity=int(max(velocity, 50)),
                pitch=int(pitch),
                start=current_time,
                end=current_time + note_duration * (2 if i == len(final_notes) - 1 else 1)
            )
            notes.append(note)
            current_time += note_duration

        return notes

    def _create_harmony(self, structure: List[str], chord_progression: List[str]) -> List[pretty_midi.Note]:
        notes = []
        current_time = 0.0
        bar_duration = self.beat_duration * 4
        scale_degrees = self.SCALE_DEGREES[self.scale_type]

        for section in structure:
            if section in ['intro', 'outro']:
                bars = 2
            else:
                bars = 4

            for _ in range(bars):
                for chord_symbol in chord_progression:
                    chord_tones = scale_degrees[chord_symbol]
                    chord_duration = self.beat_duration * 4 / len(chord_progression)

                    interval = chord_tones[1]
                    pitch = self.key_root + interval
                    velocity = random.randint(45, 60)

                    note = pretty_midi.Note(
                        velocity=int(velocity),
                        pitch=int(pitch),
                        start=current_time,
                        end=current_time + chord_duration * 0.85
                    )
                    notes.append(note)

                    current_time += chord_duration

        return notes

    def _create_bass_line(self, structure: List[str], chord_progression: List[str]) -> List[pretty_midi.Note]:
        notes = []
        current_time = 0.0
        bar_duration = self.beat_duration * 4
        scale_degrees = self.SCALE_DEGREES[self.scale_type]

        for section in structure:
            if section in ['intro', 'outro']:
                bars = 2
            else:
                bars = 4

            for _ in range(bars):
                for chord_symbol in chord_progression:
                    chord_root = scale_degrees[chord_symbol][0]
                    bass_pitch = self.key_root - 24 + chord_root

                    chord_duration = self.beat_duration * 4 / len(chord_progression)

                    velocity = random.randint(55, 70)

                    note = pretty_midi.Note(
                        velocity=int(velocity),
                        pitch=int(bass_pitch),
                        start=current_time,
                        end=current_time + chord_duration * 0.8
                    )
                    notes.append(note)
                    current_time += chord_duration

        return notes
