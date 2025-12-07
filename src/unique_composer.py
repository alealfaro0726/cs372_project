
import pretty_midi
import numpy as np
import random
from typing import List, Tuple, Dict


class UniqueComposer:

    CHORD_VOICINGS = {
        'C': [
            [0, 4, 7],
            [4, 7, 12],
            [7, 12, 16],
            [0, 4, 7, 11],
            [0, 4, 7, 10],
        ],
        'Am': [
            [0, 3, 7],
            [3, 7, 12],
            [0, 3, 7, 10],
        ],
        'F': [
            [5, 9, 12],
            [9, 12, 17],
            [5, 9, 12, 16],
        ],
        'G': [
            [7, 11, 14],
            [11, 14, 19],
            [7, 11, 14, 17],
        ],
        'Dm': [
            [2, 5, 9],
            [5, 9, 14],
            [2, 5, 9, 12],
        ],
        'Em': [
            [4, 7, 11],
            [7, 11, 16],
            [4, 7, 11, 14],
        ],
        'E': [
            [4, 8, 11],
            [8, 11, 16],
        ],
        'D': [
            [2, 6, 9],
            [6, 9, 14],
        ],
        'Bb': [
            [10, 14, 17],
            [14, 17, 22],
        ],
        'Ab': [
            [8, 12, 15],
        ],
        'Eb': [
            [3, 7, 10],
        ],
        'B': [
            [11, 15, 18],
        ],
    }

    RHYTHM_PATTERNS = {
        'straight': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'swing': [0.67, 0.33, 0.67, 0.33, 0.67, 0.33, 0.67, 0.33],
        'dotted': [0.75, 0.25, 0.75, 0.25, 0.75, 0.25],
        'syncopated': [0.25, 0.25, 0.5, 0.25, 0.25, 0.5],
        'triplet': [0.33, 0.33, 0.33, 0.33, 0.33, 0.33],
        'long-short': [1.0, 0.25, 0.25, 0.5, 1.0, 0.5],
    }

    def __init__(self, emotion: str, seed: int = None):
        self.emotion = emotion
        if seed is None:
            seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)

        self.unique_id = seed

    def compose(
        self,
        chord_progression: List[str],
        scale: List[int],
        duration: float,
        tempo: int,
        instruments: Dict,
        root_note: int = 60
    ) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        active_layers = random.sample(['melody', 'harmony', 'bass', 'texture'],
                                     k=random.randint(3, 4))

        print(f"   🎲 Unique Composition ID: {self.unique_id}")
        print(f"   🎹 Active Layers: {', '.join(active_layers)}")

        if 'melody' in active_layers:
            simple_melody = self._create_unique_melody(
                scale, duration, tempo, root_note, instruments
            )

            melody_program, melody_name = instruments.get('melody', (0, 'Piano'))
            if melody_program == 0:
                from piano_performance_enhancer import PianoPerformanceEnhancer

                scale_type = 'major' if len([s for s in scale if s in [0, 4, 7]]) >= 2 else 'minor'

                enhancer = PianoPerformanceEnhancer(
                    key_root=root_note,
                    scale_type=scale_type,
                    tempo=tempo
                )
                melody = enhancer.enhance_piano_melody(simple_melody)
            else:
                melody = simple_melody

            midi.instruments.append(melody)

        if 'harmony' in active_layers:
            harmony = self._create_unique_harmony(
                chord_progression, duration, tempo, root_note, instruments
            )
            midi.instruments.append(harmony)

        if 'bass' in active_layers:
            bass = self._create_unique_bass(
                chord_progression, duration, tempo, root_note, instruments
            )
            midi.instruments.append(bass)

        if 'texture' in active_layers:
            texture = self._create_unique_texture(
                scale, duration, tempo, root_note, instruments
            )
            midi.instruments.append(texture)

        return midi

    def _create_unique_melody(self, scale, duration, tempo, root, instruments):
        program, name = instruments.get('melody', (0, 'Piano'))
        melody_inst = pretty_midi.Instrument(program=program, name=name)

        rhythm_style = random.choice(list(self.RHYTHM_PATTERNS.keys()))
        rhythm = self.RHYTHM_PATTERNS[rhythm_style]

        direction = random.choice(['ascending', 'descending', 'wavy', 'random'])

        current_time = 0.0
        scale_degree = random.randint(0, len(scale) - 1)

        while current_time < duration:
            note_duration = random.choice(rhythm)

            if direction == 'ascending' and random.random() > 0.3:
                scale_degree = min(scale_degree + 1, len(scale) - 1)
            elif direction == 'descending' and random.random() > 0.3:
                scale_degree = max(scale_degree - 1, 0)
            elif direction == 'wavy':
                scale_degree += random.choice([-2, -1, 1, 2])
                scale_degree = max(0, min(scale_degree, len(scale) - 1))
            else:
                scale_degree = random.randint(0, len(scale) - 1)

            octave_shift = random.choice([0, 0, 12, 12, -12])
            pitch = root + scale[scale_degree] + octave_shift
            pitch = max(48, min(84, pitch))

            velocity = random.randint(70, 105)

            actual_duration = note_duration * random.uniform(0.7, 0.95)

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=current_time,
                end=min(current_time + actual_duration, duration)
            )
            melody_inst.notes.append(note)

            current_time += note_duration

        print(f"     ↳ Melody: {rhythm_style} rhythm, {direction} motion")
        return melody_inst

    def _create_unique_harmony(self, chords, duration, tempo, root, instruments):
        program, name = instruments.get('harmony', (48, 'Strings'))
        harmony_inst = pretty_midi.Instrument(program=program, name=name)

        n_chords = len(chords)
        chord_duration = duration / n_chords

        arp_style = random.choice(['block', 'up', 'down', 'up-down', 'random'])

        current_time = 0.0
        for chord_name in chords:
            voicings = self.CHORD_VOICINGS.get(chord_name, [[0, 4, 7]])
            voicing = random.choice(voicings)

            if arp_style == 'block':
                for interval in voicing:
                    pitch = root + interval - random.choice([0, 12])
                    pitch = max(36, min(72, pitch))

                    note = pretty_midi.Note(
                        velocity=random.randint(60, 85),
                        pitch=pitch,
                        start=current_time,
                        end=current_time + chord_duration * 0.95
                    )
                    harmony_inst.notes.append(note)

            else:
                arp_delay = random.uniform(0.08, 0.15)
                note_order = list(range(len(voicing)))

                if arp_style == 'down':
                    note_order.reverse()
                elif arp_style == 'random':
                    random.shuffle(note_order)
                elif arp_style == 'up-down':
                    note_order = note_order + note_order[::-1]

                for i, idx in enumerate(note_order):
                    interval = voicing[idx]
                    pitch = root + interval - 12
                    pitch = max(36, min(72, pitch))

                    note_start = current_time + (i * arp_delay)
                    if note_start >= current_time + chord_duration:
                        break

                    note = pretty_midi.Note(
                        velocity=random.randint(55, 80),
                        pitch=pitch,
                        start=note_start,
                        end=min(note_start + chord_duration * 0.4,
                               current_time + chord_duration)
                    )
                    harmony_inst.notes.append(note)

            current_time += chord_duration

        print(f"     ↳ Harmony: {arp_style} voicing")
        return harmony_inst

    def _create_unique_bass(self, chords, duration, tempo, root, instruments):
        program, name = instruments.get('bass', (32, 'Bass'))
        bass_inst = pretty_midi.Instrument(program=program, name=name)

        n_chords = len(chords)
        chord_duration = duration / n_chords

        pattern_style = random.choice(['root-fifth', 'walking', 'steady', 'syncopated'])

        current_time = 0.0
        for chord_name in chords:
            voicing = self.CHORD_VOICINGS.get(chord_name, [[0, 4, 7]])[0]
            chord_root = voicing[0]

            if pattern_style == 'root-fifth':
                bass_notes = [
                    (chord_root, 0.0, chord_duration * 0.4),
                    (voicing[2] if len(voicing) > 2 else chord_root + 7,
                     chord_duration * 0.5, chord_duration * 0.4)
                ]
            elif pattern_style == 'walking':
                bass_notes = [
                    (chord_root, 0.0, chord_duration * 0.23),
                    (voicing[1] if len(voicing) > 1 else chord_root + 3,
                     chord_duration * 0.25, chord_duration * 0.23),
                    (voicing[2] if len(voicing) > 2 else chord_root + 7,
                     chord_duration * 0.5, chord_duration * 0.23),
                    (chord_root, chord_duration * 0.75, chord_duration * 0.23),
                ]
            elif pattern_style == 'steady':
                bass_notes = [
                    (chord_root, 0.0, chord_duration * 0.9)
                ]
            else:
                bass_notes = [
                    (chord_root, 0.0, chord_duration * 0.2),
                    (chord_root, chord_duration * 0.3, chord_duration * 0.15),
                    (voicing[2] if len(voicing) > 2 else chord_root + 7,
                     chord_duration * 0.6, chord_duration * 0.3),
                ]

            for interval, offset, dur in bass_notes:
                pitch = root + interval - 24
                pitch = max(28, min(48, pitch))

                note = pretty_midi.Note(
                    velocity=random.randint(80, 100),
                    pitch=pitch,
                    start=current_time + offset,
                    end=min(current_time + offset + dur, duration)
                )
                bass_inst.notes.append(note)

            current_time += chord_duration

        print(f"     ↳ Bass: {pattern_style} pattern")
        return bass_inst

    def _create_unique_texture(self, scale, duration, tempo, root, instruments):
        program, name = instruments.get('additional', (92, 'Pad'))
        texture_inst = pretty_midi.Instrument(program=program, name=name)

        n_texture_notes = random.randint(8, 16)

        for i in range(n_texture_notes):
            time_pos = random.uniform(0, duration)

            scale_degree = random.choice(scale)
            pitch = root + scale_degree + random.choice([12, 24])
            pitch = max(60, min(96, pitch))

            note_duration = random.uniform(1.0, 3.0)

            note = pretty_midi.Note(
                velocity=random.randint(40, 60),
                pitch=pitch,
                start=time_pos,
                end=min(time_pos + note_duration, duration)
            )
            texture_inst.notes.append(note)

        print(f"     ↳ Texture: {n_texture_notes} atmospheric notes")
        return texture_inst

    def _create_unique_drums(self, duration, tempo, instruments):
        drums = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')

        KICK = 36
        SNARE = 38
        HIHAT_CLOSED = 42
        HIHAT_OPEN = 46
        CRASH = 49
        RIDE = 51
        TOM_LOW = 45
        TOM_MID = 47
        TOM_HIGH = 48

        beat_duration = 60.0 / tempo
        current_time = 0.0

        pattern_styles = ['basic_rock', 'swing', 'syncopated', 'breakbeat', 'minimal']
        pattern = random.choice(pattern_styles)

        print(f"     ↳ Drums: {pattern} pattern @ {tempo} BPM")

        while current_time < duration:
            if pattern == 'basic_rock':
                for beat in [0, 2]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=random.randint(90, 110), pitch=KICK,
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for beat in [1, 3]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=random.randint(85, 105), pitch=SNARE,
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for i in range(8):
                    vel = random.randint(60, 80) if i % 2 == 0 else random.randint(40, 60)
                    drums.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=HIHAT_CLOSED,
                        start=current_time + i * (beat_duration / 2),
                        end=current_time + i * (beat_duration / 2) + 0.05
                    ))

            elif pattern == 'swing':
                for i in range(6):
                    vel = random.randint(60, 75) if i % 2 == 0 else random.randint(45, 60)
                    drums.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=RIDE,
                        start=current_time + i * (beat_duration * 2/3),
                        end=current_time + i * (beat_duration * 2/3) + 0.2
                    ))

                drums.notes.append(pretty_midi.Note(
                    velocity=random.randint(80, 95), pitch=KICK,
                    start=current_time, end=current_time + 0.1
                ))

                for beat in [1, 3]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=random.randint(75, 90), pitch=SNARE,
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

            elif pattern == 'syncopated':
                for beat in [0, 1.5, 2.75]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=random.randint(95, 115), pitch=KICK,
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for beat in [1, 2.5]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=random.randint(90, 105), pitch=SNARE,
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for i in range(16):
                    vel = random.randint(70, 85) if i % 4 == 0 else random.randint(50, 65)
                    drums.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=HIHAT_CLOSED,
                        start=current_time + i * (beat_duration / 4),
                        end=current_time + i * (beat_duration / 4) + 0.03
                    ))

            elif pattern == 'breakbeat':
                kick_pattern = [0, 1.75, 2.5]
                for beat in kick_pattern:
                    drums.notes.append(pretty_midi.Note(
                        velocity=random.randint(100, 120), pitch=KICK,
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                snare_pattern = [1, 2, 3.5]
                for beat in snare_pattern:
                    drums.notes.append(pretty_midi.Note(
                        velocity=random.randint(95, 110), pitch=SNARE,
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for i in range(16):
                    vel = random.randint(65, 80)
                    drums.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=HIHAT_CLOSED,
                        start=current_time + i * (beat_duration / 4),
                        end=current_time + i * (beat_duration / 4) + 0.03
                    ))

            else:
                drums.notes.append(pretty_midi.Note(
                    velocity=random.randint(70, 85), pitch=KICK,
                    start=current_time, end=current_time + 0.1
                ))

                drums.notes.append(pretty_midi.Note(
                    velocity=random.randint(65, 80), pitch=SNARE,
                    start=current_time + 2 * beat_duration,
                    end=current_time + 2 * beat_duration + 0.1
                ))

                for i in [0, 2]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=random.randint(50, 65), pitch=RIDE,
                        start=current_time + i * beat_duration,
                        end=current_time + i * beat_duration + 0.3
                    ))

            current_time += beat_duration * 4

        return drums
