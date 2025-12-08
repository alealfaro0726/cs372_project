
import pretty_midi
import numpy as np
import time
from typing import List, Tuple, Optional, Dict


class EmotionMusicGenerator:

    INSTRUMENTS = {
        'happy': {
            'melody_pool': [(0, 'Acoustic Grand Piano'), (73, 'Flute'), (11, 'Vibraphone')],
            'harmony': (40, 'Violin'),
            'bass': (32, 'Acoustic Bass'),
            'additional': (73, 'Flute'),
            'woodwind': (73, 'Flute'),
            'drums': (0, 'Standard Drum Kit')
        },
        'sad': {
            'melody_pool': [(0, 'Acoustic Grand Piano'), (71, 'Clarinet'), (43, 'Cello')],
            'harmony': (48, 'String Ensemble 1'),
            'bass': (43, 'Cello'),
            'additional': (71, 'Clarinet'),
            'woodwind': (71, 'Clarinet'),
            'drums': (0, 'Standard Drum Kit')
        },
        'calm': {
            'melody_pool': [(0, 'Acoustic Grand Piano'), (73, 'Flute'), (11, 'Vibraphone')],
            'harmony': (92, 'Pad 5 (bowed)'),
            'bass': (33, 'Electric Bass (finger)'),
            'additional': (75, 'Pan Flute'),
            'woodwind': (71, 'Clarinet'),
            'drums': (0, 'Standard Drum Kit')
        },
        'angry': {
            'melody_pool': [(29, 'Overdriven Guitar'), (56, 'Trumpet'), (80, 'Square Lead')],
            'harmony': (48, 'String Ensemble 1'),
            'bass': (38, 'Synth Bass 1'),
            'additional': (61, 'Brass Section'),
            'woodwind': (70, 'Bassoon'),
            'drums': (0, 'Power Drum Kit')
        },
        'surprised': {
            'melody_pool': [(11, 'Vibraphone'), (73, 'Flute'), (0, 'Acoustic Grand Piano')],
            'harmony': (40, 'Violin'),
            'bass': (32, 'Acoustic Bass'),
            'additional': (73, 'Flute'),
            'woodwind': (73, 'Flute'),
            'drums': (0, 'Standard Drum Kit')
        }
    }

    FAMILY_PROGRAMS: Dict[str, list] = {
        'piano_keys': list(range(0, 8)), 
        'electric_keys': list(range(4, 16)),
        'mallets': list(range(9, 16)),
        'guitars': list(range(24, 32)),
        'basses': list(range(32, 40)),
        'strings': list(range(40, 48)),
        'brass': list(range(56, 64)),
        'woodwinds': [71, 72, 73, 74, 75, 68, 69],
        'pads': list(range(88, 96)),
        'leads': list(range(80, 88)),
        'plucks': [100, 101, 104],
        'synth_bass': [38, 39, 36, 37],
    }

    CHORD_PROGRESSIONS = {
        'happy': [
            [0, 4, 7],
            [5, 9, 12],
            [7, 11, 14],
            [0, 4, 7],
        ],
        'sad': [
            [0, 3, 7],
            [8, 11, 15],
            [5, 8, 12],
            [7, 10, 14],
        ],
        'calm': [
            [0, 4, 7],
            [9, 12, 16],
            [5, 9, 12],
            [0, 4, 7],
        ],
        'angry': [
            [0, 3, 7],
            [7, 10, 14],
            [0, 3, 7],
            [10, 13, 17],
        ],
        'surprised': [
            [0, 4, 7],
            [2, 6, 9],
            [7, 11, 14],
            [0, 4, 7],
        ]
    }

    SCALES = {
        'happy': [0, 2, 4, 5, 7, 9, 11],
        'sad': [0, 2, 3, 5, 7, 8, 10],
        'calm': [0, 2, 4, 5, 7, 9, 11],
        'angry': [0, 2, 3, 5, 7, 8, 10],
        'surprised': [0, 2, 4, 6, 7, 9, 11]
    }

    MELODIC_PATTERNS = {
        'happy': [
            [0, 2, 4, 2, 0, 2, 4, 2],
            [4, 2, 0, 2, 4, 4, 4, 2],
            [0, 0, 4, 4, 5, 5, 4, 2],
            [0, 2, 4, 5, 4, 2, 0, 0],
        ],
        'sad': [
            [4, 3, 2, 1, 0, 1, 2, 1],
            [0, 2, 1, 0, 3, 2, 1, 0],
            [2, 1, 0, 2, 1, 0, 1, 0],
            [0, 1, 2, 3, 2, 1, 0, 0],
        ],
        'calm': [
            [0, 2, 4, 2, 0, 2, 4, 2],
            [0, 0, 2, 2, 4, 4, 2, 0],
            [2, 0, 2, 4, 2, 0, 0, 0],
            [0, 2, 0, 4, 2, 0, 2, 0],
        ],
        'angry': [
            [0, 3, 5, 3, 0, 3, 5, 3],
            [5, 3, 0, 5, 3, 0, 3, 0],
            [0, 0, 3, 3, 5, 5, 3, 0],
            [5, 3, 2, 0, 5, 3, 2, 0],
        ],
        'surprised': [
            [0, 4, 0, 6, 4, 2, 0, 2],
            [0, 2, 6, 4, 0, 2, 6, 4],
            [0, 0, 6, 6, 4, 4, 2, 0],
            [0, 6, 2, 5, 0, 6, 2, 0],
        ]
    }

    TEMPO_BPM = {
        'happy': 120,
        'sad': 70,
        'calm': 78,
        'angry': 160,
        'surprised': 130
    }

    NOTE_DURATIONS = {
        'happy': [0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 1.0, 0.5],
        'sad': [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5],
        'calm': [1.0, 0.75, 1.0, 0.75, 1.25, 1.0, 0.75, 1.0],
        'angry': [0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.5, 0.25],
        'surprised': [0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.25]
    }

    VELOCITY = {
        'happy': 80,
        'sad': 60,
        'calm': 65,
        'angry': 95,
        'surprised': 75
    }

    def __init__(self, emotion: str = 'happy', root_note: int = 60, duration: float = 30.0, variation_seed: Optional[int] = None):
        self.emotion = emotion if emotion in self.SCALES else 'happy'
        self.root_note = root_note
        self.duration = duration
        if variation_seed is None:
            variation_seed = int(time.time() * 1000) % 1_000_000_000
        self.rng = np.random.default_rng(variation_seed)
        self.scale = self.SCALES[self.emotion]
        self.chord_progression = self.CHORD_PROGRESSIONS[self.emotion]
        self.tempo_bpm = self.TEMPO_BPM[self.emotion]
        self.melodic_patterns = self.MELODIC_PATTERNS[self.emotion]
        self.note_durations = self.NOTE_DURATIONS[self.emotion]
        self.velocity = self.VELOCITY[self.emotion]
        base_inst = dict(self.INSTRUMENTS[self.emotion])
        if 'melody_pool' in base_inst:
            base_inst['melody'] = self.rng.choice(base_inst['melody_pool'])
            del base_inst['melody_pool']
        self.instruments = self._select_instruments(base_inst)

    def _select_instruments(self, defaults: Dict) -> Dict:
        emotion_bias = {
            'happy': {'strings': 0.8, 'piano_keys': 0.8, 'woodwinds': 0.6, 'mallets': 0.6, 'brass': 0.5, 'pads': 0.4},
            'sad': {'strings': 0.8, 'woodwinds': 0.7, 'piano_keys': 0.6, 'pads': 0.6, 'basses': 0.5},
            'calm': {'pads': 0.9, 'woodwinds': 0.7, 'strings': 0.7, 'piano_keys': 0.6, 'mallets': 0.4},
            'angry': {'brass': 0.8, 'guitars': 0.8, 'synth_bass': 0.7, 'leads': 0.6, 'drums': 0.8},
            'surprised': {'mallets': 0.7, 'woodwinds': 0.6, 'strings': 0.6, 'pads': 0.5, 'leads': 0.5}
        }
        bias = emotion_bias.get(self.emotion, {})
        families = list(self.FAMILY_PROGRAMS.keys())
        scores = []
        for fam in families:
            base = bias.get(fam, 0.2)
            jitter = float(self.rng.uniform(-0.05, 0.05))
            scores.append((fam, max(0.0, base + jitter)))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        pick_n = int(self.rng.integers(3, 7))
        picked = [fam for fam, _ in scores[:pick_n]]

        def pick_program(fam_name):
            programs = self.FAMILY_PROGRAMS.get(fam_name, [])
            if not programs:
                return None
            return int(self.rng.choice(programs))

        selected = dict(defaults)
        if 'melody' in selected:
            fam = picked[0]
            prog = pick_program(fam)
            if prog is not None:
                selected['melody'] = (prog, f'Auto-{fam}')
        if 'harmony' in selected and len(picked) > 1:
            fam = picked[1]
            prog = pick_program(fam)
            if prog is not None:
                selected['harmony'] = (prog, f'Auto-{fam}')
        if 'additional' in selected and len(picked) > 2:
            fam = picked[2]
            prog = pick_program(fam)
            if prog is not None:
                selected['additional'] = (prog, f'Auto-{fam}')
        if 'woodwind' in selected and len(picked) > 3:
            fam = picked[3]
            prog = pick_program(fam)
            if prog is not None:
                selected['woodwind'] = (prog, f'Auto-{fam}')
        bass_fams = [f for f in picked if f in ('basses', 'synth_bass')]
        if bass_fams:
            prog = pick_program(bass_fams[0])
            if prog is not None:
                selected['bass'] = (prog, f'Auto-{bass_fams[0]}')

        guitar_fams = [f for f in picked if f in ('guitars',)]
        if guitar_fams:
            prog = pick_program(guitar_fams[0])
            if prog is not None:
                selected['guitar'] = (prog, f'Auto-{guitar_fams[0]}')
        return selected

    def generate_melody(self) -> pretty_midi.Instrument:
        program, name = self.instruments['melody']
        melody = pretty_midi.Instrument(program=int(program), name=name)

        current_time = 0.0
        pattern_idx = 0
        note_in_pattern = 0

        n_chords = len(self.chord_progression)
        chord_duration = self.duration / n_chords
        chord_idx = 0

        while current_time < self.duration:
            pattern = self.melodic_patterns[pattern_idx % len(self.melodic_patterns)]
            scale_degree = pattern[note_in_pattern % len(pattern)]
            duration = self.note_durations[note_in_pattern % len(self.note_durations)]

            octave_offset = 0
            if scale_degree >= 7:
                octave_offset = 12
                scale_degree = scale_degree % 7

            pitch = self.root_note + self.scale[scale_degree] + octave_offset
            pitch = max(48, min(84, pitch))

            velocity = self.velocity + self.rng.integers(-10, 10)
            velocity = max(40, min(120, velocity))

            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=current_time,
                end=min(current_time + duration, self.duration)
            )
            melody.notes.append(note)

            current_time += duration
            note_in_pattern += 1

            if current_time >= (chord_idx + 1) * chord_duration:
                pattern_idx += 1
                chord_idx += 1
                note_in_pattern = 0

        return melody

    def generate_harmony(self) -> pretty_midi.Instrument:
        program, name = self.instruments['harmony']
        harmony = pretty_midi.Instrument(program=int(program), name=name)

        n_chords = len(self.chord_progression)
        chord_duration = self.duration / n_chords
        current_time = 0.0
        style_pool = ['sustain', 'arp', 'broken', 'half-drop']
        if self.emotion in ['calm', 'sad']:
            style_pool = ['sustain', 'broken', 'sustain', 'half-drop']

        for chord_intervals in self.chord_progression:
            style = self.rng.choice(style_pool)
            inversion = int(self.rng.integers(0, 3))
            chord_voicing = chord_intervals[inversion:] + chord_intervals[:inversion]

            base_vel = max(38, self.velocity - 40)
            jitter = float(self.rng.uniform(-0.08, 0.08))
            octave_lift = int(self.rng.choice([0, 12, 12, 0])) 

            if style == 'sustain':
                for idx, interval in enumerate(chord_voicing[:3]):
                    pitch = self.root_note + interval - (12 if idx == 0 else 0) + octave_lift
                    pitch = max(36, min(76, pitch))
                    note = pretty_midi.Note(
                        velocity=int(base_vel + idx * 3),
                        pitch=int(pitch),
                        start=current_time + max(0.0, jitter + idx * 0.05),
                        end=current_time + chord_duration * 0.85
                    )
                    harmony.notes.append(note)
            elif style == 'broken':
                step = chord_duration / 3
                for idx, interval in enumerate(chord_voicing[:3]):
                    pitch = self.root_note + interval - (12 if idx == 0 else 0) + octave_lift
                    pitch = max(36, min(76, pitch))
                    start = current_time + idx * step + max(0.0, jitter)
                    note = pretty_midi.Note(
                        velocity=int(base_vel + idx * 4),
                        pitch=int(pitch),
                        start=start,
                        end=start + step * 0.9
                    )
                    harmony.notes.append(note)
            elif style == 'half-drop':
                picked = chord_voicing[:2]
                for idx, interval in enumerate(picked):
                    pitch = self.root_note + interval - (12 if idx == 0 else 0) + octave_lift
                    pitch = max(36, min(76, pitch))
                    start = current_time + idx * (chord_duration * 0.25) + max(0.0, jitter)
                    note = pretty_midi.Note(
                        velocity=int(base_vel + idx * 5),
                        pitch=int(pitch),
                        start=start,
                        end=start + chord_duration * 0.6
                    )
                    harmony.notes.append(note)
            else: 
                arp_delay = 0.18
                for idx, interval in enumerate(chord_voicing[:3]):
                    pitch = self.root_note + interval - (12 if idx == 0 else 0) + octave_lift
                    pitch = max(36, min(76, pitch))
                    note_start = current_time + (idx * arp_delay) + max(0.0, jitter)
                    note = pretty_midi.Note(
                        velocity=int(base_vel + idx * 5),
                        pitch=int(pitch),
                        start=note_start,
                        end=note_start + chord_duration * 0.55
                    )
                    harmony.notes.append(note)

            current_time += chord_duration

        return harmony

    def generate_woodwind(self) -> Optional[pretty_midi.Instrument]:
        if 'woodwind' not in self.instruments:
            return None

        program, name = self.instruments['woodwind']
        wood = pretty_midi.Instrument(program=int(program), name=name)

        n_chords = len(self.chord_progression)
        chord_duration = self.duration / n_chords
        current_time = 0.0

        for chord_intervals in self.chord_progression:
            choice = int(self.rng.choice([1, 2, 2, 0]))
            interval = chord_intervals[min(choice, len(chord_intervals) - 1)]
            octave_lift = int(self.rng.choice([12, 0, 12]))
            pitch = self.root_note + interval + octave_lift
            pitch = max(55, min(88, pitch))

            start = current_time + float(self.rng.uniform(0.0, chord_duration * 0.25))
            end = start + chord_duration * float(self.rng.uniform(0.5, 0.9))
            vel = max(40, min(72, int(self.velocity - 10 + self.rng.integers(-5, 6))))

            wood.notes.append(pretty_midi.Note(
                velocity=int(vel),
                pitch=int(pitch),
                start=start,
                end=end
            ))
            current_time += chord_duration

        return wood

    def generate_bass(self) -> pretty_midi.Instrument:
        program, name = self.instruments['bass']
        bass = pretty_midi.Instrument(program=int(program), name=name)

        n_chords = len(self.chord_progression)
        chord_duration = self.duration / n_chords
        current_time = 0.0

        for chord_intervals in self.chord_progression:
            root = self.root_note + chord_intervals[0] - 24
            root = max(28, min(48, root))

            note = pretty_midi.Note(
                velocity=int(max(45, self.velocity - 25)),
                pitch=int(root),
                start=current_time,
                end=current_time + chord_duration * 0.7
            )
            bass.notes.append(note)
            current_time += chord_duration

        return bass

    def generate_additional(self) -> pretty_midi.Instrument:
        program, name = self.instruments['additional']
        additional = pretty_midi.Instrument(program=int(program), name=name)

        current_time = 0.0
        pattern_idx = 0
        note_in_pattern = 0

        n_chords = len(self.chord_progression)
        chord_duration = self.duration / n_chords
        chord_idx = 0

        beat_duration = 0.5

        while current_time < self.duration:
            if int(current_time / beat_duration) % 2 == 0:
                pattern = self.melodic_patterns[pattern_idx % len(self.melodic_patterns)]
                scale_degree = pattern[note_in_pattern % len(pattern)]

                octave_offset = 12
                if scale_degree >= 7:
                    octave_offset = 24
                    scale_degree = scale_degree % 7

                pitch = self.root_note + self.scale[scale_degree] + octave_offset
                pitch = max(60, min(96, pitch))

                velocity = max(40, min(80, self.velocity - 30))

                note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(pitch),
                    start=current_time,
                    end=min(current_time + beat_duration * 0.8, self.duration)
                )
                additional.notes.append(note)

            current_time += beat_duration
            note_in_pattern += 1

            if current_time >= (chord_idx + 1) * chord_duration:
                pattern_idx += 1
                chord_idx += 1
                note_in_pattern = 0

        return additional

    def generate_guitar(self) -> Optional[pretty_midi.Instrument]:
        if 'guitar' not in self.instruments:
            return None
        program, name = self.instruments['guitar']
        guitar = pretty_midi.Instrument(program=int(program), name=name)

        n_chords = len(self.chord_progression)
        chord_duration = self.duration / n_chords
        current_time = 0.0

        strum_offsets = [0.0, 0.05, 0.1]

        for chord_intervals in self.chord_progression:
            chord_tones = chord_intervals[:3]
            base_pitch = self.root_note - 12
            for idx, interval in enumerate(chord_tones):
                offset = strum_offsets[idx % len(strum_offsets)]
                pitch = base_pitch + interval + (12 if idx > 0 else 0)
                pitch = max(48, min(80, pitch))
                vel = int(max(50, min(90, self.velocity - 5 + self.rng.integers(-6, 6))))
                note = pretty_midi.Note(
                    velocity=vel,
                    pitch=int(pitch),
                    start=current_time + offset,
                    end=current_time + chord_duration * 0.7
                )
                guitar.notes.append(note)
            current_time += chord_duration

        return guitar

    def generate_drums(self) -> pretty_midi.Instrument:
        if self.emotion == 'calm':
            return pretty_midi.Instrument(program=0, is_drum=True, name='Drums (muted)') 

        drums = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')

        KICK = 36
        SNARE = 38
        HIHAT_CLOSED = 42
        HIHAT_OPEN = 46
        CRASH = 49
        RIDE = 51

        if self.emotion == 'happy':
            beat_duration = 60.0 / self.tempo_bpm
            current_time = 0.0

            while current_time < self.duration:
                for beat in [0, 2]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=100, pitch=int(KICK),
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for beat in [1, 3]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=90, pitch=int(SNARE),
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for i in range(8):
                    velocity = 70 if i % 2 == 0 else 50
                    drums.notes.append(pretty_midi.Note(
                        velocity=int(velocity), pitch=int(HIHAT_CLOSED),
                        start=current_time + i * (beat_duration / 2),
                        end=current_time + i * (beat_duration / 2) + 0.05
                    ))

                current_time += beat_duration * 4

        elif self.emotion == 'sad':
            beat_duration = 60.0 / self.tempo_bpm
            current_time = 0.0

            while current_time < self.duration:
                drums.notes.append(pretty_midi.Note(
                    velocity=60, pitch=int(KICK),
                    start=current_time,
                    end=current_time + 0.1
                ))

                drums.notes.append(pretty_midi.Note(
                    velocity=50, pitch=int(SNARE),
                    start=current_time + 2 * beat_duration,
                    end=current_time + 2 * beat_duration + 0.1
                ))

                for i in [0, 2]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=40, pitch=int(RIDE),
                        start=current_time + i * beat_duration,
                        end=current_time + i * beat_duration + 0.3
                    ))

                current_time += beat_duration * 4

        elif self.emotion == 'calm':
            beat_duration = 60.0 / self.tempo_bpm
            current_time = 0.0

            while current_time < self.duration:
                for beat in [0, 2]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=65, pitch=int(KICK),
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for i in range(4):
                    drums.notes.append(pretty_midi.Note(
                        velocity=50, pitch=int(HIHAT_CLOSED),
                        start=current_time + i * beat_duration,
                        end=current_time + i * beat_duration + 0.05
                    ))

                current_time += beat_duration * 4

        elif self.emotion == 'angry':
            beat_duration = 60.0 / self.tempo_bpm
            current_time = 0.0

            while current_time < self.duration:
                for beat in [0, 1.5, 2, 3.5]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=120, pitch=int(KICK),
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for beat in [1, 3]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=115, pitch=int(SNARE),
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for i in range(16):
                    velocity = 85 if i % 4 == 0 else 70
                    drums.notes.append(pretty_midi.Note(
                        velocity=int(velocity), pitch=int(HIHAT_CLOSED),
                        start=current_time + i * (beat_duration / 4),
                        end=current_time + i * (beat_duration / 4) + 0.03
                    ))

                current_time += beat_duration * 4

        else:
            beat_duration = 60.0 / self.tempo_bpm
            current_time = 0.0

            while current_time < self.duration:
                for beat in [0, 1.5, 3]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=95, pitch=int(KICK),
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for beat in [1.5, 2.5]:
                    drums.notes.append(pretty_midi.Note(
                        velocity=85, pitch=int(SNARE),
                        start=current_time + beat * beat_duration,
                        end=current_time + beat * beat_duration + 0.1
                    ))

                for i in range(8):
                    velocity = 75 if i in [0, 3, 6] else 55
                    drums.notes.append(pretty_midi.Note(
                        velocity=int(velocity), pitch=int(HIHAT_CLOSED),
                        start=current_time + i * (beat_duration / 2),
                        end=current_time + i * (beat_duration / 2) + 0.05
                    ))

                current_time += beat_duration * 4

        return drums

    def generate(self) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI()

        melody = self.generate_melody()
        midi.instruments.append(melody)

        harmony = self.generate_harmony()
        midi.instruments.append(harmony)

        bass = self.generate_bass()
        midi.instruments.append(bass)

        additional = self.generate_additional()
        midi.instruments.append(additional)

        wood = self.generate_woodwind()
        if wood is not None:
            midi.instruments.append(wood)

        guitar = self.generate_guitar()
        if guitar is not None:
            midi.instruments.append(guitar)

        if self.emotion != 'calm': 
            from musical_expression import enhance_musical_expression
            midi = enhance_musical_expression(midi, emotion=self.emotion)

        return midi


def generate_emotion_music(emotion: str, duration: float = 30.0, root_note: int = 60) -> pretty_midi.PrettyMIDI:
    import random
    from improved_composer import ImprovedComposer

    temp_gen = EmotionMusicGenerator(
        emotion=emotion,
        root_note=root_note,
        duration=duration,
        variation_seed=int(time.time() * 1000) % 1_000_000_000
    )

    scale_type_map = {
        'happy': 'major',
        'calm': 'major',
        'surprised': 'major',
        'sad': 'minor',
        'angry': 'minor'
    }
    scale_type = scale_type_map.get(emotion, 'major')

    composer = ImprovedComposer(
        key_root=root_note,
        scale_type=scale_type,
        tempo=temp_gen.tempo_bpm
    )

    midi = composer.compose_song(
        duration=duration,
        instruments=temp_gen.instruments
    )

    from musical_expression import enhance_musical_expression
    midi = enhance_musical_expression(midi, emotion=emotion)

    return midi
