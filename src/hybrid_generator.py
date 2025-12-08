
import torch
import numpy as np
import pretty_midi
from typing import List, Tuple, Optional

from model import ConditionalTransformer
from midi_tokenizer import MIDITokenizer
from emotion_music_generator import EmotionMusicGenerator, generate_emotion_music


class HybridMusicGenerator:

    def __init__(
        self,
        model: ConditionalTransformer,
        tokenizer: MIDITokenizer,
        device: str = 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate(
        self,
        image_embeds: torch.Tensor,
        emotion_label: torch.Tensor,
        emotion_name: str,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        max_length: int = 128,
        use_theory_guidance: bool = True,
        theory_weight: float = 0.7,
        target_duration: float = 35.0
    ) -> pretty_midi.PrettyMIDI:
        if emotion_name in {"calm", "sad"}:
            temperature = min(temperature, 0.7)
            top_k = min(top_k, 12) if top_k is not None else 12
            top_p = min(top_p, 0.8) if top_p is not None else 0.8
            max_length = min(max_length, 140)

        theory_weight = max(theory_weight, 0.7)

        with torch.no_grad():
            tokens = self.model.generate(
                image_embeds=image_embeds,
                emotion_labels=emotion_label,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                bos_token=self.tokenizer.bos_id,
                eos_token=self.tokenizer.eos_id
            )

        events = self.tokenizer.decode(tokens[0].cpu().tolist())

        if use_theory_guidance:
            return self._theory_guided_generation(
                learned_events=events,
                emotion=emotion_name,
                theory_weight=max(0.7, theory_weight),
                duration=target_duration
            )

        return self._events_to_midi(events, emotion_name, max_duration=target_duration)

    def _theory_guided_generation(
        self,
        learned_events: List[Tuple[str, int, int, int]],
        emotion: str,
        theory_weight: float,
        duration: float
    ) -> pretty_midi.PrettyMIDI:
        from emotion_music_generator import generate_emotion_music
        from improved_composer import ImprovedComposer
        from emotion_music_generator import EmotionMusicGenerator

        theory_midi = generate_emotion_music(emotion=emotion, duration=duration)

        if len(learned_events) == 0:
            return self._trim_midi(self._sanitize_midi(theory_midi), duration)

        learned_midi = self._events_to_midi(learned_events, emotion, max_duration=duration)

        if theory_weight >= 0.9:
            return self._trim_midi(self._sanitize_midi(theory_midi), duration)
        elif theory_weight <= 0.1:
            return learned_midi

        hybrid_midi = pretty_midi.PrettyMIDI(initial_tempo=theory_midi.get_tempo_changes()[1][0])

        temp_gen = EmotionMusicGenerator(emotion=emotion, duration=duration)

        learned_piano = [inst for inst in learned_midi.instruments if inst.program == 0]
        learned_notes = learned_piano[0].notes if learned_piano and len(learned_piano[0].notes) > 0 else []

        for theory_inst in theory_midi.instruments:
            if theory_inst.is_drum:
                continue

            theory_notes = theory_inst.notes

            if len(learned_notes) > 0:
                blended_notes = self._blend_note_sequences(
                    theory_notes,
                    learned_notes,
                    theory_weight,
                    emotion,
                    theory_inst.program
                )

                inst = pretty_midi.Instrument(program=theory_inst.program, name=theory_inst.name)
                inst.notes.extend(blended_notes)
                hybrid_midi.instruments.append(inst)
            else:
                trimmed_inst = pretty_midi.Instrument(program=theory_inst.program, is_drum=theory_inst.is_drum, name=theory_inst.name)
                for n in theory_inst.notes:
                    if n.start >= duration:
                        continue
                    end = min(n.end, duration)
                    trimmed_inst.notes.append(pretty_midi.Note(
                        velocity=n.velocity,
                        pitch=n.pitch,
                        start=n.start,
                        end=end
                    ))
                hybrid_midi.instruments.append(trimmed_inst)

        return self._trim_midi(self._sanitize_midi(hybrid_midi), duration)

    def _blend_note_sequences(
        self,
        theory_notes: List[pretty_midi.Note],
        learned_notes: List[pretty_midi.Note],
        theory_weight: float,
        emotion: str,
        instrument_program: int
    ) -> List[pretty_midi.Note]:
        from emotion_music_generator import EmotionMusicGenerator

        blended = []

        emotion_gen = EmotionMusicGenerator(emotion=emotion)
        scale = emotion_gen.scale
        root = emotion_gen.root_note

        theory_weight = max(0.7, theory_weight)  

        max_time = max(
            max([n.end for n in theory_notes]) if theory_notes else 0,
            max([n.end for n in learned_notes]) if learned_notes else 0
        )

        time_step = 0.5
        current_time = 0.0

        if instrument_program == 0:
            octave_shift = 0
        elif instrument_program in [40, 48, 49]:
            octave_shift = 12
        else:
            octave_shift = -12

        while current_time < max_time:
            if np.random.random() < theory_weight:
                theory_in_range = [n for n in theory_notes
                                   if n.start >= current_time and n.start < current_time + time_step]
                if theory_in_range:
                    for note in theory_in_range:
                        blended.append(pretty_midi.Note(
                            velocity=note.velocity,
                            pitch=note.pitch,
                            start=note.start,
                            end=note.end
                        ))
            else:
                learned_in_range = [n for n in learned_notes
                                    if n.start >= current_time and n.start < current_time + time_step]
                if learned_in_range:
                    for note in learned_in_range:
                        pitch = note.pitch + octave_shift
                        pitch_class = pitch % 12
                        root_class = root % 12
                        relative_pitch = (pitch_class - root_class) % 12

                        if relative_pitch not in scale:
                            closest_scale_note = min(scale, key=lambda x: abs(x - relative_pitch))
                            pitch = root + (pitch // 12 - root // 12) * 12 + closest_scale_note

                        if instrument_program == 0:
                            pitch = max(48, min(84, pitch))
                        elif instrument_program in [40, 48, 49]:
                            pitch = max(55, min(88, pitch))
                        else:
                            pitch = max(28, min(55, pitch))

                        velocity_mod = 1.0
                        if instrument_program != 0:
                            velocity_mod = 0.75

                        blended.append(pretty_midi.Note(
                            velocity=int(note.velocity * velocity_mod),
                            pitch=pitch,
                            start=note.start,
                            end=note.end
                        ))

            current_time += time_step

        return sorted(blended, key=lambda n: n.start)

    def _events_to_midi(
        self,
        events: List[Tuple[str, int, int, int]],
        emotion: str,
        max_duration: Optional[float] = None
    ) -> pretty_midi.PrettyMIDI:
        from emotion_music_generator import EmotionMusicGenerator

        emotion_gen = EmotionMusicGenerator(emotion=emotion)
        tempo = emotion_gen.tempo_bpm

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0, name="Piano")

        active_notes = {}
        current_time = 0.0
        ticks_per_second = (tempo / 60.0) * self.tokenizer.ticks_per_bin
        last_pitch = None 

        for event_type, pitch, velocity, time_delta in events:
            current_time += time_delta / ticks_per_second

            if max_duration is not None and current_time > max_duration:
                break

            if event_type == 'note_on':
                if last_pitch is not None:
                    while pitch - last_pitch > 7:
                        pitch -= 12
                    while last_pitch - pitch > 7:
                        pitch += 12
                active_notes[pitch] = (current_time, velocity)
                last_pitch = pitch
            elif event_type == 'note_off' and pitch in active_notes:
                start_time, note_velocity = active_notes[pitch]
                if current_time > start_time:
                    note = pretty_midi.Note(
                        velocity=note_velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)
                del active_notes[pitch]

        for pitch, (start_time, velocity) in active_notes.items():
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=min(current_time + 0.5, max_duration if max_duration is not None else current_time + 0.5)
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        midi = self._enforce_scale_and_grid(
            midi,
            emotion_gen.scale,
            emotion_gen.root_note,
            ticks_per_second,
            emotion=emotion
        )
        if emotion in {"calm", "sad"}:
            midi = self._relaxation_filter(midi)
        midi = self._sanitize_midi(midi)
        return self._trim_midi(midi, max_duration) if max_duration is not None else midi

    def _trim_midi(self, midi: pretty_midi.PrettyMIDI, duration: float) -> pretty_midi.PrettyMIDI:
        """cut all notes to the target duration to avoid overly long songs."""
        if duration is None:
            return midi

        trimmed = pretty_midi.PrettyMIDI(initial_tempo=midi.get_tempo_changes()[1][0])
        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            for n in inst.notes:
                if n.start >= duration:
                    continue
                new_inst.notes.append(pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start,
                    end=min(n.end, duration)
                ))
            trimmed.instruments.append(new_inst)
        return trimmed

    def _sanitize_midi(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """clamp pitch/velocity to valid MIDI bytes to avoid mido errors."""
        sanitized = pretty_midi.PrettyMIDI(initial_tempo=midi.get_tempo_changes()[1][0])
        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(program=int(inst.program), is_drum=inst.is_drum, name=inst.name)
            if not inst.is_drum and inst.program == 0:
                sorted_notes = sorted(inst.notes, key=lambda n: (n.start, n.end))
                prev_pitch = None
                smoothed = []
                for n in sorted_notes:
                    pitch = int(max(0, min(127, n.pitch)))
                    if prev_pitch is not None:
                        if pitch - prev_pitch > 5:
                            pitch = prev_pitch + 5
                        elif prev_pitch - pitch > 5:
                            pitch = prev_pitch - 5
                    pitch = max(48, min(84, pitch))
                    vel = int(max(0, min(127, n.velocity)))
                    smoothed.append(pretty_midi.Note(
                        velocity=vel,
                        pitch=pitch,
                        start=float(n.start),
                        end=float(n.end)
                    ))
                    prev_pitch = pitch
                new_inst.notes.extend(smoothed)
                sanitized.instruments.append(new_inst)
                continue

            for n in inst.notes:
                pitch = int(max(0, min(127, n.pitch)))
                vel = int(max(0, min(127, n.velocity)))
                new_inst.notes.append(pretty_midi.Note(
                    velocity=vel,
                    pitch=pitch,
                    start=float(n.start),
                    end=float(n.end)
                ))
            sanitized.instruments.append(new_inst)
        return sanitized

    def _enforce_scale_and_grid(
        self,
        midi: pretty_midi.PrettyMIDI,
        scale: list,
        root: int,
        ticks_per_second: float,
        emotion: Optional[str] = None
    ) -> pretty_midi.PrettyMIDI:
        if emotion in ["calm", "sad"]:
            grid = 1.2 
            max_notes_per_second = 1
            min_duration = 0.35
            default_velocity_scale = 0.65
        else:
            grid = 0.5
            max_notes_per_second = 4
            min_duration = 0.22
            default_velocity_scale = 0.85
        new_midi = pretty_midi.PrettyMIDI(initial_tempo=midi.get_tempo_changes()[1][0])

        for inst in midi.instruments:
            velocity_scale = default_velocity_scale
            if emotion in ["calm", "sad"] and inst.program == 40:  
                velocity_scale = 0.65
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            recent = {}
            density_counter = {} 
            for n in inst.notes:
                start = round(n.start / grid) * grid
                end = max(start + 0.1, round(n.end / grid) * grid)

                if end - start < min_duration:
                    continue

                cell = round(start / grid)
                key = (cell, inst.program, inst.is_drum)
                seen = recent.setdefault(key, set())
                density_counter[key] = density_counter.get(key, 0) + 1
                if density_counter[key] > max_notes_per_second:
                    continue

                if not inst.is_drum:
                    relative = (n.pitch - root) % 12
                    closest = min(scale, key=lambda x: abs(x - relative))
                    pitch = root + ((n.pitch - root) // 12) * 12 + closest
                    pitch = max(28, min(96, pitch))
                else:
                    pitch = n.pitch

                if pitch in seen:
                    continue
                seen.add(pitch)

                new_inst.notes.append(pretty_midi.Note(
                    velocity=int(n.velocity * velocity_scale),
                    pitch=pitch,
                    start=start,
                    end=end
                ))
            new_midi.instruments.append(new_inst)

        return new_midi

    def _relaxation_filter(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """For calm moods, thin out notes, lengthen sustains, and soften velocities."""
        filtered = pretty_midi.PrettyMIDI(initial_tempo=midi.get_tempo_changes()[1][0])
        min_gap = 0.9
        sustain_min = 1.2
        sustain_max = 3.0

        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            last_start = -1e9
            for n in sorted(inst.notes, key=lambda x: x.start):
                if n.start - last_start < min_gap:
                    continue
                start = n.start
                end = max(n.end, start + sustain_min)
                end = min(end, start + sustain_max)

                if inst.program == 0: 
                    end = min(end + 0.8, start + sustain_max + 0.5)
                    vel = min(n.velocity, 68)
                else:
                    vel = min(n.velocity, 70 if not inst.is_drum else n.velocity)

                new_inst.notes.append(pretty_midi.Note(
                    velocity=vel,
                    pitch=n.pitch,
                    start=start,
                    end=end
                ))
                last_start = start
            filtered.instruments.append(new_inst)
        return filtered
