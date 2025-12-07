
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np


class MIDITokenizer:

    def __init__(
        self,
        min_pitch: int = 21,
        max_pitch: int = 108,
        n_velocity_bins: int = 32,
        n_time_bins: int = 32,
        ticks_per_bin: int = 4,
    ):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.n_velocity_bins = n_velocity_bins
        self.n_time_bins = n_time_bins
        self.ticks_per_bin = ticks_per_bin

        self.token_to_id, self.id_to_token = self._build_vocab()
        self.vocab_size = len(self.token_to_id)

        self.pad_id = self.token_to_id['PAD']
        self.bos_id = self.token_to_id['BOS']
        self.eos_id = self.token_to_id['EOS']

    def _build_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        tokens = []

        tokens.extend(['PAD', 'BOS', 'EOS'])

        for pitch in range(self.min_pitch, self.max_pitch + 1):
            tokens.append(f'NOTE_ON_{pitch}')

        for pitch in range(self.min_pitch, self.max_pitch + 1):
            tokens.append(f'NOTE_OFF_{pitch}')

        for i in range(self.n_time_bins):
            tokens.append(f'TIME_SHIFT_{i}')

        for i in range(self.n_velocity_bins):
            tokens.append(f'VELOCITY_{i}')

        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        id_to_token = {idx: token for token, idx in token_to_id.items()}

        return token_to_id, id_to_token

    def quantize_velocity(self, velocity: int) -> int:
        return min(int(velocity / 127.0 * self.n_velocity_bins), self.n_velocity_bins - 1)

    def dequantize_velocity(self, bin_idx: int) -> int:
        return int((bin_idx + 0.5) / self.n_velocity_bins * 127)

    def quantize_time(self, ticks: int) -> List[int]:
        bins = []
        remaining_ticks = ticks

        while remaining_ticks > 0:
            n_bins = min(remaining_ticks // self.ticks_per_bin, self.n_time_bins - 1)
            if n_bins > 0:
                bins.append(n_bins)
                remaining_ticks -= n_bins * self.ticks_per_bin
            else:
                if remaining_ticks > 0:
                    bins.append(1)
                break

        return bins if bins else [0]

    def encode_events(self, events: List[Tuple[str, int, int]]) -> List[int]:
        tokens = [self.bos_id]
        current_velocity = 64

        for event_type, pitch, velocity, time_delta in events:
            if time_delta > 0:
                time_bins = self.quantize_time(time_delta)
                for bin_idx in time_bins:
                    token_name = f'TIME_SHIFT_{bin_idx}'
                    if token_name in self.token_to_id:
                        tokens.append(self.token_to_id[token_name])

            if event_type == 'note_on' and velocity != current_velocity:
                vel_bin = self.quantize_velocity(velocity)
                token_name = f'VELOCITY_{vel_bin}'
                if token_name in self.token_to_id:
                    tokens.append(self.token_to_id[token_name])
                    current_velocity = velocity

            if self.min_pitch <= pitch <= self.max_pitch:
                if event_type == 'note_on':
                    token_name = f'NOTE_ON_{pitch}'
                elif event_type == 'note_off':
                    token_name = f'NOTE_OFF_{pitch}'
                else:
                    continue

                if token_name in self.token_to_id:
                    tokens.append(self.token_to_id[token_name])

        tokens.append(self.eos_id)
        return tokens

    def decode_tokens(self, token_ids: List[int]) -> List[Tuple[str, int, int, int]]:
        events = []
        current_time = 0
        current_velocity = 64
        accumulated_time = 0

        for token_id in token_ids:
            if token_id >= len(self.id_to_token):
                continue

            token = self.id_to_token[token_id]

            if token in ['PAD', 'BOS', 'EOS']:
                continue

            parts = token.split('_')
            token_type = parts[0]

            if token_type == 'TIME' and len(parts) == 3:
                bin_idx = int(parts[2])
                accumulated_time += bin_idx * self.ticks_per_bin

            elif token_type == 'VELOCITY':
                bin_idx = int(parts[1])
                current_velocity = self.dequantize_velocity(bin_idx)

            elif token_type == 'NOTE':
                event_type = parts[1].lower()
                pitch = int(parts[2]) if len(parts) > 2 else int('_'.join(parts[2:]))

                events.append((
                    f'note_{event_type}',
                    pitch,
                    current_velocity if event_type == 'on' else 0,
                    accumulated_time
                ))

                accumulated_time = 0

        return events

    def decode(self, token_ids: List[int]) -> List[Tuple[str, int, int, int]]:
        """Compatibility wrapper expected by older code paths."""
        return self.decode_tokens(token_ids)

    def save_vocab(self, path: str):
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'config': {
                'min_pitch': self.min_pitch,
                'max_pitch': self.max_pitch,
                'n_velocity_bins': self.n_velocity_bins,
                'n_time_bins': self.n_time_bins,
                'ticks_per_bin': self.ticks_per_bin,
                'vocab_size': self.vocab_size
            }
        }

        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

        print(f"Vocabulary saved to {path}")
        print(f"Vocabulary size: {self.vocab_size}")

    @classmethod
    def load_vocab(cls, path: str) -> 'MIDITokenizer':
        with open(path, 'r') as f:
            vocab_data = json.load(f)

        config = vocab_data['config']
        tokenizer = cls(
            min_pitch=config['min_pitch'],
            max_pitch=config['max_pitch'],
            n_velocity_bins=config['n_velocity_bins'],
            n_time_bins=config['n_time_bins'],
            ticks_per_bin=config['ticks_per_bin']
        )

        return tokenizer


def create_vocab(output_path: str = 'data/processed/vocab.json'):
    tokenizer = MIDITokenizer()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab(output_path)
    return tokenizer


if __name__ == '__main__':
    tokenizer = create_vocab()

    print("\n=== Example Tokenization ===")

    example_events = [
        ('note_on', 60, 80, 0),
        ('note_off', 60, 0, 120),
        ('note_on', 62, 80, 0),
        ('note_off', 62, 0, 120),
        ('note_on', 64, 80, 0),
        ('note_off', 64, 0, 120),
        ('note_on', 65, 80, 0),
        ('note_off', 65, 0, 120),
    ]

    token_ids = tokenizer.encode_events(example_events)
    print(f"\nEncoded {len(example_events)} events to {len(token_ids)} tokens")
    print(f"Token IDs: {token_ids[:10]}...")

    decoded_events = tokenizer.decode_tokens(token_ids)
    print(f"\nDecoded back to {len(decoded_events)} events")
    print("First few events:")
    for event in decoded_events[:4]:
        print(f"  {event}")
