
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

import numpy as np
import pretty_midi
from tqdm import tqdm

from midi_tokenizer import MIDITokenizer, create_vocab


def midi_to_events(midi_path: str, ticks_per_beat: int = 480) -> List[Tuple[str, int, int, int]]:
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        warnings.warn(f"Could not load {midi_path}: {e}")
        return []

    events = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            start_tick = int(note.start * ticks_per_beat * 2)
            end_tick = int(note.end * ticks_per_beat * 2)

            events.append(('note_on', note.pitch, note.velocity, start_tick))
            events.append(('note_off', note.pitch, 0, end_tick))

    events.sort(key=lambda x: x[3])

    delta_events = []
    last_time = 0

    for event_type, pitch, velocity, abs_time in events:
        delta = abs_time - last_time
        delta_events.append((event_type, pitch, velocity, delta))
        last_time = abs_time

    return delta_events


def create_sample_midis(output_dir: Path, n_samples: int = 5):
    output_dir.mkdir(parents=True, exist_ok=True)

    scales = {
        'C_major': [60, 62, 64, 65, 67, 69, 71, 72],
        'A_minor': [69, 71, 72, 74, 76, 77, 79, 81],
        'G_major': [67, 69, 71, 72, 74, 76, 78, 79],
        'E_minor': [64, 66, 67, 69, 71, 72, 74, 76],
        'D_major': [62, 64, 66, 67, 69, 71, 73, 74],
    }

    for i, (scale_name, notes) in enumerate(scales.items()):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)

        current_time = 0.0
        duration = 0.5

        for pitch in notes:
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=current_time,
                end=current_time + duration
            )
            piano.notes.append(note)
            current_time += duration

        midi.instruments.append(piano)
        output_path = output_dir / f"sample_{scale_name}.mid"
        midi.write(str(output_path))

    print(f"Created {len(scales)} sample MIDI files in {output_dir}")


def process_midi_files(
    input_dir: Path,
    output_dir: Path,
    tokenizer: MIDITokenizer,
    max_length: int = 1024,
    min_length: int = 16
) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = (
        list(input_dir.glob('*.mid')) +
        list(input_dir.glob('*.midi')) +
        list(input_dir.glob('*.MID')) +
        list(input_dir.glob('**/*.mid')) +
        list(input_dir.glob('**/*.midi')) +
        list(input_dir.glob('**/*.MID'))
    )

    midi_files = list(set(midi_files))

    if not midi_files:
        print(f"No MIDI files found in {input_dir}")
        return []

    print(f"Found {len(midi_files)} MIDI files (including subdirectories)")

    processed_data = []

    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        try:
            events = midi_to_events(str(midi_path))

            if not events:
                continue

            tokens = tokenizer.encode_events(events)

            if len(tokens) < min_length:
                continue

            for start_idx in range(0, len(tokens), max_length - 100):
                chunk = tokens[start_idx:start_idx + max_length]

                if len(chunk) >= min_length:
                    processed_data.append({
                        'tokens': chunk,
                        'length': len(chunk),
                        'source_file': midi_path.name,
                        'chunk_index': start_idx // (max_length - 100)
                    })

        except Exception as e:
            warnings.warn(f"Error processing {midi_path}: {e}")

    return processed_data


def create_splits(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    np.random.seed(seed)

    indices = np.random.permutation(len(data))

    n_train = int(len(data) * train_ratio)
    n_val = int(len(data) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data


def save_splits(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        output_path = output_dir / f'{split_name}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(split_data, f)
        print(f"Saved {len(split_data)} sequences to {output_path}")

    metadata = {
        'n_train': len(train_data),
        'n_val': len(val_data),
        'n_test': len(test_data),
        'total': len(train_data) + len(val_data) + len(test_data)
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIDI files for training')
    parser.add_argument('--input_dir', type=str, default='data/midi_files',
                       help='Directory containing MIDI files')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--sample', action='store_true',
                       help='Create and use sample MIDI files')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--min_length', type=int, default=16,
                       help='Minimum sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.sample:
        print("Creating sample MIDI files...")
        sample_dir = Path('data/sample/midi')
        create_sample_midis(sample_dir)
        input_dir = sample_dir

    print("\nCreating vocabulary...")
    vocab_path = output_dir / 'vocab.json'
    tokenizer = create_vocab(str(vocab_path))

    print("\nProcessing MIDI files...")
    processed_data = process_midi_files(
        input_dir,
        output_dir,
        tokenizer,
        max_length=args.max_length,
        min_length=args.min_length
    )

    if not processed_data:
        print("No data to process!")
        return

    print(f"\nProcessed {len(processed_data)} sequences")

    print("\nCreating train/val/test splits...")
    train_data, val_data, test_data = create_splits(
        processed_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        seed=args.seed
    )

    save_splits(train_data, val_data, test_data, output_dir)

    print("\n=== Statistics ===")
    print(f"Total sequences: {len(processed_data)}")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")

    avg_length = np.mean([d['length'] for d in processed_data])
    print(f"Average sequence length: {avg_length:.1f}")

    print(f"\nData saved to: {output_dir}")


if __name__ == '__main__':
    main()
