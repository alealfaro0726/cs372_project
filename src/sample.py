
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import pretty_midi

from model import ConditionalTransformer
from midi_tokenizer import MIDITokenizer
from preprocess_images import ImageFeatureExtractor, predict_emotion_heuristic, EMOTIONS
from emotion_music_generator import generate_emotion_music
from hybrid_generator import HybridMusicGenerator


def tokens_to_midi(
    tokens: List[int],
    tokenizer: MIDITokenizer,
    tempo_bpm: int = 120,
    emotion: str = 'happy'
) -> pretty_midi.PrettyMIDI:
    events = tokenizer.decode_tokens(tokens)

    if not events:
        return pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)

    from emotion_music_generator import EmotionMusicGenerator
    emotion_gen = EmotionMusicGenerator(emotion=emotion)
    instruments_config = emotion_gen.instruments

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)

    melody_program, melody_name = instruments_config['melody']
    melody_inst = pretty_midi.Instrument(program=melody_program, name=melody_name)

    active_notes = {}
    current_time = 0.0
    ticks_per_second = (tempo_bpm / 60.0) * tokenizer.ticks_per_bin

    for event_type, pitch, velocity, time_delta in events:
        current_time += time_delta / ticks_per_second

        if event_type == 'note_on':
            active_notes[pitch] = (current_time, velocity)

        elif event_type == 'note_off':
            if pitch in active_notes:
                start_time, note_velocity = active_notes[pitch]
                end_time = current_time

                if end_time > start_time:
                    note = pretty_midi.Note(
                        velocity=note_velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    melody_inst.notes.append(note)

                del active_notes[pitch]

    for pitch, (start_time, velocity) in active_notes.items():
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=current_time + 0.5
        )
        melody_inst.notes.append(note)

    midi.instruments.append(melody_inst)

    midi.instruments[0].notes.sort(key=lambda x: x.start)

    return midi


def sample_with_learned_model(
    image_path: str,
    model: ConditionalTransformer,
    tokenizer: MIDITokenizer,
    feature_extractor: ImageFeatureExtractor,
    device: str,
    max_length: int = 256,
    temperature: float = 0.9,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.92
) -> tuple[pretty_midi.PrettyMIDI, str, dict]:
    features = feature_extractor.extract_all_features(image_path)

    emotion, emotion_scores = predict_emotion_heuristic(features['handcrafted'])

    image_embeds = torch.tensor(features['deep_features'], dtype=torch.float32).unsqueeze(0).to(device)

    expected_dim = model.image_projection.in_features
    current_dim = image_embeds.shape[1]

    if current_dim != expected_dim:
        projection_layer = torch.nn.Linear(current_dim, expected_dim, bias=False).to(device)
        with torch.no_grad():
            chunk_size = current_dim // expected_dim
            for i in range(expected_dim):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < expected_dim - 1 else current_dim
                projection_layer.weight[i, start_idx:end_idx] = 1.0 / (end_idx - start_idx)
            image_embeds = projection_layer(image_embeds)

    emotion_label = torch.tensor([EMOTIONS.index(emotion)], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        tokens = model.generate(
            image_embeds=image_embeds,
            emotion_labels=emotion_label,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            bos_token=tokenizer.bos_id,
            eos_token=tokenizer.eos_id
        )

    generated_tokens = tokens[0].cpu().tolist()
    midi = tokens_to_midi(generated_tokens, tokenizer, tempo_bpm=120, emotion=emotion)
    from musical_expression import enhance_musical_expression
    midi = enhance_musical_expression(midi, emotion=emotion)

    metadata = {
        'emotion': emotion,
        'emotion_scores': emotion_scores,
        'handcrafted_features': features['handcrafted'],
        'mode': 'learned-transformer',
        'n_tokens': len(generated_tokens),
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'description': 'Generated with trained conditional transformer'
    }

    return midi, emotion, metadata


def sample_with_music_theory(
    image_path: str,
    feature_extractor: ImageFeatureExtractor,
    duration: float = 30.0
) -> tuple[pretty_midi.PrettyMIDI, str, dict]:
    features = feature_extractor.extract_all_features(image_path)

    emotion, emotion_scores = predict_emotion_heuristic(features['handcrafted'])
    midi = generate_emotion_music(emotion=emotion, duration=duration, root_note=60)

    metadata = {
        'emotion': emotion,
        'emotion_scores': emotion_scores,
        'handcrafted_features': features['handcrafted'],
        'mode': 'music-theory-based',
        'duration': duration,
        'description': 'High-quality music with proper melody, harmony, and bass'
    }

    return midi, emotion, metadata


def sample_with_hybrid_model(
    image_path: str,
    model: ConditionalTransformer,
    tokenizer: MIDITokenizer,
    feature_extractor: ImageFeatureExtractor,
    device: str,
    max_length: int = 128,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    theory_weight: float = 0.7,
    target_duration: float = 35.0
) -> tuple[pretty_midi.PrettyMIDI, str, dict]:
    features = feature_extractor.extract_all_features(image_path)

    emotion, emotion_scores = predict_emotion_heuristic(features['handcrafted'])

    image_embeds = torch.tensor(features['deep_features'], dtype=torch.float32).unsqueeze(0).to(device)

    expected_dim = model.image_projection.in_features
    current_dim = image_embeds.shape[1]

    if current_dim != expected_dim:
        projection_layer = torch.nn.Linear(current_dim, expected_dim, bias=False).to(device)
        with torch.no_grad():
            chunk_size = current_dim // expected_dim
            for i in range(expected_dim):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < expected_dim - 1 else current_dim
                projection_layer.weight[i, start_idx:end_idx] = 1.0 / (end_idx - start_idx)
            image_embeds = projection_layer(image_embeds)

    emotion_label = torch.tensor([EMOTIONS.index(emotion)], dtype=torch.long).to(device)

    hybrid_generator = HybridMusicGenerator(model, tokenizer, device)

    midi = hybrid_generator.generate(
        image_embeds=image_embeds,
        emotion_label=emotion_label,
        emotion_name=emotion,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_length=max_length,
        use_theory_guidance=True,
        theory_weight=theory_weight,
        target_duration=target_duration
    )

    metadata = {
        'emotion': emotion,
        'emotion_scores': emotion_scores,
        'handcrafted_features': features['handcrafted'],
        'mode': 'hybrid-learned-theory',
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'theory_weight': theory_weight,
        'description': f'Hybrid generation with {theory_weight*100:.0f}% music theory guidance'
    }

    return midi, emotion, metadata


def main():
    parser = argparse.ArgumentParser(description='Generate melodies from images')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--mode', type=str, default='learned',
                       choices=['learned', 'music_theory', 'hybrid'],
                       help='Generation mode: learned model, music-theory-based, or hybrid (learned+theory)')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint (for learned mode)')
    parser.add_argument('--vocab', type=str, default='data/processed/vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--output', type=str, default='samples/generated.mid',
                       help='Output MIDI file path')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum generation length (learned mode)')
    parser.add_argument('--n_bars', type=int, default=8,
                       help='Number of bars (rule-based mode)')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.92,
                       help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--theory_weight', type=float, default=0.7,
                       help='Music theory guidance weight for hybrid mode (0.0=pure learned, 1.0=heavy theory)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device for inference')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_extractor = ImageFeatureExtractor(use_clip=True, device=device)

    if args.mode == 'learned':
        if not args.checkpoint:
            print("Error: --checkpoint required for learned mode")
            return

        print(f"Loading tokenizer from {args.vocab}...")
        tokenizer = MIDITokenizer.load_vocab(args.vocab)

        print(f"Loading model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        config = checkpoint.get('config', {})
        model_config = config.get('model', {})

        model = ConditionalTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=model_config.get('d_model', 512),
            n_heads=model_config.get('n_heads', 8),
            n_layers=model_config.get('n_layers', 6),
            d_ff=model_config.get('d_ff', 2048),
            max_seq_len=model_config.get('max_seq_len', 1024),
            dropout=model_config.get('dropout', 0.1),
            image_embed_dim=model_config.get('image_embed_dim', 512),
            n_emotions=5,
            emotion_embed_dim=model_config.get('emotion_embed_dim', 64)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print("Model loaded successfully!")

        midi, emotion, metadata = sample_with_learned_model(
            args.image,
            model,
            tokenizer,
            feature_extractor,
            device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

    elif args.mode == 'music_theory':
        midi, emotion, metadata = sample_with_music_theory(
            args.image,
            feature_extractor,
            duration=30.0
        )

    elif args.mode == 'hybrid':
        if not args.checkpoint:
            print("Error: --checkpoint required for hybrid mode")
            return

        print(f"Loading tokenizer from {args.vocab}...")
        tokenizer = MIDITokenizer.load_vocab(args.vocab)

        print(f"Loading model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        config = checkpoint.get('config', {})
        model_config = config.get('model', {})

        model = ConditionalTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=model_config.get('d_model', 512),
            n_heads=model_config.get('n_heads', 8),
            n_layers=model_config.get('n_layers', 6),
            d_ff=model_config.get('d_ff', 2048),
            max_seq_len=model_config.get('max_seq_len', 1024),
            dropout=model_config.get('dropout', 0.1),
            image_embed_dim=model_config.get('image_embed_dim', 512),
            n_emotions=5,
            emotion_embed_dim=model_config.get('emotion_embed_dim', 64)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print("Model loaded successfully!")

        midi, emotion, metadata = sample_with_hybrid_model(
            args.image,
            model,
            tokenizer,
            feature_extractor,
            device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            theory_weight=args.theory_weight,
            target_duration=35.0
        )

    midi.write(str(output_path))
    print(f"\nSaved MIDI to: {output_path}")

    n_notes = sum(len(inst.notes) for inst in midi.instruments)
    duration = midi.get_end_time()

    print(f"\nGenerated melody:")
    print(f"  Emotion: {emotion}")
    print(f"  Notes: {n_notes}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Mode: {args.mode}")

    print(f"\nHandcrafted features:")
    for key, value in metadata['handcrafted_features'].items():
        print(f"  {key}: {value:.3f}")


if __name__ == '__main__':
    main()
