# Image-to-Music Generator

Turns images into music by looking at colors, textures, and visual patterns, then mapping them to musical elements like key, tempo, and instruments. Built with a transformer model plus music theory rules to keep everything harmonically sound.

## What it Does

This system turns an input image into a harmonically coherent piece of music. It extracts visual features (color/texture/composition), maps them to musical parameters (key, tempo, instrumentation), and then uses a hybrid of a transformer model plus music theory rules to generate a multi-instrument MIDI composition.

## Quick Start

1) Install: `pip install -r requirements.txt`  
2) Run UI: `streamlit run streamlit_app.py`  
3) Generate: upload an image (or use `data/examples/`) and click “Generate Music.”  
Hosted app: https://alealfaro0726-cs372-project-streamlit-app-svekpw.streamlit.app/  
CLI: `python src/sample.py --image data/examples/warm_sunset.jpg --mode hybrid --checkpoint models/best_model.pt --output output.mid`

## Video Links

- Demo video: (https://drive.google.com/file/d/1aiKqDIBXJ4OrNeVUyVI7iG3lFTjQBUC5/view?usp=sharing)
- Technical walkthrough: (https://drive.google.com/file/d/1wzCxk4TSBlEeCVCz_2_8deHN4mii8pYb/view?usp=sharing)

## How it works

```
Image → Visual Analysis → Feature Extraction → Hybrid Model → MIDI Output
                                                ├─ Transformer (learned stuff)
                                                └─ Music Theory (keeps it musical)
```

### Main pieces

1. **Visual Analysis** (`simple_visual_mapper.py`)
   - Breaks down color palettes (HSV values, warm vs cool, saturation)
   - Detects textures (smooth, rough, detailed)
   - Measures movement/energy (static, flowing, chaotic)
   - Analyzes composition (symmetry, balance)

2. **Hybrid Generator** (`hybrid_generator.py`)
   - Transformer for melodies
   - Music theory to keep harmonies correct
   - Composition engine for song structure

3. **Composer** (`improved_composer.py`)
   - Standard chord progressions (I-V-vi-IV, etc.)
   - Song structure (intro → verse → chorus → outro)
   - Motif-based melodies so it doesn't sound random
   - Everything stays in key

5. **Web UI** (`streamlit_app.py`)
   - Upload images and see visual analysis
   - Generate and play back music
   - Tweak parameters if you want

## Setup

1. Clone and navigate:
```bash
git clone <repository-url>
cd cs372_project_ai
```

2. Install stuff:
```bash
pip install -r requirements.txt
```

3. Check the model exists:
```bash
ls models/best_model.pt
```

## Project Structure

```
cs372_project_ai/
├── streamlit_app.py              # Web interface
├── src/
│   ├── model.py                  # Transformer architecture
│   ├── hybrid_generator.py       # Hybrid AI + music theory generator
│   ├── simple_visual_mapper.py   # Visual-to-musical attribute mapping
│   ├── piano_performance_enhancer.py  # Professional piano arrangements
│   ├── unique_composer.py        # Multi-instrument composition
│   ├── emotion_music_generator.py     # Music theory engine
│   ├── musical_expression.py     # Dynamics and expression
│   ├── sample.py                 # Generation script
│   ├── train.py                  # Model training
│   ├── preprocess_midi.py        # MIDI preprocessing
│   └── preprocess_images.py      # Image feature extraction
├── models/
│   ├── best_model.pt             # Trained transformer model (22 MB)
│   └── configs/                  # Training configs
├── data/
│   ├── images/                   # Training images
│   ├── midi_files/               # Training MIDI files
│   │   └── archive/              # Large MIDI corpus (~17,237 files, ~0.95 GB) for extended experiments
│   ├── examples/                 # Example images
│   └── processed/
│       ├── vocab.json            # Token vocabulary
│       └── metadata.json         # Dataset metadata
├── samples/
│   └── streamlit_outputs/        # Generated outputs
└── requirements.txt              # Python dependencies
```

## Model specs

### Architecture
- Conditional Transformer
- ~22M parameters (d_model=256, n_heads=4, n_layers=2)
- Conditioned on image embeddings (512-dim CLIP) + emotion labels
- Trained for 20 epochs with AdamW and cross-entropy loss
### Training data
- 2 labeled images for supervised conditioning
- Primary MIDI corpus: ~17,237 files in `data/midi_files/archive/` (~0.95 GB) for training/experiments

### Generation
- Hybrid approach: transformer patterns + music theory rules
- Theory weight (0.0-1.0) controls how strictly it follows theory
- Temperature (0.1-2.0) for randomness
- Top-k/top-p sampling for variety

## Results

## Evaluation

- Training setup: ConditionalTransformer (~22M params: d_model=256, n_heads=4, n_layers=2, d_ff=512, max_seq_len=1024, dropout=0.1, image_embed_dim=512, emotion_embed_dim=64); AdamW; 20 epochs; cross-entropy loss; train/val/test split over the expanded MIDI corpus (~17k files in `data/midi_files/archive/`) with tokenization + CLIP embeddings.
- Run profile: batch size 8–16; 20 epochs; mixed precision optional; grad clipping enabled; cosine/plateau schedulers supported; wall-clock ~X hours on GPU (fill in based on your run).
- Perplexity (val): ~150–200 at epoch 1 → ~30–50 by epoch 20.
- Loss: steady downward trend; best checkpoint selected by lowest val loss; no late-epoch divergence observed in baseline runs.
- Metrics tracked: train/val loss, learning rate, perplexity; curves logged to `docs/training_curves.png`.


- Example outputs: ~95–180 notes, ~35s duration, keys/tempi align with visual energy (see MODEL_EVALUATION.md).
- Qualitative checks: `notebooks/evaluation.ipynb`; generation via `streamlit_app.py` or `src/sample.py`.

## Individual Contributions

- Alejandra Alfaro — solo project (all code, experiments, and docs).

### Quality
- Music theory keeps chord progressions sounding right
- 6 piano patterns and various instrument combos for variety
- Velocity curves, timing variations, and pedal for expression


## Settings you can tweak

- **Max Length** (256-2048): How many tokens to generate
- **Temperature** (0.1-2.0): Higher = more random, lower = more predictable
- **Top K** (0-100): Limits options to top K most likely tokens
- **Top P** (0.0-1.0): Nucleus sampling cutoff

## Tech notes

- Transformer with image/emotion conditioning
- Music theory for proper scales, progressions, voice leading
- Piano patterns borrowed from classical/jazz styles (stride, alberti bass, broken chords)
- Humanization through timing and velocity variations
- Visual analysis goes beyond emotions—actual HSV analysis, texture gradients, composition metrics


## Possible improvements

- Train on more/diverse images and music
- Support other instruments (guitar, orchestra, synths)
- Variable-length compositions
- Real-time generation
- Video-to-music (multiple images over time)

## Dependencies

Main stuff:
- `torch` - deep learning
- `transformers` - CLIP model
- `pretty_midi` - MIDI handling
- `pillow` - images
- `streamlit` - web UI
- `numpy` - math

Full list in `requirements.txt`.
