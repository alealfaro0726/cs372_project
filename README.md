# Image-to-Music Generator

Turns images into music by looking at colors, textures, and visual patterns, then mapping them to musical elements like key, tempo, and instruments. Built with a transformer model plus music theory rules to keep everything harmonically sound.

## What it Does

This system turns an input image into a harmonically coherent piece of music. It extracts visual features (color/texture/composition), maps them to musical parameters (key, tempo, instrumentation), and then uses a hybrid of a transformer model plus music theory rules to generate a multi-instrument MIDI composition.

## Quick Start

1) Install: `pip install -r requirements.txt`  
2) Run UI: `streamlit run streamlit_app.py`  
3) Generate: upload an image (or use `data/examples/`) and click “Generate Music.”  
CLI: `python src/sample.py --image data/examples/warm_sunset.jpg --mode hybrid --checkpoint models/best_model.pt --output output.mid`

## Video Links

- Demo video: (link TBD)
- Technical walkthrough: (link TBD)

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

4. **Piano Enhancer** (`piano_performance_enhancer.py`)
   - 6 left-hand patterns (stride, alberti bass, broken chords, etc.)
   - Chord voicings that sound good
   - Timing/velocity humanization so it's not robotic
   - Pedal simulation

5. **Web UI** (`streamlit_app.py`)
   - Upload images and see visual analysis
   - Generate and play back music
   - Tweak parameters if you want

## Setup

Need Python 3.8+ and pip.

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
│   ├── examples/                 # Example images
│   └── processed/
│       ├── vocab.json            # Token vocabulary
│       └── metadata.json         # Dataset metadata
├── samples/
│   └── streamlit_outputs/        # Generated outputs
└── requirements.txt              # Python dependencies
```

## Model specs

### Training data
- 2 labeled images for training
- 233 MIDI files (classical and contemporary)
- Preprocessing via MIDI tokenization and CLIP feature extraction

### Architecture
- Conditional Transformer
- ~22M parameters (d_model=256, n_heads=4, n_layers=2)
- Conditioned on image embeddings (512-dim CLIP) + emotion labels
- Trained for 20 epochs with AdamW and cross-entropy loss

### Generation
- Hybrid approach: transformer patterns + music theory rules
- Theory weight (0.0-1.0) controls how strictly it follows theory
- Temperature (0.1-2.0) for randomness
- Top-k/top-p sampling for variety

## Results

## Evaluation

- Training curves (loss + LR schedule): see `docs/training_curves.png`.
- Qualitative outputs: generated MIDIs from `data/examples/` via `streamlit_app.py` or `src/sample.py`.
- Notebook: `notebooks/evaluation.ipynb` (load `models/best_model.pt`).

## Individual Contributions

- Alejandra Alfaro — solo project (all code, experiments, and docs).

### Quality
- Music theory keeps chord progressions sounding right
- 6 piano patterns and various instrument combos for variety
- Velocity curves, timing variations, and pedal for expression

### Visual to musical mapping

| What it sees | What it plays |
|--------------|---------------|
| Warm colors | Major keys (C, G) |
| Cool colors | Modal scales (Lydian, Phrygian) |
| Bright images | Faster tempo (100-140 BPM) |
| Dark images | Slower tempo (60-90 BPM) |
| Smooth textures | Sustained chords, gentle arpeggios |
| Rough textures | Syncopation, complex rhythms |
| High energy | Dense instrumentation, fast rhythm |
| Low energy | Minimal, sparse notes |

### Examples
Check `samples/streamlit_outputs/` for generated MIDI files.

## Settings you can tweak

- **Max Length** (256-2048): How many tokens to generate
- **Temperature** (0.1-2.0): Higher = more random, lower = more predictable
- **Top K** (0-100): Limits options to top K most likely tokens
- **Top P** (0.0-1.0): Nucleus sampling cutoff
- **Theory Weight** (0.0-1.0): How much to follow music theory vs learned patterns
  - 0.0 = all learned
  - 0.3 = balanced (default)
  - 1.0 = strict theory

## Tech notes

- Transformer with image/emotion conditioning
- Music theory for proper scales, progressions, voice leading
- Piano patterns borrowed from classical/jazz styles (stride, alberti bass, broken chords)
- Humanization through timing and velocity variations
- Visual analysis goes beyond emotions—actual HSV analysis, texture gradients, composition metrics

## Limitations

- Only trained on 2 images and 233 MIDI files (small dataset)
- Fixed 35-second outputs
- Mainly piano-focused

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

## Credits

- Transformer based on "Attention Is All You Need" (Vaswani et al., 2017)
- CLIP from OpenAI
- Music theory from standard Western harmony
- Piano patterns from classical/jazz styles

## License

CS 372 ML project
