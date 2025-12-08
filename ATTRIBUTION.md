# Credits

What this project uses and where it came from.

## AI assistance

Built with help from Claude (Anthropic) and Claude Code for:
- Debugging
- Architecture design
- Documentation

AI helped with:
- Project structure and boilerplate
- Music theory rules
- Streamlit UI
- Docs (README, SETUP, etc.)
- Refactoring

## Models

### CLIP
- From OpenAI
- Model: ViT-B/32
- Used for: Image feature extraction

### ResNet
- From Microsoft Research / torchvision
- Model: ResNet-50 (backup when CLIP not available)
- Used for: Image features

## Libraries

### Deep learning
- **PyTorch** - Neural networks 
- **torchvision** - Image stuff 
- **transformers** (Hugging Face) - Transformer utilities

### MIDI
- **pretty_midi** - MIDI generation
- **mido** - MIDI messages 
- **music21** - Music theory 

### Images
- **Pillow** - Image loading 
- **OpenCV** - Advanced image analysis
- **scikit-image** - Features

### Audio
- **soundfile** - Audio I/O
- **scipy** - Signal processing


### Web Interface
- **Streamlit** 
  - Purpose: Interactive web application framework

### Data Science and Utilities
- **NumPy**
  - Purpose: Numerical computations

- **pandas**
  - Purpose: Data manipulation

- **scikit-learn** 
  - Purpose: Machine learning utilities

- **matplotlib**
  - Purpose: Visualization

## Datasets

### MIDI Training Data
- **Source**: Public domain and creative commons MIDI files
- **Format**: Standard MIDI files (.mid)
- **Licensing**: Public domain / CC0 where applicable

No third-party code blocks were directly copied. All implementations are original or based on standard library usage following documentation.

## Originality Statement

While this project uses many external libraries and pretrained models (as listed above), the following components are original work:

1. **Architecture Design**: Hybrid model combining transformer with music theory
2. **Visual-to-Musical Mapping**: Custom algorithm mapping image features to music parameters
3. **Improved Composer**: Cohesive music composition engine with proper song structure
   - Chord progressions based on music theory (I-V-vi-IV patterns)
   - Song structure generation (intro, verse, chorus, outro)
   - Motif-based melody creation for musical consistency
   - Harmonically aligned multi-instrument arrangements
4. **Integration**: Novel combination of computer vision and music generation
5. **Web Interface**: Custom Streamlit application design
6. **Documentation**: All written documentation (README, SETUP, MODEL_EVALUATION, ATTRIBUTION)
