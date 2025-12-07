# Credits

What this project uses and where it came from.

## AI assistance

Built with help from Claude (Anthropic) and Claude Code for:
- Debugging
- Architecture design
- Documentation


AI helped with:
- Project structure and boilerplate
- Music theory rules and piano patterns
- Streamlit UI
- Docs (README, SETUP, etc.)
- Refactoring

## Models

### CLIP
- From OpenAI
- Model: ViT-B/32
- Used for: Image feature extraction
- Paper: Radford et al. (2021) "Learning Transferable Visual Models From Natural Language Supervision"
- License: MIT
- https://github.com/openai/CLIP

### ResNet
- From Microsoft Research / torchvision
- Model: ResNet-50 (backup when CLIP not available)
- Used for: Image features
- Paper: He et al. (2016) "Deep Residual Learning for Image Recognition"
- License: BSD
- https://github.com/pytorch/vision

## Libraries

### Deep learning
- **PyTorch** - Neural networks (BSD-3)
- **torchvision** - Image stuff (BSD-3)
- **transformers** (Hugging Face) - Transformer utilities (Apache 2.0)

### MIDI
- **pretty_midi** - MIDI generation (MIT) by Colin Raffel
- **mido** - MIDI messages (MIT)
- **music21** - Music theory (BSD-3)

### Images
- **Pillow** - Image loading (HPND)
- **OpenCV** - Advanced image analysis (Apache 2.0)
- **scikit-image** - Features (BSD-3)

### Audio
- **soundfile** - Audio I/O (BSD-3)
- **scipy** - Signal processing (BSD-3)

- **FluidSynth**
  - Purpose: MIDI to audio synthesis
  - License: LGPL-2.1
  - Link: https://github.com/FluidSynth/fluidsynth

### Web Interface
- **Streamlit** (v1.25+)
  - Purpose: Interactive web application framework
  - License: Apache 2.0
  - Link: https://github.com/streamlit/streamlit

### Data Science and Utilities
- **NumPy** (v1.24+)
  - Purpose: Numerical computations
  - License: BSD-3-Clause
  - Link: https://github.com/numpy/numpy

- **pandas** (v2.0+)
  - Purpose: Data manipulation
  - License: BSD-3-Clause
  - Link: https://github.com/pandas-dev/pandas

- **scikit-learn** (v1.3+)
  - Purpose: Machine learning utilities
  - License: BSD-3-Clause
  - Link: https://github.com/scikit-learn/scikit-learn

- **matplotlib** (v3.7+)
  - Purpose: Visualization
  - License: PSF License
  - Link: https://github.com/matplotlib/matplotlib

## Music Theory and Algorithmic Composition

### Piano Accompaniment Patterns
The left-hand piano patterns are based on traditional music theory:

- **Alberti Bass**: Classical keyboard technique (18th century)
  - Reference: Named after Domenico Alberti
  - Implementation: Custom, inspired by classical music pedagogy

- **Stride Piano**: Jazz/ragtime style
  - Reference: Traditional stride piano technique (1920s-1930s)
  - Implementation: Custom, inspired by Fats Waller and Art Tatum

- **Walking Bass**: Jazz bass line technique
  - Reference: Traditional jazz pedagogy
  - Implementation: Custom, chord-tone based approach

### Chord Voicings
- **Major/Minor Voicings**: Based on standard music theory
  - Reference: Traditional Western music harmony
  - Source: Public domain music theory knowledge

### Scale Modes
- **Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian**
  - Reference: Traditional Greek modal system
  - Source: Public domain music theory

## Datasets

### MIDI Training Data
- **Source**: Public domain and creative commons MIDI files
- **Collection**: 233 classical and contemporary compositions
- **Format**: Standard MIDI files (.mid)
- **Licensing**: Public domain / CC0 where applicable

### Image Training Data
- **Source**: Custom created and public domain images
- **Count**: 2 emotion-labeled training images
- **Additional**: Example images in `data/examples/` directory (custom generated)

## Research Papers and Inspiration

### Transformers
- Vaswani, A., et al. (2017). "Attention Is All You Need"
  - Link: https://arxiv.org/abs/1706.03762

### Image-to-Music Generation
- General concept inspired by cross-modal generation research
- Custom implementation combining computer vision and music generation

### Music Generation
- Various works on algorithmic composition and procedural music generation
- Custom music theory implementation based on traditional harmony

## Educational Resources

- **CS 372 Course Materials** (Duke University)
  - Course lectures and assignments
  - Instructor: [Course Instructor Name]

- **PyTorch Tutorials**
  - Link: https://pytorch.org/tutorials/

- **Music21 Documentation**
  - Link: https://web.mit.edu/music21/doc/

## Code Snippets and Techniques

### MIDI Tokenization
- Inspired by music information retrieval research
- Custom implementation for this project

### Visual Feature Extraction
- Based on standard computer vision techniques
- HSV color space analysis: Standard image processing technique
- Edge detection: Sobel/gradient-based methods (standard CV)

### Humanization Techniques
- Timing variations: Inspired by MIDI performance research
- Velocity curves: Based on music performance studies

## Tools and Development Environment

- **Python** (3.8-3.11)
  - License: PSF License
  - Link: https://www.python.org/

- **Git** - Version control
  - License: GPLv2
  - Link: https://git-scm.com/

- **Visual Studio Code** / **PyCharm** - Development environment
  - Licenses: MIT / Proprietary
  - Links: https://code.visualstudio.com/ / https://www.jetbrains.com/pycharm/

## Third-Party Code

No significant third-party code blocks were directly copied. All implementations are original or based on standard library usage following documentation.

## Images and Media

### Example Images
- `warm_sunset.jpg` - Generated programmatically using PIL
- `cool_ocean.jpg` - Generated programmatically using PIL
- `abstract_energetic.jpg` - Generated programmatically using PIL
- All example images: Created by this project (no external sources)

## Special Thanks

- **OpenAI** - For CLIP model and research
- **Anthropic** - For Claude AI assistance
- **Hugging Face** - For transformers library
- **PyTorch Team** - For deep learning framework
- **Streamlit Team** - For web framework
- **Colin Raffel** - For pretty_midi library
- **Duke University CS 372 Staff** - For course instruction and guidance

## License Compliance

This project uses only libraries compatible with academic use and open-source projects. All dependencies are available under permissive licenses (MIT, BSD, Apache 2.0, etc.).

## Originality Statement

While this project uses many external libraries and pretrained models (as listed above), the following components are original work:

1. **Architecture Design**: Hybrid model combining transformer with music theory
2. **Visual-to-Musical Mapping**: Custom algorithm mapping image features to music parameters
3. **Improved Composer**: Cohesive music composition engine with proper song structure
   - Chord progressions based on music theory (I-V-vi-IV patterns)
   - Song structure generation (intro, verse, chorus, outro)
   - Motif-based melody creation for musical consistency
   - Harmonically aligned multi-instrument arrangements
4. **Piano Performance Enhancer**: Custom implementation of arrangement techniques
   - 6 left-hand accompaniment patterns (alberti bass, stride, walking bass, etc.)
   - Dynamic voicings and humanization
5. **Integration**: Novel combination of computer vision and music generation
6. **Web Interface**: Custom Streamlit application design
7. **Documentation**: All written documentation (README, SETUP, MODEL_EVALUATION, ATTRIBUTION)

## Contact

For questions about attributions or to report missing credits, please contact the project maintainer.

---

**Last Updated**: December 2025

If you notice any missing attributions or have questions about the use of any resource, please let us know.
