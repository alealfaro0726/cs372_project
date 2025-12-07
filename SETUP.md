# Setup

Quick guide to getting this running.

## What you need

### Hardware
- 8GB RAM minimum, 16GB better
- 2-5GB free disk space
- GPU optional but speeds things up (CUDA or M1/M2)

### Software
- Python 3.8+ (3.9-3.11 works best)
- macOS, Linux, or Windows
- FluidSynth optional (for audio playback)

## Install

### 1. Clone

```bash
git clone <repository-url>
cd cs372_project_ai
```

### 2. Virtual environment (recommended)

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For Apple Silicon (M1/M2):**
```bash
pip install torch torchvision
```

### Step 4: Verify Installation

```bash
python -c "import torch; import streamlit; import pretty_midi; print('All core dependencies installed successfully!')"
```

## Project Structure

After installation, your directory should look like this:

```
cs372_project_ai/
├── models/
│   ├── best_model.pt          # Trained model (22MB)
│   └── configs/               # Training configs
├── data/
│   ├── images/                # Training images
│   ├── midi_files/            # Training MIDI files
│   └── processed/
│       ├── vocab.json         # Token vocabulary
│       └── metadata.json      # Dataset metadata
├── data/examples/
│   ├── warm_sunset.jpg        # Example image 1
│   ├── cool_ocean.jpg         # Example image 2
│   └── abstract_energetic.jpg # Example image 3
├── src/                       # Source code
├── samples/
│   └── streamlit_outputs/     # Generated outputs
├── streamlit_app.py           # Main application
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── SETUP.md                   # This file
└── ATTRIBUTION.md             # Credits and attributions
```

## Running the Application

### Method 1: Streamlit Web Interface

1. **Start the application:**
```bash
streamlit run streamlit_app.py
```

2. **Access the app:**
- Open your browser to `http://localhost:8501`
- The app should open automatically

3. **Use the interface:**
- Upload an image (or use examples from `data/examples/` folder)
- Click "Analyze Visual Features"
- Review the visual analysis results
- Click "Generate Music"
- Listen to audio in-browser or download MIDI file

### Method 2: Command Line Generation

Generate music directly from command line:

```bash
python src/sample.py \
    --image data/examples/warm_sunset.jpg \
    --mode hybrid \
    --checkpoint models/best_model.pt \
    --output output.mid

# With custom parameters
python src/sample.py \
    --image data/examples/abstract_energetic.jpg \
    --mode hybrid \
    --checkpoint models/best_model.pt \
    --temperature 0.9 \
    --top_k 40 \
    --theory_weight 0.3 \
    --output output.mid
```

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Ensure you're in the virtual environment
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Issue: CUDA out of memory

**Solution:**
The model will automatically fall back to CPU. 


### Issue: "Vocabulary not found" error

**Solution:**
Ensure `data/processed/vocab.json` exists. This file should be included in the repository.

## Advanced Configuration

### Changing Model Settings

Edit `streamlit_app.py` to modify:
- Default temperature
- Top-k/top-p values
- Theory weight
- Generation length

### Training Your Own Model

```bash
# Preprocess MIDI data
python src/preprocess_midi.py

# Preprocess images
python src/preprocess_images.py

# Train model
python src/train.py --config models/configs/transformer_small.yaml
```

## Performance Optimization

### For Faster Generation:
1. Use GPU if available (CUDA or MPS)
2. Use music_theory mode (bypasses ML model)
3. Reduce max_length parameter
4. Lower temperature value

### For Better Quality:
1. Use hybrid mode with theory_weight=0.3
2. Increase max_length for longer pieces
3. Experiment with temperature (0.8-1.0)

## Stopping the Application

### Streamlit App:
Press `Ctrl+C` in the terminal where Streamlit is running

---
