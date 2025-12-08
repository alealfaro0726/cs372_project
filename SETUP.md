# Setup

Access the hosted app directly here (no install needed): https://alealfaro0726-cs372-project-streamlit-app-svekpw.streamlit.app/

OR installation steps below:

## Install

### 1. Clone

```bash
git clone <repository-url>
cd cs372_project_ai
```

### 2. Install dependencies

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

## Stopping the Application

### Streamlit App:
Press `Ctrl+C` in the terminal where Streamlit is running

---
