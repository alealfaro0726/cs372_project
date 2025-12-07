
import streamlit as st
import torch
import numpy as np
from pathlib import Path
import tempfile
import base64

import sys
sys.path.append('src')

from model import ConditionalTransformer
from midi_tokenizer import MIDITokenizer
from preprocess_images import ImageFeatureExtractor, predict_emotion_heuristic, EMOTIONS
from sample import sample_with_learned_model, sample_with_music_theory, sample_with_hybrid_model
from simple_visual_mapper import map_visual_to_musical_simple
import json

try:
    from midi_to_audio import midi_to_audio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


st.set_page_config(
    page_title="Image-to-Music Generator",
    layout="wide"
)

st.title("Image-to-Music Generator")
st.markdown("""
Upload an image and get music based on colors, textures, movement, and composition.

In depth analysis of visual features and maps them to musical elements.
""")

CHECKPOINT_PATH = "models/best_model.pt"
VOCAB_PATH = "data/processed/vocab.json"

@st.cache_resource
def load_model_and_tokenizer(device='cpu'):
    if not Path(CHECKPOINT_PATH).exists():
        st.error(f"Model checkpoint not found: {CHECKPOINT_PATH}")
        st.info("Please train a model first by running: python src/train.py")
        return None, None, None

    if not Path(VOCAB_PATH).exists():
        st.error(f"Vocabulary not found: {VOCAB_PATH}")
        st.info("Please run preprocessing first: python src/preprocess_midi.py")
        return None, None, None

    tokenizer = MIDITokenizer.load_vocab(VOCAB_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    model = ConditionalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_config.get('d_model', 256),
        n_heads=model_config.get('n_heads', 4),
        n_layers=model_config.get('n_layers', 2),
        d_ff=model_config.get('d_ff', 512),
        max_seq_len=model_config.get('max_seq_len', 1024),
        dropout=model_config.get('dropout', 0.1),
        image_embed_dim=model_config.get('image_embed_dim', 512),
        n_emotions=5,
        emotion_embed_dim=model_config.get('emotion_embed_dim', 64)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    feature_extractor = ImageFeatureExtractor(use_clip=True, device=device)

    return model, tokenizer, feature_extractor

@st.cache_resource
def load_feature_extractor_only(device='cpu'):
    return ImageFeatureExtractor(use_clip=True, device=device)

def get_midi_download_link(midi_file_path, filename="generated_music.mid"):
    with open(midi_file_path, "rb") as f:
        midi_bytes = f.read()
    b64 = base64.b64encode(midi_bytes).decode()
    return f'<a href="data:audio/midi;base64,{b64}" download="{filename}">Download MIDI File</a>'

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Image")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to generate music from"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name

        if st.button("Analyze Visual Features", use_container_width=True):
            with st.spinner("Analyzing visual attributes..."):
                try:
                    result_json = map_visual_to_musical_simple(image_path, variation_seed=42)
                    result = json.loads(result_json)
                    st.session_state['visual_analysis'] = result
                    st.session_state['image_path'] = image_path
                    st.success("Done!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

with col2:
    st.header("Visual Analysis")

    if 'visual_analysis' in st.session_state:
        result = st.session_state['visual_analysis']

        st.subheader("Musical Mapping")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Key/Mode", f"{result['key_suggestion']} {result['scale_mode']}")
        col_m2.metric("Tempo", f"{result['tempo_bpm']} BPM")
        col_m3.metric("Intensity", f"{result['intensity']:.2f}")

        st.markdown("**Instruments:**")
        st.write(", ".join(result['primary_instruments']))

        with st.expander("Color Analysis"):
            color_info = result['color_analysis']
            st.write(f"**Temperature:** {color_info['temperature']}")
            st.write(f"**Brightness:** {color_info['avg_brightness']:.2f}")
            st.write(f"**Saturation:** {color_info['avg_saturation']:.2f}")
            st.write(f"**Harmony:** {color_info['harmony_type']}")

        with st.expander("Texture & Movement"):
            st.write(f"**Texture:** {result['texture_analysis']['type']}")
            st.write(f"**Detail Level:** {result['texture_analysis']['detail_level']:.2f}")
            st.write(f"**Movement:** {result['movement_analysis']['type']}")
            st.write(f"**Energy:** {result['movement_analysis']['energy']:.2f}")

        with st.expander("Style & Atmosphere"):
            st.write(f"**Style:** {', '.join(result['style_tags'])}")
            st.write(f"**Atmosphere:** {', '.join(result['atmosphere_tags'])}")

        with st.expander("Rationale"):
            st.write(result['rationale'])

if 'visual_analysis' in st.session_state and 'image_path' in st.session_state:
    st.markdown("---")
    st.header("Generate Music")

    with st.expander("Advanced Settings"):
        max_length = st.slider("Max Length", 256, 2048, 1024)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.9, 0.1)
        top_k = st.slider("Top K", 0, 100, 40)
        top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.05)
        theory_weight = st.slider("Theory Weight", 0.0, 1.0, 0.3, 0.05)

    if st.button("Generate Music", type="primary", use_container_width=True):
        with st.spinner("Generating music..."):
            try:
                image_path = st.session_state['image_path']

                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'

                model, tokenizer, feature_extractor = load_model_and_tokenizer(device)

                if model is None or tokenizer is None:
                    st.error("Failed to load model and tokenizer")
                else:
                    midi, emotion, metadata = sample_with_hybrid_model(
                        image_path,
                        model,
                        tokenizer,
                        feature_extractor,
                        device,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        theory_weight=theory_weight,
                        target_duration=35.0
                    )

                output_dir = Path("samples/streamlit_outputs")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "generated.mid"
                midi.write(str(output_path))

                st.success("Music generated successfully!")

                n_notes = sum(len(inst.notes) for inst in midi.instruments)
                duration = midi.get_end_time()

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Notes", n_notes)
                col_b.metric("Duration", f"{duration:.1f}s")
                col_c.metric("Instruments", len(midi.instruments))

                if 'visual_analysis' in st.session_state:
                    visual = st.session_state['visual_analysis']
                    st.info(f"**Style:** {', '.join(visual['style_tags'][:3])} | **Atmosphere:** {', '.join(visual['atmosphere_tags'][:2])}")

                if AUDIO_AVAILABLE:
                    try:
                        audio_path = output_dir / "generated.wav"
                        midi_to_audio(str(output_path), str(audio_path))
                        st.audio(str(audio_path), format='audio/wav')
                    except Exception as audio_error:
                        st.warning(f"Audio playback unavailable: {str(audio_error)}")
                        st.info("Download the MIDI file below to listen")

                timestamp = st.session_state.get('visual_analysis', {}).get('key_suggestion', 'generated')
                st.markdown(
                    get_midi_download_link(output_path, f"{timestamp}_music.mid"),
                    unsafe_allow_html=True
                )

                with st.expander("📊 Generation Details"):
                    st.json(metadata)

            except Exception as e:
                st.error(f"Error generating music: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")
st.header("How it works")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.subheader("Visual analysis")
    st.markdown("""

1. **Color** - HSV values, warm vs cool, harmony
2. **Texture** - Smooth, rough, grainy, swirly
3. **Movement** - Static, flowing, turbulent
4. **Composition** - Symmetry, spirals, chaos, focal points
5. **Style** - Impressionist, surreal, abstract, realistic

Maps to musical choices:
- **Key/mode** (major/minor/modal)
- **Tempo** (40-160 BPM)
- **Instruments** (piano, strings, pads, bells, etc.)
- **Rhythm** (arpeggios, sustained notes, syncopation)
- **Structure** (A-B-A, verse-chorus, etc.)
- **Texture** (smooth, swirling, lo-fi)
""")
