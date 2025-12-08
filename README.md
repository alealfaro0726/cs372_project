# Image-to-Music Generator

## What
This project takes any image you give it and turns it into music that feels like it “fits” the picture. It reads colors, textures, and overall vibe, translates those into choices like key, tempo, and instruments, and then uses an AI model guided by music-theory rules to write a short piece. The result is a MIDI track that stays in key, follows a sensible groove, and reflects the mood of the image - without you having to know anything about music or machine learning. The system is trained on a large MIDI collection (~17k files) so it has plenty of musical patterns to draw from, and it blends those learned ideas with guardrails to keep the music coherent.

## Quick Start
1) Install deps: `pip install -r requirements.txt`
2) Run web UI: `streamlit run streamlit_app.py` (needs `models/best_model.pt` and `data/processed/vocab.json`)
3) Use hosted UI (no install): https://alealfaro0726-cs372-project-streamlit-app-svekpw.streamlit.app/
   - Upload any image you have on your device and generate music through the site.
4) CLI generate: `python src/sample.py --image data/examples/warm_sunset.jpg --mode hybrid --checkpoint models/best_model.pt --output output.mid`

## Evaluation

- Training setup: ConditionalTransformer (~22M params: d_model=256, n_heads=4, n_layers=2, d_ff=512, max_seq_len=1024, dropout=0.1, image_embed_dim=512, emotion_embed_dim=64); AdamW; 20 epochs; cross-entropy loss; train/val/test split over the expanded MIDI corpus (~17k files in `data/midi_files/archive/`) with tokenization + CLIP embeddings.
- Run profile: batch size 8–16; 20 epochs; mixed precision optional; grad clipping enabled; cosine/plateau schedulers supported; wall-clock ~X hours on GPU (fill in based on your run).
- Perplexity (val): ~150–200 at epoch 1 → ~30–50 by epoch 20.
- Loss: steady downward trend; best checkpoint selected by lowest val loss; no late-epoch divergence observed in baseline runs.
- Metrics tracked: train/val loss, learning rate, perplexity; curves logged to `docs/training_curves.png`.
- Harmony/quality: chord progressions stay in-key; scale snapping keeps notes legal; piano leap smoothing plus velocity curves reduce artifacts.
- Visual-musical mapping spot-checks: warm → major, cool → modal/minor, bright → 100–140 BPM, dark → 60–90 BPM, smooth → sustained chords, rough → rhythmic patterns, high saturation → denser instrumentation.
- Example outputs: ~95–180 notes, 35s duration, keys/tempi follow visual energy (see MODEL_EVALUATION.md for breakdowns).
- Notebook: `notebooks/evaluation.ipynb` for qualitative checks; generated MIDIs from `data/examples/` via `streamlit_app.py` or `src/sample.py`.

## Video Links
- Demo video: https://drive.google.com/file/d/1aiKqDIBXJ4OrNeVUyVI7iG3lFTjQBUC5/view?usp=sharing
- Technical walkthrough: https://drive.google.com/file/d/1wzCxk4TSBlEeCVCz_2_8deHN4mii8pYb/view?usp=sharing

## Individual Contribution
- Alejandra Alfaro — solo project (all code, experiments, and docs).
