
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
from tqdm import tqdm

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("CLIP not available, will use ResNet fallback")

from torchvision import models, transforms


EMOTIONS = ['happy', 'sad', 'calm', 'angry', 'surprised']


class ImageFeatureExtractor:

    def __init__(self, use_clip: bool = True, device: str = 'cpu'):
        self.device = device
        self.use_clip = use_clip and CLIP_AVAILABLE

        if self.use_clip:
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            self.embed_dim = 512
        else:
            print("Loading ResNet-50 model...")
            self.resnet = models.resnet50(pretrained=True).to(device)
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
            self.resnet.eval()
            self.embed_dim = 2048

            self.resnet_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def extract_handcrafted_features(self, image_path: str) -> Dict[str, float]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        brightness = img_gray.mean() / 255.0

        contrast = img_gray.std() / 255.0

        rg = img_rgb[:, :, 0].astype(float) - img_rgb[:, :, 1].astype(float)
        yb = 0.5 * (img_rgb[:, :, 0].astype(float) + img_rgb[:, :, 1].astype(float)) - img_rgb[:, :, 2].astype(float)
        colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        colorfulness = min(colorfulness / 100.0, 1.0)

        hue_values = img_hsv[:, :, 0]
        dominant_hue = float(np.mean(hue_values) * 2)

        edges = cv2.Canny(img_gray, 100, 200)
        edge_density = (edges > 0).sum() / edges.size

        has_face = 0.0
        smile_confidence = 0.0
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

            faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

            if len(faces) > 0:
                has_face = 1.0
                for (x, y, w, h) in faces:
                    roi_gray = img_gray[y:y+h, x:x+w]
                    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                    if len(smiles) > 0:
                        smile_confidence = min(len(smiles) / 2.0, 1.0)
                        break
        except:
            pass

        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'colorfulness': float(colorfulness),
            'dominant_hue': float(dominant_hue),
            'edge_density': float(edge_density),
            'has_face': float(has_face),
            'smile_confidence': float(smile_confidence)
        }

    def extract_deep_features(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')

        with torch.no_grad():
            if self.use_clip:
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.clip_model.get_image_features(**inputs)
                features = outputs.cpu().numpy()[0]
            else:
                img_tensor = self.resnet_transform(img).unsqueeze(0).to(self.device)
                features = self.resnet(img_tensor).squeeze().cpu().numpy()

        return features

    def extract_all_features(self, image_path: str) -> Dict:
        handcrafted = self.extract_handcrafted_features(image_path)
        deep_features = self.extract_deep_features(image_path)

        return {
            'handcrafted': handcrafted,
            'deep_features': deep_features.tolist(),
            'embed_dim': self.embed_dim
        }


class SimpleEmotionClassifier(nn.Module):

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, n_emotions: int = 5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_emotions)
        )

    def forward(self, x):
        return self.classifier(x)


def predict_emotion_heuristic(handcrafted_features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    brightness = handcrafted_features['brightness']
    contrast = handcrafted_features['contrast']
    colorfulness = handcrafted_features['colorfulness']
    edge_density = handcrafted_features['edge_density']
    has_face = handcrafted_features.get('has_face', 0.0)
    smile_confidence = handcrafted_features.get('smile_confidence', 0.0)

    scores = {}

    hue = handcrafted_features.get('dominant_hue', 0)

    warm_boost = 0.0
    if 30 <= hue <= 90:
        warm_boost = 1.0
    elif 90 < hue <= 140:
        warm_boost = 0.8

    cool_boost = 0.0
    if 180 <= hue <= 240:
        cool_boost = 1.0
    elif 150 <= hue < 180 or 240 < hue <= 270:
        cool_boost = 0.5

    scores['happy'] = brightness * 0.45 + colorfulness * 0.3 + warm_boost * 0.5

    face_uncertainty_penalty = 1.0
    if has_face == 0:
        face_uncertainty_penalty = 0.4

    if has_face > 0 and smile_confidence > 0:
        scores['happy'] += smile_confidence * 0.7
        face_uncertainty_penalty = 1.0

    scores['happy'] *= face_uncertainty_penalty

    if has_face > 0 or brightness < 0.6:
        if brightness > 0.6:
            scores['happy'] *= 1.5
        if brightness > 0.7:
            scores['happy'] *= 1.8
        if warm_boost > 0.5:
            scores['happy'] *= 1.5
        if brightness > 0.65 and warm_boost > 0.7:
            scores['happy'] *= 1.8
        if brightness > 0.75 and warm_boost >= 0.8:
            scores['happy'] *= 2.0
        if colorfulness > 0.4:
            scores['happy'] *= 1.3
    else:
        if brightness > 0.7 and warm_boost > 0.7:
            scores['happy'] *= 1.2

    scores['sad'] = (1 - brightness) * 0.4 + (1 - colorfulness) * 0.3 + cool_boost * 0.3

    if has_face > 0 and smile_confidence < 0.2:
        scores['sad'] += 0.2

    if has_face == 0 and brightness > 0.6:
        scores['sad'] *= 2.5

    calm_brightness = (1 - abs(brightness - 0.5))
    scores['calm'] = calm_brightness * 0.15 + (1 - edge_density) * 0.15 + cool_boost * 0.25 + (1 - contrast) * 0.15

    if brightness > 0.6:
        scores['calm'] *= 0.3
    if warm_boost > 0.3:
        scores['calm'] *= 0.3
    if brightness > 0.7 and warm_boost > 0.5:
        scores['calm'] *= 0.1
    if brightness > 0.75 and warm_boost > 0.7:
        scores['calm'] *= 0.05
    if colorfulness > 0.4:
        scores['calm'] *= 0.4
    if has_face == 0 and brightness > 0.5:
        scores['calm'] *= 1.8

    red_boost = 1.0 if (hue < 30 or hue > 330) else 0.0
    scores['angry'] = contrast * 0.4 + edge_density * 0.4 + red_boost * 0.2

    scores['surprised'] = colorfulness * 0.3 + edge_density * 0.3 + contrast * 0.3 + brightness * 0.1

    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    predicted_emotion = max(scores.items(), key=lambda x: x[1])[0]

    return predicted_emotion, scores


def process_image_sample(image_path: str, output_path: Optional[str] = None, device: str = 'cpu'):
    extractor = ImageFeatureExtractor(use_clip=CLIP_AVAILABLE, device=device)

    print(f"Processing: {image_path}")
    features = extractor.extract_all_features(image_path)

    emotion, emotion_scores = predict_emotion_heuristic(features['handcrafted'])
    features['emotion'] = emotion
    features['emotion_scores'] = emotion_scores

    print(f"\nHandcrafted Features:")
    for k, v in features['handcrafted'].items():
        print(f"  {k}: {v:.3f}")

    print(f"\nPredicted Emotion: {emotion}")
    print(f"Emotion Scores:")
    for emo, score in emotion_scores.items():
        print(f"  {emo}: {score:.3f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"\nFeatures saved to: {output_path}")

    return features


def main():
    parser = argparse.ArgumentParser(description='Extract image features and predict emotions')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--input_dir', type=str, default='data/images',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='data/processed/images',
                       help='Output directory for features')
    parser.add_argument('--sample', action='store_true',
                       help='Use bundled sample images')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device for inference')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    print(f"CLIP available: {CLIP_AVAILABLE}")

    if args.image:
        output_path = Path(args.output_dir) / (Path(args.image).stem + '_features.json')
        process_image_sample(args.image, str(output_path), device)
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"No images found in {input_dir}")
        if args.sample:
            print("\nTo use this with sample images, add some images to data/images/")
            print("Or test with a single image: python src/preprocess_images.py --image path/to/image.jpg")
        return

    print(f"Found {len(image_files)} images")

    extractor = ImageFeatureExtractor(use_clip=CLIP_AVAILABLE, device=device)

    all_features = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            features = extractor.extract_all_features(str(img_path))

            emotion, emotion_scores = predict_emotion_heuristic(features['handcrafted'])
            features['emotion'] = emotion
            features['emotion_scores'] = emotion_scores
            features['image_path'] = str(img_path)

            all_features.append(features)

            output_path = output_dir / (img_path.stem + '_features.json')
            with open(output_path, 'w') as f:
                json.dump(features, f, indent=2)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'n_images': len(all_features),
            'clip_used': CLIP_AVAILABLE,
            'embed_dim': extractor.embed_dim,
            'emotions': EMOTIONS,
            'images': [f['image_path'] for f in all_features]
        }, f, indent=2)

    print(f"\nProcessed {len(all_features)} images")
    print(f"Features saved to: {output_dir}")


if __name__ == '__main__':
    main()
