
import numpy as np
from PIL import Image
import json
import colorsys


def map_visual_to_musical_simple(image_path, variation_seed=42):
    np.random.seed(variation_seed)

    img = Image.open(image_path)
    img = img.convert('RGB')
    img_array = np.array(img)

    img_small = img.resize((256, 256))
    img_small_array = np.array(img_small)

    pixels = img_small_array.reshape(-1, 3) / 255.0

    brightness = np.mean(pixels)

    hsv_pixels = []
    for r, g, b in pixels[::10]:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_pixels.append([h, s, v])
    hsv_pixels = np.array(hsv_pixels)

    avg_saturation = np.mean(hsv_pixels[:, 1])
    avg_hue = np.mean(hsv_pixels[:, 0])

    warm_score = np.mean(pixels[:, 0]) - np.mean(pixels[:, 2])
    if warm_score > 0.1:
        temperature = 'warm'
    elif warm_score < -0.1:
        temperature = 'cool'
    else:
        temperature = 'neutral'

    dominant_hues = []
    hue_bins = np.histogram(hsv_pixels[:, 0], bins=12)[0]
    top_hue_indices = np.argsort(hue_bins)[-3:]
    for idx in top_hue_indices:
        hue_name = ['red', 'orange', 'yellow', 'yellow-green', 'green', 'cyan',
                   'blue', 'blue-violet', 'violet', 'magenta', 'red-magenta', 'red'][idx]
        dominant_hues.append(hue_name)

    gray = np.mean(img_small_array, axis=2)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 2

    if edge_density < 5:
        texture_type = 'smooth'
        detail_level = 0.2
    elif edge_density < 15:
        texture_type = 'soft'
        detail_level = 0.5
    elif edge_density < 30:
        texture_type = 'textured'
        detail_level = 0.7
    else:
        texture_type = 'rough'
        detail_level = 0.9

    contrast = np.std(pixels)

    energy = min(1.0, (edge_density / 40.0 + contrast) / 2)

    if energy < 0.3:
        movement_type = 'static'
    elif energy < 0.5:
        movement_type = 'gentle'
    elif energy < 0.7:
        movement_type = 'flowing'
    else:
        movement_type = 'energetic'


    if brightness > 0.6 and temperature == 'warm':
        key = 'C'
        scale_mode = 'Ionian (Major)'
    elif brightness > 0.6 and temperature == 'cool':
        key = 'D'
        scale_mode = 'Lydian'
    elif brightness < 0.4 and temperature == 'warm':
        key = 'A'
        scale_mode = 'Aeolian (Minor)'
    elif brightness < 0.4 and temperature == 'cool':
        key = 'E'
        scale_mode = 'Phrygian'
    elif avg_saturation > 0.5:
        key = 'G'
        scale_mode = 'Mixolydian'
    else:
        key = 'D'
        scale_mode = 'Dorian'

    base_tempo = 90
    tempo_modifier = (energy - 0.5) * 60
    tempo_bpm = int(np.clip(base_tempo + tempo_modifier, 60, 140))

    intensity = min(1.0, (energy + contrast) / 2)

    if texture_type == 'smooth':
        instruments = ['Piano', 'Soft Pad', 'Strings']
    elif texture_type == 'soft':
        instruments = ['Piano', 'Violin', 'Flute']
    elif texture_type == 'textured':
        instruments = ['Piano', 'Strings', 'Bass', 'Brass']
    else:
        instruments = ['Piano', 'Strings', 'Bass', 'Clarinet']

    if movement_type == 'static':
        rhythm = 'sustained long notes'
    elif movement_type == 'gentle':
        rhythm = 'gentle arpeggios'
    elif movement_type == 'flowing':
        rhythm = 'flowing melody'
    else:
        rhythm = 'rhythmic syncopation'

    if contrast < 0.15:
        structure = 'minimal A-A'
    elif contrast < 0.25:
        structure = 'simple A-B-A'
    else:
        structure = 'dynamic verse-chorus'

    style_tags = []
    if brightness > 0.6:
        style_tags.append('bright')
    if avg_saturation > 0.5:
        style_tags.append('colorful')
    if energy > 0.6:
        style_tags.append('energetic')
    if contrast < 0.2:
        style_tags.append('minimalist')
    if texture_type in ['smooth', 'soft']:
        style_tags.append('ambient')

    if not style_tags:
        style_tags = ['balanced']

    atmosphere_tags = []
    if brightness > 0.6 and energy < 0.5:
        atmosphere_tags.append('peaceful')
    elif brightness > 0.6 and energy > 0.5:
        atmosphere_tags.append('uplifting')
    elif brightness < 0.4 and energy < 0.5:
        atmosphere_tags.append('contemplative')
    elif brightness < 0.4 and energy > 0.5:
        atmosphere_tags.append('intense')

    if avg_saturation > 0.5:
        atmosphere_tags.append('vibrant')

    if not atmosphere_tags:
        atmosphere_tags = ['neutral']

    semantic_elements = ['abstract visual composition']

    rationale = (
        f"The image has {temperature} colors with {brightness:.2f} brightness and "
        f"{avg_saturation:.2f} saturation. The {texture_type} texture suggests {rhythm}, "
        f"while the {movement_type} movement indicates {tempo_bpm} BPM tempo. "
        f"The {key} {scale_mode} scale matches the overall visual character."
    )

    result = {
        'key_suggestion': key,
        'scale_mode': scale_mode,
        'tempo_bpm': tempo_bpm,
        'intensity': round(intensity, 2),
        'primary_instruments': instruments,
        'rhythmic_movement': rhythm,
        'texture_type': structure,
        'structure': structure,
        'color_analysis': {
            'temperature': temperature,
            'avg_brightness': round(brightness, 2),
            'avg_saturation': round(avg_saturation, 2),
            'harmony_type': 'varied',
            'dominant_hues': dominant_hues
        },
        'texture_analysis': {
            'type': texture_type,
            'detail_level': round(detail_level, 2)
        },
        'movement_analysis': {
            'type': movement_type,
            'energy': round(energy, 2)
        },
        'composition': {
            'type': 'balanced',
            'symmetry': 0.5,
            'balance': 'centered'
        },
        'style_tags': style_tags,
        'atmosphere_tags': atmosphere_tags,
        'semantic_elements': semantic_elements,
        'rationale': rationale
    }

    return json.dumps(result, indent=2)
